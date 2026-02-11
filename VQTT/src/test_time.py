import copy
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from utils import pairwise_cosine_similarity
from tqdm.auto import tqdm
import torch.nn as nn
import utils



def test_time_scaling(agent, sample, message_length=4, n=10, sampling_temperature=1e-1):
    """
    Test-time scaling for a single sample.
    This function fine-tunes a clone of the agent's text generation GRU modules on a single sample.
    """
    agent_clone = copy.deepcopy(agent)
    
    agent_clone.eval()
    agent_clone.vq.train()
        
    img = sample.unsqueeze(0).to('cuda')  # Add batch dimension
    
    # Try different message lengths and return the one with highest similarity to image
    best_similarity = -float('inf')
    best_message = None
    best_discretized = None
    
    for _ in range(n):
        sender_result = agent_clone.forward_text_generation(
            img, 
            message_length=message_length, 
            freeze_codebook=True, 
            mode='discrete',
            sampling_temperature=sampling_temperature
        )

        text_repr = agent_clone.forward_text_perception(sender_result['discretized'])
                
        img_repr = agent_clone.forward_image_encoder(img)
        similarity = F.cosine_similarity(img_repr, text_repr).item()
        
        if similarity > best_similarity:
            best_similarity = similarity
            best_message = sender_result['indices']
            best_discretized = sender_result['discretized'].detach()
    
    del agent_clone
    return best_message, best_discretized



def test_time_sample_adaptation(agent, sample, num_iterations=100, lr=1e-4, sampling_temperature=1e-5, entropy_factor=0, length_message=4):
    """
    Test-time training for a single sample.
    This function fine-tunes a clone of the agent's text generation GRU modules on a single sample.
    """
    # Clone the agent to avoid modifying the original
    agent_clone = copy.deepcopy(agent)
    agent_clone.object_encoder.eval()

    optimizer = torch.optim.Adam([
        {'params': agent_clone.text_generation_gru.parameters()},
        {'params': agent_clone.text_generation_gru_head.parameters()}
    ], lr=lr)
    
    img = sample.unsqueeze(0).to('cuda')  # Add batch dimension
    
    for iter_num in range(num_iterations):
        optimizer.zero_grad()
        
        img_repr = agent_clone.forward_image_encoder(img)
        text_generation_result = agent_clone.forward_text_generation(
            img, message_length=length_message, 
            freeze_codebook=True, 
            mode='discrete', 
            sampling_temperature=sampling_temperature
        )

        text_repr = agent_clone.forward_text_perception(text_generation_result['discretized'])
        
        # Since we have only one sample, we need to create a contrastive objective differently
        # We can use the similarity between the image and text representations
        similarity = F.cosine_similarity(img_repr, text_repr)
        contrastive_loss = -similarity.mean()  # Maximize similarity
        
        # Add commit loss from VQ-VAE
        loss = contrastive_loss + text_generation_result['commit_loss']

        # Add entropy regularization if needed
        if entropy_factor > 0:
            entropy_loss = -Categorical(F.softmax(text_generation_result['words_logits'], dim=2)).entropy().mean()
            loss = loss + entropy_factor * entropy_loss
        
        loss.backward()
        optimizer.step()
    
    # Get the final message after training
    agent_clone.eval()
    sender_result = agent_clone.forward_text_generation(
        img, message_length=length_message, 
        freeze_codebook=True,
        mode='discrete', 
        sampling_temperature=sampling_temperature
    )

    return sender_result['indices'], sender_result['discretized'].detach()


def test_time_dataset_adaptation(
    agent, 
    test_loader, 
    lr, 
    num_epochs, 
    message_length, 
    sampling_temperature=1e-5, 
    entropy_factor=0, 
    contrastive_loss_temperature=0.1, 
    device="cuda",
    refrential=True,
):      
    
    agent_clone = agent
    agent_clone.object_encoder.eval()

    optimizer = torch.optim.Adam([
        {'params': agent_clone.object_encoder.parameters()},
        {'params': agent_clone.object_decoder.parameters()},
        {'params': agent_clone.text_generation_gru.parameters()},
        {'params': agent_clone.text_generation_gru_head.parameters()}
    ], lr=lr)
    freeze_codebook = True
    
    for epoch_num in range(num_epochs):
        progress_bar = tqdm(test_loader, desc=f'Training Progress Epoch {epoch_num}/{num_epochs}')
        
        total_task_loss = 0.0
        total_commit_loss = 0.0
        total_corrects = 0
        
        for iter_num, (imgs, labels) in enumerate(progress_bar):
            optimizer.zero_grad()
            imgs = imgs.to(device)
            # imgs: (B, 1, 28, 56)



            # Forward pass through agent
            img_repr, entropy = agent_clone.forward_image_encoder(imgs, message_length=message_length)
            text_generation_result = agent_clone.forward_text_generation(
                img_repr, 
                message_length=message_length, 
                freeze_codebook=freeze_codebook, 
                mode='discrete', 
                sampling_temperature=sampling_temperature
            )
            text_repr = agent_clone.forward_text_perception(text_generation_result['discretized'])
            
            if refrential:
                # Compute contrastive loss
                similarities_messages_to_objects = pairwise_cosine_similarity(text_repr, img_repr) / contrastive_loss_temperature
                task_loss = F.cross_entropy(
                    similarities_messages_to_objects, 
                    torch.arange(imgs.shape[0], device=device)
                )
            else:
                reconstructed_imgs, recons = agent_clone.forward_image_decoder(text_generation_result['discretized'], message_length=message_length, return_recons=True)
                criterion = nn.MSELoss()
                task_loss = criterion(reconstructed_imgs, imgs) + 1.0 * entropy
                
                # Total loss with commitment and entropy regularization
            loss = task_loss + text_generation_result['commit_loss']
            if entropy_factor > 0:
                entropy_loss = -Categorical(
                    F.softmax(text_generation_result['words_logits'], dim=2)
                ).entropy().mean()
                loss = loss + entropy_factor * entropy_loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update metrics
            total_task_loss += task_loss.item()
            total_commit_loss += text_generation_result['commit_loss'].item()
            
            # Compute accuracy
            predicted_labels = pairwise_cosine_similarity(text_repr, img_repr[:, 0, :]).argmax(1).cpu()
            total_corrects += (predicted_labels == torch.arange(imgs.shape[0])).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                task_loss=f"{total_task_loss / (iter_num + 1):.4f}",
                commit_loss=f"{total_commit_loss / (iter_num + 1):.4f}",
                acc=f"{total_corrects / ((iter_num + 1) * imgs.shape[0]):.4f}"
            )
            progress_bar.refresh()
        
        if epoch_num % 5 == 0:
            utils.show_recons(imgs, reconstructed_imgs, recons, idx=0)
            utils.show_recons(imgs, reconstructed_imgs, recons, idx=1)
            utils.show_recons(imgs, reconstructed_imgs, recons, idx=2)

def test_time_batch_adaptation(
    agent, 
    batch, 
    lr, 
    num_epochs, 
    message_length, 
    sampling_temperature=1e-5, 
    entropy_factor=0, 
    contrastive_loss_temperature=0.1, 
    device="cuda", 
    refrential=True,
):      
    
    agent_clone = copy.deepcopy(agent)
    agent_clone.object_encoder.eval()

    optimizer = torch.optim.Adam([
        {'params': agent_clone.text_generation_gru.parameters()},
        {'params': agent_clone.text_generation_gru_head.parameters()}
    ], lr=lr)
    
    for epoch_num in range(num_epochs):        
        optimizer.zero_grad()
        imgs = batch.to(device)
                
        # Forward pass through agent
        img_repr = agent_clone.forward_image_encoder(imgs)
        text_generation_result = agent_clone.forward_text_generation(
            imgs, 
            message_length=message_length, 
            freeze_codebook=True, 
            mode='discrete', 
            sampling_temperature=sampling_temperature
        )
        text_repr = agent_clone.forward_text_perception(text_generation_result['discretized'])
        
        if refrential:
            # Compute contrastive loss
            similarities_messages_to_objects = pairwise_cosine_similarity(text_repr, img_repr) / contrastive_loss_temperature
            contrastive_loss = F.cross_entropy(
                similarities_messages_to_objects, 
                torch.arange(imgs.shape[0], device=device)
            )

            # Total loss with commitment and entropy regularization
            loss = contrastive_loss + text_generation_result['commit_loss']
        else:
            reconstructed_imgs = agent_clone.forward_image_decoder(text_repr)
            criterion = nn.MSELoss()
            reconstruction_loss = criterion(reconstructed_imgs, imgs)
            
            # Total loss with commitment and entropy regularization
            loss = reconstruction_loss + text_generation_result['commit_loss']
            
        if entropy_factor > 0:
            entropy_loss = -Categorical(
                F.softmax(text_generation_result['words_logits'], dim=2)
            ).entropy().mean()
            loss = loss + entropy_factor * entropy_loss
        
        # Backward pass
        loss.backward()
        optimizer.step()        
    
    agent_clone.eval()
    sender_result = agent_clone.forward_text_generation(
        batch.to(device), message_length=message_length, 
        freeze_codebook=True,
        mode='discrete', 
        sampling_temperature=sampling_temperature
    )

    return sender_result['indices'], sender_result['discretized'].detach()
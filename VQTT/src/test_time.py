import copy
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from utils import pairwise_cosine_similarity
from tqdm.auto import tqdm
from utils import compute_contrastive_loss, compute_corrects


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
):      
    
    agent_clone = agent
    agent_clone.object_encoder.eval()

    optimizer = torch.optim.Adam([
        {'params': agent_clone.text_generation_gru.parameters()},
        {'params': agent_clone.text_generation_gru_head.parameters()}
    ], lr=lr)
    
    for epoch_num in range(num_epochs):
        progress_bar = tqdm(test_loader, desc=f'Training Progress Epoch {epoch_num}/{num_epochs}')
        
        total_contrastive_loss = 0.0
        total_commit_loss = 0.0
        total_corrects = 0
        
        for iter_num, (imgs, labels) in enumerate(progress_bar):
            optimizer.zero_grad()
            imgs = imgs.to(device)
            
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
            
            contrastive_loss = compute_contrastive_loss(text_repr, img_repr, contrastive_loss_temperature=contrastive_loss_temperature, idx=labels)
            
            # Total loss with commitment and entropy regularization
            loss = contrastive_loss + text_generation_result['commit_loss']
            if entropy_factor > 0:
                entropy_loss = -Categorical(
                    F.softmax(text_generation_result['words_logits'], dim=2)
                ).entropy().mean()
                loss = loss + entropy_factor * entropy_loss
            
            loss.backward()
            optimizer.step()
            
            total_contrastive_loss += contrastive_loss.item()
            total_commit_loss += text_generation_result['commit_loss'].item()
            
            corrects, total = compute_corrects(text_repr, img_repr, idx=labels)
            total_corrects += corrects
            
            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                contrastive_loss=f"{total_contrastive_loss / (iter_num + 1):.4f}",
                commit_loss=f"{total_commit_loss / (iter_num + 1):.4f}",
                acc=f"{total_corrects / ((iter_num + 1) * total):.4f}"
            )
            progress_bar.refresh()
            
def test_time_full_sender_dataset_adaptation(
    agent, 
    test_loader,
    lr, 
    num_epochs, 
    message_length, 
    sampling_temperature=1e-5, 
    entropy_factor=0, 
    contrastive_loss_temperature=0.1, 
    device="cuda", 
):      
    
    agent_clone = agent

    optimizer = torch.optim.Adam([
        {'params': agent_clone.parameters()},
    ], lr=lr)
    
    for epoch_num in range(num_epochs):
        progress_bar = tqdm(test_loader, desc=f'Training Progress Epoch {epoch_num}/{num_epochs}')
        
        total_contrastive_loss = 0.0
        total_commit_loss = 0.0
        total_corrects = 0
        
        for iter_num, (imgs, labels) in enumerate(progress_bar):
            optimizer.zero_grad()
            imgs = imgs.to(device)
            
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
            
            contrastive_loss = compute_contrastive_loss(text_repr, img_repr, contrastive_loss_temperature=contrastive_loss_temperature, idx=labels)
            
            # Total loss with commitment and entropy regularization
            loss = contrastive_loss + text_generation_result['commit_loss']
            if entropy_factor > 0:
                entropy_loss = -Categorical(
                    F.softmax(text_generation_result['words_logits'], dim=2)
                ).entropy().mean()
                loss = loss + entropy_factor * entropy_loss
            
            loss.backward()
            optimizer.step()
            
            total_contrastive_loss += contrastive_loss.item()
            total_commit_loss += text_generation_result['commit_loss'].item()
            
            corrects, total = compute_corrects(text_repr, img_repr, idx=labels)
            total_corrects += corrects
            
            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                contrastive_loss=f"{total_contrastive_loss / (iter_num + 1):.4f}",
                commit_loss=f"{total_commit_loss / (iter_num + 1):.4f}",
                acc=f"{total_corrects / ((iter_num + 1) * total):.4f}"
            )
            progress_bar.refresh()
         
   
def oracle_dataset_adaptation(
    agent,
    agent_b, 
    test_loader, 
    lr, 
    num_epochs, 
    message_length, 
    sampling_temperature=1e-5, 
    entropy_factor=0, 
    contrastive_loss_temperature=0.1, 
    device="cuda", 
):      
    
    agent_clone = agent
    agent_clone.object_encoder.eval()
    
    agent_clone_b = copy.deepcopy(agent_b)

    optimizer = torch.optim.Adam([
        {'params': agent_clone.text_generation_gru.parameters()},
        {'params': agent_clone.text_generation_gru_head.parameters()},
        # {'params': agent_b.text_perception_gru.parameters()},
        {'params': agent_clone_b.text_perception_gru_head.parameters()}
    ], lr=lr)
    
    for epoch_num in range(num_epochs):
        progress_bar = tqdm(test_loader, desc=f'Training Progress Epoch {epoch_num}/{num_epochs}')
        
        total_contrastive_loss = 0.0
        total_commit_loss = 0.0
        total_corrects = 0
        
        for iter_num, (imgs, labels) in enumerate(progress_bar):
            optimizer.zero_grad()
            imgs = imgs.to(device)
            
            # Forward pass through agent
            img_repr = agent_clone.forward_image_encoder(imgs)
            text_generation_result = agent_clone.forward_text_generation(
                imgs, 
                message_length=message_length, 
                freeze_codebook=True, 
                mode='discrete', 
                sampling_temperature=sampling_temperature
            )
            text_repr = agent_clone_b.forward_text_perception(text_generation_result['discretized'])
            
            contrastive_loss = compute_contrastive_loss(text_repr, img_repr, contrastive_loss_temperature=contrastive_loss_temperature, idx=labels)
            
            # Total loss with commitment and entropy regularization
            loss = contrastive_loss + text_generation_result['commit_loss']
            if entropy_factor > 0:
                entropy_loss = -Categorical(
                    F.softmax(text_generation_result['words_logits'], dim=2)
                ).entropy().mean()
                loss = loss + entropy_factor * entropy_loss
            
            loss.backward()
            optimizer.step()
            
            total_contrastive_loss += contrastive_loss.item()
            total_commit_loss += text_generation_result['commit_loss'].item()
            
            corrects, total = compute_corrects(text_repr, img_repr, idx=labels)
            total_corrects += corrects
            
            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                contrastive_loss=f"{total_contrastive_loss / (iter_num + 1):.4f}",
                commit_loss=f"{total_commit_loss / (iter_num + 1):.4f}",
                acc=f"{total_corrects / ((iter_num + 1) * total):.4f}"
            )
            progress_bar.refresh()

            
def test_time_batch_adaptation(
    agent, 
    batch, 
    lr, 
    num_epochs, 
    message_length, 
    labels=None,
    sampling_temperature=1e-5, 
    entropy_factor=0, 
    contrastive_loss_temperature=0.1, 
    device="cuda", 
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
        
        contrastive_loss = compute_contrastive_loss(text_repr, img_repr, contrastive_loss_temperature=contrastive_loss_temperature, idx=labels)
        
        # Total loss with commitment and entropy regularization
        loss = contrastive_loss + text_generation_result['commit_loss']
        if entropy_factor > 0:
            entropy_loss = -Categorical(
                F.softmax(text_generation_result['words_logits'], dim=2)
            ).entropy().mean()
            loss = loss + entropy_factor * entropy_loss
        
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


def test_time_full_sender_batch_adaptation(
    agent, 
    batch, 
    lr, 
    num_epochs, 
    message_length, 
    labels=None,
    sampling_temperature=1e-5, 
    entropy_factor=0, 
    contrastive_loss_temperature=0.1, 
    device="cuda", 
):      
    
    agent_clone = copy.deepcopy(agent)
    # agent_clone.object_encoder.eval()

    optimizer = torch.optim.Adam([
        {'params': agent_clone.parameters()},
    ], lr=lr)
    
    for epoch_num in range(num_epochs):        
        optimizer.zero_grad()
        imgs = batch.to(device)
                
        img_repr = agent_clone.forward_image_encoder(imgs)
        text_generation_result = agent_clone.forward_text_generation(
            imgs, 
            message_length=message_length, 
            freeze_codebook=True, 
            mode='discrete', 
            sampling_temperature=sampling_temperature
        )
        text_repr = agent_clone.forward_text_perception(text_generation_result['discretized'])
        
        contrastive_loss = compute_contrastive_loss(text_repr, img_repr, contrastive_loss_temperature=contrastive_loss_temperature, idx=labels)
        
        # Total loss with commitment and entropy regularization
        loss = contrastive_loss + text_generation_result['commit_loss']
        if entropy_factor > 0:
            entropy_loss = -Categorical(
                F.softmax(text_generation_result['words_logits'], dim=2)
            ).entropy().mean()
            loss = loss + entropy_factor * entropy_loss
        
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

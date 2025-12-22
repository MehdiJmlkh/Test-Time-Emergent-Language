import copy
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

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



def test_time_adaptation(agent, sample, num_iterations=100, lr=1e-4, sampling_temperature=1e-5, entropy_factor=0, length_message=4):
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

import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from abc import ABC, abstractmethod
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import torch
import pickle


def set_seed(seed):
    """
    Set random seeds for reproducible results across all libraries and hardware.
    
    Args:
        seed (int): Random seed value to use for all random number generators
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    # Ensure deterministic behavior for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pairwise_cosine_similarity(x1, x2):
    """
    Compute pairwise cosine similarity between two sets of vectors.
    
    Args:
        x1: First set of vectors [N, D]
        x2: Second set of vectors [M, D]
        
        Similarity matrix [N, M] where entry (i,j) is cosine similarity between x1[i] and x2[j]
    """
    # Normalize vectors to unit length
    x1_norm = F.normalize(x1, p=2, dim=1)
    x2_norm = F.normalize(x2, p=2, dim=1)
    
    # Compute cosine similarity via matrix multiplication
    similarity_matrix = torch.mm(x1_norm, x2_norm.t())
    
    return similarity_matrix


def evaluate_self_communicate(agent, test_dataset, device, message_length, number_of_candidates=100, batch_sampler=None, collate_fn=None):
    """Evaluate agent's ability to match images with their emergent language representations"""
    total_correct_matches = 0
    total_samples = 0

    # Create test data loader
    if batch_sampler == None:
        test_loader = DataLoader(test_dataset, batch_size=number_of_candidates, shuffle=False, num_workers=0, collate_fn=collate_fn)
    else:
        test_loader = DataLoader(test_dataset, batch_sampler=batch_sampler, collate_fn=collate_fn)

    # Evaluate matching accuracy
    for images, true_labels in tqdm(test_loader, desc="Evaluating image-text matching"):
        images = images.to(device)
        
        # Get image representations
        image_representations = agent.forward_image_encoder(images)
        
        # Generate emergent language for images
        text_generation_result = agent.forward_text_generation(
            images, 
            message_length=message_length, 
            freeze_codebook=True, 
            mode='discrete', 
            sampling_temperature=1e-5
        )
        
        # Get text representations from generated language
        text_representations = agent.forward_text_perception(text_generation_result['discretized'])
        
        # Find best matches between text and image representations
        similarity_matrix = pairwise_cosine_similarity(text_representations, image_representations)
        predicted_matches = similarity_matrix.argmax(dim=1).cpu()
        
        # Count correct matches (diagonal should be highest)
        correct_matches = (predicted_matches == torch.arange(images.shape[0])).sum().item()
        total_correct_matches += correct_matches
        total_samples += len(images)

    # Calculate and display accuracy
    matching_accuracy = total_correct_matches / total_samples
    print(f'Image-text matching accuracy: {matching_accuracy:.3f}')
    return matching_accuracy


def evaluate_cross_communicate(agent_a, agent_b, test_dataset, device, message_length, batch_size=100, num_workers=0, batch_sampler=None, collate_fn=None):
    """
    Evaluate test accuracy of the emergent language communication system.
    
    Works with both BaselineAgent and VQELAgent thanks to standardized interface.
    
    Args:
        agent_a: Sender agent (BaselineAgent or VQELAgent)
        agent_b: Receiver agent (BaselineAgent or VQELAgent)
        test_dataset: Test dataset
        device: Device to run evaluation on
        message_length: Length of generated messages
        batch_size: Batch size for evaluation (default: 100)
        num_workers: Number of workers for data loading (default: 0)
    
    Returns:
        test_accuracy: Test accuracy as a float
    """
    if batch_sampler == None:
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    else:
        test_loader = DataLoader(test_dataset, batch_sampler=batch_sampler, collate_fn=collate_fn)

    total_correct = 0
    total_samples = 0

    agent_a.eval()
    agent_b.eval()

    with torch.no_grad():
        for iter_num, (imgs, labels) in enumerate(tqdm(test_loader, desc="Evaluating")):
            imgs = imgs.to(device)
            
            # Generate messages from images using agent A
            generated_output = agent_a.forward_text_generation(
                imgs, 
                message_length=message_length, 
                freeze_codebook=True, 
                mode='discrete', 
                sampling_temperature=1e-5
            )
            words = generated_output['indices']
            
            # Process messages and images through agent B
            listener_messages_repr = agent_b.forward_external_text_perception(words).squeeze(1)
            listener_objects_repr = agent_b.forward_image_encoder(imgs)
            
            # Calculate similarity matrix between messages and objects
            similarities_messages_to_objects = pairwise_cosine_similarity(
                listener_messages_repr, 
                listener_objects_repr
            )

            # Count correct matches (diagonal should be highest)
            batch_size_actual = imgs.shape[0]
            predicted_matches = similarities_messages_to_objects.argmax(dim=-1)
            correct_matches = torch.arange(batch_size_actual, device=device)
            
            total_correct += (predicted_matches == correct_matches).sum().item()
            total_samples += batch_size_actual

    agent_a.train()
    agent_b.train()

    test_accuracy = total_correct / total_samples
    return test_accuracy


def load_dataset(path):
    with open(os.path.join(path, "data.pkl"), "rb") as f:
        return pickle.load(f)
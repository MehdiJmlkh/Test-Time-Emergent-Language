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
from numpy.linalg import norm


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


def compute_contrastive_loss(msg_repr, img_repr, contrastive_loss_temperature, idx=None, device='cuda'):
    if isinstance(idx, torch.Tensor):
        similarities_messages_to_objects = pairwise_cosine_similarity(msg_repr, img_repr) / contrastive_loss_temperature 
        contrastive_loss = F.cross_entropy( 
            similarities_messages_to_objects, torch.arange(img_repr.shape[0], device=device)
        )

    else:
        msg_i = msg_repr[idx].unsqueeze(0)

        similarities = pairwise_cosine_similarity(msg_i, img_repr) / contrastive_loss_temperature

        target = torch.tensor([idx], device=device)
            
        contrastive_loss = F.cross_entropy(
            similarities,
            target
        )
    return contrastive_loss

def compute_corrects(msg_repr, img_repr, idx=None):
    if isinstance(idx, torch.Tensor):
        predicted_labels = pairwise_cosine_similarity(
            msg_repr, img_repr
        ).argmax(1).cpu()

        corrects = (
            predicted_labels == torch.arange(img_repr.shape[0])
        ).sum().item()

        total = img_repr.shape[0]

    else:
        msg_i = msg_repr[idx].unsqueeze(0)
        
        similarities = pairwise_cosine_similarity(
            msg_i, img_repr
        )
        predicted_label = similarities.argmax(dim=1).item()
        corrects = int(predicted_label == idx)
        total = 1

    return corrects, total

        
def compute_rewards(msg_repr, img_repr, contrastive_loss_temperature, idx=None, device='cuda'):
    if isinstance(idx, torch.Tensor):
        similarities_messages_to_objects = pairwise_cosine_similarity(msg_repr, img_repr) / contrastive_loss_temperature
        rewards = -1 * F.cross_entropy(similarities_messages_to_objects, torch.arange(0, img_repr.shape[0]).to(device), reduction='none').detach()
    else:
        text_i = msg_repr[idx].unsqueeze(0)
        similarities = pairwise_cosine_similarity(
            text_i, 
            img_repr
        ) / contrastive_loss_temperature
        target = torch.tensor([idx], device=device)
        rewards = -1 * F.cross_entropy(similarities, target, reduction='none').detach()
    
    return rewards

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
    for images, labels in tqdm(test_loader, desc="Evaluating image-text matching"):
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
     
        # Count correct matches (diagonal should be highest)
        correct_matches, total = compute_corrects(text_representations, image_representations, idx=labels)
        total_correct_matches += correct_matches
        total_samples += total

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
            
            corrects, total = compute_corrects(listener_messages_repr, listener_objects_repr, idx=labels)
            
            total_correct += corrects
            total_samples += total

    agent_a.train()
    agent_b.train()

    test_accuracy = total_correct / total_samples
    return test_accuracy

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader


def center_gram(X):
    """
    Center the Gram matrix: K = X X^T, then apply double-centering.
    For linear CKA, we can just center X directly instead.
    """
    # Not used in the simplest linear CKA formula below,
    # but kept for completeness if you want kernel CKA later.
    n = X.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ X @ X.T @ H

def linear_cka(X, Y):
    """
    Compute linear CKA between two representations X and Y.
    X: shape (n_samples, d_x)
    Y: shape (n_samples, d_y)
    """
    # Center features (subtract mean over samples)
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    # Compute cross-covariance-like matrix
    dot_product = Y.T @ X  # shape (d_y, d_x)

    # Frobenius norms
    numerator = np.linalg.norm(dot_product, "fro") ** 2
    denom_x = np.linalg.norm(X.T @ X, "fro")
    denom_y = np.linalg.norm(Y.T @ Y, "fro")

    # To avoid division by zero
    if denom_x == 0 or denom_y == 0:
        return 0.0

    return numerator / (denom_x * denom_y)

def evaluate_cosine_sim_between_text_perceptions(agent_a, agent_b, test_dataset, device, message_length, batch_size=100, num_workers=0, batch_sampler=None, collate_fn=None, num_permutations=1000):
    if batch_sampler is None:
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    else:
        test_loader = DataLoader(test_dataset, batch_sampler=batch_sampler, collate_fn=collate_fn)

    agent_a.eval()
    agent_b.eval()

    # Store all representations to compute global significance at the end
    all_repr_a = []
    all_repr_b = []
    
    with torch.no_grad():
        for iter_num, (imgs, labels) in enumerate(tqdm(test_loader, desc="Evaluating")):
            imgs = imgs.to(device)
            
            text_generation_result = agent_a.forward_text_generation(
                imgs, 
                message_length=message_length, 
                freeze_codebook=True, 
                mode='discrete', 
                sampling_temperature=1e-5
            )
            
            # Squeeze and extract features
            messages_repr_a = agent_b.forward_external_text_perception(text_generation_result['indices']).squeeze(1)
            messages_repr_b = agent_a.forward_text_perception(text_generation_result['discretized'])
            
            # Append batch to our lists (keep them on CPU to save GPU memory if dataset is large)
            all_repr_a.append(messages_repr_a.cpu())
            all_repr_b.append(messages_repr_b.cpu())

    agent_a.train()
    agent_b.train()

    # Concatenate all batches into two large tensors: shape (Total_Samples, Embedding_Dim)
    tensor_a = torch.cat(all_repr_a, dim=0)
    tensor_b = torch.cat(all_repr_b, dim=0)
    
    num_samples = tensor_a.size(0)

    observed_sim = linear_cka(tensor_a, tensor_b)

    # 2. Compute Baseline and p-value via Permutation Test
    print(f"Running {num_permutations} permutations for p-value calculation...")
    random_sims = []
    
    for _ in tqdm(range(num_permutations), desc="Permutations"):
        # Shuffle the indices of tensor_b to break the alignment
        shuffled_indices = torch.randperm(num_samples)
        shuffled_tensor_b = tensor_b[shuffled_indices]
        
        # Compute similarity of the mismatched pairs
        random_sims.append(linear_cka(tensor_a, shuffled_tensor_b))
        
    random_sims = np.array(random_sims)
    
    # Baseline statistics
    baseline_mean = np.mean(random_sims)
    baseline_std = np.std(random_sims)
    
    # Calculate empirical p-value: (Count of random means >= observed mean + 1) / (N + 1)
    # The +1 is a standard statistical correction to avoid p=0.
    extreme_count = np.sum(random_sims >= observed_sim)
    p_value = (extreme_count + 1) / (num_permutations + 1)

    print("\n--- Significance Report ---")
    print(f"Observed similarity: {observed_sim:.4f}")
    print(f"Random Baseline Similarity: {baseline_mean:.4f} ± {baseline_std:.4f}")
    print(f"p-value: {p_value:.5f}")

    results = {
        "observed_similarity": observed_sim,
        "baseline_mean": baseline_mean,
        "baseline_std": baseline_std,
        "p_value": p_value
    }

    return results


def load_dataset(path):
    with open(os.path.join(path, "data.pkl"), "rb") as f:
        return pickle.load(f)
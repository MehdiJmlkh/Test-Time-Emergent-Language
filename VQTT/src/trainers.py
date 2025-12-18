from torch.distributions.categorical import Categorical
from collections import deque
from torch.utils.tensorboard import SummaryWriter
import copy
import os
import logging
from typing import Tuple, Dict, Optional, Any
import numpy as np
from models import AbstractAgent
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from tqdm.auto import tqdm
import torch.nn.functional as F
from utils import pairwise_cosine_similarity
from torchvision import transforms
from PIL import Image
import numpy as np

def train_agents_baseline_reinforce(
    agent_a: AbstractAgent, 
    agent_b: AbstractAgent, 
    train_loader: DataLoader[Any], 
    val_loader: DataLoader[Any], 
    optimizer: torch.optim.Optimizer, 
    lr_scheduler: Any, 
    device: torch.device,
    num_epochs: int = 10, 
    message_length: int = 4,
    sampling_temperature: float = 1e0, 
    entropy_regularization_factor: float = 0e0, 
    contrastive_loss_temperature: float = 0.1,
    ckpt_dir: str = "checkpoints", 
    tensorboard_writer: Optional[SummaryWriter] = None, 
    logger: Optional[logging.Logger] = None
) -> Tuple[float, Optional[Dict[str, Any]]]:
    """
    Train the agents using REINFORCE algorithm.
    
    Args:
        agent_a: Sender agent
        agent_b: Listener agent
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer for both agents
        lr_scheduler: Learning rate scheduler
        device: Device to run training on
        num_epochs: Number of training epochs
        message_length: Length of generated messages
        sampling_temperature: Temperature for sampling during text generation
        entropy_regularization_factor: Factor for entropy regularization
        contrastive_loss_temperature: Temperature for contrastive loss
        ckpt_dir: Directory to save checkpoints
        tensorboard_writer: TensorBoard writer (optional)
        logger: Logger instance (optional)
    
    Returns:
        best_val_acc: Best validation accuracy achieved
        best_model_state: State dict of the best model
    """
    if tensorboard_writer is None:
        tensorboard_writer = SummaryWriter('runs/baseline')
    
    if logger is None:
        logger = logging.getLogger(__name__)
    
    reward_deque: deque = deque(maxlen=100)
    best_val_acc: float = 0.0
    best_model_state: Optional[Dict[str, Any]] = None
    
    os.makedirs(ckpt_dir, exist_ok=True)
    
    def save_checkpoint(
        epoch: int, 
        agent_a: AbstractAgent, 
        agent_b: AbstractAgent, 
        optimizer: torch.optim.Optimizer, 
        lr_scheduler: Any, 
        is_best: bool = False, 
        val_acc: Optional[float] = None
    ) -> None:
        checkpoint = {
            'agent_a': agent_a.state_dict(),
            'agent_b': agent_b.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'val_acc': val_acc
        }
        # ckpt_path = os.path.join(ckpt_dir, f'epoch_{epoch:03d}.pth')
        # torch.save(checkpoint, ckpt_path)
        if is_best:
            best_ckpt_path = os.path.join(ckpt_dir, 'best_model.pth')
            torch.save(checkpoint, best_ckpt_path)
    
    for epoch_num in range(num_epochs):
        progress_bar = tqdm(train_loader, desc=f'Training Progress Epoch {epoch_num}/{num_epochs}')
        total_m1_loss: float = 0.0
        total_m2_loss: float = 0.0
        total_correct: int = 0
        total_ins: int = 0 
        for iter_num, (imgs, labels) in enumerate(progress_bar):
            batch_size = imgs.shape[0]
            optimizer.zero_grad()
            imgs = imgs.to(device)
            sender_result = agent_a.forward_text_generation(imgs, message_length=message_length, sampling_temperature=sampling_temperature)
            words = sender_result['indices']
            sender_words_logits = sender_result['words_logits']
            probs = sender_words_logits
            #####
            listener_messages_repr = agent_b.forward_external_text_perception(words).squeeze(1)
            listener_objects_repr = agent_b.forward_image_encoder(imgs)
            #similarities_messages_to_objects = (listener_messages_repr @ listener_objects_repr.T)
            similarities_messages_to_objects = pairwise_cosine_similarity(listener_messages_repr, listener_objects_repr) / contrastive_loss_temperature
            m2_loss = F.cross_entropy(similarities_messages_to_objects, torch.arange(0, batch_size).to(device))
            m2_loss.backward()
            ###
            rewards = -1*F.cross_entropy(similarities_messages_to_objects, torch.arange(0, batch_size).to(device), reduction='none').detach()
            #rewards = (similarities_messages_to_objects.argmax(-1) == torch.arange(0, batch_size).to(device)).float()
            reward_deque.append(rewards.detach().cpu().numpy())
            avg_reward: float = float(np.mean(reward_deque))
            #####
            log_probs = torch.log(torch.gather(probs, -1, words.unsqueeze(-1)))
            returns = ((rewards - rewards.mean()) / (rewards.std() + 1e-5)).detach()
            # returns = (rewards - avg_reward).detach()
            # returns = (rewards - avg_reward) / (rewards.std() + 1e-5)
            # returns = (rewards) / (rewards.std() + 1e-5)
            returns = (rewards)
            returns = returns.detach()
            #returns = (rewards - avg_reward)
            # returns = rewards.detach()
            m1_loss = -torch.mean(returns.unsqueeze(1) * log_probs.squeeze(-1)) #+ sender_result['commit_loss']
            entropy_loss = -Categorical(F.softmax(sender_words_logits, dim=2)).entropy().mean()
            m1_loss = m1_loss + entropy_regularization_factor*entropy_loss
            m1_loss.backward()
            optimizer.step()
            lr_scheduler.step()
            ###
            total_correct += (similarities_messages_to_objects.argmax(-1) == torch.arange(0, batch_size).to(device)).sum().item()
            total_ins += len(imgs)
            total_m1_loss += m1_loss.item()
            total_m2_loss += m2_loss.item()
            progress_bar.set_postfix(
                m1r_loss=f"{total_m1_loss/(iter_num+1):.4f}", 
                m2r_loss=f"{total_m2_loss/(iter_num+1):.4f}", 
                accr=f"{total_correct/(total_ins):.4f}"
            )
            progress_bar.refresh()
            global_step = epoch_num * len(train_loader) + iter_num
            tensorboard_writer.add_scalar("Loss/train", (total_correct/(total_ins)), global_step)
        total_correct, total_ins = 0, 0
        for iter_num, (imgs, labels) in enumerate(val_loader):
            optimizer.zero_grad()
            imgs = imgs.to(device)
            sender_result = agent_a.forward_text_generation(imgs, message_length=message_length, sampling_temperature=1e-5)
            words = sender_result['indices']
            listener_messages_repr = agent_b.forward_external_text_perception(words).squeeze(1)
            listener_objects_repr = agent_b.forward_image_encoder(imgs)
            #similarities_messages_to_objects = (listener_messages_repr @ listener_objects_repr.T)
            similarities_messages_to_objects = pairwise_cosine_similarity(listener_messages_repr, listener_objects_repr)
            total_correct += (similarities_messages_to_objects.argmax(-1) == torch.arange(0, batch_size).to(device)).sum().item()
            total_ins += len(imgs)
        val_acc: float = total_correct / total_ins
        logger.info(f'val accuracy: {round(val_acc, 3)}')
    
        # Save checkpoint every epoch
        save_checkpoint(
            epoch=epoch_num,
            agent_a=agent_a,
            agent_b=agent_b,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            is_best=False,
            val_acc=val_acc
        )
    
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {
                'agent_a': copy.deepcopy(agent_a.state_dict()),
                'agent_b': copy.deepcopy(agent_b.state_dict()),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch_num
            }
            # Save best model checkpoint
            save_checkpoint(
                epoch=epoch_num,
                agent_a=agent_a,
                agent_b=agent_b,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                is_best=True,
                val_acc=val_acc
            )
            logger.info(f'New best validation accuracy: {best_val_acc:.3f}')
    logger.info(best_val_acc)
    
    return best_val_acc, best_model_state




from typing import Any, Optional
import logging


def train_self_play(
    agent, 
    train_loader, 
    val_loader, 
    optimizer, 
    lr_scheduler, 
    device, 
    num_pretrain_epochs, 
    length_message, 
    sampling_temperature=0.1, 
    entropy_factor=0, 
    contrastive_loss_temperature=0.1, 
    ckpt_dir='checkpoints',
    logger: Optional[logging.Logger] = None,
):
    """
    Self-play an agent using contrastive learning between image and text representations.
    
    Args:
        agent: The agent to train
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer for training
        lr_scheduler: Learning rate scheduler
        device: Device to run training on
        num_pretrain_epochs: Number of training epochs
        length_message: Length of generated messages
        sampling_temperature: Temperature for text generation sampling
        entropy_factor: Weight for entropy regularization loss
        contrastive_loss_temperature: Temperature for contrastive loss
        ckpt_dir: Directory to save checkpoints
        logger: Logger instance (optional)
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    best_val_acc = 0.0
    best_model_state = None
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(ckpt_dir, exist_ok=True)
    
    def save_checkpoint(
        epoch: int, 
        agent: AbstractAgent, 
        optimizer: torch.optim.Optimizer, 
        lr_scheduler: Any, 
        is_best: bool = False, 
        val_acc: Optional[float] = None
    ) -> None:
        checkpoint = {
            'agent': agent.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'val_acc': val_acc
        }
        # ckpt_path = os.path.join(ckpt_dir, f'epoch_{epoch:03d}.pth')
        # torch.save(checkpoint, ckpt_path)
        if is_best:
            best_ckpt_path = os.path.join(ckpt_dir, 'best_model.pth')
            torch.save(checkpoint, best_ckpt_path)
    
    for epoch_num in range(num_pretrain_epochs):
        # Training loop
        progress_bar = tqdm(train_loader, desc=f'Training Progress Epoch {epoch_num}/{num_pretrain_epochs}')
        
        # Initialize metrics
        total_contrastive_loss = 0.0
        total_commit_loss = 0.0
        total_corrects = 0
        
        for iter_num, (imgs, labels) in enumerate(progress_bar):
            optimizer.zero_grad()
            imgs = imgs.to(device)
            
            # Forward pass through agent
            img_repr = agent.forward_image_encoder(imgs)
            text_generation_result = agent.forward_text_generation(
                imgs, 
                message_length=length_message, 
                freeze_codebook=False, 
                mode='discrete', 
                sampling_temperature=sampling_temperature
            )
            text_repr = agent.forward_text_perception(text_generation_result['discretized'])
            
            # Compute contrastive loss
            similarities_messages_to_objects = pairwise_cosine_similarity(text_repr, img_repr) / contrastive_loss_temperature
            contrastive_loss = F.cross_entropy(
                similarities_messages_to_objects, 
                torch.arange(imgs.shape[0], device=device)
            )
            
            # Total loss with commitment and entropy regularization
            loss = contrastive_loss + text_generation_result['commit_loss']
            if entropy_factor > 0:
                entropy_loss = -Categorical(
                    F.softmax(text_generation_result['words_logits'], dim=2)
                ).entropy().mean()
                loss = loss + entropy_factor * entropy_loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            
            # Update metrics
            total_contrastive_loss += contrastive_loss.item()
            total_commit_loss += text_generation_result['commit_loss'].item()
            
            # Compute accuracy
            predicted_labels = pairwise_cosine_similarity(text_repr, img_repr).argmax(1).cpu()
            total_corrects += (predicted_labels == torch.arange(imgs.shape[0])).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                contrastive_loss=f"{total_contrastive_loss / (iter_num + 1):.4f}",
                commit_loss=f"{total_commit_loss / (iter_num + 1):.4f}",
                acc=f"{total_corrects / ((iter_num + 1) * imgs.shape[0]):.4f}"
            )
            progress_bar.refresh()
        
        # Validation loop
        total_correct = 0
        total_instances = 0
        
        with torch.no_grad():
            for imgs, _ in tqdm(val_loader, desc="Validation"):
                imgs = imgs.to(device)
                
                # Forward pass
                img_repr = agent.forward_image_encoder(imgs)
                text_generation_result = agent.forward_text_generation(
                    imgs, 
                    message_length=length_message, 
                    freeze_codebook=True, 
                    mode='discrete', 
                    sampling_temperature=sampling_temperature
                )
                text_repr = agent.forward_text_perception(text_generation_result['discretized'])
                
                # Compute accuracy
                predicted_labels = pairwise_cosine_similarity(text_repr, img_repr).argmax(1).cpu()
                total_correct += (predicted_labels == torch.arange(imgs.shape[0])).sum().item()
                total_instances += imgs.shape[0]
        
        # Update best model if validation accuracy improved
        val_acc = total_correct / total_instances
        logger.info(f'Validation accuracy: {val_acc:.3f}')
        
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(agent.state_dict())
            logger.info(f'New best validation accuracy: {best_val_acc:.3f}')
        
        # Save checkpoint every epoch
        save_checkpoint(
            epoch=epoch_num,
            agent=agent,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            is_best=is_best,
            val_acc=val_acc
        )
    
    # Load best model state
    if best_model_state is not None:
        agent.load_state_dict(best_model_state)
        logger.info(f'Loaded best model with validation accuracy: {best_val_acc:.3f}')


def train_mutual_play(
    agent_a, 
    agent_b, 
    train_loader, 
    val_loader, 
    device, 
    message_length, 
    num_pretrain_epochs, 
    optimizer, 
    lr_scheduler, 
    num_dialogue_epochs=1, 
    sampling_temperature=1e-5, 
    entropy_regularization_factor=0.0, 
    contrastive_loss_temperature=0.1, 
    ckpt_dir='checkpoints_dialogue', 
    agent_a_training_mode='reinforce_with_preservation',
    freeze_codebook=True,
    logger: Optional[logging.Logger] = None,
    tensorboard_writer: Optional[SummaryWriter] = None, 
):
    """
    Train agents in dialogue phase with multiple training modes for agent A.
    
    Args:
        agent_a: Sender agent
        agent_b: Receiver agent
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to run training on
        message_length: Length of generated messages
        num_pretrain_epochs: Number of completed pretraining epochs (for epoch numbering)
        optimizer: Optimizer for training
        lr_scheduler: Learning rate scheduler
        num_dialogue_epochs: Number of dialogue training epochs
        sampling_temperature: Temperature for text generation sampling
        entropy_regularization_factor: Weight for entropy regularization loss
        contrastive_loss_temperature: Temperature for contrastive loss
        ckpt_dir: Directory to save checkpoints
        agent_a_training_mode: Training mode for agent A:
            - 'frozen': Agent A parameters frozen
            - 'reinforce_only': Agent A trained only with REINFORCE
            - 'reinforce_with_preservation': Agent A trained with REINFORCE + language preservation (default)
        freeze_codebook: Whether to freeze the codebook during text generation (default: True)
        logger: Logger instance (optional)
    
    Returns:
        best_model_state: State dict of best model
        best_val_acc: Best validation accuracy achieved
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Create checkpoint directory
    os.makedirs(ckpt_dir, exist_ok=True)
    
    def save_checkpoint(epoch, agent_a, agent_b, optimizer, lr_scheduler, is_best=False, val_acc=None):
        """Save model checkpoint."""
        checkpoint = {
            'agent_a': agent_a.state_dict(),
            'agent_b': agent_b.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'val_acc': val_acc
        }
        # ckpt_path = os.path.join(ckpt_dir, f'epoch_{epoch:03d}.pth')
        # torch.save(checkpoint, ckpt_path)
        if is_best:
            best_ckpt_path = os.path.join(ckpt_dir, 'best_model.pth')
            torch.save(checkpoint, best_ckpt_path)
    
    # Initialize tracking variables
    best_val_acc = 0.0
    best_model_state = None
    reward_deque = deque(maxlen=100)
    
    writer = tensorboard_writer
    if writer == None:
        writer = SummaryWriter('runs/vqel_entropy0')
    
    # Determine if agent A should be frozen
    freeze_agent_a = (agent_a_training_mode == 'frozen')
    preserve_language = (agent_a_training_mode == 'reinforce_with_preservation')

    for epoch_num in range(num_pretrain_epochs, num_pretrain_epochs + num_dialogue_epochs):
        progress_bar = tqdm(
            train_loader, 
            desc=f'Training Progress Epoch {epoch_num}/{num_pretrain_epochs + num_dialogue_epochs}'
        )
        
        # Initialize epoch metrics
        metrics = {
            'm1_loss': 0.0,
            'm2_loss': 0.0,
            'commit_loss': 0.0,
            'correct': 0,
            'total': 0,
            'agent_a_self_play_correct': 0
        }
        
        for iter_num, (imgs, labels) in enumerate(progress_bar):
            optimizer.zero_grad()
            imgs = imgs.to(device)
            batch_size = imgs.shape[0]
            
            # ===== Agent A (Sender): Generate messages =====
            sender_result = agent_a.forward_text_generation(
                imgs, 
                message_length=message_length, 
                freeze_codebook=freeze_codebook, 
                mode='discrete', 
                sampling_temperature=sampling_temperature
            )
            words = sender_result['indices']
            sender_words_logits = sender_result['words_logits']
            
            # ===== Agent B (Receiver): Interpret messages =====
            listener_messages_repr = agent_b.forward_external_text_perception(words).squeeze(1)
            listener_objects_repr = agent_b.forward_image_encoder(imgs)
            
            # Compute receiver loss (cross-entropy on similarity matrix)
            similarities = pairwise_cosine_similarity(listener_messages_repr, listener_objects_repr) / contrastive_loss_temperature
            target_indices = torch.arange(batch_size, device=device)
            m2_loss = F.cross_entropy(similarities, target_indices)
            m2_loss.backward()
            
            # ===== Agent A (Sender): Compute loss based on training mode =====
            m1_loss = torch.tensor(0.0, device=device)
            
            if not freeze_agent_a:
                # Compute REINFORCE rewards
                with torch.no_grad():
                    rewards = -F.cross_entropy(similarities, target_indices, reduction='none')
                    reward_deque.append(rewards.cpu().numpy())
                    baseline = float(np.mean(reward_deque))
                
                # Compute policy gradient loss
                # log_probs = F.log_softmax(sender_words_logits, dim=-1)
                # selected_log_probs = torch.gather(log_probs, -1, words.unsqueeze(-1)).squeeze(-1)
                selected_log_probs = torch.log(torch.gather(sender_words_logits, -1, words.unsqueeze(-1))).squeeze(-1)

                # Normalize returns with baseline
                # normalized_returns = (rewards - baseline) / (rewards.std() + 1e-5)
                # normalized_returns = (rewards) / (rewards.std() + 1e-5)
                normalized_returns = (rewards)
                
                # REINFORCE loss
                m1_loss = -torch.mean(normalized_returns.unsqueeze(1) * selected_log_probs)
                
                # Add entropy regularization if specified
                if entropy_regularization_factor > 0:
                    entropy = Categorical(F.softmax(sender_words_logits, dim=2)).entropy().mean()
                    m1_loss = m1_loss - entropy_regularization_factor * entropy
                
                # Add commitment loss if present (only for VQ-based agents)
                if 'commit_loss' in sender_result:
                    m1_loss = m1_loss + sender_result['commit_loss']
                
                # Add language preservation loss if specified (only for VQ-based agents)
                if preserve_language and 'discretized' in sender_result:
                    # Self-play contrastive loss for agent A
                    img_repr_a = agent_a.forward_image_encoder(imgs)
                    text_repr_a = agent_a.forward_text_perception(sender_result['discretized'])
                    agent_a_similarities = pairwise_cosine_similarity(text_repr_a, img_repr_a) / contrastive_loss_temperature
                    
                    self_play_loss = F.cross_entropy(agent_a_similarities, target_indices)
                    
                    # Normalize and combine losses
                    m1_normalized = m1_loss / (m1_loss.detach() + 1e-8)
                    self_play_normalized = self_play_loss / (self_play_loss.detach() + 1e-8)
                    total_sender_loss = m1_normalized + self_play_normalized
                    total_sender_loss.backward()
                    
                    # Track self-play accuracy
                    metrics['agent_a_self_play_correct'] += (agent_a_similarities.argmax(-1) == target_indices).sum().item()
                else:
                    m1_loss.backward()
            
            optimizer.step()
            lr_scheduler.step()
            
            # Update metrics
            metrics['m1_loss'] += m1_loss.item()
            metrics['m2_loss'] += m2_loss.item()
            if 'commit_loss' in sender_result:
                metrics['commit_loss'] += sender_result['commit_loss'].item()
            metrics['correct'] += (similarities.argmax(-1) == target_indices).sum().item()
            metrics['total'] += batch_size
            
            # Update progress bar
            postfix = {
                'm1_loss': metrics['m1_loss'] / (iter_num + 1),
                'm2_loss': metrics['m2_loss'] / (iter_num + 1),
                'acc': metrics['correct'] / metrics['total'],
                'commit_loss': metrics['commit_loss'] / (iter_num + 1)
            }
            if preserve_language:
                postfix['agent_a_self_play_acc'] = metrics['agent_a_self_play_correct'] / metrics['total']
            
            progress_bar.set_postfix(postfix)
            progress_bar.refresh()
            
            # Log to tensorboard
            global_step = epoch_num * len(train_loader) + iter_num
            writer.add_scalar("Accuracy/train", metrics['correct'] / metrics['total'], global_step)
        
        # ===== Validation Phase =====
        agent_a.eval()
        agent_b.eval()
        
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc="Validation"):
                imgs = imgs.to(device)
                batch_size = imgs.shape[0]
                
                # Generate messages and compute similarities
                sender_result = agent_a.forward_text_generation(
                    imgs, 
                    message_length=message_length, 
                    mode='discrete',
                    freeze_codebook=freeze_codebook, 
                    sampling_temperature=sampling_temperature
                )
                
                words = sender_result['indices']
                listener_messages_repr = agent_b.forward_external_text_perception(words).squeeze(1)
                listener_objects_repr = agent_b.forward_image_encoder(imgs)
                similarities = pairwise_cosine_similarity(listener_messages_repr, listener_objects_repr) 
                
                val_correct += (similarities.argmax(-1) == torch.arange(batch_size, device=device)).sum().item()
                val_total += batch_size
        
        agent_a.train()
        agent_b.train()
        
        val_acc = val_correct / val_total
        logger.info(f'Validation accuracy: {val_acc:.3f}')
        
        # Save checkpoint
        save_checkpoint(
            epoch=epoch_num,
            agent_a=agent_a,
            agent_b=agent_b,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            is_best=False,
            val_acc=val_acc
        )
        
        # Update best model if improved
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {
                'agent_a': copy.deepcopy(agent_a.state_dict()),
                'agent_b': copy.deepcopy(agent_b.state_dict()),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch_num
            }
            save_checkpoint(
                epoch=epoch_num,
                agent_a=agent_a,
                agent_b=agent_b,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                is_best=True,
                val_acc=val_acc
            )
            logger.info(f'New best validation accuracy: {best_val_acc:.3f}')

    writer.close()
    
    return best_model_state, best_val_acc


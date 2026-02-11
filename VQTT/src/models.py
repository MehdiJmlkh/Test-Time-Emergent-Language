from torch.distributions.categorical import Categorical
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from typing import Any
from abc import ABC
from vector_quantize_pytorch import VectorQuantize


# Standard output format for text generation across all agent types
# Required fields: indices, words_logits
# Optional fields: commit_loss, continuous, discretized (only present for VQ-based agents)

class AbstractAgent(ABC, nn.Module):
    """
    Abstract Agent class for VQEL.
    This class can act as a placeholder and can be expanded with environment interaction,
    action selection, and learning logic as needed.
    """
    def __init__(self) -> None:
        super().__init__()
        pass

    def forward_image_encoder(self, x) -> Any:
        pass

    def forward_text_generation(self, x) -> Any:
        pass

    def forward_text_perception(self, x) -> Any:
        pass

    def forward_external_text_perception(self, x) -> Any:
        pass

class BaselineAgent(AbstractAgent):
    def __init__(self, input_dim: int, representation_dim: int, vocab_size: int, object_encoder: nn.Module) -> None:
        super(BaselineAgent, self).__init__()
        self.input_dim = input_dim
        self.representation_dim = representation_dim
        #### ConvNet
        self.object_encoder = object_encoder
        ### Text Generation
        self.text_generation_gru = nn.GRU(representation_dim, representation_dim, batch_first=True)
        self.text_generation_gru_head = nn.Linear(representation_dim, representation_dim)
        self.vocab_logits = nn.Linear(representation_dim, vocab_size)
        self.text_generation_word_embedding = nn.Embedding(vocab_size, representation_dim)
        ### Text Perception
        self.text_perception_gru = nn.GRU(representation_dim, representation_dim, batch_first=True)
        self.text_perception_gru_head = nn.Linear(representation_dim, representation_dim)
        ### Others Text Perception
        self.external_token_embedding = nn.Embedding(vocab_size, representation_dim)
    
    def forward_image_encoder(self, x):
        """
        Forward pass through the ConvNet and Image Encoder.

        Args:
        - x (torch.Tensor): Input image tensor of shape (batch_size, channels, height, width).

        Returns:
        - torch.Tensor: Output representation tensor of shape (batch_size, representation_dim).
        """
        x = self.object_encoder(x)
        return x
    
    def forward_text_generation(self, x, message_length=4, sampling_temperature=1, 
                              freeze_codebook=False, mode='discrete') -> dict:
        """
        Forward pass for text generation.

        Args:
        - x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
        - message_length (int): Number of time steps for generation.
        - sampling_temperature (float): Temperature for sampling.
        - freeze_codebook (bool): No-op for baseline (for interface compatibility).
        - mode (str): Generation mode - only 'discrete' is implemented for baseline.

        Returns:
        - TextGenerationOutput: Dictionary containing generation results.
        """
        if mode != 'discrete':
            raise NotImplementedError(f"BaselineAgent only supports mode='discrete', got mode='{mode}'")
        
        x = self.object_encoder(x)
        x = einops.repeat(x, 'b d -> b l d', l=1) 
        generated_andices = []
        logit_scores = []
        batch_size = x.shape[0]
        h = torch.zeros(1, batch_size, self.representation_dim).to(next(self.parameters()).device)
        h[0, :, :] = x[:, 0, :]
        x = torch.zeros_like(x)
        for i in range(message_length):
            x, h = self.text_generation_gru(x, h)
            last_hidden_state = x[:, -1:, :]
            logit_score = self.vocab_logits(self.text_generation_gru_head(last_hidden_state))
            logit_score = F.softmax(logit_score/sampling_temperature, dim=2)
            logit_scores.append(logit_score)
            next_word = Categorical(logit_score).sample()
            next_word_embeddings = self.text_generation_word_embedding(next_word)
            x = next_word_embeddings
            generated_andices.append(next_word)
        # Concatenate tensors along dimension 1
        generated_andices = torch.cat(generated_andices, dim=1) 
        logit_scores = torch.cat(logit_scores, dim=1)# 1 is because l in [b, l, score]
        
        # Return standardized output format
        # BaselineAgent only returns required fields (no VQ-specific fields)
        return {
            'indices': generated_andices,
            'words_logits': logit_scores
        }
    
    def forward_external_text_perception(self, x):
        """
        Forward pass for text perception.

        Args:
        - x (torch.Tensor): Input tensor of shape (batch_size, sequence_length).

        Returns:
        - torch.Tensor: Output representation tensor of shape (batch_size, representation_dim).
        """
        batch_size = x.shape[0]
        x = self.external_token_embedding(x)
        h = torch.zeros(1, batch_size, self.representation_dim).to(next(self.parameters()).device)
        x, h = self.text_perception_gru(x, h)
        x = x[:, -1, :]
        x = self.text_perception_gru_head(x)
        return x



class VQELAgent(AbstractAgent):
    def __init__(self, input_dim: int, representation_dim: int, threshold_ema_dead_code, vocab_size: int, object_encoder: nn.Module, decay=0.97, commitment_weight=0.25, orthogonal_reg_weight=0, use_cosine_sim=False, object_decoder: nn.Module=None):
        super().__init__()
        self.input_dim = input_dim
        self.use_cosine_sim = use_cosine_sim
        self.representation_dim = representation_dim
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.vocab_size = vocab_size
        self.decay = decay
        self.commitment_weight = commitment_weight
        self.orthogonal_reg_weight = orthogonal_reg_weight
        
        # Object encoder
        self.object_encoder = object_encoder
        self.object_decoder = object_decoder
        
        # Text generation components
        self.text_generation_gru = nn.GRU(representation_dim, representation_dim, batch_first=True)
        self.text_generation_gru_head = nn.Linear(representation_dim, representation_dim)
        self.text_generation = nn.ModuleList([
            self.text_generation_gru,
            self.text_generation_gru_head
        ])
        
        # Vector quantization
        self.vq = VectorQuantize(
            dim=representation_dim,
            codebook_size=vocab_size,
            decay=decay,
            commitment_weight=commitment_weight,
            use_cosine_sim=use_cosine_sim,
            threshold_ema_dead_code=threshold_ema_dead_code,
            stochastic_sample_codes=True,
            orthogonal_reg_weight=orthogonal_reg_weight,
        )
        
        # Text perception components
        self.text_perception_gru = nn.GRU(representation_dim, representation_dim, batch_first=True)
        self.text_perception_gru_head = nn.Linear(representation_dim, representation_dim)
        self.text_perception_modules = nn.ModuleList([
            self.text_perception_gru, 
            self.text_perception_gru_head
        ])
        
        # External token embedding
        self.external_token_embedding = nn.Embedding(vocab_size, representation_dim)
        
    def reset_codebook(self):
        self.vq = VectorQuantize(
            dim=self.representation_dim,
            codebook_size=self.vocab_size,
            decay=self.decay,
            commitment_weight=self.commitment_weight,
            use_cosine_sim=self.use_cosine_sim,
            threshold_ema_dead_code=self.threshold_ema_dead_code,
            stochastic_sample_codes=True,
            orthogonal_reg_weight=self.orthogonal_reg_weight,
        ).to(self.vq.codebook.device)
    
    def forward_image_decoder(self, x, message_length=None, return_recons=False):
        recon_combined, recons = self.object_decoder(x, num_slots=message_length)
        if return_recons:
            return recon_combined, recons
        return recon_combined
    
    def forward_image_encoder(self, x, message_length=None):
        """
        Encode input through the object encoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Encoded representation of shape (batch_size, representation_dim).
        """
        return self.object_encoder(x, num_slots=message_length)
    
    def forward_text_generation(self, x, message_length=4, freeze_codebook=False, 
                              sampling_temperature=1, mode='continuous') -> dict:
        """
        Generate text representations through GRU and vector quantization.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
            message_length (int): Number of generation steps.
            freeze_codebook (bool): Whether to freeze the VQ codebook.
            sampling_temperature (float): Temperature for sampling.
            mode (str): Generation mode ('continuous' or 'discrete').

        Returns:
            TextGenerationOutput: Dictionary containing generation results.
        """
        continuous_outputs = []
        discretized_outputs = []
        indices_outputs = []
        commit_losses = []
        word_logits = []

        for i in range(x.shape[1]):
            # Vector quantization
            slot = x[:, i, :]
            x_discretized, x_indices, x_commit_loss = self.vq(
                slot, freeze_codebook=freeze_codebook, sample_codebook_temp=sampling_temperature
            )
            
            # Compute word logits using distance to codebook
            codebook = self.vq.codebook
            # similarities = -torch.cdist(x, codebook, p=2.0)[:, 0, :]
            
            if self.use_cosine_sim:
                # First reshape if needed
                x_flat = slot.view(slot.size(0), -1)               # (B, D)
                codebook_flat = codebook.view(codebook.size(0), -1)  # (K, D)

                # Normalize
                x_norm = x_flat / x_flat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                code_norm = codebook_flat / codebook_flat.norm(dim=-1, keepdim=True).clamp(min=1e-8)

                # Cosine similarity: (B, D) @ (D, K) -> (B, K)
                similarities = torch.matmul(x_norm, code_norm.transpose(0, 1))

                word_logits_step = F.softmax(similarities / sampling_temperature, dim=-1)
            else:
                distances = -torch.cdist(slot, codebook, p=2.0)[:, 0, :]
                word_logits_step = F.softmax(distances / sampling_temperature, dim=-1)

            
            # Store results
            continuous_outputs.append(slot)
            discretized_outputs.append(x_discretized)
            indices_outputs.append(x_indices.unsqueeze(1))
            commit_losses.append(x_commit_loss)
            word_logits.append(word_logits_step)
        # Combine results
        result = {
            'continuous': torch.cat(continuous_outputs, dim=1),
            'discretized': torch.stack(discretized_outputs, dim=1),
            'indices': torch.cat(indices_outputs, dim=1),
            'commit_loss': torch.stack(commit_losses).sum(),
            'words_logits': torch.stack(word_logits, dim=1)
        }
        
        return result
    
    def forward_text_perception(self, x):
        """
        Process text representations through perception GRU.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, representation_dim).

        Returns:
            torch.Tensor: Final representation of shape (batch_size, representation_dim).
        """
        batch_size = x.shape[0]
        device = next(self.parameters()).device
        
        # Initialize hidden state
        h = torch.zeros(1, batch_size, self.representation_dim, device=device)
        
        # Process sequence
        x, h = self.text_perception_gru(x, h)
        x = x[:, -1, :]  # Take final output
        x = self.text_perception_gru_head(x)
        
        return x
    
    def forward_external_text_perception(self, x):
        """
        Process external token indices through embedding and perception.

        Args:
            x (torch.Tensor): Token indices of shape (batch_size, sequence_length).

        Returns:
            torch.Tensor: Final representation of shape (batch_size, representation_dim).
        """
        batch_size = x.shape[0]
        device = next(self.parameters()).device
        
        # Embed tokens
        x = self.external_token_embedding(x)
        
        # Initialize hidden state
        h = torch.zeros(1, batch_size, self.representation_dim, device=device)
        
        # Process sequence
        x, h = self.text_perception_gru(x, h)
        x = x[:, -1, :]  # Take final output
        x = self.text_perception_gru_head(x)
        
        return x

    def reset_perception(self):
        """Reset perception modules to their initial state."""
        device = next(self.parameters()).device
        self.text_perception_gru = nn.GRU(
            self.representation_dim, self.representation_dim, batch_first=True
        ).to(device)
        self.text_perception_gru_head = nn.Linear(
            self.representation_dim, self.representation_dim
        ).to(device)
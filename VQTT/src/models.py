from torch.distributions.categorical import Categorical
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from typing import Any
from abc import ABC
from vector_quantize_pytorch import VectorQuantize


class AbstractAgent(ABC, nn.Module):
    """
    Abstract Agent class for VQEL.
    This class can act as a placeholder and can be expanded with environment interaction,
    action selection, and learning logic as needed.
    """

    def __init__(
        self, hidden_dim: int = 1024, tau_0: float = 0.1, gumbel: bool = False
    ) -> None:
        super(AbstractAgent, self).__init__()

        self.hidden_dim = hidden_dim
        self.tau_0 = tau_0
        if gumbel:
            self.inv_tau_mlp = torch.nn.Linear(hidden_dim, 1)

    def forward_image_encoder(self, x) -> Any:
        pass

    def forward_text_generation(self, x) -> Any:
        pass

    def forward_text_perception(self, x) -> Any:
        pass

    def forward_external_text_perception(self, x) -> Any:
        pass

    def compute_temperature(self, h: torch.Tensor, eval=False) -> torch.Tensor:
        """
        h: [B, L, H] sender hidden states
        returns tau: [B, L, 1]

        During eval mode, gradients are not computed.
        """
        if eval:
            with torch.no_grad():
                inv_tau = torch.log1p(torch.exp(self.inv_tau_mlp(h))) + self.tau_0
                tau = 1.0 / inv_tau
            return tau

        inv_tau = torch.log1p(torch.exp(self.inv_tau_mlp(h))) + self.tau_0

        tau = 1.0 / inv_tau
        return tau


class BaselineAgent(AbstractAgent):
    def __init__(
        self,
        input_dim: int,
        representation_dim: int,
        vocab_size: int,
        object_encoder: nn.Module,
        tau_0: float = 1.0,
        gumbel: bool = False,
    ) -> None:
        super(BaselineAgent, self).__init__(
            tau_0=tau_0, hidden_dim=representation_dim, gumbel=gumbel
        )
        self.input_dim = input_dim
        self.representation_dim = representation_dim

        self.object_encoder = object_encoder

        self.text_generation_gru = nn.GRU(
            representation_dim, representation_dim, batch_first=True
        )
        self.text_generation_gru_head = nn.Linear(
            representation_dim, representation_dim
        )
        self.vocab_logits = nn.Linear(representation_dim, vocab_size)
        self.text_generation_word_embedding = nn.Embedding(
            vocab_size, representation_dim
        )

        self.text_perception_gru = nn.GRU(
            representation_dim, representation_dim, batch_first=True
        )
        self.text_perception_gru_head = nn.Linear(
            representation_dim, representation_dim
        )

        self.external_token_embedding = nn.Embedding(vocab_size, representation_dim)

    def forward_image_encoder(self, x):
        x = self.object_encoder(x)
        return x

    def forward_text_generation(
        self,
        x,
        message_length=4,
        sampling_temperature=1,
        freeze_codebook=False,
        mode="discrete",
    ) -> dict:
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
        if mode != "discrete":
            raise NotImplementedError(
                f"BaselineAgent only supports mode='discrete', got mode='{mode}'"
            )

        x = self.object_encoder(x)
        x = einops.repeat(x, "b d -> b l d", l=1)
        batch_size = x.shape[0]

        generated_andices = []
        logit_scores = []
        hidden_states = []

        h = torch.zeros(1, batch_size, self.representation_dim).to(
            next(self.parameters()).device
        )
        h[0, :, :] = x[:, 0, :]

        x = torch.zeros_like(x)

        for i in range(message_length):
            x, h = self.text_generation_gru(x, h)
            last_hidden_state = x[:, -1:, :]

            logit_score = self.vocab_logits(
                self.text_generation_gru_head(last_hidden_state)
            )
            logit_scores.append(logit_score)

            logit_score = F.softmax(logit_score / sampling_temperature, dim=2)
            next_word = Categorical(logit_score).sample()
            next_word_embeddings = self.text_generation_word_embedding(next_word)

            x = next_word_embeddings

            generated_andices.append(next_word)
            hidden_states.append(h[0])

        generated_andices = torch.cat(generated_andices, dim=1)
        logit_scores = torch.cat(logit_scores, dim=1)  # 1 is because l in [b, l, score]

        return {
            "indices": generated_andices,
            "words_logits": logit_scores,
            "hidden_states": torch.stack(hidden_states, dim=1),
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
        if x.dim() == 2:
            x = self.external_token_embedding(x)
        else:
            x = torch.matmul(x, self.external_token_embedding.weight)

        h = torch.zeros(1, batch_size, self.representation_dim).to(
            next(self.parameters()).device
        )
        x, h = self.text_perception_gru(x, h)
        x = x[:, -1, :]
        x = self.text_perception_gru_head(x)

        return x


class VQELAgent(AbstractAgent):
    def __init__(
        self,
        input_dim: int,
        representation_dim: int,
        threshold_ema_dead_code,
        vocab_size: int,
        object_encoder: nn.Module,
        decay=0.97,
        commitment_weight=0.25,
        orthogonal_reg_weight=0,
        use_cosine_sim=False,
        gumbel: bool = False,
    ) -> None:
        super(VQELAgent, self).__init__(gumbel=gumbel)
        self.input_dim = input_dim
        self.use_cosine_sim = use_cosine_sim
        self.representation_dim = representation_dim
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.vocab_size = vocab_size
        self.decay = decay
        self.commitment_weight = commitment_weight
        self.orthogonal_reg_weight = orthogonal_reg_weight

        self.object_encoder = object_encoder

        self.text_generation_gru = nn.GRU(
            representation_dim, representation_dim, batch_first=True
        )
        self.text_generation_gru_head = nn.Linear(
            representation_dim, representation_dim
        )
        self.text_generation = nn.ModuleList(
            [self.text_generation_gru, self.text_generation_gru_head]
        )

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

        self.text_perception_gru = nn.GRU(
            representation_dim, representation_dim, batch_first=True
        )
        self.text_perception_gru_head = nn.Linear(
            representation_dim, representation_dim
        )
        self.text_perception_modules = nn.ModuleList(
            [self.text_perception_gru, self.text_perception_gru_head]
        )

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

    def forward_image_encoder(self, x):
        """
        Encode input through the object encoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Encoded representation of shape (batch_size, representation_dim).
        """
        return self.object_encoder(x)

    def forward_text_generation(
        self,
        x,
        message_length=4,
        freeze_codebook=False,
        sampling_temperature=1,
        mode="continuous",
    ) -> dict:
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
        x = self.object_encoder(x)
        batch_size = x.shape[0]
        device = next(self.parameters()).device

        # Initialize GRU hidden state
        h = torch.zeros(1, batch_size, self.representation_dim, device=device)
        h[0, :, :] = x

        # Initialize input for generation
        x = torch.zeros(batch_size, 1, self.representation_dim, device=device)

        continuous_outputs = []
        discretized_outputs = []
        indices_outputs = []
        commit_losses = []
        word_logits = []

        for _ in range(message_length):
            x, h = self.text_generation_gru(x, h)
            x = x[:, -1:, :]  # Take last output
            x = self.text_generation_gru_head(x)

            # Vector quantization
            x_discretized, x_indices, x_commit_loss = self.vq(
                x,
                freeze_codebook=freeze_codebook,
                sample_codebook_temp=sampling_temperature,
            )

            # Compute word logits using distance to codebook
            codebook = self.vq.codebook

            if self.use_cosine_sim:
                x_flat = x.view(x.size(0), -1)  # (B, D)
                codebook_flat = codebook.view(codebook.size(0), -1)  # (K, D)

                # Normalize
                x_norm = x_flat / x_flat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                code_norm = codebook_flat / codebook_flat.norm(
                    dim=-1, keepdim=True
                ).clamp(min=1e-8)

                # Cosine similarity: (B, D) @ (D, K) -> (B, K)
                similarities = torch.matmul(x_norm, code_norm.transpose(0, 1))

                word_logits_step = F.softmax(
                    similarities / sampling_temperature, dim=-1
                )
            else:
                distances = -torch.cdist(x, codebook, p=2.0)[:, 0, :]
                word_logits_step = F.softmax(distances / sampling_temperature, dim=-1)

            continuous_outputs.append(x)
            discretized_outputs.append(x_discretized)
            indices_outputs.append(x_indices)
            commit_losses.append(x_commit_loss)
            word_logits.append(word_logits_step)

            if mode == "continuous":
                pass  # Keep continuous representation
            elif mode == "discrete":
                x = x_discretized
            else:
                raise ValueError(
                    f"Invalid mode: {mode}. Must be 'continuous' or 'discrete'."
                )

        result = {
            "continuous": torch.cat(continuous_outputs, dim=1),
            "discretized": torch.cat(discretized_outputs, dim=1),
            "indices": torch.cat(indices_outputs, dim=1),
            "commit_loss": torch.stack(commit_losses).sum(),
            "words_logits": torch.stack(word_logits, dim=1),
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
        if x.dim() == 2:
            x = self.external_token_embedding(x)
        else:
            x = torch.matmul(x, self.external_token_embedding.weight)

        # Initialize hidden state
        h = torch.zeros(1, batch_size, self.representation_dim, device=device)

        x, h = self.text_perception_gru(x, h)
        x = x[:, -1, :]  # Take final output
        x = self.text_perception_gru_head(x)

        return x

    def get_external_embedding(self, x):
        if x.dim() == 2:
            x = self.external_token_embedding(x)
        else:
            x = torch.matmul(x, self.external_token_embedding.weight)

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

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from zeta import FeedForward

class MultiQueryAttention(nn.Module):
    """
    Multi-Query Attention module with rotary positional embeddings.

    This module implements a variant of multi-head attention where the queries are projected
    into multiple heads, while keys and values are projected only once (i.e. shared across heads).
    Rotary positional embeddings are applied to queries and keys before attention computation.

    Args:
        d_model (int): Dimensionality of the input and output features.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout probability applied to the attention weights.
    """

    def __init__(
        self, d_model: int, num_heads: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.head_dim = d_model // num_heads

        # Separate query projection for multi-heads
        self.q_proj = nn.Linear(d_model, d_model)
        # Shared key and value projections (multi-query: one projection each)
        self.k_proj = nn.Linear(d_model, self.head_dim)
        self.v_proj = nn.Linear(d_model, self.head_dim)
        # Output projection to combine heads
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _rotary_emb(self, x: Tensor, seq_len: int) -> Tensor:
        """Apply rotary positional embeddings to input tensor."""
        # Create position indices
        position = torch.arange(seq_len, device=x.device).unsqueeze(1)
        # Create dimension indices
        dim = torch.arange(
            self.head_dim // 2, device=x.device
        ).unsqueeze(0)
        # Compute angles
        angle = position * (
            1.0 / torch.pow(10000, (2.0 * dim) / self.head_dim)
        )
        # Create rotation matrices
        cos = torch.cos(angle).unsqueeze(0)  # [1, seq_len, dim]
        sin = torch.sin(angle).unsqueeze(0)  # [1, seq_len, dim]

        # Split input into real and imaginary parts
        x1 = x[..., : self.head_dim // 2]
        x2 = x[..., self.head_dim // 2 : self.head_dim]

        # Apply complex rotation
        rotated = torch.cat(
            [x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1
        )
        return rotated

    def forward(
        self, x: Tensor, attn_mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        Compute multi-query attention with rotary embeddings.

        Args:
            x (Tensor): Input tensor of shape (batch, seq_length, d_model).
            attn_mask (Optional[Tensor]): Optional attention mask of shape
                (batch, num_heads, seq_length, seq_length).

        Returns:
            Tensor: Output tensor of shape (batch, seq_length, d_model).
        """
        B, T, _ = x.size()
        # Project queries and reshape to (B, num_heads, T, head_dim)
        q = (
            self.q_proj(x)
            .view(B, T, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        # Shared keys and values: shape (B, T, head_dim), then unsqueeze to add head dim.
        k = self.k_proj(x).unsqueeze(1)  # (B, 1, T, head_dim)
        v = self.v_proj(x).unsqueeze(1)  # (B, 1, T, head_dim)

        # Apply rotary embeddings to queries and keys
        q = self._rotary_emb(q, T)
        k = self._rotary_emb(k, T)

        # Scaled dot-product attention (broadcasting over heads)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(
            self.head_dim
        )
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Compute weighted sum of values and combine heads.
        context = torch.matmul(attn, v)  # (B, num_heads, T, head_dim)
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(B, T, self.d_model)
        )
        out = self.out_proj(context)
        return out


class SwitchMultiQueryAttention(nn.Module):
    """
    Switch Mixture-of-Experts Multi-Query Attention module.

    This module routes each token to a subset of multi-query attention experts using top-k routing.
    Each token may be dispatched to multiple experts (weighted by the router probabilities)
    while enforcing a capacity constraint per expert (dropping tokens if necessary).
    Additionally, a load-balancing loss is computed to encourage a uniform token distribution.
    A residual connection and layer normalization are applied to the expert outputs.

    Args:
        d_model (int): Dimensionality of the input and output features.
        num_heads (int): Number of attention heads for each expert.
        num_experts (int): Number of experts.
        dropout (float): Dropout probability for attention weights.
        capacity_factor (float): Factor to compute capacity per expert as
            capacity = int(capacity_factor * (seq_length / num_experts)).
        top_k (int): Number of experts to which each token is routed.
        load_loss_coef (float): Coefficient for the load balancing loss.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_experts: int,
        dropout: float = 0.0,
        capacity_factor: float = 1.0,
        top_k: int = 2,
        load_loss_coef: float = 1e-2,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.load_loss_coef = load_loss_coef

        # Create experts as a ModuleList of MultiQueryAttention modules.
        self.experts = nn.ModuleList(
            [
                MultiQueryAttention(d_model, num_heads, dropout)
                for _ in range(num_experts)
            ]
        )
        # Router that produces logits over experts for each token.
        self.router = nn.Linear(d_model, num_experts)
        # Layer norm for stabilization and a residual connection.
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self, x: Tensor, attn_mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass for the Switch Multi-Query Attention module.

        Each token in the input is assigned to top-k experts (via soft gating). Tokens are
        dispatched to each expert with a capacity constraint per expert (tokens beyond capacity
        are dropped). Each expert processes its subset of tokens independently and outputs are
        aggregated (weighted by the gating scores). A load balancing loss is computed as well.

        Args:
            x (Tensor): Input tensor of shape (B, T, d_model).
            attn_mask (Optional[Tensor]): Optional attention mask (applied within experts if needed).

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing:
                - Output tensor of shape (B, T, d_model) with a residual connection applied.
                - Scalar load balancing loss.
        """
        B, T, _ = x.size()
        # Compute router logits and convert to probabilities.
        router_logits = self.router(x)  # (B, T, num_experts)
        router_probs = F.softmax(
            router_logits, dim=-1
        )  # (B, T, num_experts)

        # Compute load balancing loss.
        # importance: total probability mass for each expert.
        importance = router_probs.sum(dim=(0, 1))  # (num_experts,)
        # Count how many times each expert is selected in top-k.
        # Create an index tensor of shape (num_experts,) to compare.
        expert_range = torch.arange(
            self.num_experts, device=x.device
        ).view(1, 1, -1)
        # Boolean mask: shape (B, T, num_experts) where True if expert is in top-k.
        topk_probs, topk_indices = torch.topk(
            router_probs, k=self.top_k, dim=-1
        )
        # Create a mask for each expert.
        expert_mask = topk_indices.unsqueeze(
            -1
        ) == expert_range.unsqueeze(0).unsqueeze(0)
        # Each token may be assigned multiple times; count each occurrence.
        load = expert_mask.float().sum(
            dim=(1, 2)
        )  # (B,) then sum over batch:
        load = load.sum()  # scalar count
        # Normalize counts and probabilities.
        importance_norm = importance / (B * T)
        load_norm = load / (B * T * self.top_k)
        load_loss = (
            self.num_experts
            * torch.sum(importance_norm * load_norm)
            * self.load_loss_coef
        )

        # Prepare an output tensor.
        out = torch.zeros_like(x)

        # Compute capacity per expert per batch.
        capacity = int(self.capacity_factor * (T / self.num_experts))
        capacity = max(capacity, 1)

        # For each expert, dispatch tokens per batch.
        for expert_id, expert in enumerate(self.experts):
            for b in range(B):
                # Get top-k assignments for this batch sample.
                # topk_indices[b]: shape (T, top_k), topk_probs[b]: shape (T, top_k)
                token_mask = (
                    topk_indices[b] == expert_id
                )  # shape (T, top_k) boolean mask.
                # For each token, if expert_id is among top-k, get its gating weight.
                gating_scores = topk_probs[b][
                    token_mask
                ]  # 1D tensor of scores.
                # Get token indices in the sequence that are assigned to expert_id.
                token_indices = token_mask.nonzero(as_tuple=False)[
                    :, 0
                ]  # shape (n_tokens,)
                if token_indices.numel() == 0:
                    continue
                # If more tokens are assigned than capacity, select the ones with highest gating.
                if token_indices.numel() > capacity:
                    gating_subset = gating_scores
                    top_capacity = torch.topk(
                        gating_subset, k=capacity
                    )[1]
                    token_indices = token_indices[top_capacity]
                    gating_scores = gating_scores[top_capacity]
                # Gather tokens for this expert.
                x_expert = x[b, token_indices, :].unsqueeze(
                    0
                )  # shape (1, n_tokens, d_model)
                # Process through expert.
                expert_out = expert(
                    x_expert, attn_mask=None
                )  # shape (1, n_tokens, d_model)
                # Weight the expert output by the gating score(s).
                # Note: if a token is assigned to multiple experts, contributions will be summed.
                weighted_out = expert_out.squeeze(
                    0
                ) * gating_scores.unsqueeze(-1)
                # Scatter the weighted output back into the output tensor.
                out[b, token_indices, :] += weighted_out

        # Apply residual connection and layer normalization.
        out = x + self.norm(out)
        return out, load_loss
    
    
class Block(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_experts: int, dropout: float = 0.0, capacity_factor: float = 1.0, top_k: int = 2, load_loss_coef: float = 1e-2):
        super().__init__()
        self.attn = SwitchMultiQueryAttention(d_model, num_heads, num_experts, dropout, capacity_factor, top_k, load_loss_coef)
        self.ffn = FeedForward(d_model, d_model, swish=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, attn_mask: Optional[Tensor] = None):
        # First residual block with attention
        residual = x
        x = self.norm1(x)
        attn_out, load_loss = self.attn(x, attn_mask)
        x = residual + attn_out
        
        # Second residual block with FFN
        residual = x
        x = self.norm2(x)
        x = residual + self.ffn(x)
        
        return x
    


class Transformer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_experts: int, num_layers: int, vocab_size: int, dropout: float = 0.0, capacity_factor: float = 1.0, top_k: int = 2, load_loss_coef: float = 1e-2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.empty(1, 1024, d_model))
        nn.init.normal_(self.pos_embedding, std=0.02)
        
        self.layers = nn.ModuleList([Block(d_model, num_heads, num_experts, dropout, capacity_factor, top_k, load_loss_coef) for _ in range(num_layers)])
        
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
    def forward(self, x: Tensor, attn_mask: Optional[Tensor] = None):
        # Input embedding
        x = self.embedding(x)
        x = x + self.pos_embedding[:, :x.size(1), :]
        
        # Transformer layers
        for layer in self.layers:
            x = layer(x, attn_mask)
            
        # Output head
        x = self.norm(x)
        x = self.head(x)
        return F.log_softmax(x, dim=-1)
    
    
# Example usage
vocab_size = 1000
x = torch.randint(0, vocab_size, (2, 16)) # Input tokens
transformer = Transformer(d_model=64, num_heads=8, num_experts=4, num_layers=2, vocab_size=vocab_size)
output = transformer(x)
print(output) # [2, 16, vocab_size]

# # Example usage:
# if __name__ == "__main__":
#     # Define dimensions.
#     batch_size = 2
#     seq_length = 16
#     d_model = 64
#     num_heads = 8
#     num_experts = 4

#     # Create dummy input.
#     dummy_input = torch.randn(batch_size, seq_length, d_model)

#     # Instantiate the optimized Switch Multi-Query Attention module.
#     switch_mqa = SwitchMultiQueryAttention(
#         d_model=d_model,
#         num_heads=num_heads,
#         num_experts=num_experts,
#         dropout=0.1,
#         capacity_factor=1.0,
#         top_k=2,
#         load_loss_coef=1e-2,
#     )

#     # Compute output and load balancing loss.
#     output, load_loss = switch_mqa(dummy_input)
#     print(output)
#     print(load_loss)

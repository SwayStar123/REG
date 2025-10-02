import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ConceptBank(nn.Module):
    def __init__(
        self, 
        n_concepts=10_000,
        concept_dim=768,
        key_dim=128,
        n_registers=64,
        topk_per_register=1,
    ):
        super().__init__()
        self.n_concepts = n_concepts
        self.n_registers = n_registers
        self.topk = topk_per_register
        self.concept_dim = concept_dim
        
        # Product key structure for O(√N) retrieval
        self.n_keys = int(np.sqrt(n_concepts))
        self.row_keys = nn.Parameter(torch.randn(self.n_keys, key_dim // 2))
        self.col_keys = nn.Parameter(torch.randn(self.n_keys, key_dim // 2))
        
        # The actual concept bank
        self.concepts = nn.Parameter(torch.randn(n_concepts, concept_dim))
        
        # Per-register query generation
        self.query_net = nn.Sequential(
            nn.LayerNorm(concept_dim),
            nn.Linear(concept_dim, key_dim),
        )
        
        # Query to concept space projection (for commitment loss)
        self.query_to_concept = nn.Linear(key_dim, concept_dim)
        
        # Initialize keys uniformly on hypersphere
        with torch.no_grad():
            self.row_keys.div_(self.row_keys.norm(dim=-1, keepdim=True))
            self.col_keys.div_(self.col_keys.norm(dim=-1, keepdim=True))
            # Initialize concepts with small norm
            self.concepts.mul_(0.01)
    
    def forward(self, registers):
        """
        registers: [B, n_registers, D] - e.g. [B, 64, 768]
        returns: concepts [B, n_registers * topk, D], info dict
        """
        B, R, D = registers.shape
        
        # Generate query for each register
        queries = self.query_net(registers)  # [B, R, key_dim]
        q_row, q_col = queries.chunk(2, dim=-1)  # [B, R, key_dim/2] each
        
        # Product key scoring
        row_scores = torch.einsum('brd,kd->brk', q_row, self.row_keys)
        col_scores = torch.einsum('brd,kd->brk', q_col, self.col_keys)
        
        # Combine: scores[b,r,i,j] = row_scores[b,r,i] + col_scores[b,r,j]
        scores = row_scores.unsqueeze(3) + col_scores.unsqueeze(2)  # [B, R, n_keys, n_keys]
        scores = scores.flatten(2)  # [B, R, n_concepts]
        
        # Top-k per register
        topk_scores, topk_idx = scores.topk(self.topk, dim=-1)  # [B, R, topk]
        
        # Gather concepts - need to handle batched indexing properly
        B, R, K = topk_idx.shape
        # Flatten batch and register dims for gathering
        flat_idx = topk_idx.view(B * R * K)  # [B*R*K]
        selected_concepts = self.concepts[flat_idx]  # [B*R*K, D]
        selected_concepts = selected_concepts.view(B, R, K, D)  # [B, R, K, D]
        
        # For output, flatten to [B, R*K, D]
        output_concepts = selected_concepts.view(B, R * K, D)
        
        # Project queries to concept space for commitment loss
        query_projected = self.query_to_concept(queries)  # [B, R, D]
        
        info = {
            'indices': topk_idx,  # [B, R, K]
            'scores': scores,  # [B, R, n_concepts]
            'topk_scores': topk_scores,  # [B, R, K]
            'selected_concepts': selected_concepts,  # [B, R, K, D]
            'query_projected': query_projected,  # [B, R, D]
        }
        
        return output_concepts, info


def compute_concept_losses(info, n_concepts, 
                           lambda_load=0.01,
                           lambda_entropy=0.001, 
                           lambda_codebook=0.25,
                           lambda_commitment=0.25,
                           lambda_diversity=0.01,
                           temp=0.1):
    """
    Compute all auxiliary losses for the concept memory layer.
    
    Args:
        info: dict from SparseConceptMemory forward pass
        n_concepts: total number of concepts
        lambda_*: loss weights
        temp: temperature for softmax
    """
    indices = info['indices']  # [B, R, K]
    scores = info['scores']  # [B, R, N]
    selected_concepts = info['selected_concepts']  # [B, R, K, D]
    query_projected = info['query_projected']  # [B, R, D]
    
    B, R, K, D = selected_concepts.shape
    
    # ========== 1. LOAD BALANCING LOSS ==========
    # Encourage uniform concept usage across batch
    usage_counts = torch.zeros(n_concepts, device=scores.device)
    usage_counts.scatter_add_(0, indices.flatten(), 
                              torch.ones_like(indices.flatten(), dtype=torch.float))
    
    # Target: each concept selected equally often
    target_usage = (B * R * K) / n_concepts
    load_balance_loss = ((usage_counts - target_usage) ** 2).mean()
    
    # ========== 2. ENTROPY REGULARIZATION ==========
    # Each register should maintain diverse options (high entropy distribution)
    probs = F.softmax(scores / temp, dim=-1)  # [B, R, N]
    entropy = -(probs * torch.clamp(probs, 1e-10).log()).sum(dim=-1).mean()
    entropy_loss = -entropy  # Maximize entropy = minimize negative entropy
    
    # ========== 3. CODEBOOK LOSS ==========
    # Move concept embeddings toward queries that select them
    # Stop gradient on queries so only concepts move
    query_projected_expanded = query_projected.unsqueeze(2)  # [B, R, 1, D]
    codebook_loss = F.mse_loss(selected_concepts, query_projected_expanded.detach())
    
    # ========== 4. COMMITMENT LOSS ==========
    # Move queries toward the concepts they selected
    # Stop gradient on concepts so only query network moves
    # Average over K selected concepts per register
    selected_mean = selected_concepts.mean(dim=2)  # [B, R, D]
    commitment_loss = F.mse_loss(query_projected, selected_mean.detach())
    
    # ========== 5. DIVERSITY LOSS ==========
    # Selected concepts within a batch shouldn't be too similar
    selected_flat = selected_concepts.view(B * R * K, D)  # [B*R*K, D]
    
    if selected_flat.shape[0] > 1:
        # Normalize for cosine similarity
        normalized = F.normalize(selected_flat, dim=-1)
        # Only compute for a subset to save memory for large batches
        n_samples = min(512, selected_flat.shape[0])
        indices_sample = torch.randperm(selected_flat.shape[0], device=normalized.device)[:n_samples]
        normalized_sample = normalized[indices_sample]
        
        # Pairwise cosine similarity
        similarity_matrix = normalized_sample @ normalized_sample.T
        # Penalize high off-diagonal similarities
        mask = ~torch.eye(n_samples, device=scores.device, dtype=torch.bool)
        diversity_loss = (similarity_matrix[mask].clamp(min=0) ** 2).mean()
    else:
        diversity_loss = torch.tensor(0.0, device=scores.device)
    
    # ========== TOTAL LOSS ==========
    total_loss = (
        lambda_load * load_balance_loss +
        lambda_entropy * entropy_loss +
        lambda_codebook * codebook_loss +
        lambda_commitment * commitment_loss +
        lambda_diversity * diversity_loss
    )
    
    # ========== METRICS FOR LOGGING ==========
    with torch.no_grad():
        metrics = {
            'concept_usage_std': usage_counts.std().item(),
            'concept_usage_max': usage_counts.max().item(),
            'concept_usage_min': usage_counts.min().item(),
            'concept_usage_mean': usage_counts.mean().item(),
            'dead_concepts': (usage_counts == 0).sum().item(),
            'entropy': entropy.item(),
            'load_balance_loss': load_balance_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'codebook_loss': codebook_loss.item(),
            'commitment_loss': commitment_loss.item(),
            'diversity_loss': diversity_loss.item() if isinstance(diversity_loss, torch.Tensor) else diversity_loss,
        }
    
    return total_loss, metrics


# ========== USAGE EXAMPLE ==========
if __name__ == "__main__":
    concept_layer = ConceptBank(
        n_concepts=100_000,
        concept_dim=1152,
        key_dim=256,
        n_registers=64,
        topk_per_register=1,
    )
    
    print("Parameters:", sum(p.numel() for p in concept_layer.parameters()))

    # In your training loop
    registers = torch.randn(4, 64, 1152)  # [B, n_registers, D]
    
    # Forward pass
    concepts, info = concept_layer(registers)
    # concepts: [4, 64, 768] - retrieved concept tokens
    
    # Compute auxiliary losses
    aux_loss, metrics = compute_concept_losses(
        info, 
        n_concepts=concept_layer.n_concepts
    )
    
    # Your main diffusion loss
    # main_loss = ...
    
    # Total loss
    # total_loss = main_loss + aux_loss
    
    # Log metrics
    print(f"Dead concepts: {metrics['dead_concepts']}")
    print(f"Usage std: {metrics['concept_usage_std']:.4f}")
    print(f"Entropy: {metrics['entropy']:.4f}")

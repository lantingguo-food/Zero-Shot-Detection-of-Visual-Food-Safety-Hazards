"""
Multi-Source Graph Fusion Module
Implements Equations 10-11 from the paper
"""

import torch
import torch.nn as nn
from .gcn import GCN


class MultiSourceGraphFusion(nn.Module):
    """
    Fuses knowledge from multiple graph sources using multi-head attention
    Implements: S = φ(Q, E_f, E_w2v) = MHA(QW_Q, E_fW_K, E_w2vW_V)
    """
    
    def __init__(self,
                 num_classes: int,
                 semantic_dim: int,
                 hidden_dim: int,
                 embedding_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        
        # GCN for each knowledge source (Equation 10)
        self.gcn_fskg = GCN(semantic_dim, hidden_dim, embedding_dim, dropout)
        self.gcn_hyperclass = GCN(semantic_dim, hidden_dim, embedding_dim, dropout)
        self.gcn_cooccur = GCN(semantic_dim, hidden_dim, embedding_dim, dropout)
        
        # Learnable queries
        self.queries = nn.Parameter(torch.randn(num_classes, hidden_dim))
        
        # Multi-head attention for fusion (Equation 11)
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Projection layers for attention
        self.query_proj = nn.Linear(hidden_dim, embedding_dim)
        self.key_proj = nn.Linear(embedding_dim, embedding_dim)
        self.value_proj = nn.Linear(embedding_dim, embedding_dim)
        
        # Cross-attention for word and attribute fusion
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
    def forward(self,
                semantic_vectors: torch.Tensor,
                adj_fskg: torch.Tensor,
                adj_hyperclass: torch.Tensor,
                adj_cooccur: torch.Tensor) -> torch.Tensor:
        """
        Args:
            semantic_vectors: [C, semantic_dim] - BERT embeddings for classes
            adj_fskg: [C, C] - FSKG adjacency matrix
            adj_hyperclass: [C, C] - Hyperclass adjacency matrix
            adj_cooccur: [C, C] - Co-occurrence adjacency matrix
            
        Returns:
            knowledge_repr: [C, embedding_dim] - Fused knowledge representations
        """
        
        # Get graph embeddings from each source (Equation 10)
        E_fskg = self.gcn_fskg(semantic_vectors, adj_fskg)
        E_hyperclass = self.gcn_hyperclass(semantic_vectors, adj_hyperclass)
        E_cooccur = self.gcn_cooccur(semantic_vectors, adj_cooccur)
        
        # Combine embeddings from all sources
        E_combined = (E_fskg + E_hyperclass + E_cooccur) / 3.0
        
        # Also get word embeddings (direct semantic)
        E_word = self.key_proj(semantic_vectors)
        
        # Cross-attention between graph embeddings and word embeddings
        E_fused, _ = self.cross_attn(
            E_combined.unsqueeze(0),
            E_word.unsqueeze(0),
            E_word.unsqueeze(0)
        )
        E_fused = E_fused.squeeze(0)
        
        # Project queries
        Q = self.query_proj(self.queries)
        
        # Multi-head attention fusion (Equation 11)
        # S = MHA(QW_Q, E_fW_K, E_w2vW_V)
        K = self.key_proj(E_fused)
        V = self.value_proj(E_word)
        
        knowledge_repr, _ = self.multihead_attn(
            Q.unsqueeze(0),
            K.unsqueeze(0),
            V.unsqueeze(0)
        )
        
        return knowledge_repr.squeeze(0)

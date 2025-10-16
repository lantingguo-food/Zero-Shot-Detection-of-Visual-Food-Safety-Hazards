"""
Graph Convolutional Network for Knowledge Graph Embedding
Implements Equation 3 from the paper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(nn.Module):
    """Single layer of graph convolution"""
    
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features [N, in_features]
            adj: Normalized adjacency matrix [N, N]
        Returns:
            Updated node features [N, out_features]
        """
        support = torch.mm(x, self.weight)
        output = torch.spmm(adj, support) if adj.is_sparse else torch.mm(adj, support)
        return output + self.bias


class GCN(nn.Module):
    """
    Two-layer Graph Convolutional Network
    Implements: E = ψ(X, A) = Ã·σ(ÃXW1)W2
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int, 
                 output_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        
        self.gc1 = GraphConvolution(input_dim, hidden_dim)
        self.gc2 = GraphConvolution(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Initial node features [N, input_dim]
            adj: Normalized adjacency matrix [N, N]
        Returns:
            Node embeddings [N, output_dim]
        """
        # First layer with ReLU
        h = self.gc1(x, adj)
        h = F.relu(h)
        h = self.dropout(h)
        
        # Second layer
        h = self.gc2(h, adj)
        
        return h


class KnowledgeGraphEmbedding(nn.Module):
    """
    Embeds knowledge graph with multiple relation types
    Implements Equations 3-5 from the paper
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_relations: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        
        self.num_relations = num_relations
        
        # Separate GCN for each relation type
        self.gcns = nn.ModuleList([
            GCN(input_dim, hidden_dim, output_dim, dropout)
            for _ in range(num_relations)
        ])
        
        # Learnable attention weights for relation aggregation (Equation 5)
        self.relation_attention = nn.Parameter(torch.ones(num_relations) / num_relations)
        
    def forward(self, 
                x: torch.Tensor, 
                adjacencies: list) -> torch.Tensor:
        """
        Args:
            x: Initial node features [N, input_dim]
            adjacencies: List of normalized adjacency matrices, one per relation
        Returns:
            Final node embeddings [N, output_dim]
        """
        # Compute embeddings for each relation type (Equation 4)
        embeddings = []
        for gcn, adj in zip(self.gcns, adjacencies):
            emb = gcn(x, adj)
            embeddings.append(emb)
            
        # Stack embeddings [num_relations, N, output_dim]
        embeddings = torch.stack(embeddings, dim=0)
        
        # Normalize attention weights
        alpha = F.softmax(self.relation_attention, dim=0)
        
        # Weighted aggregation (Equation 5)
        # E_final = Σ α_r * E^(r)
        final_embeddings = torch.sum(
            embeddings * alpha.view(-1, 1, 1),
            dim=0
        )
        
        return final_embeddings

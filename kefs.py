"""
Knowledge-Enhanced Feature Synthesizer (KEFS)
Combines MSGF and RFDM for feature synthesis
"""

import torch
import torch.nn as nn
from .msgf import MultiSourceGraphFusion
from .rfdm import RegionFeatureDiffusion


class KEFS(nn.Module):
    """
    Complete Knowledge-Enhanced Feature Synthesizer
    Combines Multi-Source Graph Fusion and Region Feature Diffusion
    """
    
    def __init__(self, config: dict):
        super().__init__()
        
        # Multi-Source Graph Fusion Module
        self.msgf = MultiSourceGraphFusion(
            num_classes=config['dataset']['num_seen_classes'] + 
                       config['dataset']['num_unseen_classes'],
            semantic_dim=config['semantic']['projection_dim'],
            hidden_dim=config['kefs']['hidden_dim'],
            embedding_dim=config['kefs']['knowledge_dim'],
            num_heads=config['kefs']['num_heads'],
            dropout=config['kefs']['dropout']
        )
        
        # Region Feature Diffusion Model
        self.rfdm = RegionFeatureDiffusion(
            feature_dim=config['detector']['feature_dim'],
            condition_dim=config['kefs']['knowledge_dim'],
            num_timesteps=config['rfdm']['num_timesteps'],
            beta_start=config['rfdm']['beta_start'],
            beta_end=config['rfdm']['beta_end'],
            hidden_dims=config['rfdm']['hidden_dims']
        )
        
        # Discriminator for adversarial training (optional)
        self.discriminator = nn.Sequential(
            nn.Linear(config['detector']['feature_dim'], 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
        
    def get_knowledge_representations(self,
                                     semantic_vectors: torch.Tensor,
                                     adj_fskg: torch.Tensor,
                                     adj_hyperclass: torch.Tensor,
                                     adj_cooccur: torch.Tensor) -> torch.Tensor:
        """Get knowledge representations from MSGF"""
        return self.msgf(semantic_vectors, adj_fskg, adj_hyperclass, adj_cooccur)
    
    def synthesize_features(self,
                          knowledge_repr: torch.Tensor,
                          num_samples: int = 1) -> torch.Tensor:
        """
        Synthesize features using RFDM
        
        Args:
            knowledge_repr: [B, knowledge_dim] or [1, knowledge_dim]
            num_samples: Number of features to generate per class
            
        Returns:
            Synthesized features [B * num_samples, feature_dim]
        """
        return self.rfdm.sample(knowledge_repr, num_samples)
    
    def forward_train(self,
                     real_features: torch.Tensor,
                     semantic_vectors: torch.Tensor,
                     class_ids: torch.Tensor,
                     adj_fskg: torch.Tensor,
                     adj_hyperclass: torch.Tensor,
                     adj_cooccur: torch.Tensor) -> dict:
        """
        Training forward pass
        
        Args:
            real_features: Real region features [B, feature_dim]
            semantic_vectors: Semantic vectors for all classes [C, semantic_dim]
            class_ids: Class IDs for each feature [B]
            adj_fskg, adj_hyperclass, adj_cooccur: Adjacency matrices [C, C]
            
        Returns:
            Dictionary with losses and outputs
        """
        # Get knowledge representations
        knowledge_repr = self.msgf(
            semantic_vectors, adj_fskg, adj_hyperclass, adj_cooccur
        )
        
        # Get knowledge for current batch classes
        batch_knowledge = knowledge_repr[class_ids]
        
        # RFDM training
        rfdm_output = self.rfdm(real_features, batch_knowledge)
        
        # Discriminator on real features
        real_validity = self.discriminator(real_features)
        
        # Generate fake features for discriminator training
        with torch.no_grad():
            fake_features = self.rfdm.sample(batch_knowledge, num_samples=1)
        fake_validity = self.discriminator(fake_features)
        
        return {
            'rfdm_loss': rfdm_output['loss'],
            'knowledge_repr': knowledge_repr,
            'real_validity': real_validity,
            'fake_validity': fake_validity,
            'predicted_noise': rfdm_output['predicted_noise'],
            'true_noise': rfdm_output['true_noise']
        }
    
    @torch.no_grad()
    def synthesize_dataset(self,
                          semantic_vectors: torch.Tensor,
                          class_ids: list,
                          adj_fskg: torch.Tensor,
                          adj_hyperclass: torch.Tensor,
                          adj_cooccur: torch.Tensor,
                          num_samples_per_class: int = 500) -> tuple:
        """
        Synthesize complete dataset for unseen classes
        
        Args:
            semantic_vectors: [C, semantic_dim]
            class_ids: List of class IDs to synthesize
            adjacency matrices
            num_samples_per_class: Number of features per class
            
        Returns:
            features: [N, feature_dim]
            labels: [N]
        """
        # Get knowledge representations
        knowledge_repr = self.msgf(
            semantic_vectors, adj_fskg, adj_hyperclass, adj_cooccur
        )
        
        all_features = []
        all_labels = []
        
        for class_id in class_ids:
            # Get knowledge for this class
            class_knowledge = knowledge_repr[class_id:class_id+1]
            
            # Synthesize features
            features = self.rfdm.sample(class_knowledge, num_samples_per_class)
            
            # Store
            all_features.append(features)
            all_labels.append(torch.full((num_samples_per_class,), 
                                        class_id, 
                                        dtype=torch.long,
                                        device=features.device))
        
        return torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0)

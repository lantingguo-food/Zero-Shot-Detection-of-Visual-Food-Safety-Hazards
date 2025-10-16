"""
Loss functions for KEFS training
Implements Equations 15-17 from the paper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphDenoisingLoss(nn.Module):
    """
    Graph Denoising Loss (Equation 16)
    L_G = -1/C * Σ[y_i*log(ŝ_i) - α*ŝ_i*log(σ(b̂_i^k))]
    """
    
    def __init__(self, alpha: float = 0.5, num_sources: int = 3):
        super().__init__()
        self.alpha = alpha
        self.num_sources = num_sources
        
    def forward(self,
               knowledge_repr: torch.Tensor,
               adjacencies: list,
               labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            knowledge_repr: S matrix [C, d] 
            adjacencies: List of A^k matrices [C, C]
            labels: One-hot labels [C, C] for class predictions
            
        Returns:
            Graph denoising loss scalar
        """
        C = knowledge_repr.shape[0]
        
        # ŝ_i = σ(s_i)
        s_hat = torch.sigmoid(knowledge_repr)
        
        # First term: -y_i * log(ŝ_i)
        # For diagonal, y_i = 1 for correct class
        first_term = -torch.mean(labels * torch.log(s_hat + 1e-8))
        
        # Second term: -α * ŝ_i * log(σ(b̂_i^k))
        # b̂_i^k = i-th row of A^k * S
        second_term = 0
        for adj in adjacencies:
            # Compute b̂^k = A^k @ S
            b_hat = torch.mm(adj, knowledge_repr)
            b_hat_sigmoid = torch.sigmoid(b_hat)
            
            # -α * ŝ_i * log(σ(b̂_i^k))
            second_term += -self.alpha * torch.mean(
                s_hat * torch.log(b_hat_sigmoid + 1e-8)
            )
        
        second_term = second_term / self.num_sources
        
        return first_term + second_term


class WassersteinLoss(nn.Module):
    """Wasserstein GAN loss for feature generation"""
    
    def __init__(self, lambda_gp: float = 10.0):
        super().__init__()
        self.lambda_gp = lambda_gp
        
    def discriminator_loss(self,
                          real_validity: torch.Tensor,
                          fake_validity: torch.Tensor,
                          real_features: torch.Tensor,
                          fake_features: torch.Tensor) -> dict:
        """
        Compute discriminator loss with gradient penalty
        
        Returns:
            Dictionary with loss components
        """
        # Wasserstein loss
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity)
        
        # Gradient penalty
        gp = self._gradient_penalty(real_features, fake_features)
        
        total_loss = d_loss + self.lambda_gp * gp
        
        return {
            'total': total_loss,
            'wasserstein': d_loss,
            'gradient_penalty': gp
        }
    
    def generator_loss(self, fake_validity: torch.Tensor) -> torch.Tensor:
        """Generator loss (want discriminator to think fakes are real)"""
        return -torch.mean(fake_validity)
    
    def _gradient_penalty(self,
                         real_features: torch.Tensor,
                         fake_features: torch.Tensor) -> torch.Tensor:
        """Compute gradient penalty for WGAN-GP"""
        batch_size = real_features.shape[0]
        
        # Random weight term for interpolation
        alpha = torch.rand(batch_size, 1, device=real_features.device)
        
        # Get random interpolation between real and fake
        interpolates = (alpha * real_features + 
                       (1 - alpha) * fake_features).requires_grad_(True)
        
        # Get discriminator output for interpolated features
        d_interpolates = self.discriminator(interpolates)
        
        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Compute gradient penalty
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        
        return gradient_penalty


class KEFSLoss(nn.Module):
    """
    Complete KEFS loss combining all components (Equation 19)
    L_KEFS = L_W + λ1*L_R + λ2*L_G
    """
    
    def __init__(self, 
                 lambda1: float = 0.1, 
                 lambda2: float = 0.1,
                 alpha: float = 0.5):
        super().__init__()
        
        self.lambda1 = lambda1  # Region diffusion loss weight
        self.lambda2 = lambda2  # Graph denoising loss weight
        
        self.wasserstein_loss = WassersteinLoss()
        self.graph_loss = GraphDenoisingLoss(alpha=alpha)
        
    def forward(self,
               real_features: torch.Tensor,
               fake_features: torch.Tensor,
               real_validity: torch.Tensor,
               fake_validity: torch.Tensor,
               rfdm_loss: torch.Tensor,
               knowledge_repr: torch.Tensor,
               adjacencies: list,
               labels: torch.Tensor,
               mode: str = 'generator') -> dict:
        """
        Compute total KEFS loss
        
        Args:
            mode: 'generator' or 'discriminator'
            
        Returns:
            Dictionary with all loss components
        """
        
        if mode == 'discriminator':
            # Discriminator loss (Wasserstein)
            d_losses = self.wasserstein_loss.discriminator_loss(
                real_validity, fake_validity, real_features, fake_features
            )
            return {
                'total': d_losses['total'],
                'wasserstein': d_losses['wasserstein'],
                'gradient_penalty': d_losses['gradient_penalty']
            }
        
        else:  # generator mode
            # Generator Wasserstein loss
            gen_loss = self.wasserstein_loss.generator_loss(fake_validity)
            
            # Region diffusion loss (L_R)
            region_loss = rfdm_loss
            
            # Graph denoising loss (L_G)
            graph_loss = self.graph_loss(knowledge_repr, adjacencies, labels)
            
            # Total loss (Equation 19)
            total_loss = gen_loss + self.lambda1 * region_loss + self.lambda2 * graph_loss
            
            return {
                'total': total_loss,
                'wasserstein': gen_loss,
                'region_diffusion': region_loss,
                'graph_denoising': graph_loss
            }


class DetectionLoss(nn.Module):
    """Detection losses for Faster R-CNN (Equation 17)"""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, 
                cls_scores: torch.Tensor,
                bbox_preds: torch.Tensor,
                labels: torch.Tensor,
                bbox_targets: torch.Tensor,
                bbox_weights: torch.Tensor) -> dict:
        """
        Args:
            cls_scores: [N, num_classes]
            bbox_preds: [N, num_classes*4]
            labels: [N]
            bbox_targets: [N, 4]
            bbox_weights: [N, 4]
            
        Returns:
            Dictionary with classification and regression losses
        """
        # Classification loss
        cls_loss = F.cross_entropy(cls_scores, labels, reduction='mean')
        
        # Regression loss (Smooth L1)
        # Only compute for positive samples
        pos_inds = labels > 0
        if pos_inds.sum() > 0:
            pos_bbox_preds = bbox_preds[pos_inds]
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_weights = bbox_weights[pos_inds]
            
            bbox_loss = F.smooth_l1_loss(
                pos_bbox_preds * pos_bbox_weights,
                pos_bbox_targets * pos_bbox_weights,
                reduction='sum'
            ) / max(pos_inds.sum(), 1.0)
        else:
            bbox_loss = torch.tensor(0.0, device=cls_scores.device)
        
        return {
            'cls_loss': cls_loss,
            'bbox_loss': bbox_loss,
            'total': cls_loss + bbox_loss
        }

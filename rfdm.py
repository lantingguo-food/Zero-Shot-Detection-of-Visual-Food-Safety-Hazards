"""
Region Feature Diffusion Model (RFDM)
Implements the diffusion-based feature synthesis (Equations 12-15)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SinusoidalPositionEmbeddings(nn.Module):
    """Timestep embeddings"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class NoisePredictor(nn.Module):
    """Neural network to predict noise z_θ(h_t, t, s)"""
    
    def __init__(self,
                 feature_dim: int,
                 condition_dim: int,
                 hidden_dims: list,
                 time_dim: int = 128):
        super().__init__()
        
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.ReLU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # Network to predict noise
        layers = []
        in_dim = feature_dim + condition_dim + time_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            in_dim = hidden_dim
            
        layers.append(nn.Linear(in_dim, feature_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, 
                h_t: torch.Tensor,
                t: torch.Tensor,
                condition: torch.Tensor) -> torch.Tensor:
        """
        Predict noise component
        
        Args:
            h_t: Noisy features [B, feature_dim]
            t: Timestep [B]
            condition: Knowledge representation [B, condition_dim]
            
        Returns:
            Predicted noise [B, feature_dim]
        """
        t_emb = self.time_embed(t)
        x = torch.cat([h_t, condition, t_emb], dim=-1)
        return self.network(x)


class RegionFeatureDiffusion(nn.Module):
    """
    Region Feature Diffusion Model
    Implements forward diffusion (Eq 12) and reverse denoising (Eq 13-14)
    """
    
    def __init__(self,
                 feature_dim: int = 2048,
                 condition_dim: int = 512,
                 num_timesteps: int = 100,
                 beta_start: float = 0.00085,
                 beta_end: float = 0.012,
                 hidden_dims: list = [512, 512, 512]):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.condition_dim = condition_dim
        self.num_timesteps = num_timesteps
        
        # Noise schedule (linear) - Equation 12
        self.register_buffer('betas', 
                            torch.linspace(beta_start, beta_end, num_timesteps))
        
        # Pre-compute diffusion parameters
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        self.register_buffer('alphas_cumprod_prev',
                            F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0))
        
        # Equation 12: sqrt(1 - gamma_t) where gamma_t = 1 - alpha_t
        self.register_buffer('sqrt_alphas_cumprod', 
                            torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                            torch.sqrt(1.0 - self.alphas_cumprod))
        
        # For Equation 14
        self.register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / self.alphas))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                            torch.sqrt(1.0 / self.alphas_cumprod - 1))
        
        # Noise predictor network
        self.noise_predictor = NoisePredictor(
            feature_dim, condition_dim, hidden_dims
        )
        
    def q_sample(self, 
                 h_0: torch.Tensor,
                 t: torch.Tensor,
                 noise: torch.Tensor = None) -> torch.Tensor:
        """
        Forward diffusion process (Equation 12)
        h_t = sqrt(1 - gamma_t) * h_{t-1} + sqrt(gamma_t) * z_t
        
        Or in closed form:
        h_t = sqrt(alpha_bar_t) * h_0 + sqrt(1 - alpha_bar_t) * z
        """
        if noise is None:
            noise = torch.randn_like(h_0)
            
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, h_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, h_0.shape
        )
        
        return sqrt_alphas_cumprod_t * h_0 + sqrt_one_minus_alphas_cumprod_t * noise
    
    def predict_start_from_noise(self,
                                 h_t: torch.Tensor,
                                 t: torch.Tensor,
                                 noise: torch.Tensor) -> torch.Tensor:
        """Predict h_0 from h_t and noise"""
        sqrt_recip_alphas_cumprod = self._extract(
            torch.sqrt(1.0 / self.alphas_cumprod), t, h_t.shape
        )
        sqrt_recipm1_alphas_cumprod = self._extract(
            self.sqrt_recipm1_alphas_cumprod, t, h_t.shape
        )
        
        return sqrt_recip_alphas_cumprod * h_t - sqrt_recipm1_alphas_cumprod * noise
    
    def p_mean_variance(self,
                       h_t: torch.Tensor,
                       t: torch.Tensor,
                       condition: torch.Tensor) -> tuple:
        """
        Compute mean and variance for reverse step (Equation 13-14)
        """
        # Predict noise
        predicted_noise = self.noise_predictor(h_t, t, condition)
        
        # Predict h_0
        h_0_pred = self.predict_start_from_noise(h_t, t, predicted_noise)
        
        # Compute mean (Equation 14)
        sqrt_recip_alphas_t = self._extract(self.sqrt_recip_alphas, t, h_t.shape)
        betas_t = self._extract(self.betas, t, h_t.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, h_t.shape
        )
        
        mean = sqrt_recip_alphas_t * (
            h_t - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
        )
        
        # Variance (simplified, use betas)
        variance = self._extract(self.betas, t, h_t.shape)
        
        return mean, variance, predicted_noise
    
    @torch.no_grad()
    def p_sample(self,
                 h_t: torch.Tensor,
                 t: torch.Tensor,
                 condition: torch.Tensor) -> torch.Tensor:
        """
        Single reverse diffusion step
        Sample h_{t-1} from p_θ(h_{t-1}|h_t, s)
        """
        mean, variance, _ = self.p_mean_variance(h_t, t, condition)
        
        noise = torch.randn_like(h_t)
        # No noise when t == 0
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(h_t.shape) - 1)))
        
        return mean + nonzero_mask * torch.sqrt(variance) * noise
    
    @torch.no_grad()
    def sample(self,
              condition: torch.Tensor,
              num_samples: int = None) -> torch.Tensor:
        """
        Generate features from noise using reverse diffusion
        
        Args:
            condition: Knowledge representations [B, condition_dim] or [1, condition_dim]
            num_samples: Number of features to generate per condition
            
        Returns:
            Generated features [B * num_samples, feature_dim]
        """
        device = condition.device
        batch_size = condition.shape[0]
        
        if num_samples is not None and num_samples > 1:
            # Repeat condition for multiple samples
            condition = condition.repeat_interleave(num_samples, dim=0)
            batch_size = condition.shape[0]
        
        # Start from pure noise
        h_t = torch.randn(batch_size, self.feature_dim, device=device)
        
        # Reverse diffusion
        for t_idx in reversed(range(self.num_timesteps)):
            t = torch.full((batch_size,), t_idx, device=device, dtype=torch.long)
            h_t = self.p_sample(h_t, t, condition)
            
        return h_t
    
    def forward(self,
               h_0: torch.Tensor,
               condition: torch.Tensor,
               noise: torch.Tensor = None) -> dict:
        """
        Training forward pass
        
        Args:
            h_0: Clean region features [B, feature_dim]
            condition: Knowledge representations [B, condition_dim]
            noise: Optional pre-sampled noise
            
        Returns:
            Dictionary with loss components
        """
        batch_size = h_0.shape[0]
        device = h_0.device
        
        # Sample random timesteps
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=device).long()
        
        # Sample noise
        if noise is None:
            noise = torch.randn_like(h_0)
        
        # Forward diffusion
        h_t = self.q_sample(h_0, t, noise)
        
        # Predict noise
        predicted_noise = self.noise_predictor(h_t, t, condition)
        
        # Compute loss (Equation 15)
        loss = F.mse_loss(predicted_noise, noise)
        
        return {
            'loss': loss,
            'predicted_noise': predicted_noise,
            'true_noise': noise
        }
    
    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: tuple) -> torch.Tensor:
        """Extract values from a at indices t and reshape for broadcasting"""
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

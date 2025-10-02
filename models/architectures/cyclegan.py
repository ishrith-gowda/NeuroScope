"""CycleGAN model architecture implementation.

This module provides the complete CycleGAN model implementation including
generators, discriminators, and loss computation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

from neuroscope.core.logging import get_logger
from neuroscope.models.generators.resnet_generator import ResNetGenerator
from neuroscope.models.discriminators.patch_discriminator import PatchDiscriminator

logger = get_logger(__name__)


class CycleGAN(nn.Module):
    """Complete CycleGAN model with generators and discriminators."""
    
    def __init__(
        self,
        input_channels: int = 4,
        output_channels: int = 4,
        generator_channels: int = 64,
        discriminator_channels: int = 64,
        n_residual_blocks: int = 9,
        lambda_cycle: float = 10.0,
        lambda_identity: float = 5.0
    ):
        """Initialize CycleGAN model.
        
        Args:
            input_channels: Number of input channels
            output_channels: Number of output channels
            generator_channels: Number of generator channels
            discriminator_channels: Number of discriminator channels
            n_residual_blocks: Number of residual blocks in generators
            lambda_cycle: Cycle consistency loss weight
            lambda_identity: Identity loss weight
        """
        super(CycleGAN, self).__init__()
        
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity
        
        # Initialize generators
        self.G_A2B = ResNetGenerator(
            input_channels=input_channels,
            output_channels=output_channels,
            channels=generator_channels,
            n_residual_blocks=n_residual_blocks
        )
        
        self.G_B2A = ResNetGenerator(
            input_channels=input_channels,
            output_channels=output_channels,
            channels=generator_channels,
            n_residual_blocks=n_residual_blocks
        )
        
        # Initialize discriminators
        self.D_A = PatchDiscriminator(
            input_channels=input_channels,
            channels=discriminator_channels
        )
        
        self.D_B = PatchDiscriminator(
            input_channels=input_channels,
            channels=discriminator_channels
        )
        
        # Initialize weights
        self.apply(self._weights_init_normal)
    
    def _weights_init_normal(self, m):
        """Initialize weights using normal distribution."""
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)
    
    def forward(self, real_a: torch.Tensor, real_b: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through the CycleGAN model.
        
        Args:
            real_a: Real images from domain A
            real_b: Real images from domain B
            
        Returns:
            Dictionary containing generated images and reconstructions
        """
        # Generate fake images
        fake_b = self.G_A2B(real_a)
        fake_a = self.G_B2A(real_b)
        
        # Reconstruct images
        rec_a = self.G_B2A(fake_b)
        rec_b = self.G_A2B(fake_a)
        
        # Identity images
        id_a = self.G_B2A(real_a)
        id_b = self.G_A2B(real_b)
        
        return {
            'fake_a': fake_a,
            'fake_b': fake_b,
            'rec_a': rec_a,
            'rec_b': rec_b,
            'id_a': id_a,
            'id_b': id_b
        }
    
    def compute_generator_losses(
        self,
        real_a: torch.Tensor,
        real_b: torch.Tensor,
        fake_a: torch.Tensor,
        fake_b: torch.Tensor,
        rec_a: torch.Tensor,
        rec_b: torch.Tensor,
        id_a: torch.Tensor,
        id_b: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute generator losses.
        
        Args:
            real_a: Real images from domain A
            real_b: Real images from domain B
            fake_a: Generated images for domain A
            fake_b: Generated images for domain B
            rec_a: Reconstructed images for domain A
            rec_b: Reconstructed images for domain B
            id_a: Identity images for domain A
            id_b: Identity images for domain B
            
        Returns:
            Dictionary of generator losses
        """
        # Adversarial losses
        pred_fake_a = self.D_A(fake_a)
        pred_fake_b = self.D_B(fake_b)
        
        loss_G_A2B = F.mse_loss(pred_fake_b, torch.ones_like(pred_fake_b))
        loss_G_B2A = F.mse_loss(pred_fake_a, torch.ones_like(pred_fake_a))
        
        # Cycle consistency losses
        loss_cycle_A = F.l1_loss(rec_a, real_a)
        loss_cycle_B = F.l1_loss(rec_b, real_b)
        
        # Identity losses
        loss_identity_A = F.l1_loss(id_a, real_a)
        loss_identity_B = F.l1_loss(id_b, real_b)
        
        # Total generator losses
        loss_G_A2B_total = (
            loss_G_A2B + 
            self.lambda_cycle * loss_cycle_B + 
            self.lambda_identity * loss_identity_B
        )
        
        loss_G_B2A_total = (
            loss_G_B2A + 
            self.lambda_cycle * loss_cycle_A + 
            self.lambda_identity * loss_identity_A
        )
        
        return {
            'G_A2B': loss_G_A2B_total,
            'G_B2A': loss_G_B2A_total,
            'G_A2B_adv': loss_G_A2B,
            'G_B2A_adv': loss_G_B2A,
            'cycle_A': loss_cycle_A,
            'cycle_B': loss_cycle_B,
            'identity_A': loss_identity_A,
            'identity_B': loss_identity_B,
            'total': loss_G_A2B_total + loss_G_B2A_total
        }
    
    def compute_discriminator_losses(
        self,
        real_a: torch.Tensor,
        real_b: torch.Tensor,
        fake_a: torch.Tensor,
        fake_b: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute discriminator losses.
        
        Args:
            real_a: Real images from domain A
            real_b: Real images from domain B
            fake_a: Generated images for domain A
            fake_b: Generated images for domain B
            
        Returns:
            Dictionary of discriminator losses
        """
        # Real images
        pred_real_a = self.D_A(real_a)
        pred_real_b = self.D_B(real_b)
        
        # Fake images
        pred_fake_a = self.D_A(fake_a.detach())
        pred_fake_b = self.D_B(fake_b.detach())
        
        # Discriminator losses
        loss_D_A = (
            F.mse_loss(pred_real_a, torch.ones_like(pred_real_a)) +
            F.mse_loss(pred_fake_a, torch.zeros_like(pred_fake_a))
        ) * 0.5
        
        loss_D_B = (
            F.mse_loss(pred_real_b, torch.ones_like(pred_real_b)) +
            F.mse_loss(pred_fake_b, torch.zeros_like(pred_fake_b))
        ) * 0.5
        
        return {
            'D_A': loss_D_A,
            'D_B': loss_D_B,
            'total': loss_D_A + loss_D_B
        }
    
    def generate_a2b(self, real_a: torch.Tensor) -> torch.Tensor:
        """Generate images from domain A to domain B.
        
        Args:
            real_a: Real images from domain A
            
        Returns:
            Generated images for domain B
        """
        with torch.no_grad():
            return self.G_A2B(real_a)
    
    def generate_b2a(self, real_b: torch.Tensor) -> torch.Tensor:
        """Generate images from domain B to domain A.
        
        Args:
            real_b: Real images from domain B
            
        Returns:
            Generated images for domain A
        """
        with torch.no_grad():
            return self.G_B2A(real_b)
    
    def get_model_info(self) -> Dict[str, any]:
        """Get model information including parameter counts.
        
        Returns:
            Dictionary containing model information
        """
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'G_A2B_parameters': count_parameters(self.G_A2B),
            'G_B2A_parameters': count_parameters(self.G_B2A),
            'D_A_parameters': count_parameters(self.D_A),
            'D_B_parameters': count_parameters(self.D_B),
            'total_parameters': count_parameters(self),
            'lambda_cycle': self.lambda_cycle,
            'lambda_identity': self.lambda_identity
        }
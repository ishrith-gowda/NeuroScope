"""
3d cyclegan architecture.

complete 3d cyclegan implementation for volumetric medical image
harmonization, including self-attention enhanced variants.
"""

from typing import Dict, Optional, Any, Tuple, List
from dataclasses import dataclass, field
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from .generator_3d import Generator3D, SAGenerator3D, UNetGenerator3D
from .discriminator_3d import Discriminator3D, MultiScaleDiscriminator3D


@dataclass
class CycleGAN3DConfig:
    """configuration for 3d cyclegan."""
    
    # input/output settings
    in_channels: int = 1
    out_channels: int = 1
    
    # volume dimensions (for memory estimation)
    volume_size: Tuple[int, int, int] = (64, 128, 128)  # d, h, w
    
    # generator settings
    ngf: int = 32  # reduced for 3d memory constraints
    n_residual: int = 6
    n_downsampling: int = 2
    generator_type: str = 'resnet'  # 'resnet', 'unet', 'sa_resnet'
    
    # discriminator settings
    ndf: int = 32
    n_layers: int = 3
    n_scales: int = 2  # fewer scales for 3d
    use_multi_scale: bool = True
    
    # loss weights
    lambda_cycle: float = 10.0
    lambda_identity: float = 0.5
    lambda_adversarial: float = 1.0
    
    # training settings
    use_amp: bool = True
    use_checkpoint: bool = True  # gradient checkpointing
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in self.__annotations__}


class CycleGAN3D(nn.Module):
    """
    3d cyclegan for volumetric image translation.
    
    enables unpaired image-to-image translation on 3d medical volumes
    for harmonization across different scanner/site domains.
    """
    
    def __init__(self, config: Optional[CycleGAN3DConfig] = None):
        super().__init__()
        
        self.config = config or CycleGAN3DConfig()
        
        # build generators
        self.G_A2B = self._build_generator()
        self.G_B2A = self._build_generator()
        
        # build discriminators
        if self.config.use_multi_scale:
            self.D_A = MultiScaleDiscriminator3D(
                in_channels=self.config.in_channels,
                ndf=self.config.ndf,
                n_layers=self.config.n_layers,
                n_scales=self.config.n_scales
            )
            self.D_B = MultiScaleDiscriminator3D(
                in_channels=self.config.in_channels,
                ndf=self.config.ndf,
                n_layers=self.config.n_layers,
                n_scales=self.config.n_scales
            )
        else:
            self.D_A = Discriminator3D(
                in_channels=self.config.in_channels,
                ndf=self.config.ndf,
                n_layers=self.config.n_layers
            )
            self.D_B = Discriminator3D(
                in_channels=self.config.in_channels,
                ndf=self.config.ndf,
                n_layers=self.config.n_layers
            )
        
        # loss functions
        self.criterion_gan = nn.MSELoss()  # lsgan
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()
    
    def _build_generator(self) -> nn.Module:
        """build generator based on config."""
        if self.config.generator_type == 'unet':
            return UNetGenerator3D(
                in_channels=self.config.in_channels,
                out_channels=self.config.out_channels,
                ngf=self.config.ngf,
                n_levels=self.config.n_downsampling,
                use_checkpoint=self.config.use_checkpoint
            )
        elif self.config.generator_type == 'sa_resnet':
            return SAGenerator3D(
                in_channels=self.config.in_channels,
                out_channels=self.config.out_channels,
                ngf=self.config.ngf,
                n_downsampling=self.config.n_downsampling,
                n_residual=self.config.n_residual,
                use_checkpoint=self.config.use_checkpoint
            )
        else:
            return Generator3D(
                in_channels=self.config.in_channels,
                out_channels=self.config.out_channels,
                ngf=self.config.ngf,
                n_downsampling=self.config.n_downsampling,
                n_residual=self.config.n_residual,
                use_checkpoint=self.config.use_checkpoint
            )
    
    def forward(
        self,
        real_A: torch.Tensor,
        real_B: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        forward pass computing all generated volumes.
        
        args:
            real_a: volumes from domain a [b, c, d, h, w]
            real_b: volumes from domain b [b, c, d, h, w]
            
        returns:
            dictionary of generated volumes
        """
        # forward cycle a -> b -> a
        fake_B = self.G_A2B(real_A)
        rec_A = self.G_B2A(fake_B)
        
        # forward cycle b -> a -> b
        fake_A = self.G_B2A(real_B)
        rec_B = self.G_A2B(fake_A)
        
        # identity mapping
        idt_A = self.G_B2A(real_A)
        idt_B = self.G_A2B(real_B)
        
        return {
            'fake_B': fake_B,
            'rec_A': rec_A,
            'fake_A': fake_A,
            'rec_B': rec_B,
            'idt_A': idt_A,
            'idt_B': idt_B
        }
    
    def compute_generator_loss(
        self,
        real_A: torch.Tensor,
        real_B: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        compute total generator loss.
        
        args:
            real_a: volumes from domain a
            real_b: volumes from domain b
            
        returns:
            tuple of (total_loss, loss_dict)
        """
        outputs = self.forward(real_A, real_B)
        
        # adversarial losses
        pred_fake_B = self.D_B(outputs['fake_B'])
        pred_fake_A = self.D_A(outputs['fake_A'])
        
        if isinstance(pred_fake_B, list):
            loss_gan_A2B = sum(
                self.criterion_gan(p, torch.ones_like(p)) 
                for p in pred_fake_B
            ) / len(pred_fake_B)
            loss_gan_B2A = sum(
                self.criterion_gan(p, torch.ones_like(p))
                for p in pred_fake_A
            ) / len(pred_fake_A)
        else:
            loss_gan_A2B = self.criterion_gan(
                pred_fake_B, torch.ones_like(pred_fake_B)
            )
            loss_gan_B2A = self.criterion_gan(
                pred_fake_A, torch.ones_like(pred_fake_A)
            )
        
        loss_gan = (loss_gan_A2B + loss_gan_B2A) * self.config.lambda_adversarial
        
        # cycle consistency losses
        loss_cycle_A = self.criterion_cycle(outputs['rec_A'], real_A)
        loss_cycle_B = self.criterion_cycle(outputs['rec_B'], real_B)
        loss_cycle = (loss_cycle_A + loss_cycle_B) * self.config.lambda_cycle
        
        # identity losses
        loss_idt_A = self.criterion_identity(outputs['idt_A'], real_A)
        loss_idt_B = self.criterion_identity(outputs['idt_B'], real_B)
        loss_identity = (loss_idt_A + loss_idt_B) * self.config.lambda_identity
        
        # total loss
        total_loss = loss_gan + loss_cycle + loss_identity
        
        loss_dict = {
            'loss_G': total_loss.item(),
            'loss_gan': loss_gan.item(),
            'loss_cycle': loss_cycle.item(),
            'loss_identity': loss_identity.item()
        }
        
        return total_loss, loss_dict
    
    def compute_discriminator_loss(
        self,
        real_A: torch.Tensor,
        real_B: torch.Tensor,
        fake_A: torch.Tensor,
        fake_B: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        compute discriminator loss.
        
        args:
            real_a: real volumes from domain a
            real_b: real volumes from domain b
            fake_a: generated volumes for domain a
            fake_b: generated volumes for domain b
            
        returns:
            tuple of (total_loss, loss_dict)
        """
        # d_a
        pred_real_A = self.D_A(real_A)
        pred_fake_A = self.D_A(fake_A.detach())
        
        if isinstance(pred_real_A, list):
            loss_D_A = sum(
                self.criterion_gan(p, torch.ones_like(p)) +
                self.criterion_gan(f, torch.zeros_like(f))
                for p, f in zip(pred_real_A, pred_fake_A)
            ) / (2 * len(pred_real_A))
        else:
            loss_D_A = (
                self.criterion_gan(pred_real_A, torch.ones_like(pred_real_A)) +
                self.criterion_gan(pred_fake_A, torch.zeros_like(pred_fake_A))
            ) / 2
        
        # d_b
        pred_real_B = self.D_B(real_B)
        pred_fake_B = self.D_B(fake_B.detach())
        
        if isinstance(pred_real_B, list):
            loss_D_B = sum(
                self.criterion_gan(p, torch.ones_like(p)) +
                self.criterion_gan(f, torch.zeros_like(f))
                for p, f in zip(pred_real_B, pred_fake_B)
            ) / (2 * len(pred_real_B))
        else:
            loss_D_B = (
                self.criterion_gan(pred_real_B, torch.ones_like(pred_real_B)) +
                self.criterion_gan(pred_fake_B, torch.zeros_like(pred_fake_B))
            ) / 2
        
        total_loss = (loss_D_A + loss_D_B) / 2
        
        loss_dict = {
            'loss_D': total_loss.item(),
            'loss_D_A': loss_D_A.item(),
            'loss_D_B': loss_D_B.item()
        }
        
        return total_loss, loss_dict
    
    def translate_A2B(self, x: torch.Tensor) -> torch.Tensor:
        """translate volume from domain a to b."""
        return self.G_A2B(x)
    
    def translate_B2A(self, x: torch.Tensor) -> torch.Tensor:
        """translate volume from domain b to a."""
        return self.G_B2A(x)
    
    def get_generator_params(self) -> List[torch.nn.Parameter]:
        """get all generator parameters."""
        return list(self.G_A2B.parameters()) + list(self.G_B2A.parameters())
    
    def get_discriminator_params(self) -> List[torch.nn.Parameter]:
        """get all discriminator parameters."""
        return list(self.D_A.parameters()) + list(self.D_B.parameters())
    
    def count_parameters(self) -> Dict[str, int]:
        """count parameters per component."""
        return {
            'G_A2B': sum(p.numel() for p in self.G_A2B.parameters()),
            'G_B2A': sum(p.numel() for p in self.G_B2A.parameters()),
            'D_A': sum(p.numel() for p in self.D_A.parameters()),
            'D_B': sum(p.numel() for p in self.D_B.parameters()),
            'total': sum(p.numel() for p in self.parameters())
        }
    
    def estimate_memory(self, batch_size: int = 1) -> Dict[str, float]:
        """
        estimate gpu memory requirements.
        
        returns memory estimates in gb.
        """
        D, H, W = self.config.volume_size
        C = self.config.in_channels
        
        # input tensor size
        input_size = batch_size * C * D * H * W * 4 / (1024**3)  # gb
        
        # model parameters
        param_size = sum(p.numel() * 4 for p in self.parameters()) / (1024**3)
        
        # gradients (same size as parameters)
        grad_size = param_size
        
        # activations (rough estimate: 3x input for encoder-decoder)
        activation_size = input_size * 10
        
        return {
            'input_gb': input_size,
            'params_gb': param_size,
            'gradients_gb': grad_size,
            'activations_gb': activation_size,
            'total_estimated_gb': input_size + param_size + grad_size + activation_size
        }


@dataclass
class SACycleGAN3DConfig(CycleGAN3DConfig):
    """configuration for self-attention 3d cyclegan."""
    
    # attention settings
    attention_type: str = 'self'  # 'self', 'multi_head', 'axial', 'cbam'
    attention_positions: List[int] = field(default_factory=lambda: [2, 4])
    num_attention_heads: int = 4
    
    # additional loss weights
    lambda_perceptual: float = 1.0
    lambda_tumor: float = 5.0


class SACycleGAN3D(CycleGAN3D):
    """
    self-attention enhanced 3d cyclegan.
    
    extends basic 3d cyclegan with self-attention mechanisms
    and additional medical imaging-specific losses.
    """
    
    def __init__(self, config: Optional[SACycleGAN3DConfig] = None):
        self.sa_config = config or SACycleGAN3DConfig()
        
        # override generator type
        self.sa_config.generator_type = 'sa_resnet'
        
        super().__init__(self.sa_config)
        
        # build additional loss functions
        self._build_losses()
    
    def _build_generator(self) -> nn.Module:
        """build self-attention generator."""
        return SAGenerator3D(
            in_channels=self.config.in_channels,
            out_channels=self.config.out_channels,
            ngf=self.config.ngf,
            n_downsampling=self.config.n_downsampling,
            n_residual=self.config.n_residual,
            attention_positions=self.sa_config.attention_positions,
            attention_type=self.sa_config.attention_type,
            use_checkpoint=self.config.use_checkpoint
        )
    
    def _build_losses(self):
        """build additional loss functions."""
        # perceptual loss (3d vgg-style)
        self.perceptual_loss = Perceptual3DLoss()
        
        # tumor preservation loss
        self.tumor_loss = TumorPreservation3DLoss()
    
    def compute_generator_loss(
        self,
        real_A: torch.Tensor,
        real_B: torch.Tensor,
        tumor_mask_A: Optional[torch.Tensor] = None,
        tumor_mask_B: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        compute generator loss with additional terms.
        """
        outputs = self.forward(real_A, real_B)
        
        # base losses
        total_loss, loss_dict = super().compute_generator_loss.__wrapped__(
            self, real_A, real_B
        )
        
        # perceptual loss
        if hasattr(self, 'perceptual_loss'):
            loss_perc_A2B = self.perceptual_loss(outputs['fake_B'], real_B)
            loss_perc_B2A = self.perceptual_loss(outputs['fake_A'], real_A)
            loss_perceptual = (loss_perc_A2B + loss_perc_B2A) * self.sa_config.lambda_perceptual
            total_loss = total_loss + loss_perceptual
            loss_dict['loss_perceptual'] = loss_perceptual.item()
        
        # tumor preservation loss
        if tumor_mask_A is not None and hasattr(self, 'tumor_loss'):
            loss_tumor_A = self.tumor_loss(
                outputs['fake_B'], real_A, tumor_mask_A
            )
            total_loss = total_loss + loss_tumor_A * self.sa_config.lambda_tumor
            loss_dict['loss_tumor'] = loss_tumor_A.item()
        
        if tumor_mask_B is not None and hasattr(self, 'tumor_loss'):
            loss_tumor_B = self.tumor_loss(
                outputs['fake_A'], real_B, tumor_mask_B
            )
            total_loss = total_loss + loss_tumor_B * self.sa_config.lambda_tumor
            loss_dict['loss_tumor'] = loss_dict.get('loss_tumor', 0) + loss_tumor_B.item()
        
        loss_dict['loss_G'] = total_loss.item()
        
        return total_loss, loss_dict


class Perceptual3DLoss(nn.Module):
    """
    3d perceptual loss.
    
    uses 3d feature extraction for perceptual similarity.
    """
    
    def __init__(self, layers: List[int] = None):
        super().__init__()
        
        self.layers = layers or [2, 5, 8]
        
        # simple 3d feature extractor
        self.features = nn.Sequential(
            nn.Conv3d(1, 32, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 128, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 256, 3, 2, 1),
            nn.ReLU(inplace=True),
        )
        
        # freeze
        for param in self.features.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """compute perceptual loss."""
        loss = 0.0
        
        hx, hy = x, y
        for i, layer in enumerate(self.features):
            hx = layer(hx)
            hy = layer(hy)
            
            if i in self.layers:
                loss = loss + F.l1_loss(hx, hy)
        
        return loss / len(self.layers)


class TumorPreservation3DLoss(nn.Module):
    """
    3d tumor preservation loss.
    
    ensures tumor regions are preserved during harmonization.
    """
    
    def __init__(
        self,
        intensity_weight: float = 1.0,
        gradient_weight: float = 0.5
    ):
        super().__init__()
        
        self.intensity_weight = intensity_weight
        self.gradient_weight = gradient_weight
    
    def forward(
        self,
        generated: torch.Tensor,
        original: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        compute tumor preservation loss.
        
        args:
            generated: generated volume
            original: original volume
            mask: tumor mask
            
        returns:
            preservation loss
        """
        # intensity preservation in tumor region
        masked_gen = generated * mask
        masked_orig = original * mask
        
        intensity_loss = F.l1_loss(masked_gen, masked_orig)
        
        # gradient preservation
        grad_gen = self._compute_gradient(generated) * mask
        grad_orig = self._compute_gradient(original) * mask
        
        gradient_loss = F.l1_loss(grad_gen, grad_orig)
        
        return (
            self.intensity_weight * intensity_loss +
            self.gradient_weight * gradient_loss
        )
    
    def _compute_gradient(self, x: torch.Tensor) -> torch.Tensor:
        """compute 3d gradient magnitude."""
        # sobel-like gradients
        grad_d = torch.abs(x[:, :, 1:, :, :] - x[:, :, :-1, :, :])
        grad_h = torch.abs(x[:, :, :, 1:, :] - x[:, :, :, :-1, :])
        grad_w = torch.abs(x[:, :, :, :, 1:] - x[:, :, :, :, :-1])
        
        # pad to original size
        grad_d = F.pad(grad_d, (0, 0, 0, 0, 0, 1))
        grad_h = F.pad(grad_h, (0, 0, 0, 1, 0, 0))
        grad_w = F.pad(grad_w, (0, 1, 0, 0, 0, 0))
        
        return torch.sqrt(grad_d ** 2 + grad_h ** 2 + grad_w ** 2 + 1e-8)

"""
Model Architecture Tests.

Unit tests for generators, discriminators,
and model components.
"""

import pytest
import torch
import torch.nn as nn


class TestResidualBlock:
    """Test residual block."""
    
    def test_residual_block_shape(self):
        """Test output shape matches input."""
        from ..models.blocks import ResidualBlock
        
        block = ResidualBlock(64)
        x = torch.randn(2, 64, 32, 32, 32)
        y = block(x)
        
        assert y.shape == x.shape
    
    def test_residual_connection(self):
        """Test residual connection gradient flow."""
        from ..models.blocks import ResidualBlock
        
        block = ResidualBlock(64)
        x = torch.randn(2, 64, 16, 16, 16, requires_grad=True)
        y = block(x)
        loss = y.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestSelfAttention:
    """Test self-attention mechanism."""
    
    def test_attention_shape(self):
        """Test attention output shape."""
        from ..models.attention import SelfAttention3D
        
        attn = SelfAttention3D(64)
        x = torch.randn(2, 64, 8, 8, 8)
        y, _ = attn(x)
        
        assert y.shape == x.shape
    
    def test_attention_weights_sum_to_one(self):
        """Test attention weights are valid probabilities."""
        from ..models.attention import SelfAttention3D
        
        attn = SelfAttention3D(64)
        x = torch.randn(2, 64, 4, 4, 4)
        _, attention_weights = attn(x)
        
        # Attention weights should sum to 1 along key dimension
        if attention_weights is not None:
            sums = attention_weights.sum(dim=-1)
            assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


class TestSAGenerator:
    """Test SA-CycleGAN generator."""
    
    @pytest.fixture
    def generator(self):
        """Create generator instance."""
        from ..models.generators import SAGenerator
        return SAGenerator(
            in_channels=4,
            out_channels=4,
            base_channels=32,
            n_blocks=3,
            use_attention=True
        )
    
    def test_generator_shape(self, generator):
        """Test generator output shape."""
        x = torch.randn(1, 4, 64, 64, 64)
        y = generator(x)
        
        assert y.shape == x.shape
    
    def test_generator_output_range(self, generator):
        """Test generator output is in valid range."""
        x = torch.randn(1, 4, 32, 32, 32)
        y = generator(x)
        
        # With tanh activation, output should be in [-1, 1]
        assert y.min() >= -1.0
        assert y.max() <= 1.0
    
    def test_generator_gradient_flow(self, generator):
        """Test gradients flow through generator."""
        x = torch.randn(1, 4, 32, 32, 32, requires_grad=True)
        y = generator(x)
        loss = y.mean()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
    
    @pytest.mark.slow
    def test_generator_large_input(self, generator):
        """Test generator with large input."""
        x = torch.randn(1, 4, 128, 128, 128)
        y = generator(x)
        
        assert y.shape == x.shape


class TestMultiScaleDiscriminator:
    """Test multi-scale discriminator."""
    
    @pytest.fixture
    def discriminator(self):
        """Create discriminator instance."""
        from ..models.discriminators import MultiScaleDiscriminator
        return MultiScaleDiscriminator(
            in_channels=4,
            base_channels=32,
            n_scales=2
        )
    
    def test_discriminator_output_list(self, discriminator):
        """Test discriminator returns list of outputs."""
        x = torch.randn(1, 4, 64, 64, 64)
        outputs = discriminator(x)
        
        assert isinstance(outputs, list)
        assert len(outputs) == 2  # n_scales
    
    def test_discriminator_output_shapes(self, discriminator):
        """Test discriminator output shapes decrease."""
        x = torch.randn(1, 4, 64, 64, 64)
        outputs = discriminator(x)
        
        # Each scale should have smaller spatial dimensions
        for i in range(len(outputs) - 1):
            assert outputs[i].shape[2] >= outputs[i+1].shape[2]
    
    def test_discriminator_gradient_flow(self, discriminator):
        """Test gradients flow through discriminator."""
        x = torch.randn(1, 4, 32, 32, 32, requires_grad=True)
        outputs = discriminator(x)
        loss = sum(o.mean() for o in outputs)
        loss.backward()
        
        assert x.grad is not None


class TestLossFunctions:
    """Test loss functions."""
    
    def test_adversarial_loss(self):
        """Test adversarial loss computation."""
        from ..models.losses import AdversarialLoss
        
        loss_fn = AdversarialLoss(mode='lsgan')
        
        pred_real = torch.ones(2, 1, 8, 8, 8)
        pred_fake = torch.zeros(2, 1, 8, 8, 8)
        
        loss_real = loss_fn(pred_real, True)
        loss_fake = loss_fn(pred_fake, False)
        
        assert loss_real >= 0
        assert loss_fake >= 0
    
    def test_cycle_consistency_loss(self):
        """Test cycle consistency loss."""
        from ..models.losses import CycleConsistencyLoss
        
        loss_fn = CycleConsistencyLoss()
        
        real = torch.randn(2, 4, 16, 16, 16)
        reconstructed = real + 0.1 * torch.randn_like(real)
        
        loss = loss_fn(reconstructed, real)
        
        assert loss >= 0
        assert loss < 1.0  # Should be small for similar images
    
    def test_perceptual_loss(self):
        """Test perceptual loss."""
        from ..models.losses import PerceptualLoss
        
        loss_fn = PerceptualLoss()
        
        pred = torch.randn(2, 4, 32, 32, 32)
        target = torch.randn(2, 4, 32, 32, 32)
        
        loss = loss_fn(pred, target)
        
        assert loss >= 0
    
    def test_ssim_loss(self):
        """Test SSIM loss."""
        from ..models.losses import MultiScaleSSIMLoss
        
        loss_fn = MultiScaleSSIMLoss()
        
        # Identical images should have low loss
        x = torch.randn(2, 4, 32, 32, 32)
        loss_identical = loss_fn(x, x)
        
        # Different images should have higher loss
        y = torch.randn(2, 4, 32, 32, 32)
        loss_different = loss_fn(x, y)
        
        assert loss_identical < loss_different


class TestModelIntegration:
    """Integration tests for complete model."""
    
    @pytest.mark.integration
    def test_full_forward_pass(self):
        """Test complete forward pass through model."""
        from ..models import SAGenerator, MultiScaleDiscriminator
        
        G_A2B = SAGenerator(4, 4, 32, 3)
        G_B2A = SAGenerator(4, 4, 32, 3)
        D_A = MultiScaleDiscriminator(4, 32, 2)
        D_B = MultiScaleDiscriminator(4, 32, 2)
        
        # Forward cycle
        real_A = torch.randn(1, 4, 32, 32, 32)
        fake_B = G_A2B(real_A)
        rec_A = G_B2A(fake_B)
        
        # Backward cycle
        real_B = torch.randn(1, 4, 32, 32, 32)
        fake_A = G_B2A(real_B)
        rec_B = G_A2B(fake_A)
        
        # Discriminator
        d_fake_B = D_B(fake_B)
        d_real_B = D_B(real_B)
        
        assert rec_A.shape == real_A.shape
        assert rec_B.shape == real_B.shape
        assert isinstance(d_fake_B, list)


class TestParameterCounts:
    """Test model parameter counts."""
    
    def test_generator_params(self):
        """Test generator parameter count is reasonable."""
        from ..models.generators import SAGenerator
        
        generator = SAGenerator(4, 4, 64, 9)
        n_params = sum(p.numel() for p in generator.parameters())
        
        # Should be in reasonable range (5M - 50M)
        assert 1_000_000 < n_params < 100_000_000
    
    def test_discriminator_params(self):
        """Test discriminator parameter count."""
        from ..models.discriminators import MultiScaleDiscriminator
        
        discriminator = MultiScaleDiscriminator(4, 64, 3)
        n_params = sum(p.numel() for p in discriminator.parameters())
        
        # Should be smaller than generator
        assert 500_000 < n_params < 50_000_000

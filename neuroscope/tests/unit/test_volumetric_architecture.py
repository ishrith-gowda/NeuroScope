"""
Comprehensive Unit Tests for 3D Volumetric Architecture.

Tests cover:
- 3D Generator architectures
- 3D Discriminator architectures
- 3D CycleGAN integration
- 3D building blocks (residual, attention)
- Memory efficiency
- Input/output shape validation
"""

import pytest
import torch
import torch.nn as nn
from typing import Tuple
import gc

# Import 3D architecture components
from src.models.architectures.volumetric import (
    ResidualBlock3D,
    DownsampleBlock3D,
    UpsampleBlock3D,
    SelfAttention3D,
    ResNetGenerator3D,
    SAGenerator3D,
    MemoryEfficientGenerator3D,
    PatchDiscriminator3D,
    MultiScaleDiscriminator3D,
    SpectralDiscriminator3D,
    CycleGAN3D,
    CycleGAN3DConfig,
)


# Test fixtures
@pytest.fixture
def device():
    """Get available device."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def small_volume(device):
    """Create small test volume (B, C, D, H, W)."""
    return torch.randn(1, 1, 32, 32, 32, device=device)


@pytest.fixture
def medium_volume(device):
    """Create medium test volume."""
    return torch.randn(1, 1, 64, 64, 64, device=device)


@pytest.fixture
def batch_volume(device):
    """Create batched test volumes."""
    return torch.randn(2, 1, 32, 32, 32, device=device)


class TestResidualBlock3D:
    """Tests for 3D residual block."""
    
    def test_basic_forward(self, device):
        """Test basic forward pass."""
        block = ResidualBlock3D(64, 64).to(device)
        x = torch.randn(1, 64, 16, 16, 16, device=device)
        
        output = block(x)
        
        assert output.shape == x.shape
        assert not torch.isnan(output).any()
    
    def test_channel_change(self, device):
        """Test with channel number change."""
        block = ResidualBlock3D(32, 64).to(device)
        x = torch.randn(1, 32, 16, 16, 16, device=device)
        
        output = block(x)
        
        assert output.shape == (1, 64, 16, 16, 16)
    
    def test_different_norm(self, device):
        """Test different normalization types."""
        for norm in ['instance', 'batch', 'group']:
            block = ResidualBlock3D(64, 64, norm_type=norm).to(device)
            x = torch.randn(2, 64, 8, 8, 8, device=device)
            
            output = block(x)
            
            assert output.shape == x.shape


class TestDownsampleBlock3D:
    """Tests for 3D downsample block."""
    
    def test_spatial_reduction(self, device):
        """Test that spatial dimensions are halved."""
        block = DownsampleBlock3D(32, 64).to(device)
        x = torch.randn(1, 32, 16, 16, 16, device=device)
        
        output = block(x)
        
        assert output.shape == (1, 64, 8, 8, 8)
    
    def test_strided_conv(self, device):
        """Test strided convolution mode."""
        block = DownsampleBlock3D(32, 64, mode='strided').to(device)
        x = torch.randn(1, 32, 16, 16, 16, device=device)
        
        output = block(x)
        
        assert output.shape == (1, 64, 8, 8, 8)
    
    def test_maxpool(self, device):
        """Test maxpool mode."""
        block = DownsampleBlock3D(32, 64, mode='maxpool').to(device)
        x = torch.randn(1, 32, 16, 16, 16, device=device)
        
        output = block(x)
        
        assert output.shape == (1, 64, 8, 8, 8)


class TestUpsampleBlock3D:
    """Tests for 3D upsample block."""
    
    def test_spatial_increase(self, device):
        """Test that spatial dimensions are doubled."""
        block = UpsampleBlock3D(64, 32).to(device)
        x = torch.randn(1, 64, 8, 8, 8, device=device)
        
        output = block(x)
        
        assert output.shape == (1, 32, 16, 16, 16)
    
    def test_different_modes(self, device):
        """Test different upsampling modes."""
        for mode in ['transpose', 'nearest', 'trilinear']:
            block = UpsampleBlock3D(64, 32, mode=mode).to(device)
            x = torch.randn(1, 64, 8, 8, 8, device=device)
            
            output = block(x)
            
            assert output.shape == (1, 32, 16, 16, 16)


class TestSelfAttention3D:
    """Tests for 3D self-attention module."""
    
    def test_basic_attention(self, device):
        """Test basic attention forward pass."""
        attn = SelfAttention3D(64).to(device)
        x = torch.randn(1, 64, 8, 8, 8, device=device)
        
        output = attn(x)
        
        assert output.shape == x.shape
        assert not torch.isnan(output).any()
    
    def test_attention_weights(self, device):
        """Test that gamma parameter is trainable."""
        attn = SelfAttention3D(64).to(device)
        
        # Gamma should be learnable
        assert attn.gamma.requires_grad
        
        # Initially close to zero
        assert attn.gamma.abs().item() < 0.1
    
    def test_larger_volume(self, device):
        """Test with larger volume (memory check)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for memory-intensive test")
        
        attn = SelfAttention3D(32, use_checkpoint=True).to(device)
        x = torch.randn(1, 32, 16, 16, 16, device=device)
        
        output = attn(x)
        
        assert output.shape == x.shape


class TestResNetGenerator3D:
    """Tests for 3D ResNet generator."""
    
    def test_basic_forward(self, small_volume, device):
        """Test basic forward pass."""
        gen = ResNetGenerator3D(input_nc=1, output_nc=1).to(device)
        
        output = gen(small_volume)
        
        assert output.shape == small_volume.shape
        assert not torch.isnan(output).any()
    
    def test_output_range(self, small_volume, device):
        """Test that output is in [-1, 1] due to tanh."""
        gen = ResNetGenerator3D(input_nc=1, output_nc=1).to(device)
        
        output = gen(small_volume)
        
        assert output.min() >= -1.0
        assert output.max() <= 1.0
    
    def test_different_channels(self, device):
        """Test with different input/output channels."""
        gen = ResNetGenerator3D(input_nc=1, output_nc=4).to(device)
        x = torch.randn(1, 1, 32, 32, 32, device=device)
        
        output = gen(x)
        
        assert output.shape == (1, 4, 32, 32, 32)
    
    def test_different_num_blocks(self, device):
        """Test with different number of residual blocks."""
        for num_blocks in [3, 6, 9]:
            gen = ResNetGenerator3D(
                input_nc=1,
                output_nc=1,
                num_residual_blocks=num_blocks
            ).to(device)
            x = torch.randn(1, 1, 32, 32, 32, device=device)
            
            output = gen(x)
            
            assert output.shape == x.shape


class TestSAGenerator3D:
    """Tests for 3D Self-Attention generator."""
    
    def test_basic_forward(self, small_volume, device):
        """Test basic forward pass."""
        gen = SAGenerator3D(input_nc=1, output_nc=1).to(device)
        
        output = gen(small_volume)
        
        assert output.shape == small_volume.shape
    
    def test_attention_positions(self, device):
        """Test different attention layer positions."""
        gen = SAGenerator3D(
            input_nc=1,
            output_nc=1,
            attention_positions=[2, 4, 6]
        ).to(device)
        x = torch.randn(1, 1, 32, 32, 32, device=device)
        
        output = gen(x)
        
        assert output.shape == x.shape
    
    def test_batched_input(self, batch_volume, device):
        """Test with batched input."""
        gen = SAGenerator3D(input_nc=1, output_nc=1).to(device)
        
        output = gen(batch_volume)
        
        assert output.shape == batch_volume.shape


class TestMemoryEfficientGenerator3D:
    """Tests for memory-efficient 3D generator."""
    
    def test_basic_forward(self, small_volume, device):
        """Test basic forward pass."""
        gen = MemoryEfficientGenerator3D(input_nc=1, output_nc=1).to(device)
        
        output = gen(small_volume)
        
        assert output.shape == small_volume.shape
    
    def test_checkpointing(self, device):
        """Test gradient checkpointing."""
        gen = MemoryEfficientGenerator3D(
            input_nc=1,
            output_nc=1,
            use_checkpoint=True
        ).to(device)
        x = torch.randn(1, 1, 32, 32, 32, device=device, requires_grad=True)
        
        output = gen(x)
        loss = output.mean()
        loss.backward()
        
        assert x.grad is not None


class TestPatchDiscriminator3D:
    """Tests for 3D PatchGAN discriminator."""
    
    def test_basic_forward(self, small_volume, device):
        """Test basic forward pass."""
        disc = PatchDiscriminator3D(input_nc=1).to(device)
        
        output = disc(small_volume)
        
        # Output should be smaller due to patch-wise computation
        assert len(output.shape) == 5
        assert output.shape[0] == small_volume.shape[0]
    
    def test_output_for_real_fake(self, small_volume, device):
        """Test discriminator output for real and fake."""
        disc = PatchDiscriminator3D(input_nc=1).to(device)
        fake_volume = torch.randn_like(small_volume)
        
        real_out = disc(small_volume)
        fake_out = disc(fake_volume)
        
        assert real_out.shape == fake_out.shape
    
    def test_different_n_layers(self, device):
        """Test with different number of layers."""
        for n_layers in [2, 3, 4]:
            disc = PatchDiscriminator3D(input_nc=1, n_layers=n_layers).to(device)
            x = torch.randn(1, 1, 32, 32, 32, device=device)
            
            output = disc(x)
            
            assert len(output.shape) == 5


class TestMultiScaleDiscriminator3D:
    """Tests for multi-scale 3D discriminator."""
    
    def test_basic_forward(self, small_volume, device):
        """Test basic forward pass."""
        disc = MultiScaleDiscriminator3D(input_nc=1, num_scales=2).to(device)
        
        outputs = disc(small_volume)
        
        assert isinstance(outputs, list)
        assert len(outputs) == 2
    
    def test_scale_outputs(self, device):
        """Test that each scale produces valid output."""
        disc = MultiScaleDiscriminator3D(input_nc=1, num_scales=3).to(device)
        x = torch.randn(1, 1, 64, 64, 64, device=device)
        
        outputs = disc(x)
        
        for i, out in enumerate(outputs):
            assert not torch.isnan(out).any(), f"NaN in scale {i}"


class TestSpectralDiscriminator3D:
    """Tests for spectral normalized 3D discriminator."""
    
    def test_basic_forward(self, small_volume, device):
        """Test basic forward pass."""
        disc = SpectralDiscriminator3D(input_nc=1).to(device)
        
        output = disc(small_volume)
        
        assert len(output.shape) == 5
    
    def test_spectral_norm_applied(self, device):
        """Test that spectral normalization is applied."""
        disc = SpectralDiscriminator3D(input_nc=1).to(device)
        
        # Check for spectral norm hooks
        has_spectral_norm = False
        for module in disc.modules():
            if hasattr(module, 'weight_orig'):
                has_spectral_norm = True
                break
        
        assert has_spectral_norm


class TestCycleGAN3D:
    """Tests for complete 3D CycleGAN."""
    
    @pytest.fixture
    def cyclegan(self, device):
        """Create CycleGAN3D instance."""
        config = CycleGAN3DConfig(
            input_nc=1,
            output_nc=1,
            ngf=32,
            ndf=32,
            num_residual_blocks=4
        )
        return CycleGAN3D(config).to(device)
    
    def test_forward(self, cyclegan, small_volume, device):
        """Test forward pass."""
        fake_B = cyclegan.forward(small_volume, direction='A2B')
        fake_A = cyclegan.forward(small_volume, direction='B2A')
        
        assert fake_B.shape == small_volume.shape
        assert fake_A.shape == small_volume.shape
    
    def test_cycle_consistency(self, cyclegan, small_volume, device):
        """Test cycle consistency property."""
        # A -> B -> A
        fake_B = cyclegan.forward(small_volume, direction='A2B')
        rec_A = cyclegan.forward(fake_B, direction='B2A')
        
        # Reconstructed should be similar to original
        assert rec_A.shape == small_volume.shape
    
    def test_training_step_structure(self, cyclegan, small_volume, device):
        """Test training step returns expected structure."""
        real_A = small_volume
        real_B = torch.randn_like(small_volume)
        
        # This tests that model components work together
        fake_B = cyclegan.G_A2B(real_A)
        fake_A = cyclegan.G_B2A(real_B)
        rec_A = cyclegan.G_B2A(fake_B)
        rec_B = cyclegan.G_A2B(fake_A)
        
        assert fake_B.shape == real_A.shape
        assert fake_A.shape == real_B.shape
        assert rec_A.shape == real_A.shape
        assert rec_B.shape == real_B.shape
    
    def test_discriminator_outputs(self, cyclegan, small_volume, device):
        """Test discriminator forward pass."""
        disc_out = cyclegan.D_A(small_volume)
        
        assert isinstance(disc_out, (list, torch.Tensor))


class TestGradientFlow:
    """Tests for gradient flow through 3D architectures."""
    
    def test_generator_gradients(self, device):
        """Test gradients flow through generator."""
        gen = ResNetGenerator3D(input_nc=1, output_nc=1).to(device)
        x = torch.randn(1, 1, 32, 32, 32, device=device, requires_grad=True)
        
        output = gen(x)
        loss = output.mean()
        loss.backward()
        
        # Check that gradients exist
        for name, param in gen.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
    
    def test_discriminator_gradients(self, device):
        """Test gradients flow through discriminator."""
        disc = PatchDiscriminator3D(input_nc=1).to(device)
        x = torch.randn(1, 1, 32, 32, 32, device=device, requires_grad=True)
        
        output = disc(x)
        loss = output.mean()
        loss.backward()
        
        for name, param in disc.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
    
    def test_full_cyclegan_gradients(self, device):
        """Test gradients flow through full CycleGAN."""
        config = CycleGAN3DConfig(
            input_nc=1, output_nc=1, ngf=16, ndf=16, num_residual_blocks=2
        )
        model = CycleGAN3D(config).to(device)
        
        real_A = torch.randn(1, 1, 16, 16, 16, device=device, requires_grad=True)
        real_B = torch.randn(1, 1, 16, 16, 16, device=device, requires_grad=True)
        
        # Generator forward
        fake_B = model.G_A2B(real_A)
        fake_A = model.G_B2A(real_B)
        
        # Cycle
        rec_A = model.G_B2A(fake_B)
        rec_B = model.G_A2B(fake_A)
        
        # Loss and backward
        loss = fake_B.mean() + fake_A.mean() + rec_A.mean() + rec_B.mean()
        loss.backward()
        
        assert real_A.grad is not None
        assert real_B.grad is not None


class TestMemoryUsage:
    """Tests for memory usage of 3D architectures."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_generator_memory(self, device):
        """Test generator memory footprint."""
        torch.cuda.reset_peak_memory_stats(device)
        
        gen = SAGenerator3D(input_nc=1, output_nc=1, ngf=32).to(device)
        x = torch.randn(1, 1, 64, 64, 64, device=device)
        
        _ = gen(x)
        
        memory_mb = torch.cuda.max_memory_allocated(device) / 1024**2
        
        # Should be reasonable (< 8GB for a single forward pass)
        assert memory_mb < 8192, f"Memory usage too high: {memory_mb:.1f} MB"
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_checkpointing_reduces_memory(self, device):
        """Test that checkpointing reduces memory usage."""
        torch.cuda.reset_peak_memory_stats(device)
        
        # Without checkpointing
        gen_no_ckpt = MemoryEfficientGenerator3D(
            input_nc=1, output_nc=1, use_checkpoint=False
        ).to(device)
        x = torch.randn(1, 1, 48, 48, 48, device=device, requires_grad=True)
        
        output = gen_no_ckpt(x)
        output.mean().backward()
        
        memory_no_ckpt = torch.cuda.max_memory_allocated(device)
        
        # Clear
        del gen_no_ckpt, x, output
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        
        # With checkpointing
        gen_ckpt = MemoryEfficientGenerator3D(
            input_nc=1, output_nc=1, use_checkpoint=True
        ).to(device)
        x = torch.randn(1, 1, 48, 48, 48, device=device, requires_grad=True)
        
        output = gen_ckpt(x)
        output.mean().backward()
        
        memory_ckpt = torch.cuda.max_memory_allocated(device)
        
        # Checkpointing should use less memory
        assert memory_ckpt < memory_no_ckpt


class TestNumericalStability:
    """Tests for numerical stability."""
    
    def test_no_nan_in_generator(self, device):
        """Test generator doesn't produce NaN."""
        gen = SAGenerator3D(input_nc=1, output_nc=1).to(device)
        
        # Test with various input ranges
        for scale in [0.01, 1.0, 100.0]:
            x = torch.randn(1, 1, 32, 32, 32, device=device) * scale
            output = gen(x)
            
            assert not torch.isnan(output).any(), f"NaN at scale {scale}"
            assert not torch.isinf(output).any(), f"Inf at scale {scale}"
    
    def test_no_nan_in_discriminator(self, device):
        """Test discriminator doesn't produce NaN."""
        disc = PatchDiscriminator3D(input_nc=1).to(device)
        
        for scale in [0.01, 1.0, 100.0]:
            x = torch.randn(1, 1, 32, 32, 32, device=device) * scale
            output = disc(x)
            
            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-x'])

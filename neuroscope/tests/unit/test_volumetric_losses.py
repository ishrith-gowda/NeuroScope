"""
Unit Tests for Volumetric (3D) Loss Functions.

Tests cover:
- VolumetricSSIM and MultiScale variants
- VolumetricCycleConsistencyLoss
- VolumetricGradientLoss
- VolumetricPerceptualLoss
- AnatomicalConsistencyLoss
- TissuePreservationLoss
- VolumetricNCELoss
- CombinedVolumetricLoss
"""

import pytest
import torch
import torch.nn as nn
from typing import Tuple

from src.models.losses.volumetric import (
    VolumetricSSIM,
    VolumetricMultiScaleSSIM,
    VolumetricCycleConsistencyLoss,
    VolumetricGradientLoss,
    VolumetricPerceptualLoss,
    AnatomicalConsistencyLoss,
    TissuePreservationLoss,
    VolumetricNCELoss,
    VolumetricIdentityLoss,
    CombinedVolumetricLoss,
)


@pytest.fixture
def device():
    """Get available device."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def volume_pair(device):
    """Create pair of test volumes."""
    vol1 = torch.randn(2, 1, 32, 32, 32, device=device)
    vol2 = torch.randn(2, 1, 32, 32, 32, device=device)
    return vol1, vol2


@pytest.fixture
def identical_volumes(device):
    """Create identical volumes for testing."""
    vol = torch.randn(2, 1, 32, 32, 32, device=device)
    return vol, vol.clone()


@pytest.fixture
def small_volume_pair(device):
    """Smaller volumes for faster testing."""
    vol1 = torch.randn(1, 1, 16, 16, 16, device=device)
    vol2 = torch.randn(1, 1, 16, 16, 16, device=device)
    return vol1, vol2


class TestVolumetricSSIM:
    """Tests for 3D SSIM implementation."""
    
    def test_basic_computation(self, volume_pair, device):
        """Test basic SSIM computation."""
        vol1, vol2 = volume_pair
        ssim = VolumetricSSIM().to(device)
        
        result = ssim(vol1, vol2)
        
        assert isinstance(result, torch.Tensor)
        assert result.ndim == 0  # Scalar
        assert not torch.isnan(result)
    
    def test_identical_volumes(self, identical_volumes, device):
        """Test SSIM of identical volumes is 1."""
        vol1, vol2 = identical_volumes
        ssim = VolumetricSSIM().to(device)
        
        result = ssim(vol1, vol2)
        
        assert result.item() > 0.99, f"SSIM of identical images: {result.item()}"
    
    def test_ssim_range(self, volume_pair, device):
        """Test SSIM is in valid range [-1, 1]."""
        vol1, vol2 = volume_pair
        ssim = VolumetricSSIM().to(device)
        
        result = ssim(vol1, vol2)
        
        assert -1 <= result.item() <= 1
    
    def test_different_window_sizes(self, small_volume_pair, device):
        """Test different window sizes."""
        vol1, vol2 = small_volume_pair
        
        for window_size in [3, 5, 7]:
            ssim = VolumetricSSIM(window_size=window_size).to(device)
            result = ssim(vol1, vol2)
            
            assert not torch.isnan(result)
    
    def test_multi_channel(self, device):
        """Test with multi-channel volumes."""
        vol1 = torch.randn(1, 4, 16, 16, 16, device=device)
        vol2 = torch.randn(1, 4, 16, 16, 16, device=device)
        
        ssim = VolumetricSSIM(channel=4).to(device)
        result = ssim(vol1, vol2)
        
        assert not torch.isnan(result)
    
    def test_batch_processing(self, device):
        """Test batch processing."""
        vol1 = torch.randn(4, 1, 16, 16, 16, device=device)
        vol2 = torch.randn(4, 1, 16, 16, 16, device=device)
        
        ssim = VolumetricSSIM(size_average=False).to(device)
        result = ssim(vol1, vol2)
        
        assert result.shape == (4,)


class TestVolumetricMultiScaleSSIM:
    """Tests for multi-scale 3D SSIM."""
    
    def test_basic_computation(self, device):
        """Test basic MS-SSIM computation."""
        vol1 = torch.randn(1, 1, 64, 64, 64, device=device)
        vol2 = torch.randn(1, 1, 64, 64, 64, device=device)
        
        msssim = VolumetricMultiScaleSSIM().to(device)
        result = msssim(vol1, vol2)
        
        assert isinstance(result, torch.Tensor)
        assert not torch.isnan(result)
    
    def test_identical_volumes(self, device):
        """Test MS-SSIM of identical volumes."""
        vol = torch.randn(1, 1, 64, 64, 64, device=device)
        
        msssim = VolumetricMultiScaleSSIM().to(device)
        result = msssim(vol, vol.clone())
        
        assert result.item() > 0.99


class TestVolumetricCycleConsistencyLoss:
    """Tests for 3D cycle consistency loss."""
    
    def test_l1_loss(self, volume_pair, device):
        """Test L1 cycle consistency loss."""
        vol1, vol2 = volume_pair
        loss_fn = VolumetricCycleConsistencyLoss(loss_type='l1').to(device)
        
        result = loss_fn(vol1, vol2)
        
        assert isinstance(result, torch.Tensor)
        assert result.item() >= 0
    
    def test_l2_loss(self, volume_pair, device):
        """Test L2 cycle consistency loss."""
        vol1, vol2 = volume_pair
        loss_fn = VolumetricCycleConsistencyLoss(loss_type='l2').to(device)
        
        result = loss_fn(vol1, vol2)
        
        assert result.item() >= 0
    
    def test_ssim_loss(self, volume_pair, device):
        """Test SSIM-based cycle consistency loss."""
        vol1, vol2 = volume_pair
        loss_fn = VolumetricCycleConsistencyLoss(loss_type='ssim').to(device)
        
        result = loss_fn(vol1, vol2)
        
        assert not torch.isnan(result)
    
    def test_combined_loss(self, volume_pair, device):
        """Test combined L1 + SSIM loss."""
        vol1, vol2 = volume_pair
        loss_fn = VolumetricCycleConsistencyLoss(loss_type='combined').to(device)
        
        result = loss_fn(vol1, vol2)
        
        assert not torch.isnan(result)
    
    def test_multi_scale(self, volume_pair, device):
        """Test multi-scale loss computation."""
        vol1, vol2 = volume_pair
        loss_fn = VolumetricCycleConsistencyLoss(
            loss_type='l1',
            multi_scale=True,
            num_scales=3
        ).to(device)
        
        result = loss_fn(vol1, vol2)
        
        assert not torch.isnan(result)
    
    def test_zero_for_identical(self, identical_volumes, device):
        """Test that loss is ~0 for identical volumes."""
        vol1, vol2 = identical_volumes
        loss_fn = VolumetricCycleConsistencyLoss(loss_type='l1').to(device)
        
        result = loss_fn(vol1, vol2)
        
        assert result.item() < 1e-5
    
    def test_with_mask(self, volume_pair, device):
        """Test with spatial mask."""
        vol1, vol2 = volume_pair
        mask = torch.ones_like(vol1)
        mask[:, :, :16, :, :] = 0  # Mask half the volume
        
        loss_fn = VolumetricCycleConsistencyLoss(loss_type='l1').to(device)
        result = loss_fn(vol1, vol2, mask=mask)
        
        assert not torch.isnan(result)


class TestVolumetricGradientLoss:
    """Tests for 3D gradient matching loss."""
    
    def test_basic_computation(self, volume_pair, device):
        """Test basic gradient loss computation."""
        vol1, vol2 = volume_pair
        loss_fn = VolumetricGradientLoss().to(device)
        
        result = loss_fn(vol1, vol2)
        
        assert isinstance(result, torch.Tensor)
        assert result.item() >= 0
    
    def test_identical_volumes(self, identical_volumes, device):
        """Test gradient loss for identical volumes."""
        vol1, vol2 = identical_volumes
        loss_fn = VolumetricGradientLoss().to(device)
        
        result = loss_fn(vol1, vol2)
        
        assert result.item() < 1e-5
    
    def test_l1_vs_l2(self, volume_pair, device):
        """Test L1 vs L2 gradient loss."""
        vol1, vol2 = volume_pair
        
        loss_l1 = VolumetricGradientLoss(loss_type='l1').to(device)
        loss_l2 = VolumetricGradientLoss(loss_type='l2').to(device)
        
        result_l1 = loss_l1(vol1, vol2)
        result_l2 = loss_l2(vol1, vol2)
        
        assert not torch.isnan(result_l1)
        assert not torch.isnan(result_l2)


class TestVolumetricPerceptualLoss:
    """Tests for 3D perceptual loss."""
    
    def test_basic_computation(self, small_volume_pair, device):
        """Test basic perceptual loss computation."""
        vol1, vol2 = small_volume_pair
        loss_fn = VolumetricPerceptualLoss(feature_extractor='custom').to(device)
        
        result = loss_fn(vol1, vol2)
        
        assert isinstance(result, torch.Tensor)
        assert result.item() >= 0
    
    def test_identical_volumes(self, device):
        """Test perceptual loss for identical volumes."""
        vol = torch.randn(1, 1, 16, 16, 16, device=device)
        loss_fn = VolumetricPerceptualLoss(feature_extractor='custom').to(device)
        
        result = loss_fn(vol, vol.clone())
        
        assert result.item() < 1e-4
    
    def test_gradients_flow(self, small_volume_pair, device):
        """Test that gradients flow through perceptual loss."""
        vol1, vol2 = small_volume_pair
        vol1.requires_grad = True
        
        loss_fn = VolumetricPerceptualLoss(feature_extractor='custom').to(device)
        result = loss_fn(vol1, vol2)
        result.backward()
        
        assert vol1.grad is not None


class TestAnatomicalConsistencyLoss:
    """Tests for anatomical consistency loss."""
    
    def test_basic_computation(self, volume_pair, device):
        """Test basic anatomical consistency loss."""
        vol1, vol2 = volume_pair
        loss_fn = AnatomicalConsistencyLoss().to(device)
        
        result = loss_fn(vol1, vol2)
        
        assert isinstance(result, torch.Tensor)
        assert result.item() >= 0
    
    def test_with_tissue_maps(self, volume_pair, device):
        """Test with explicit tissue probability maps."""
        vol1, vol2 = volume_pair
        
        tissue_maps = {
            'gm': torch.ones_like(vol1) * 0.5,
            'wm': torch.ones_like(vol1) * 0.3,
            'csf': torch.ones_like(vol1) * 0.2,
        }
        
        loss_fn = AnatomicalConsistencyLoss().to(device)
        result = loss_fn(vol1, vol2, tissue_maps=tissue_maps)
        
        assert not torch.isnan(result)
    
    def test_different_weights(self, volume_pair, device):
        """Test with different tissue weights."""
        vol1, vol2 = volume_pair
        
        loss_fn = AnatomicalConsistencyLoss(
            gm_weight=2.0,
            wm_weight=1.5,
            csf_weight=0.5
        ).to(device)
        
        result = loss_fn(vol1, vol2)
        
        assert not torch.isnan(result)


class TestTissuePreservationLoss:
    """Tests for tissue boundary preservation loss."""
    
    def test_basic_computation(self, volume_pair, device):
        """Test basic tissue preservation loss."""
        vol1, vol2 = volume_pair
        loss_fn = TissuePreservationLoss().to(device)
        
        result = loss_fn(vol1, vol2)
        
        assert isinstance(result, torch.Tensor)
        assert result.item() >= 0
    
    def test_identical_boundaries(self, identical_volumes, device):
        """Test that identical volumes have zero boundary loss."""
        vol1, vol2 = identical_volumes
        loss_fn = TissuePreservationLoss().to(device)
        
        result = loss_fn(vol1, vol2)
        
        assert result.item() < 1e-5


class TestVolumetricNCELoss:
    """Tests for 3D contrastive loss."""
    
    def test_basic_computation(self, device):
        """Test basic NCE loss computation."""
        # Create feature lists (simulating encoder outputs)
        feat_q = [torch.randn(2, 64, 8, 8, 8, device=device)]
        feat_k = [torch.randn(2, 64, 8, 8, 8, device=device)]
        
        loss_fn = VolumetricNCELoss(num_patches=16).to(device)
        result = loss_fn(feat_q, feat_k)
        
        assert isinstance(result, torch.Tensor)
        assert result.item() >= 0
    
    def test_positive_pairs(self, device):
        """Test NCE with identical features (positive pairs)."""
        feat = torch.randn(2, 64, 8, 8, 8, device=device)
        feat_q = [feat]
        feat_k = [feat.clone()]
        
        loss_fn = VolumetricNCELoss(num_patches=16).to(device)
        result = loss_fn(feat_q, feat_k)
        
        # With identical features, loss should be lower
        assert result.item() < 5.0


class TestVolumetricIdentityLoss:
    """Tests for 3D identity loss."""
    
    def test_l1_identity(self, volume_pair, device):
        """Test L1 identity loss."""
        vol1, vol2 = volume_pair
        loss_fn = VolumetricIdentityLoss(loss_type='l1').to(device)
        
        result = loss_fn(vol1, vol2)
        
        assert isinstance(result, torch.Tensor)
        assert result.item() >= 0
    
    def test_with_ssim(self, volume_pair, device):
        """Test identity loss with SSIM component."""
        vol1, vol2 = volume_pair
        loss_fn = VolumetricIdentityLoss(loss_type='l1', lambda_ssim=0.5).to(device)
        
        result = loss_fn(vol1, vol2)
        
        assert not torch.isnan(result)


class TestCombinedVolumetricLoss:
    """Tests for combined volumetric loss."""
    
    @pytest.fixture
    def cyclegan_outputs(self, device):
        """Create simulated CycleGAN outputs."""
        B, C, D, H, W = 1, 1, 16, 16, 16
        
        return {
            'real_A': torch.randn(B, C, D, H, W, device=device),
            'real_B': torch.randn(B, C, D, H, W, device=device),
            'fake_A': torch.randn(B, C, D, H, W, device=device),
            'fake_B': torch.randn(B, C, D, H, W, device=device),
            'rec_A': torch.randn(B, C, D, H, W, device=device),
            'rec_B': torch.randn(B, C, D, H, W, device=device),
            'idt_A': torch.randn(B, C, D, H, W, device=device),
            'idt_B': torch.randn(B, C, D, H, W, device=device),
        }
    
    def test_basic_computation(self, cyclegan_outputs, device):
        """Test combined loss computation."""
        loss_fn = CombinedVolumetricLoss().to(device)
        
        losses = loss_fn(**cyclegan_outputs)
        
        assert isinstance(losses, dict)
        assert 'total' in losses
        assert not torch.isnan(losses['total'])
    
    def test_all_components_present(self, cyclegan_outputs, device):
        """Test that all loss components are computed."""
        loss_fn = CombinedVolumetricLoss().to(device)
        
        losses = loss_fn(**cyclegan_outputs)
        
        expected_keys = [
            'cycle_A', 'cycle_B',
            'identity_A', 'identity_B',
            'ssim_A', 'ssim_B',
            'gradient_A', 'gradient_B',
            'perceptual_A', 'perceptual_B',
            'anatomical_A', 'anatomical_B',
            'tissue_A', 'tissue_B',
            'total'
        ]
        
        for key in expected_keys:
            assert key in losses, f"Missing loss component: {key}"
    
    def test_lambda_weights(self, cyclegan_outputs, device):
        """Test that lambda weights affect total loss."""
        # High cycle weight
        loss_fn_high = CombinedVolumetricLoss(lambda_cycle=100.0).to(device)
        losses_high = loss_fn_high(**cyclegan_outputs)
        
        # Low cycle weight
        loss_fn_low = CombinedVolumetricLoss(lambda_cycle=0.1).to(device)
        losses_low = loss_fn_low(**cyclegan_outputs)
        
        # High weight should produce higher total loss
        assert losses_high['total'] != losses_low['total']
    
    def test_without_identity(self, device):
        """Test combined loss without identity outputs."""
        B, C, D, H, W = 1, 1, 16, 16, 16
        
        inputs = {
            'real_A': torch.randn(B, C, D, H, W, device=device),
            'real_B': torch.randn(B, C, D, H, W, device=device),
            'fake_A': torch.randn(B, C, D, H, W, device=device),
            'fake_B': torch.randn(B, C, D, H, W, device=device),
            'rec_A': torch.randn(B, C, D, H, W, device=device),
            'rec_B': torch.randn(B, C, D, H, W, device=device),
        }
        
        loss_fn = CombinedVolumetricLoss().to(device)
        losses = loss_fn(**inputs)
        
        assert 'total' in losses
        assert not torch.isnan(losses['total'])


class TestLossGradients:
    """Tests for gradient computation through losses."""
    
    def test_cycle_loss_gradients(self, device):
        """Test gradients flow through cycle loss."""
        vol1 = torch.randn(1, 1, 16, 16, 16, device=device, requires_grad=True)
        vol2 = torch.randn(1, 1, 16, 16, 16, device=device)
        
        loss_fn = VolumetricCycleConsistencyLoss().to(device)
        result = loss_fn(vol1, vol2)
        result.backward()
        
        assert vol1.grad is not None
    
    def test_ssim_gradients(self, device):
        """Test gradients flow through SSIM."""
        vol1 = torch.randn(1, 1, 16, 16, 16, device=device, requires_grad=True)
        vol2 = torch.randn(1, 1, 16, 16, 16, device=device)
        
        ssim = VolumetricSSIM().to(device)
        result = ssim(vol1, vol2)
        loss = 1 - result
        loss.backward()
        
        assert vol1.grad is not None
    
    def test_gradient_loss_gradients(self, device):
        """Test gradients flow through gradient loss."""
        vol1 = torch.randn(1, 1, 16, 16, 16, device=device, requires_grad=True)
        vol2 = torch.randn(1, 1, 16, 16, 16, device=device)
        
        loss_fn = VolumetricGradientLoss().to(device)
        result = loss_fn(vol1, vol2)
        result.backward()
        
        assert vol1.grad is not None


class TestLossNumericalStability:
    """Tests for numerical stability of losses."""
    
    def test_ssim_with_constant_input(self, device):
        """Test SSIM doesn't produce NaN with constant input."""
        vol = torch.ones(1, 1, 16, 16, 16, device=device)
        
        ssim = VolumetricSSIM().to(device)
        result = ssim(vol, vol)
        
        # Should not be NaN (stability constants prevent division by zero)
        assert not torch.isnan(result)
    
    def test_gradient_loss_with_constant(self, device):
        """Test gradient loss with constant input."""
        vol = torch.ones(1, 1, 16, 16, 16, device=device)
        
        loss_fn = VolumetricGradientLoss().to(device)
        result = loss_fn(vol, vol)
        
        assert not torch.isnan(result)
        assert result.item() < 1e-5  # Should be ~0 for constant
    
    def test_losses_with_extreme_values(self, device):
        """Test losses with extreme input values."""
        vol1 = torch.randn(1, 1, 16, 16, 16, device=device) * 100
        vol2 = torch.randn(1, 1, 16, 16, 16, device=device) * 0.01
        
        losses = [
            VolumetricCycleConsistencyLoss(loss_type='l1'),
            VolumetricGradientLoss(),
            TissuePreservationLoss(),
        ]
        
        for loss_fn in losses:
            loss_fn = loss_fn.to(device)
            result = loss_fn(vol1, vol2)
            
            assert not torch.isnan(result), f"NaN in {type(loss_fn).__name__}"
            assert not torch.isinf(result), f"Inf in {type(loss_fn).__name__}"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-x'])

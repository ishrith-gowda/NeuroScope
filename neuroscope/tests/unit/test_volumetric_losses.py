"""
unit tests for volumetric (3d) loss functions.

tests cover:
- volumetricssim and multiscale variants
- volumetriccycleconsistencyloss
- volumetricgradientloss
- volumetricperceptualloss
- anatomicalconsistencyloss
- tissuepreservationloss
- volumetricnceloss
- combinedvolumetricloss
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
    """get available device."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def volume_pair(device):
    """create pair of test volumes."""
    vol1 = torch.randn(2, 1, 32, 32, 32, device=device)
    vol2 = torch.randn(2, 1, 32, 32, 32, device=device)
    return vol1, vol2


@pytest.fixture
def identical_volumes(device):
    """create identical volumes for testing."""
    vol = torch.randn(2, 1, 32, 32, 32, device=device)
    return vol, vol.clone()


@pytest.fixture
def small_volume_pair(device):
    """smaller volumes for faster testing."""
    vol1 = torch.randn(1, 1, 16, 16, 16, device=device)
    vol2 = torch.randn(1, 1, 16, 16, 16, device=device)
    return vol1, vol2


class TestVolumetricSSIM:
    """tests for 3d ssim implementation."""
    
    def test_basic_computation(self, volume_pair, device):
        """test basic ssim computation."""
        vol1, vol2 = volume_pair
        ssim = VolumetricSSIM().to(device)
        
        result = ssim(vol1, vol2)
        
        assert isinstance(result, torch.Tensor)
        assert result.ndim == 0  # scalar
        assert not torch.isnan(result)
    
    def test_identical_volumes(self, identical_volumes, device):
        """test ssim of identical volumes is 1."""
        vol1, vol2 = identical_volumes
        ssim = VolumetricSSIM().to(device)
        
        result = ssim(vol1, vol2)
        
        assert result.item() > 0.99, f"SSIM of identical images: {result.item()}"
    
    def test_ssim_range(self, volume_pair, device):
        """test ssim is in valid range [-1, 1]."""
        vol1, vol2 = volume_pair
        ssim = VolumetricSSIM().to(device)
        
        result = ssim(vol1, vol2)
        
        assert -1 <= result.item() <= 1
    
    def test_different_window_sizes(self, small_volume_pair, device):
        """test different window sizes."""
        vol1, vol2 = small_volume_pair
        
        for window_size in [3, 5, 7]:
            ssim = VolumetricSSIM(window_size=window_size).to(device)
            result = ssim(vol1, vol2)
            
            assert not torch.isnan(result)
    
    def test_multi_channel(self, device):
        """test with multi-channel volumes."""
        vol1 = torch.randn(1, 4, 16, 16, 16, device=device)
        vol2 = torch.randn(1, 4, 16, 16, 16, device=device)
        
        ssim = VolumetricSSIM(channel=4).to(device)
        result = ssim(vol1, vol2)
        
        assert not torch.isnan(result)
    
    def test_batch_processing(self, device):
        """test batch processing."""
        vol1 = torch.randn(4, 1, 16, 16, 16, device=device)
        vol2 = torch.randn(4, 1, 16, 16, 16, device=device)
        
        ssim = VolumetricSSIM(size_average=False).to(device)
        result = ssim(vol1, vol2)
        
        assert result.shape == (4,)


class TestVolumetricMultiScaleSSIM:
    """tests for multi-scale 3d ssim."""
    
    def test_basic_computation(self, device):
        """test basic ms-ssim computation."""
        vol1 = torch.randn(1, 1, 64, 64, 64, device=device)
        vol2 = torch.randn(1, 1, 64, 64, 64, device=device)
        
        msssim = VolumetricMultiScaleSSIM().to(device)
        result = msssim(vol1, vol2)
        
        assert isinstance(result, torch.Tensor)
        assert not torch.isnan(result)
    
    def test_identical_volumes(self, device):
        """test ms-ssim of identical volumes."""
        vol = torch.randn(1, 1, 64, 64, 64, device=device)
        
        msssim = VolumetricMultiScaleSSIM().to(device)
        result = msssim(vol, vol.clone())
        
        assert result.item() > 0.99


class TestVolumetricCycleConsistencyLoss:
    """tests for 3d cycle consistency loss."""
    
    def test_l1_loss(self, volume_pair, device):
        """test l1 cycle consistency loss."""
        vol1, vol2 = volume_pair
        loss_fn = VolumetricCycleConsistencyLoss(loss_type='l1').to(device)
        
        result = loss_fn(vol1, vol2)
        
        assert isinstance(result, torch.Tensor)
        assert result.item() >= 0
    
    def test_l2_loss(self, volume_pair, device):
        """test l2 cycle consistency loss."""
        vol1, vol2 = volume_pair
        loss_fn = VolumetricCycleConsistencyLoss(loss_type='l2').to(device)
        
        result = loss_fn(vol1, vol2)
        
        assert result.item() >= 0
    
    def test_ssim_loss(self, volume_pair, device):
        """test ssim-based cycle consistency loss."""
        vol1, vol2 = volume_pair
        loss_fn = VolumetricCycleConsistencyLoss(loss_type='ssim').to(device)
        
        result = loss_fn(vol1, vol2)
        
        assert not torch.isnan(result)
    
    def test_combined_loss(self, volume_pair, device):
        """test combined l1 + ssim loss."""
        vol1, vol2 = volume_pair
        loss_fn = VolumetricCycleConsistencyLoss(loss_type='combined').to(device)
        
        result = loss_fn(vol1, vol2)
        
        assert not torch.isnan(result)
    
    def test_multi_scale(self, volume_pair, device):
        """test multi-scale loss computation."""
        vol1, vol2 = volume_pair
        loss_fn = VolumetricCycleConsistencyLoss(
            loss_type='l1',
            multi_scale=True,
            num_scales=3
        ).to(device)
        
        result = loss_fn(vol1, vol2)
        
        assert not torch.isnan(result)
    
    def test_zero_for_identical(self, identical_volumes, device):
        """test that loss is ~0 for identical volumes."""
        vol1, vol2 = identical_volumes
        loss_fn = VolumetricCycleConsistencyLoss(loss_type='l1').to(device)
        
        result = loss_fn(vol1, vol2)
        
        assert result.item() < 1e-5
    
    def test_with_mask(self, volume_pair, device):
        """test with spatial mask."""
        vol1, vol2 = volume_pair
        mask = torch.ones_like(vol1)
        mask[:, :, :16, :, :] = 0  # mask half the volume
        
        loss_fn = VolumetricCycleConsistencyLoss(loss_type='l1').to(device)
        result = loss_fn(vol1, vol2, mask=mask)
        
        assert not torch.isnan(result)


class TestVolumetricGradientLoss:
    """tests for 3d gradient matching loss."""
    
    def test_basic_computation(self, volume_pair, device):
        """test basic gradient loss computation."""
        vol1, vol2 = volume_pair
        loss_fn = VolumetricGradientLoss().to(device)
        
        result = loss_fn(vol1, vol2)
        
        assert isinstance(result, torch.Tensor)
        assert result.item() >= 0
    
    def test_identical_volumes(self, identical_volumes, device):
        """test gradient loss for identical volumes."""
        vol1, vol2 = identical_volumes
        loss_fn = VolumetricGradientLoss().to(device)
        
        result = loss_fn(vol1, vol2)
        
        assert result.item() < 1e-5
    
    def test_l1_vs_l2(self, volume_pair, device):
        """test l1 vs l2 gradient loss."""
        vol1, vol2 = volume_pair
        
        loss_l1 = VolumetricGradientLoss(loss_type='l1').to(device)
        loss_l2 = VolumetricGradientLoss(loss_type='l2').to(device)
        
        result_l1 = loss_l1(vol1, vol2)
        result_l2 = loss_l2(vol1, vol2)
        
        assert not torch.isnan(result_l1)
        assert not torch.isnan(result_l2)


class TestVolumetricPerceptualLoss:
    """tests for 3d perceptual loss."""
    
    def test_basic_computation(self, small_volume_pair, device):
        """test basic perceptual loss computation."""
        vol1, vol2 = small_volume_pair
        loss_fn = VolumetricPerceptualLoss(feature_extractor='custom').to(device)
        
        result = loss_fn(vol1, vol2)
        
        assert isinstance(result, torch.Tensor)
        assert result.item() >= 0
    
    def test_identical_volumes(self, device):
        """test perceptual loss for identical volumes."""
        vol = torch.randn(1, 1, 16, 16, 16, device=device)
        loss_fn = VolumetricPerceptualLoss(feature_extractor='custom').to(device)
        
        result = loss_fn(vol, vol.clone())
        
        assert result.item() < 1e-4
    
    def test_gradients_flow(self, small_volume_pair, device):
        """test that gradients flow through perceptual loss."""
        vol1, vol2 = small_volume_pair
        vol1.requires_grad = True
        
        loss_fn = VolumetricPerceptualLoss(feature_extractor='custom').to(device)
        result = loss_fn(vol1, vol2)
        result.backward()
        
        assert vol1.grad is not None


class TestAnatomicalConsistencyLoss:
    """tests for anatomical consistency loss."""
    
    def test_basic_computation(self, volume_pair, device):
        """test basic anatomical consistency loss."""
        vol1, vol2 = volume_pair
        loss_fn = AnatomicalConsistencyLoss().to(device)
        
        result = loss_fn(vol1, vol2)
        
        assert isinstance(result, torch.Tensor)
        assert result.item() >= 0
    
    def test_with_tissue_maps(self, volume_pair, device):
        """test with explicit tissue probability maps."""
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
        """test with different tissue weights."""
        vol1, vol2 = volume_pair
        
        loss_fn = AnatomicalConsistencyLoss(
            gm_weight=2.0,
            wm_weight=1.5,
            csf_weight=0.5
        ).to(device)
        
        result = loss_fn(vol1, vol2)
        
        assert not torch.isnan(result)


class TestTissuePreservationLoss:
    """tests for tissue boundary preservation loss."""
    
    def test_basic_computation(self, volume_pair, device):
        """test basic tissue preservation loss."""
        vol1, vol2 = volume_pair
        loss_fn = TissuePreservationLoss().to(device)
        
        result = loss_fn(vol1, vol2)
        
        assert isinstance(result, torch.Tensor)
        assert result.item() >= 0
    
    def test_identical_boundaries(self, identical_volumes, device):
        """test that identical volumes have zero boundary loss."""
        vol1, vol2 = identical_volumes
        loss_fn = TissuePreservationLoss().to(device)
        
        result = loss_fn(vol1, vol2)
        
        assert result.item() < 1e-5


class TestVolumetricNCELoss:
    """tests for 3d contrastive loss."""
    
    def test_basic_computation(self, device):
        """test basic nce loss computation."""
        # create feature lists (simulating encoder outputs)
        feat_q = [torch.randn(2, 64, 8, 8, 8, device=device)]
        feat_k = [torch.randn(2, 64, 8, 8, 8, device=device)]
        
        loss_fn = VolumetricNCELoss(num_patches=16).to(device)
        result = loss_fn(feat_q, feat_k)
        
        assert isinstance(result, torch.Tensor)
        assert result.item() >= 0
    
    def test_positive_pairs(self, device):
        """test nce with identical features (positive pairs)."""
        feat = torch.randn(2, 64, 8, 8, 8, device=device)
        feat_q = [feat]
        feat_k = [feat.clone()]
        
        loss_fn = VolumetricNCELoss(num_patches=16).to(device)
        result = loss_fn(feat_q, feat_k)
        
        # with identical features, loss should be lower
        assert result.item() < 5.0


class TestVolumetricIdentityLoss:
    """tests for 3d identity loss."""
    
    def test_l1_identity(self, volume_pair, device):
        """test l1 identity loss."""
        vol1, vol2 = volume_pair
        loss_fn = VolumetricIdentityLoss(loss_type='l1').to(device)
        
        result = loss_fn(vol1, vol2)
        
        assert isinstance(result, torch.Tensor)
        assert result.item() >= 0
    
    def test_with_ssim(self, volume_pair, device):
        """test identity loss with ssim component."""
        vol1, vol2 = volume_pair
        loss_fn = VolumetricIdentityLoss(loss_type='l1', lambda_ssim=0.5).to(device)
        
        result = loss_fn(vol1, vol2)
        
        assert not torch.isnan(result)


class TestCombinedVolumetricLoss:
    """tests for combined volumetric loss."""
    
    @pytest.fixture
    def cyclegan_outputs(self, device):
        """create simulated cyclegan outputs."""
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
        """test combined loss computation."""
        loss_fn = CombinedVolumetricLoss().to(device)
        
        losses = loss_fn(**cyclegan_outputs)
        
        assert isinstance(losses, dict)
        assert 'total' in losses
        assert not torch.isnan(losses['total'])
    
    def test_all_components_present(self, cyclegan_outputs, device):
        """test that all loss components are computed."""
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
        """test that lambda weights affect total loss."""
        # high cycle weight
        loss_fn_high = CombinedVolumetricLoss(lambda_cycle=100.0).to(device)
        losses_high = loss_fn_high(**cyclegan_outputs)
        
        # low cycle weight
        loss_fn_low = CombinedVolumetricLoss(lambda_cycle=0.1).to(device)
        losses_low = loss_fn_low(**cyclegan_outputs)
        
        # high weight should produce higher total loss
        assert losses_high['total'] != losses_low['total']
    
    def test_without_identity(self, device):
        """test combined loss without identity outputs."""
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
    """tests for gradient computation through losses."""
    
    def test_cycle_loss_gradients(self, device):
        """test gradients flow through cycle loss."""
        vol1 = torch.randn(1, 1, 16, 16, 16, device=device, requires_grad=True)
        vol2 = torch.randn(1, 1, 16, 16, 16, device=device)
        
        loss_fn = VolumetricCycleConsistencyLoss().to(device)
        result = loss_fn(vol1, vol2)
        result.backward()
        
        assert vol1.grad is not None
    
    def test_ssim_gradients(self, device):
        """test gradients flow through ssim."""
        vol1 = torch.randn(1, 1, 16, 16, 16, device=device, requires_grad=True)
        vol2 = torch.randn(1, 1, 16, 16, 16, device=device)
        
        ssim = VolumetricSSIM().to(device)
        result = ssim(vol1, vol2)
        loss = 1 - result
        loss.backward()
        
        assert vol1.grad is not None
    
    def test_gradient_loss_gradients(self, device):
        """test gradients flow through gradient loss."""
        vol1 = torch.randn(1, 1, 16, 16, 16, device=device, requires_grad=True)
        vol2 = torch.randn(1, 1, 16, 16, 16, device=device)
        
        loss_fn = VolumetricGradientLoss().to(device)
        result = loss_fn(vol1, vol2)
        result.backward()
        
        assert vol1.grad is not None


class TestLossNumericalStability:
    """tests for numerical stability of losses."""
    
    def test_ssim_with_constant_input(self, device):
        """test ssim doesn't produce nan with constant input."""
        vol = torch.ones(1, 1, 16, 16, 16, device=device)
        
        ssim = VolumetricSSIM().to(device)
        result = ssim(vol, vol)
        
        # should not be nan (stability constants prevent division by zero)
        assert not torch.isnan(result)
    
    def test_gradient_loss_with_constant(self, device):
        """test gradient loss with constant input."""
        vol = torch.ones(1, 1, 16, 16, 16, device=device)
        
        loss_fn = VolumetricGradientLoss().to(device)
        result = loss_fn(vol, vol)
        
        assert not torch.isnan(result)
        assert result.item() < 1e-5  # should be ~0 for constant
    
    def test_losses_with_extreme_values(self, device):
        """test losses with extreme input values."""
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

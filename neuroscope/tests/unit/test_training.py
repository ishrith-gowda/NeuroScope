"""
Training Pipeline Tests.

Unit tests for trainer, optimizers, and schedulers.
"""

import pytest
import torch
import torch.nn as nn


class TestOptimizers:
    """Test optimizer configurations."""
    
    def test_adam_optimizer_creation(self):
        """Test Adam optimizer creation."""
        from ..training.optimizers import create_optimizer
        
        model = nn.Linear(10, 10)
        optimizer = create_optimizer(
            model.parameters(),
            optimizer_type='adam',
            lr=0.0002,
            betas=(0.5, 0.999)
        )
        
        assert isinstance(optimizer, torch.optim.Adam)
        assert optimizer.defaults['lr'] == 0.0002
    
    def test_adamw_optimizer(self):
        """Test AdamW optimizer."""
        from ..training.optimizers import create_optimizer
        
        model = nn.Linear(10, 10)
        optimizer = create_optimizer(
            model.parameters(),
            optimizer_type='adamw',
            lr=0.0002,
            weight_decay=0.01
        )
        
        assert isinstance(optimizer, torch.optim.AdamW)


class TestSchedulers:
    """Test learning rate schedulers."""
    
    def test_cosine_scheduler(self):
        """Test cosine annealing scheduler."""
        from ..training.schedulers import create_scheduler
        
        model = nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        scheduler = create_scheduler(
            optimizer,
            scheduler_type='cosine',
            T_max=100
        )
        
        initial_lr = optimizer.param_groups[0]['lr']
        
        # Step through some epochs
        for _ in range(50):
            scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # LR should decrease
        assert current_lr < initial_lr
    
    def test_linear_warmup(self):
        """Test linear warmup scheduler."""
        from ..training.schedulers import LinearWarmupScheduler
        
        model = nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        scheduler = LinearWarmupScheduler(
            optimizer,
            warmup_epochs=5,
            total_epochs=100
        )
        
        lrs = []
        for epoch in range(10):
            lrs.append(optimizer.param_groups[0]['lr'])
            scheduler.step()
        
        # LR should increase during warmup
        assert lrs[4] > lrs[0]
    
    def test_warmup_cosine_scheduler(self):
        """Test warmup + cosine decay."""
        from ..training.schedulers import create_scheduler
        
        model = nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        scheduler = create_scheduler(
            optimizer,
            scheduler_type='warmup_cosine',
            warmup_epochs=5,
            total_epochs=100
        )
        
        # Warmup phase
        for _ in range(5):
            scheduler.step()
        
        peak_lr = optimizer.param_groups[0]['lr']
        
        # Decay phase
        for _ in range(50):
            scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        
        assert current_lr < peak_lr


class TestCallbacks:
    """Test training callbacks."""
    
    def test_early_stopping(self):
        """Test early stopping callback."""
        from ..training.callbacks import EarlyStopping
        
        callback = EarlyStopping(patience=3, min_delta=0.01)
        
        # Improving metrics
        assert not callback(0.90)
        assert not callback(0.91)
        assert not callback(0.92)
        
        # Stagnating metrics
        assert not callback(0.92)
        assert not callback(0.92)
        assert not callback(0.92)
        assert callback(0.92)  # Should trigger after patience
    
    def test_model_checkpoint(self, tmp_path):
        """Test model checkpointing."""
        from ..training.callbacks import ModelCheckpoint
        
        callback = ModelCheckpoint(
            dirpath=str(tmp_path),
            filename='best',
            monitor='val_ssim',
            mode='max'
        )
        
        model = nn.Linear(10, 10)
        
        # Save first checkpoint
        callback.on_epoch_end(0, {'val_ssim': 0.90}, {'model': model})
        
        # Better metric - should save
        callback.on_epoch_end(1, {'val_ssim': 0.92}, {'model': model})
        
        # Worse metric - should not save
        callback.on_epoch_end(2, {'val_ssim': 0.88}, {'model': model})
        
        assert callback.best_value == 0.92
    
    def test_lr_logger(self):
        """Test learning rate logging callback."""
        from ..training.callbacks import LearningRateLogger
        
        model = nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        callback = LearningRateLogger()
        
        lr = callback.get_lr(optimizer)
        
        assert lr == 0.001


class TestTrainer:
    """Test training loop."""
    
    @pytest.fixture
    def mock_trainer(self):
        """Create mock trainer."""
        from ..models.generators import SAGenerator
        from ..models.discriminators import MultiScaleDiscriminator
        from ..training.trainer import HarmonizationTrainer
        
        G_A2B = SAGenerator(4, 4, 16, 2, use_attention=False)
        G_B2A = SAGenerator(4, 4, 16, 2, use_attention=False)
        D_A = MultiScaleDiscriminator(4, 16, 2)
        D_B = MultiScaleDiscriminator(4, 16, 2)
        
        trainer = HarmonizationTrainer(
            generator_a2b=G_A2B,
            generator_b2a=G_B2A,
            discriminator_a=D_A,
            discriminator_b=D_B
        )
        
        return trainer
    
    def test_train_step(self, mock_trainer):
        """Test single training step."""
        batch = {
            'source': torch.randn(1, 4, 16, 16, 16),
            'target': torch.randn(1, 4, 16, 16, 16)
        }
        
        losses = mock_trainer.train_step(batch)
        
        assert 'g_loss' in losses
        assert 'd_loss' in losses
        assert not torch.isnan(torch.tensor(losses['g_loss']))
    
    def test_validation_step(self, mock_trainer):
        """Test validation step."""
        batch = {
            'source': torch.randn(1, 4, 16, 16, 16),
            'target': torch.randn(1, 4, 16, 16, 16)
        }
        
        metrics = mock_trainer.validate_step(batch)
        
        assert 'ssim' in metrics or 'val_loss' in metrics


class TestGradientHandling:
    """Test gradient handling utilities."""
    
    def test_gradient_clipping(self):
        """Test gradient clipping."""
        from ..training.trainer import clip_gradients
        
        model = nn.Linear(100, 100)
        x = torch.randn(10, 100)
        y = model(x)
        loss = (y ** 2).sum()
        loss.backward()
        
        # Before clipping
        grad_norm_before = sum(
            p.grad.norm() ** 2 for p in model.parameters()
        ) ** 0.5
        
        clip_gradients(model.parameters(), max_norm=1.0)
        
        grad_norm_after = sum(
            p.grad.norm() ** 2 for p in model.parameters()
        ) ** 0.5
        
        assert grad_norm_after <= 1.0 + 1e-6
    
    def test_gradient_accumulation(self):
        """Test gradient accumulation."""
        model = nn.Linear(10, 10)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        
        accumulation_steps = 4
        
        for i in range(accumulation_steps):
            x = torch.randn(2, 10)
            y = model(x)
            loss = y.sum() / accumulation_steps
            loss.backward()
        
        # Gradients should be accumulated
        assert model.weight.grad is not None
        
        optimizer.step()
        optimizer.zero_grad()
        
        assert model.weight.grad is None or model.weight.grad.sum() == 0


class TestMixedPrecision:
    """Test mixed precision training."""
    
    @pytest.mark.gpu
    def test_amp_forward(self):
        """Test AMP forward pass."""
        if not torch.cuda.is_available():
            pytest.skip("GPU required")
        
        from ..models.generators import SAGenerator
        
        model = SAGenerator(4, 4, 32, 3).cuda()
        x = torch.randn(1, 4, 32, 32, 32).cuda()
        
        with torch.cuda.amp.autocast():
            y = model(x)
        
        assert y.dtype == torch.float16 or y.dtype == torch.float32
    
    @pytest.mark.gpu
    def test_amp_backward(self):
        """Test AMP backward pass with scaler."""
        if not torch.cuda.is_available():
            pytest.skip("GPU required")
        
        model = nn.Linear(100, 100).cuda()
        optimizer = torch.optim.Adam(model.parameters())
        scaler = torch.cuda.amp.GradScaler()
        
        x = torch.randn(10, 100).cuda()
        
        with torch.cuda.amp.autocast():
            y = model(x)
            loss = y.sum()
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Training should complete without errors
        assert True


class TestEMA:
    """Test Exponential Moving Average."""
    
    def test_ema_update(self):
        """Test EMA weight update."""
        from ..training.trainer import ExponentialMovingAverage
        
        model = nn.Linear(10, 10)
        ema = ExponentialMovingAverage(model, decay=0.9)
        
        # Get initial EMA weights
        initial_weight = ema.shadow['weight'].clone()
        
        # Update model weights
        model.weight.data = model.weight.data + 1.0
        
        # Update EMA
        ema.update()
        
        # EMA should move towards new weights
        assert not torch.equal(ema.shadow['weight'], initial_weight)
        assert not torch.equal(ema.shadow['weight'], model.weight)
    
    def test_ema_apply_restore(self):
        """Test applying and restoring EMA weights."""
        from ..training.trainer import ExponentialMovingAverage
        
        model = nn.Linear(10, 10)
        original_weight = model.weight.data.clone()
        
        ema = ExponentialMovingAverage(model, decay=0.9)
        
        # Modify model
        model.weight.data = model.weight.data + 1.0
        ema.update()
        
        # Apply EMA
        ema.apply_shadow()
        ema_weight = model.weight.data.clone()
        
        # Restore original
        ema.restore()
        
        assert not torch.equal(ema_weight, model.weight.data)

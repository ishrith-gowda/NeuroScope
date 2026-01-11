"""
TensorBoard Logger.

Comprehensive TensorBoard logging for 2.5D SA-CycleGAN training.
Logs scalars, images, histograms, and model graphs.

Author: NeuroScope Research Team
"""

from typing import Optional, Dict, Any, Union, List
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


class TensorBoardLogger:
    """
    Professional TensorBoard logger with comprehensive logging capabilities.
    
    Features:
    - Scalar logging (losses, metrics, learning rates)
    - Image logging (samples, comparisons, attention maps)
    - Histogram logging (weights, gradients, activations)
    - Graph logging (model architecture)
    - Hyperparameter logging
    - Custom layout for organized dashboards
    """
    
    def __init__(
        self,
        log_dir: Union[str, Path],
        experiment_name: Optional[str] = None,
        flush_secs: int = 30,
        comment: str = "",
        purge_step: Optional[int] = None,
        max_queue: int = 10,
        filename_suffix: str = ""
    ):
        """
        Initialize TensorBoard logger.
        
        Args:
            log_dir: Root directory for TensorBoard logs
            experiment_name: Name for this experiment run
            flush_secs: Flush frequency in seconds
            comment: Comment to append to log directory name
            purge_step: Purge all events after this global step
            max_queue: Maximum queue size before flushing
            filename_suffix: Suffix for log files
        """
        log_dir = Path(log_dir)
        
        if experiment_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_dir = log_dir / f"{experiment_name}_{timestamp}"
        else:
            self.log_dir = log_dir
            
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.writer = SummaryWriter(
            log_dir=str(self.log_dir),
            comment=comment,
            purge_step=purge_step,
            max_queue=max_queue,
            flush_secs=flush_secs,
            filename_suffix=filename_suffix
        )
        
        self._setup_custom_layout()
        self.step = 0
        
    def _setup_custom_layout(self):
        """Set up custom TensorBoard layout for organized viewing."""
        layout = {
            "Training Losses": {
                "Generator": ["Multiline", ["Loss/Generator/total", "Loss/Generator/gan", 
                                           "Loss/Generator/cycle", "Loss/Generator/identity"]],
                "Discriminator": ["Multiline", ["Loss/Discriminator/total", 
                                                "Loss/Discriminator/A", "Loss/Discriminator/B"]],
            },
            "Validation Metrics": {
                "SSIM": ["Multiline", ["Metrics/SSIM/A2B", "Metrics/SSIM/B2A", "Metrics/SSIM/mean"]],
                "PSNR": ["Multiline", ["Metrics/PSNR/A2B", "Metrics/PSNR/B2A", "Metrics/PSNR/mean"]],
            },
            "Learning Rates": {
                "Optimizers": ["Multiline", ["LR/Generator", "LR/Discriminator"]],
            },
            "Gradients": {
                "Norms": ["Multiline", ["Gradients/Generator/norm", "Gradients/Discriminator/norm"]],
            },
        }
        self.writer.add_custom_scalars(layout)
        
    # =========================================================================
    # Scalar Logging
    # =========================================================================
    
    def log_scalar(
        self,
        tag: str,
        value: float,
        step: Optional[int] = None,
        walltime: Optional[float] = None
    ):
        """Log a single scalar value."""
        step = step if step is not None else self.step
        self.writer.add_scalar(tag, value, step, walltime)
        
    def log_scalars(
        self,
        main_tag: str,
        tag_scalar_dict: Dict[str, float],
        step: Optional[int] = None,
        walltime: Optional[float] = None
    ):
        """Log multiple scalars under a common main tag."""
        step = step if step is not None else self.step
        self.writer.add_scalars(main_tag, tag_scalar_dict, step, walltime)
        
    def log_training_losses(
        self,
        losses: Dict[str, float],
        step: Optional[int] = None
    ):
        """
        Log training losses with organized structure.
        
        Expected keys:
        - G_total, G_gan, G_cycle, G_identity, G_ssim, G_gradient
        - D_total, D_A, D_B
        """
        step = step if step is not None else self.step
        
        # Generator losses
        if 'G_total' in losses:
            self.log_scalar("Loss/Generator/total", losses['G_total'], step)
        if 'G_gan' in losses:
            self.log_scalar("Loss/Generator/gan", losses['G_gan'], step)
        if 'G_cycle' in losses:
            self.log_scalar("Loss/Generator/cycle", losses['G_cycle'], step)
        if 'G_identity' in losses:
            self.log_scalar("Loss/Generator/identity", losses['G_identity'], step)
        if 'G_ssim' in losses:
            self.log_scalar("Loss/Generator/ssim", losses['G_ssim'], step)
        if 'G_gradient' in losses:
            self.log_scalar("Loss/Generator/gradient", losses['G_gradient'], step)
            
        # Discriminator losses
        if 'D_total' in losses:
            self.log_scalar("Loss/Discriminator/total", losses['D_total'], step)
        if 'D_A' in losses:
            self.log_scalar("Loss/Discriminator/A", losses['D_A'], step)
        if 'D_B' in losses:
            self.log_scalar("Loss/Discriminator/B", losses['D_B'], step)
            
    def log_validation_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None
    ):
        """
        Log validation metrics.
        
        Expected keys:
        - ssim_A2B, ssim_B2A
        - psnr_A2B, psnr_B2A
        - ncc_A2B, ncc_B2A (optional)
        """
        step = step if step is not None else self.step
        
        # SSIM
        if 'ssim_A2B' in metrics:
            self.log_scalar("Metrics/SSIM/A2B", metrics['ssim_A2B'], step)
        if 'ssim_B2A' in metrics:
            self.log_scalar("Metrics/SSIM/B2A", metrics['ssim_B2A'], step)
        if 'ssim_A2B' in metrics and 'ssim_B2A' in metrics:
            self.log_scalar("Metrics/SSIM/mean", 
                          (metrics['ssim_A2B'] + metrics['ssim_B2A']) / 2, step)
            
        # PSNR
        if 'psnr_A2B' in metrics:
            self.log_scalar("Metrics/PSNR/A2B", metrics['psnr_A2B'], step)
        if 'psnr_B2A' in metrics:
            self.log_scalar("Metrics/PSNR/B2A", metrics['psnr_B2A'], step)
        if 'psnr_A2B' in metrics and 'psnr_B2A' in metrics:
            self.log_scalar("Metrics/PSNR/mean",
                          (metrics['psnr_A2B'] + metrics['psnr_B2A']) / 2, step)
            
        # NCC (if available)
        if 'ncc_A2B' in metrics:
            self.log_scalar("Metrics/NCC/A2B", metrics['ncc_A2B'], step)
        if 'ncc_B2A' in metrics:
            self.log_scalar("Metrics/NCC/B2A", metrics['ncc_B2A'], step)
            
    def log_learning_rates(
        self,
        lr_G: float,
        lr_D: float,
        step: Optional[int] = None
    ):
        """Log learning rates."""
        step = step if step is not None else self.step
        self.log_scalar("LR/Generator", lr_G, step)
        self.log_scalar("LR/Discriminator", lr_D, step)
        
    # =========================================================================
    # Image Logging
    # =========================================================================
    
    def log_image(
        self,
        tag: str,
        img_tensor: torch.Tensor,
        step: Optional[int] = None,
        dataformats: str = 'CHW'
    ):
        """
        Log a single image.
        
        Args:
            tag: Image tag
            img_tensor: Image tensor (C, H, W) or (H, W, C)
            step: Global step
            dataformats: Data format ('CHW', 'HWC', 'HW')
        """
        step = step if step is not None else self.step
        
        if isinstance(img_tensor, torch.Tensor):
            img_tensor = img_tensor.detach().cpu()
            
        self.writer.add_image(tag, img_tensor, step, dataformats=dataformats)
        
    def log_images(
        self,
        tag: str,
        img_tensor: torch.Tensor,
        step: Optional[int] = None,
        dataformats: str = 'NCHW'
    ):
        """
        Log multiple images as a grid.
        
        Args:
            tag: Image tag
            img_tensor: Image tensor (N, C, H, W)
            step: Global step
            dataformats: Data format
        """
        step = step if step is not None else self.step
        
        if isinstance(img_tensor, torch.Tensor):
            img_tensor = img_tensor.detach().cpu()
            
        self.writer.add_images(tag, img_tensor, step, dataformats=dataformats)
        
    def log_sample_comparison(
        self,
        real_A: torch.Tensor,
        fake_B: torch.Tensor,
        rec_A: torch.Tensor,
        real_B: torch.Tensor,
        fake_A: torch.Tensor,
        rec_B: torch.Tensor,
        step: Optional[int] = None,
        modality_idx: int = 0,
        max_samples: int = 4
    ):
        """
        Log side-by-side comparison of translations.
        
        Creates grid showing:
        Row 1: real_A -> fake_B -> rec_A
        Row 2: real_B -> fake_A -> rec_B
        
        Args:
            real_A: Real domain A images [B, C, H, W]
            fake_B: Generated B images [B, C, H, W]
            rec_A: Reconstructed A images [B, C, H, W]
            real_B: Real domain B images [B, C, H, W]
            fake_A: Generated A images [B, C, H, W]
            rec_B: Reconstructed B images [B, C, H, W]
            step: Global step
            modality_idx: Which modality channel to visualize
            max_samples: Maximum number of samples to show
        """
        step = step if step is not None else self.step
        n = min(real_A.size(0), max_samples)
        
        # Extract single modality for visualization
        def get_modality(x, idx):
            if x.size(1) > idx:
                return x[:n, idx:idx+1]
            return x[:n, 0:1]
        
        # Create comparison grids
        # A -> B -> A cycle
        grid_A2B = torch.cat([
            get_modality(real_A, modality_idx),
            get_modality(fake_B, modality_idx),
            get_modality(rec_A, modality_idx)
        ], dim=0)  # [3*n, 1, H, W]
        
        # B -> A -> B cycle
        grid_B2A = torch.cat([
            get_modality(real_B, modality_idx),
            get_modality(fake_A, modality_idx),
            get_modality(rec_B, modality_idx)
        ], dim=0)
        
        self.log_images(f"Samples/A2B_cycle/modality_{modality_idx}", grid_A2B, step)
        self.log_images(f"Samples/B2A_cycle/modality_{modality_idx}", grid_B2A, step)
        
    def log_attention_maps(
        self,
        attention_maps: Dict[str, torch.Tensor],
        step: Optional[int] = None
    ):
        """
        Log attention maps from self-attention layers.
        
        Args:
            attention_maps: Dictionary of attention maps
            step: Global step
        """
        step = step if step is not None else self.step
        
        for name, attn in attention_maps.items():
            if attn is not None and attn.numel() > 0:
                # Normalize attention for visualization
                attn = attn.detach().cpu()
                if attn.dim() == 4:  # [B, heads, H*W, H*W]
                    # Take first sample, mean over heads
                    attn = attn[0].mean(0)  # [H*W, H*W]
                    
                # Normalize to [0, 1]
                attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)
                
                self.writer.add_image(
                    f"Attention/{name}",
                    attn.unsqueeze(0),  # [1, H*W, H*W]
                    step,
                    dataformats='CHW'
                )
                
    # =========================================================================
    # Histogram Logging
    # =========================================================================
    
    def log_histogram(
        self,
        tag: str,
        values: Union[torch.Tensor, np.ndarray],
        step: Optional[int] = None,
        bins: str = 'tensorflow'
    ):
        """Log a histogram of values."""
        step = step if step is not None else self.step
        
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
            
        self.writer.add_histogram(tag, values, step, bins=bins)
        
    def log_model_weights(
        self,
        model: nn.Module,
        step: Optional[int] = None,
        prefix: str = ""
    ):
        """
        Log histograms of all model weights.
        
        Args:
            model: PyTorch model
            step: Global step
            prefix: Prefix for tags
        """
        step = step if step is not None else self.step
        
        for name, param in model.named_parameters():
            if param.requires_grad and param.data is not None:
                tag = f"{prefix}Weights/{name}" if prefix else f"Weights/{name}"
                self.log_histogram(tag, param.data, step)
                
    def log_model_gradients(
        self,
        model: nn.Module,
        step: Optional[int] = None,
        prefix: str = ""
    ):
        """
        Log histograms of all model gradients.
        
        Args:
            model: PyTorch model
            step: Global step
            prefix: Prefix for tags
        """
        step = step if step is not None else self.step
        
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                tag = f"{prefix}Gradients/{name}" if prefix else f"Gradients/{name}"
                self.log_histogram(tag, param.grad.data, step)
                
    def log_gradient_norms(
        self,
        models: Dict[str, nn.Module],
        step: Optional[int] = None
    ):
        """
        Log gradient L2 norms for models.
        
        Args:
            models: Dictionary of model_name -> model
            step: Global step
        """
        step = step if step is not None else self.step
        
        for name, model in models.items():
            total_norm = 0.0
            for param in model.parameters():
                if param.grad is not None:
                    total_norm += param.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5
            
            self.log_scalar(f"Gradients/{name}/norm", total_norm, step)
            
    # =========================================================================
    # Model Graph & Hyperparameters
    # =========================================================================
    
    def log_graph(
        self,
        model: nn.Module,
        input_shape: tuple,
        verbose: bool = False
    ):
        """
        Log model computational graph.
        
        Args:
            model: PyTorch model
            input_shape: Shape of input tensor (without batch dimension)
            verbose: Whether to print graph info
        """
        import inspect
        device = next(model.parameters()).device
        dummy_input = torch.zeros(1, *input_shape, device=device)
        try:
            # Inspect the forward signature
            sig = inspect.signature(model.forward)
            params = list(sig.parameters.values())
            # Exclude 'self'
            n_args = len([p for p in params if p.name != 'self'])
            if n_args == 2:
                # Multi-input: pass tuple of two dummy tensors
                dummy_input2 = torch.zeros(1, *input_shape, device=device)
                self.writer.add_graph(model, (dummy_input, dummy_input2), verbose=verbose)
            else:
                self.writer.add_graph(model, dummy_input, verbose=verbose)
        except Exception as e:
            print(f"Warning: Could not log model graph: {e}")
            
    def log_hyperparameters(
        self,
        hparam_dict: Dict[str, Any],
        metric_dict: Optional[Dict[str, float]] = None
    ):
        """
        Log hyperparameters.
        
        Args:
            hparam_dict: Dictionary of hyperparameters
            metric_dict: Dictionary of metrics (optional)
        """
        metric_dict = metric_dict or {}
        
        # Filter to supported types
        filtered_hparams = {}
        for k, v in hparam_dict.items():
            if isinstance(v, (int, float, str, bool)):
                filtered_hparams[k] = v
            elif isinstance(v, (list, tuple)):
                filtered_hparams[k] = str(v)
            else:
                filtered_hparams[k] = str(v)
                
        self.writer.add_hparams(filtered_hparams, metric_dict)
        
    # =========================================================================
    # Text Logging
    # =========================================================================
    
    def log_text(
        self,
        tag: str,
        text: str,
        step: Optional[int] = None
    ):
        """Log text data."""
        step = step if step is not None else self.step
        self.writer.add_text(tag, text, step)
        
    def log_config(self, config: Dict[str, Any]):
        """Log configuration as formatted text."""
        config_text = "## Training Configuration\n\n"
        for section, values in config.items():
            config_text += f"### {section}\n"
            if isinstance(values, dict):
                for k, v in values.items():
                    config_text += f"- **{k}**: {v}\n"
            else:
                config_text += f"- {values}\n"
            config_text += "\n"
            
        self.log_text("Config/training", config_text, 0)
        
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def set_step(self, step: int):
        """Set the current global step."""
        self.step = step
        
    def increment_step(self, n: int = 1):
        """Increment the global step."""
        self.step += n
        
    def flush(self):
        """Flush pending events to disk."""
        self.writer.flush()
        
    def close(self):
        """Close the writer."""
        self.writer.close()
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

"""cyclegan model for unpaired image-to-image translation."""

import torch
import torch.nn as nn
import itertools
from typing import Dict, List, Optional, Tuple, Union, Any

from neuroscope.models.components import ResNetGenerator, PatchDiscriminator, ReplayBuffer
from neuroscope.core.logging import get_logger

logger = get_logger(__name__)


class CycleGAN:
    """cyclegan model for unpaired image-to-image translation.
    
    this model learns mappings between two domains a and b using unpaired data.
    it consists of two generators (g_a2b and g_b2a) and two discriminators (d_a and d_b).
    the model uses cycle consistency loss to enforce the mappings.
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        n_residual_blocks: int = 9,
        device: torch.device = None,
    ):
        """initialize cyclegan model.
        
        args:
            in_channels: number of input channels.
            out_channels: number of output channels.
            n_residual_blocks: number of residual blocks in generators.
            device: device to run model on.
        """
        # set device
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # initialize generators and discriminators
        self.G_A2B = ResNetGenerator(in_channels, out_channels, n_residual_blocks).to(self.device)
        self.G_B2A = ResNetGenerator(in_channels, out_channels, n_residual_blocks).to(self.device)
        self.D_A = PatchDiscriminator(in_channels).to(self.device)
        self.D_B = PatchDiscriminator(in_channels).to(self.device)
        
        logger.info(f"Generator A2B parameters: {self.G_A2B.get_num_params():,}")
        logger.info(f"Generator B2A parameters: {self.G_B2A.get_num_params():,}")
        logger.info(f"Discriminator A parameters: {self.D_A.get_num_params():,}")
        logger.info(f"Discriminator B parameters: {self.D_B.get_num_params():,}")
        
        # initialize replay buffers for discriminators
        self.fake_A_buffer = ReplayBuffer(max_size=50)
        self.fake_B_buffer = ReplayBuffer(max_size=50)
        
        # initialize loss functions
        self.criterion_gan = nn.MSELoss()
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()
    
    def setup_optimizers(
        self,
        lr: float = 0.0002,
        beta1: float = 0.5,
        beta2: float = 0.999,
        weight_decay: float = 0.0001
    ) -> Tuple[torch.optim.Optimizer, torch.optim.Optimizer, torch.optim.Optimizer]:
        """set up optimizers for generators and discriminators.
        
        args:
            lr: learning rate.
            beta1: beta1 parameter for adam optimizer.
            beta2: beta2 parameter for adam optimizer.
            weight_decay: weight decay for regularization.
            
        returns:
            tuple of optimizers (optimizer_g, optimizer_d_a, optimizer_d_b).
        """
        # set up optimizers
        optimizer_G = torch.optim.Adam(
            itertools.chain(self.G_A2B.parameters(), self.G_B2A.parameters()),
            lr=lr, betas=(beta1, beta2), weight_decay=weight_decay
        )
        optimizer_D_A = torch.optim.Adam(
            self.D_A.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=weight_decay
        )
        optimizer_D_B = torch.optim.Adam(
            self.D_B.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=weight_decay
        )
        
        return optimizer_G, optimizer_D_A, optimizer_D_B
    
    def setup_schedulers(
        self,
        optimizer_G: torch.optim.Optimizer,
        optimizer_D_A: torch.optim.Optimizer,
        optimizer_D_B: torch.optim.Optimizer,
        n_epochs: int = 200,
        decay_epoch: int = 100
    ) -> Tuple[torch.optim.lr_scheduler._LRScheduler, ...]:
        """set up learning rate schedulers.
        
        args:
            optimizer_g: generator optimizer.
            optimizer_d_a: discriminator a optimizer.
            optimizer_d_b: discriminator b optimizer.
            n_epochs: total number of epochs.
            decay_epoch: epoch to start learning rate decay.
            
        returns:
            tuple of schedulers (scheduler_g, scheduler_d_a, scheduler_d_b).
        """
        def lambda_rule(epoch: int) -> float:
            """calculate learning rate decay factor."""
            return 1.0 - max(0, epoch - decay_epoch) / float(n_epochs - decay_epoch)
        
        scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lambda_rule)
        scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=lambda_rule)
        scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=lambda_rule)
        
        return scheduler_G, scheduler_D_A, scheduler_D_B
    
    def forward(self, x: torch.Tensor, direction: str = "a2b") -> torch.Tensor:
        """forward pass through generator.
        
        args:
            x: input image tensor.
            direction: translation direction, either "a2b" or "b2a".
            
        returns:
            generated image tensor.
        """
        if direction.lower() == "a2b":
            return self.G_A2B(x)
        elif direction.lower() == "b2a":
            return self.G_B2A(x)
        else:
            raise ValueError(f"Invalid direction: {direction}. Must be 'a2b' or 'b2a'.")
    
    def training_step(
        self,
        real_A: torch.Tensor,
        real_B: torch.Tensor,
        optimizer_G: torch.optim.Optimizer,
        optimizer_D_A: torch.optim.Optimizer,
        optimizer_D_B: torch.optim.Optimizer,
        lambda_identity: float = 5.0,
        lambda_cycle: float = 10.0,
    ) -> Dict[str, float]:
        """perform one training step.
        
        args:
            real_a: real images from domain a.
            real_b: real images from domain b.
            optimizer_g: generator optimizer.
            optimizer_d_a: discriminator a optimizer.
            optimizer_d_b: discriminator b optimizer.
            lambda_identity: weight for identity loss.
            lambda_cycle: weight for cycle consistency loss.
            
        returns:
            dictionary of loss values.
        """
        # move tensors to device
        real_A = real_A.to(self.device)
        real_B = real_B.to(self.device)
        
        # get batch size
        batch_size = real_A.size(0)
        
        # create target tensors (1s for real, 0s for fake)
        real_label = torch.ones((batch_size, 1, 16, 16), device=self.device)
        fake_label = torch.zeros((batch_size, 1, 16, 16), device=self.device)
        
        #------------------
        # train generators
        #------------------
        optimizer_G.zero_grad()
        
        # identity loss
        if lambda_identity > 0:
            # g_a2b should be identity if real_b is fed: ||g_a2b(b) - b||
            identity_B = self.G_A2B(real_B)
            loss_identity_B = self.criterion_identity(identity_B, real_B) * lambda_identity
            
            # g_b2a should be identity if real_a is fed: ||g_b2a(a) - a||
            identity_A = self.G_B2A(real_A)
            loss_identity_A = self.criterion_identity(identity_A, real_A) * lambda_identity
            
            # total identity loss
            loss_identity = loss_identity_A + loss_identity_B
        else:
            loss_identity = torch.tensor(0.0, device=self.device)
        
        # gan loss for generators
        fake_B = self.G_A2B(real_A)
        pred_fake_B = self.D_B(fake_B)
        loss_GAN_A2B = self.criterion_gan(pred_fake_B, real_label)
        
        fake_A = self.G_B2A(real_B)
        pred_fake_A = self.D_A(fake_A)
        loss_GAN_B2A = self.criterion_gan(pred_fake_A, real_label)
        
        # total gan loss
        loss_GAN = loss_GAN_A2B + loss_GAN_B2A
        
        # cycle consistency loss
        # forward cycle: a -> b -> a
        recovered_A = self.G_B2A(fake_B)
        loss_cycle_A = self.criterion_cycle(recovered_A, real_A) * lambda_cycle
        
        # backward cycle: b -> a -> b
        recovered_B = self.G_A2B(fake_A)
        loss_cycle_B = self.criterion_cycle(recovered_B, real_B) * lambda_cycle
        
        # total cycle loss
        loss_cycle = loss_cycle_A + loss_cycle_B
        
        # total generator loss
        loss_G = loss_GAN + loss_cycle + loss_identity
        
        # backward pass and optimize
        loss_G.backward()
        optimizer_G.step()
        
        #-----------------------
        # train discriminator a
        #-----------------------
        optimizer_D_A.zero_grad()
        
        # real loss
        pred_real_A = self.D_A(real_A)
        loss_D_real_A = self.criterion_gan(pred_real_A, real_label)
        
        # fake loss (using buffer)
        fake_A_buffer = self.fake_A_buffer.push_and_pop(fake_A.detach())
        pred_fake_A_buffer = self.D_A(fake_A_buffer)
        loss_D_fake_A = self.criterion_gan(pred_fake_A_buffer, fake_label)
        
        # total discriminator a loss
        loss_D_A = (loss_D_real_A + loss_D_fake_A) * 0.5
        
        # backward pass and optimize
        loss_D_A.backward()
        optimizer_D_A.step()
        
        #-----------------------
        # train discriminator b
        #-----------------------
        optimizer_D_B.zero_grad()
        
        # real loss
        pred_real_B = self.D_B(real_B)
        loss_D_real_B = self.criterion_gan(pred_real_B, real_label)
        
        # fake loss (using buffer)
        fake_B_buffer = self.fake_B_buffer.push_and_pop(fake_B.detach())
        pred_fake_B_buffer = self.D_B(fake_B_buffer)
        loss_D_fake_B = self.criterion_gan(pred_fake_B_buffer, fake_label)
        
        # total discriminator b loss
        loss_D_B = (loss_D_real_B + loss_D_fake_B) * 0.5
        
        # backward pass and optimize
        loss_D_B.backward()
        optimizer_D_B.step()
        
        # return losses
        return {
            "loss_G": loss_G.item(),
            "loss_G_GAN": loss_GAN.item(),
            "loss_G_cycle": loss_cycle.item(),
            "loss_G_identity": loss_identity.item(),
            "loss_D_A": loss_D_A.item(),
            "loss_D_B": loss_D_B.item(),
        }
    
    def save_models(self, save_path: str, epoch: int = 0):
        """save models to disk.
        
        args:
            save_path: directory to save models to.
            epoch: current epoch number.
        """
        import os
        
        # create directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        
        # save generators
        torch.save(self.G_A2B.state_dict(), os.path.join(save_path, f"G_A2B_{epoch}.pth"))
        torch.save(self.G_B2A.state_dict(), os.path.join(save_path, f"G_B2A_{epoch}.pth"))
        
        # save discriminators
        torch.save(self.D_A.state_dict(), os.path.join(save_path, f"D_A_{epoch}.pth"))
        torch.save(self.D_B.state_dict(), os.path.join(save_path, f"D_B_{epoch}.pth"))
        
        logger.info(f"Models saved to {save_path} (epoch {epoch})")
    
    def load_models(self, load_path: str, epoch: int = 0):
        """load models from disk.
        
        args:
            load_path: directory to load models from.
            epoch: epoch number to load.
        """
        import os
        
        # load generators
        self.G_A2B.load_state_dict(torch.load(os.path.join(load_path, f"G_A2B_{epoch}.pth")))
        self.G_B2A.load_state_dict(torch.load(os.path.join(load_path, f"G_B2A_{epoch}.pth")))
        
        # load discriminators
        self.D_A.load_state_dict(torch.load(os.path.join(load_path, f"D_A_{epoch}.pth")))
        self.D_B.load_state_dict(torch.load(os.path.join(load_path, f"D_B_{epoch}.pth")))
        
        logger.info(f"Models loaded from {load_path} (epoch {epoch})")
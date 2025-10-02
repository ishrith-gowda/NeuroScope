"""Buffer for CycleGAN replay memory."""

import random
import torch
from typing import List, Optional, Union, Tuple


class ReplayBuffer:
    """Replay buffer for CycleGAN to store generated images.
    
    This buffer helps stabilize training by providing a history of generated images
    to the discriminator. It randomly returns either a new generated image or an
    image from the buffer history.
    """
    
    def __init__(self, max_size: int = 50):
        """Initialize replay buffer.
        
        Args:
            max_size: Maximum number of images to store in buffer.
        """
        self.max_size = max_size
        self.buffer: List[torch.Tensor] = []
    
    def push_and_pop(self, images: torch.Tensor) -> torch.Tensor:
        """Push new images to buffer and return same number of images.
        
        For each image in the batch, there's a 50% chance it will be replaced with
        an image from the buffer. The replaced image is then added to the buffer.
        If the buffer is not full, the new image is added to the buffer and returned.
        
        Args:
            images: Batch of images to push to buffer.
            
        Returns:
            Batch of images from buffer with same batch size.
        """
        result_images = []
        for image in images:
            # Add single image dimension
            image = image.unsqueeze(0)
            
            # If buffer is not full, add image to buffer and return it
            if len(self.buffer) < self.max_size:
                self.buffer.append(image)
                result_images.append(image)
            else:
                # Randomly decide whether to return image from buffer
                if random.random() < 0.5:
                    # Choose random image from buffer
                    idx = random.randint(0, len(self.buffer) - 1)
                    # Retrieve image from buffer
                    result_images.append(self.buffer[idx].clone())
                    # Replace image in buffer
                    self.buffer[idx] = image
                else:
                    # Return new image
                    result_images.append(image)
        
        # Concatenate result images
        return torch.cat(result_images, dim=0)
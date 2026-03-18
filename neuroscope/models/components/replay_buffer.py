"""buffer for cyclegan replay memory."""

import random
import torch
from typing import List, Optional, Union, Tuple


class ReplayBuffer:
    """replay buffer for cyclegan to store generated images.
    
    this buffer helps stabilize training by providing a history of generated images
    to the discriminator. it randomly returns either a new generated image or an
    image from the buffer history.
    """
    
    def __init__(self, max_size: int = 50):
        """initialize replay buffer.
        
        args:
            max_size: maximum number of images to store in buffer.
        """
        self.max_size = max_size
        self.buffer: List[torch.Tensor] = []
    
    def push_and_pop(self, images: torch.Tensor) -> torch.Tensor:
        """push new images to buffer and return same number of images.
        
        for each image in the batch, there's a 50% chance it will be replaced with
        an image from the buffer. the replaced image is then added to the buffer.
        if the buffer is not full, the new image is added to the buffer and returned.
        
        args:
            images: batch of images to push to buffer.
            
        returns:
            batch of images from buffer with same batch size.
        """
        result_images = []
        for image in images:
            # add single image dimension
            image = image.unsqueeze(0)
            
            # if buffer is not full, add image to buffer and return it
            if len(self.buffer) < self.max_size:
                self.buffer.append(image)
                result_images.append(image)
            else:
                # randomly decide whether to return image from buffer
                if random.random() < 0.5:
                    # choose random image from buffer
                    idx = random.randint(0, len(self.buffer) - 1)
                    # retrieve image from buffer
                    result_images.append(self.buffer[idx].clone())
                    # replace image in buffer
                    self.buffer[idx] = image
                else:
                    # return new image
                    result_images.append(image)
        
        # concatenate result images
        return torch.cat(result_images, dim=0)
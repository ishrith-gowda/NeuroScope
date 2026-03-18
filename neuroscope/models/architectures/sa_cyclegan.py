"""
self-attention cyclegan (sa-cyclegan) architecture

novel architecture combining cyclegan with multi-scale self-attention for
brain mri domain translation between brats and upenn-gbm datasets.

key innovations:
1. multi-scale self-attention in generator bottleneck
2. cbam attention in encoder/decoder paths
3. spectral-normalized discriminator with attention
4. modality-aware feature extraction

reference: this is a novel contribution for neurips-level publication.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math


class SpectralNorm(nn.Module):
    """spectral normalization for stabilized training."""
    
    def __init__(self, module: nn.Module, name: str = 'weight', n_power_iterations: int = 1):
        super().__init__()
        self.module = module
        self.name = name
        self.n_power_iterations = n_power_iterations
        self._make_params()
    
    def _make_params(self):
        w = getattr(self.module, self.name)
        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]
        
        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = F.normalize(u.data, dim=0)
        v.data = F.normalize(v.data, dim=0)
        
        delattr(self.module, self.name)
        self.module.register_parameter(self.name + "_bar", nn.Parameter(w.data))
        self.module.register_buffer(self.name + "_u", u.data)
        self.module.register_buffer(self.name + "_v", v.data)
    
    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")
        
        height = w.data.shape[0]
        for _ in range(self.n_power_iterations):
            v.data = F.normalize(torch.mv(w.view(height, -1).data.t(), u.data), dim=0)
            u.data = F.normalize(torch.mv(w.view(height, -1).data, v.data), dim=0)
        
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))
    
    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


class SelfAttention2D(nn.Module):
    """
    self-attention mechanism for 2d feature maps.
    
    computes attention between all spatial positions, allowing the model
    to capture long-range dependencies crucial for anatomical consistency.
    """
    
    def __init__(self, in_channels: int, reduction: int = 8):
        super().__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        
        # reduced channel dimension for efficiency
        self.query_conv = nn.Conv2d(in_channels, in_channels // reduction, 1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // reduction, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        
        # learnable attention scaling
        self.gamma = nn.Parameter(torch.zeros(1))
        
        # layer norm for stability
        self.norm = nn.InstanceNorm2d(in_channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        args:
            x: input tensor [b, c, h, w]
        returns:
            attention-weighted output [b, c, h, w]
        """
        B, C, H, W = x.shape
        
        # compute query, key, value projections
        query = self.query_conv(x).view(B, -1, H * W).permute(0, 2, 1)  # [b, hw, c']
        key = self.key_conv(x).view(B, -1, H * W)  # [b, c', hw]
        value = self.value_conv(x).view(B, -1, H * W)  # [b, c, hw]
        
        # scaled dot-product attention
        attention = torch.bmm(query, key)  # [b, hw, hw]
        attention = F.softmax(attention / math.sqrt(C // self.reduction), dim=-1)
        
        # apply attention to values
        out = torch.bmm(value, attention.permute(0, 2, 1))  # [b, c, hw]
        out = out.view(B, C, H, W)
        
        # residual connection with learnable scale
        out = self.gamma * self.norm(out) + x
        
        return out


class ChannelAttention(nn.Module):
    """channel attention module (squeeze-and-excitation style)."""
    
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return torch.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """spatial attention module."""
    
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        return torch.sigmoid(self.conv(concat))


class CBAM(nn.Module):
    """convolutional block attention module combining channel and spatial attention."""
    
    def __init__(self, in_channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


class ResidualBlockWithAttention(nn.Module):
    """residual block with optional attention mechanism."""
    
    def __init__(
        self,
        channels: int,
        use_attention: bool = False,
        attention_type: str = 'cbam'
    ):
        super().__init__()
        
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, bias=False),
            nn.InstanceNorm2d(channels)
        )
        
        self.use_attention = use_attention
        if use_attention:
            if attention_type == 'cbam':
                self.attention = CBAM(channels)
            elif attention_type == 'self':
                self.attention = SelfAttention2D(channels)
            else:
                self.attention = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        if self.use_attention:
            out = self.attention(out)
        return out + x


class ModalityEncoder(nn.Module):
    """
    modality-specific encoder that processes each mri modality separately
    before combining them, preserving modality-specific features.
    """
    
    def __init__(self, out_channels: int = 16):
        super().__init__()
        
        # separate encoder for each modality
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(1, out_channels, 3, bias=False),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for _ in range(4)  # t1, t1ce, t2, flair
        ])
        
        # fusion layer with attention
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 4, out_channels * 4, 1, bias=False),
            nn.InstanceNorm2d(out_channels * 4),
            nn.ReLU(inplace=True)
        )
        self.fusion_attention = CBAM(out_channels * 4)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        args:
            x: input tensor [b, 4, h, w] (4 mri modalities)
        returns:
            fused features [b, out_channels*4, h, w]
        """
        # process each modality separately
        modality_features = []
        for i, encoder in enumerate(self.encoders):
            modality_features.append(encoder(x[:, i:i+1, :, :]))
        
        # concatenate and fuse with attention
        fused = torch.cat(modality_features, dim=1)
        fused = self.fusion(fused)
        fused = self.fusion_attention(fused)
        
        return fused


class SAGenerator(nn.Module):
    """
    self-attention generator (sa-generator) for mri domain translation.
    
    architecture:
    1. modality-aware encoder with separate processing per modality
    2. multi-scale encoder with cbam attention
    3. self-attention bottleneck for long-range dependencies
    4. decoder with skip connections and cbam attention
    
    this architecture is specifically designed for multi-modal mri translation,
    preserving anatomical structures and modality-specific information.
    """
    
    def __init__(
        self,
        input_channels: int = 4,
        output_channels: int = 4,
        ngf: int = 64,
        n_residual_blocks: int = 9,
        use_modality_encoder: bool = True,
        attention_layers: List[int] = [3, 4, 5]  # which residual blocks get self-attention
    ):
        super().__init__()
        
        self.use_modality_encoder = use_modality_encoder
        
        # initial modality-aware encoding (optional)
        if use_modality_encoder:
            self.modality_encoder = ModalityEncoder(out_channels=16)
            first_layer_in = 64  # 16 * 4 modalities
        else:
            first_layer_in = input_channels
        
        # initial convolution
        self.initial = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(first_layer_in, ngf, 7, bias=False),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True)
        )
        
        # encoder (downsampling with attention)
        self.encoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ngf, ngf * 2, 3, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(ngf * 2),
                nn.ReLU(inplace=True),
                CBAM(ngf * 2)
            ),
            nn.Sequential(
                nn.Conv2d(ngf * 2, ngf * 4, 3, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(ngf * 4),
                nn.ReLU(inplace=True),
                CBAM(ngf * 4)
            )
        ])
        
        # bottleneck with self-attention at specified layers
        self.bottleneck = nn.ModuleList()
        for i in range(n_residual_blocks):
            use_self_attention = i in attention_layers
            self.bottleneck.append(
                ResidualBlockWithAttention(
                    ngf * 4,
                    use_attention=True,
                    attention_type='self' if use_self_attention else 'cbam'
                )
            )
        
        # global self-attention in bottleneck
        self.global_attention = SelfAttention2D(ngf * 4, reduction=4)
        
        # decoder (upsampling with attention)
        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, stride=2, padding=1, output_padding=1, bias=False),
                nn.InstanceNorm2d(ngf * 2),
                nn.ReLU(inplace=True),
                CBAM(ngf * 2)
            ),
            nn.Sequential(
                nn.ConvTranspose2d(ngf * 2, ngf, 3, stride=2, padding=1, output_padding=1, bias=False),
                nn.InstanceNorm2d(ngf),
                nn.ReLU(inplace=True),
                CBAM(ngf)
            )
        ])
        
        # output layer
        self.output = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_channels, 7),
            nn.Tanh()
        )
        
        # skip connections fusion
        self.skip_fuse1 = nn.Conv2d(ngf * 4, ngf * 2, 1, bias=False)
        self.skip_fuse2 = nn.Conv2d(ngf * 2, ngf, 1, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward pass through sa-generator.
        
        args:
            x: input mri [b, 4, h, w]
        returns:
            translated mri [b, 4, h, w]
        """
        # modality-aware encoding
        if self.use_modality_encoder:
            x = self.modality_encoder(x)
        
        # initial convolution
        x = self.initial(x)
        
        # encoder with skip connections
        skips = [x]
        for enc in self.encoder:
            x = enc(x)
            skips.append(x)
        
        # bottleneck with self-attention
        for block in self.bottleneck:
            x = block(x)
        
        # global attention
        x = self.global_attention(x)
        
        # decoder with skip connections
        x = self.decoder[0](x)
        # add skip connection from encoder
        skip = self.skip_fuse1(skips[-1])
        x = x + F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        x = self.decoder[1](x)
        skip = self.skip_fuse2(skips[-2])
        x = x + F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        # output
        x = self.output(x)
        
        return x


class SADiscriminator(nn.Module):
    """
    self-attention patchgan discriminator.
    
    multi-scale discriminator with spectral normalization and self-attention
    for improved training stability and realistic outputs.
    """
    
    def __init__(
        self,
        input_channels: int = 4,
        ndf: int = 64,
        n_layers: int = 3,
        use_spectral_norm: bool = True
    ):
        super().__init__()
        
        def get_norm(layer):
            if use_spectral_norm:
                return SpectralNorm(layer)
            return layer
        
        # initial layer (no normalization)
        sequence = [
            get_norm(nn.Conv2d(input_channels, ndf, 4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        # hidden layers
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            
            sequence.extend([
                get_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 4, stride=2, padding=1, bias=False)),
                nn.InstanceNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, inplace=True)
            ])
        
        # self-attention layer
        sequence.append(SelfAttention2D(ndf * nf_mult))
        
        # final layers
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        
        sequence.extend([
            get_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 4, stride=1, padding=1, bias=False)),
            nn.InstanceNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, inplace=True)
        ])
        
        # output layer (patchgan)
        sequence.append(
            get_norm(nn.Conv2d(ndf * nf_mult, 1, 4, stride=1, padding=1))
        )
        
        self.model = nn.Sequential(*sequence)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        args:
            x: input image [b, c, h, w]
        returns:
            patch-wise discrimination scores [b, 1, h', w']
        """
        return self.model(x)


class MultiScaleDiscriminator(nn.Module):
    """
    multi-scale discriminator for better handling of different frequency content.
    
    processes images at multiple scales to capture both fine details and
    global structure.
    """
    
    def __init__(
        self,
        input_channels: int = 4,
        ndf: int = 64,
        n_layers: int = 3,
        n_scales: int = 2
    ):
        super().__init__()
        
        self.n_scales = n_scales
        self.discriminators = nn.ModuleList([
            SADiscriminator(input_channels, ndf, n_layers)
            for _ in range(n_scales)
        ])
        
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1)
        
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        args:
            x: input image [b, c, h, w]
        returns:
            list of discrimination scores at each scale
        """
        outputs = []
        for i, disc in enumerate(self.discriminators):
            if i > 0:
                x = self.downsample(x)
            outputs.append(disc(x))
        return outputs


class SACycleGAN(nn.Module):
    """
    complete sa-cyclegan model for bidirectional domain translation.
    
    combines:
    - two sa-generators (a->b and b->a)
    - two multi-scale discriminators (for domain a and b)
    """
    
    def __init__(
        self,
        input_channels: int = 4,
        ngf: int = 64,
        ndf: int = 64,
        n_residual_blocks: int = 9,
        n_discriminator_layers: int = 3,
        n_discriminator_scales: int = 2,
        use_modality_encoder: bool = True
    ):
        super().__init__()
        
        # generators
        self.G_A2B = SAGenerator(
            input_channels=input_channels,
            output_channels=input_channels,
            ngf=ngf,
            n_residual_blocks=n_residual_blocks,
            use_modality_encoder=use_modality_encoder
        )
        
        self.G_B2A = SAGenerator(
            input_channels=input_channels,
            output_channels=input_channels,
            ngf=ngf,
            n_residual_blocks=n_residual_blocks,
            use_modality_encoder=use_modality_encoder
        )
        
        # discriminators
        self.D_A = MultiScaleDiscriminator(
            input_channels=input_channels,
            ndf=ndf,
            n_layers=n_discriminator_layers,
            n_scales=n_discriminator_scales
        )
        
        self.D_B = MultiScaleDiscriminator(
            input_channels=input_channels,
            ndf=ndf,
            n_layers=n_discriminator_layers,
            n_scales=n_discriminator_scales
        )
        
    def forward(
        self,
        real_A: torch.Tensor,
        real_B: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        forward pass for training.
        
        args:
            real_a: real images from domain a [b, c, h, w]
            real_b: real images from domain b [b, c, h, w]
            
        returns:
            fake_b: generated images in domain b
            fake_a: generated images in domain a
            rec_a: reconstructed images in domain a
            rec_b: reconstructed images in domain b
        """
        # forward cycle: a -> b -> a
        fake_B = self.G_A2B(real_A)
        rec_A = self.G_B2A(fake_B)
        
        # backward cycle: b -> a -> b
        fake_A = self.G_B2A(real_B)
        rec_B = self.G_A2B(fake_A)
        
        return fake_B, fake_A, rec_A, rec_B
    
    @torch.no_grad()
    def translate_A2B(self, x: torch.Tensor) -> torch.Tensor:
        """translate from domain a to domain b."""
        return self.G_A2B(x)
    
    @torch.no_grad()
    def translate_B2A(self, x: torch.Tensor) -> torch.Tensor:
        """translate from domain b to domain a."""
        return self.G_B2A(x)


def count_parameters(model: nn.Module) -> int:
    """count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_sa_cyclegan(
    input_channels: int = 4,
    ngf: int = 64,
    ndf: int = 64,
    n_residual_blocks: int = 9
) -> SACycleGAN:
    """
    factory function to create sa-cyclegan model.
    
    args:
        input_channels: number of input channels (4 for multi-modal mri)
        ngf: number of generator filters
        ndf: number of discriminator filters
        n_residual_blocks: number of residual blocks in generator
        
    returns:
        sacyclegan model instance
    """
    model = SACycleGAN(
        input_channels=input_channels,
        ngf=ngf,
        ndf=ndf,
        n_residual_blocks=n_residual_blocks
    )
    
    # print parameter counts
    g_params = count_parameters(model.G_A2B)
    d_params = count_parameters(model.D_A)
    total = count_parameters(model)
    
    print(f"sa-cyclegan architecture summary:")
    print(f"  generator parameters: {g_params:,}")
    print(f"  discriminator parameters: {d_params:,}")
    print(f"  total parameters: {total:,}")
    
    return model


if __name__ == '__main__':
    # test the architecture
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # create model
    model = create_sa_cyclegan()
    model = model.to(device)
    
    # test forward pass
    batch_size = 2
    x_a = torch.randn(batch_size, 4, 256, 256).to(device)
    x_b = torch.randn(batch_size, 4, 256, 256).to(device)
    
    fake_b, fake_a, rec_a, rec_b = model(x_a, x_b)
    
    print(f"\ninput shape: {x_a.shape}")
    print(f"output shape: {fake_b.shape}")
    print(f"reconstruction shape: {rec_a.shape}")
    
    # test discriminator
    d_out = model.D_A(x_a)
    print(f"discriminator outputs: {[o.shape for o in d_out]}")

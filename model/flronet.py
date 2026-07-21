from typing import List, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from .transolver2D import Model as TransolverModel

class SpectralConv2d(nn.Module):

    def __init__(self, embedding_dim: int, n_hmodes: int, n_wmodes: int):
        super().__init__()
        self.embedding_dim: int = embedding_dim
        self.n_hmodes: int = n_hmodes
        self.n_wmodes: int = n_wmodes
        self.scale: float = 0.02    
        self.weights_real = nn.Parameter(
            self.scale * torch.randn(2, embedding_dim, embedding_dim, n_hmodes, n_wmodes, dtype=torch.float)
        )
        self.weights_imag = nn.Parameter(
            self.scale * torch.randn(2, embedding_dim, embedding_dim, n_hmodes, n_wmodes, dtype=torch.float)
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.ndim == 4
        n_frames, embedding_dim, H, W = input.shape
        assert embedding_dim == self.embedding_dim

        padded_H: int = self.next_power_of_2(H)
        padded_W: int = self.next_power_of_2(W)
        padded_input: torch.Tensor = F.pad(input=input, pad=(0, padded_W - W, 0, padded_H - H), mode='constant', value=0)
        # FFT
        fourier_coeff: torch.Tensor = torch.fft.rfft2(padded_input, dim=(2, 3), norm="ortho")
        output_real = torch.zeros((n_frames, embedding_dim, H, W), device=input.device)
        output_imag = torch.zeros((n_frames, embedding_dim, H, W), device=input.device)

        pos_freq_slice: Tuple[slice, slice, slice, slice] = (
            slice(None), slice(None), slice(None, self.n_hmodes), slice(None, self.n_wmodes)
        )   # [:, :, :self.n_hmodes, :self.n_wmodes] 
        neg_freq_slice: Tuple[slice, slice, slice, slice] = (
            slice(None), slice(None), slice(-self.n_hmodes, None), slice(None, self.n_wmodes)
        )   # [:, :, -self.n_hmodes:, :self.n_wmodes]
        output_real[pos_freq_slice], output_imag[pos_freq_slice] = self.complex_mul(
            input_real=fourier_coeff.real[pos_freq_slice], 
            input_imag=fourier_coeff.imag[pos_freq_slice],
            weights_real=self.weights_real[0],
            weights_imag=self.weights_imag[0],
        )
        output_real[neg_freq_slice], output_imag[neg_freq_slice] = self.complex_mul(
            input_real=fourier_coeff.real[neg_freq_slice], 
            input_imag=fourier_coeff.imag[neg_freq_slice],
            weights_real=self.weights_real[1],
            weights_imag=self.weights_imag[1],
        )
        # IFFT
        output: torch.Tensor = torch.complex(real=output_real, imag=output_imag)
        output = torch.fft.irfft2(input=output, s=(H, W), dim=(2, 3), norm="ortho")
        assert output.shape == input.shape == (n_frames, embedding_dim, H, W)
        return output

    @staticmethod
    def complex_mul(
        input_real: torch.Tensor,
        input_imag: torch.Tensor,
        weights_real: torch.Tensor,
        weights_imag: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ops: str = 'nihw,iohw->nohw'
        real_part: torch.Tensor = (
            torch.einsum(ops, input_real, weights_real) - torch.einsum(ops, input_imag, weights_imag)
        )
        imag_part: torch.Tensor = (
            torch.einsum(ops, input_real, weights_imag) + torch.einsum(ops, input_imag, weights_real)
        )
        return real_part, imag_part

    @staticmethod
    def next_power_of_2(x: int) -> int:
        return 1 if x == 0 else 2 ** (x - 1).bit_length()


###
class AFNOLayer(nn.Module):
    def __init__(self, embedding_dim, img_size=(48, 128), patch_size=(8, 8)):
        super().__init__()
        self.hidden_size = embedding_dim
        self.img_size = img_size
        self.patch_size = patch_size
        self.a = img_size[0] // patch_size[0]
        self.b = img_size[1] // patch_size[1]

        self.num_blocks = 8 # afno默认为8
        self.block_size = self.hidden_size // self.num_blocks
        assert self.hidden_size % self.num_blocks == 0

        self.scale = 0.02
        self.w1 = torch.nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size))
        self.b1 = torch.nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))
        self.w2 = torch.nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size))
        self.b2 = torch.nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))
        self.relu = nn.ReLU()
        self.bias = nn.Conv1d(self.hidden_size, self.hidden_size, 1)
        self.softshrink = 0.01

    def multiply(self, input, weights):
        return torch.einsum('...bd,bdk->...bk', input, weights)

    def forward(self, x):
        B, N, C = x.shape
        a, b = self.a, self.b

        if self.bias:
            bias = self.bias(x.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            bias = torch.zeros(x.shape, device=x.device)

        x = x.reshape(B, a, b, C).float()

        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        x = x.reshape(B, x.shape[1], x.shape[2], self.num_blocks, self.block_size)

        x_real_1 = F.relu(self.multiply(x.real, self.w1[0]) - self.multiply(x.imag, self.w1[1]) + self.b1[0])
        x_imag_1 = F.relu(self.multiply(x.real, self.w1[1]) + self.multiply(x.imag, self.w1[0]) + self.b1[1])
        x_real_2 = self.multiply(x_real_1, self.w2[0]) - self.multiply(x_imag_1, self.w2[1]) + self.b2[0]
        x_imag_2 = self.multiply(x_real_1, self.w2[1]) + self.multiply(x_imag_1, self.w2[0]) + self.b2[1]

        x = torch.stack([x_real_2, x_imag_2], dim=-1).float()
        x = F.softshrink(x, lambd=self.softshrink) if self.softshrink else x

        x = torch.view_as_complex(x)
        x = x.reshape(B, x.shape[1], x.shape[2], self.hidden_size)
        x = torch.fft.irfft2(x, s=(a, b), dim=(1, 2), norm='ortho')
        x = x.reshape(B, N, C)

        x = x + bias
        return x

class PatchEmbed(nn.Module):
    def __init__(self, img_size=(48, 128), patch_size=(8, 8), in_chans=64, embed_dim=768):
        super().__init__()
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class DePatchEmbed(nn.Module):
    def __init__(self, img_size=(48, 128), patch_size=(8, 8), out_chans=64, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.out_chans = out_chans
        self.patch_dim = patch_size[0] * patch_size[1] * out_chans
        self.proj = nn.Linear(embed_dim, self.patch_dim)

    def forward(self, x):
        x = self.proj(x)
        h_grid = self.img_size[0] // self.patch_size[0]
        w_grid = self.img_size[1] // self.patch_size[1]
        x = rearrange(
            x,
            "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
            h=h_grid,
            w=w_grid,
            p1=self.patch_size[0],
            p2=self.patch_size[1],
            c=self.out_chans
        )
        return x

class FNOBranchNet(nn.Module):

    def __init__(self, n_channels: int, n_fno_layers: int, n_hmodes: int, n_wmodes: int, embedding_dim: int, 
                 resolution: Tuple[int, int] = (48, 128), n_timeframes: int = 5, is_afno: bool = False, is_TC: bool = True):
        super().__init__()
        self.n_channels: int = n_channels
        self.n_timeframes: int = n_timeframes
        self.n_fno_layers: int = n_fno_layers
        self.n_hmodes: int = n_hmodes
        self.n_wmodes: int = n_wmodes
        self.embedding_dim: int = embedding_dim
        self.resolution = resolution
        self.is_afno = is_afno
        self.is_TC = is_TC
        
        in_chans_total = n_timeframes * n_channels if is_TC else n_channels
        
        # Determine patch size: should be divisor of H and W
        patch_size = (8, 8) if resolution[0] % 8 == 0 and resolution[1] % 8 == 0 else (resolution[0], resolution[1])

        self.embedding_layer = nn.Sequential(
            nn.Linear(in_features=in_chans_total, out_features=128),
            nn.GELU(),
            nn.Linear(in_features=128, out_features=256),
            nn.GELU(),
            nn.Linear(in_features=256, out_features=embedding_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=128),
            nn.GELU(),
            nn.Linear(in_features=128, out_features=128),
            nn.GELU(),
            nn.Linear(in_features=128, out_features=in_chans_total),
        ) 

        if self.is_afno:
            self.patch_embed = PatchEmbed(img_size=resolution, patch_size=patch_size, in_chans=in_chans_total, embed_dim=768)
            self.num_patches = self.patch_embed.num_patches
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, 768))
            self.de_patch_embed = DePatchEmbed(img_size=resolution, patch_size=patch_size, out_chans=in_chans_total, embed_dim=768)

            self.spectral_conv_layers = nn.ModuleList(
                modules=[AFNOLayer(embedding_dim=768, img_size=resolution, patch_size=patch_size) for _ in range(n_fno_layers)]
            )

        else:
            ###  fno
            self.spectral_conv_layers = nn.ModuleList(
                modules=[SpectralConv2d(embedding_dim=embedding_dim, n_hmodes=n_hmodes, n_wmodes=n_wmodes) for _ in range(n_fno_layers)]
            )
            self.Ws = nn.ModuleList([
                nn.Conv2d(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=1  )
                for _ in range(n_fno_layers)
            ])

            

    def forward(self, sensor_values: torch.Tensor, out_resolution: Tuple[int, int]) -> torch.Tensor:
        batch_size, in_timeframes, n_channels, in_H, in_W = sensor_values.shape  # (B, T, C, H, W)
        assert n_channels == self.n_channels
        assert in_timeframes == self.n_timeframes
        
        # Merge Dimensions
        if self.is_TC:
            flattened_sensor_value: torch.Tensor = sensor_values.flatten(start_dim=1, end_dim=2)  # Merge T and C: (B, T*C, H, W)
        else:
            flattened_sensor_value: torch.Tensor = sensor_values.flatten(start_dim=0, end_dim=1)  # Merge B and T: (B*T, C, H, W)
        
        # embedding
        if self.is_afno:
            output = flattened_sensor_value
        else:
            output = self.embedding_layer(flattened_sensor_value.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        # interpolate embeddings to output resolution
        if out_resolution != (in_H, in_W):
            output = F.interpolate(input=output, size=out_resolution, mode='bilinear', align_corners=False)


        # afno
        if self.is_afno:
            output = self.patch_embed(output)  # patch embedding
            output = output + self.pos_embed  # position embedding
            for i in range(self.n_fno_layers):
                spectral_conv_layer: AFNO = self.spectral_conv_layers[i]
                output = spectral_conv_layer(output)
            output = self.de_patch_embed(output)  # depatch embedding

        # fno
        else:   
            for i in range(self.n_fno_layers):
                spectral_conv_layer: SpectralConv2d = self.spectral_conv_layers[i]
                out1: torch.Tensor = spectral_conv_layer(input=output)
                W: nn.Conv2d = self.Ws[i]
                out2: torch.Tensor = W(input=output)
                output = out1 + out2
                if i < self.n_fno_layers - 1:   # not the last layer
                    output = F.gelu(output)
            output = self.decoder(output.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)      
            
        # Reshape back to (B, T, C, H, W)
        out_H, out_W = out_resolution
        if self.is_TC:
            output = output.reshape(batch_size, self.n_timeframes, self.n_channels, out_H, out_W)
        else:
            output = output.reshape(batch_size, self.n_timeframes, self.n_channels, out_H, out_W)  # B*T -> B, T handles natively by reshape

        return output


class UNetBranchNet(nn.Module):

    def __init__(self, n_channels: int, embedding_dim: int):
        super().__init__()
        self.n_channels: int = n_channels
        self.embedding_dim: int = embedding_dim
        # Encoder
        self.enc_conv1 = self.conv_block(in_channels=n_channels, out_channels=embedding_dim)
        self.enc_conv2 = self.conv_block(in_channels=embedding_dim, out_channels=embedding_dim * 2)
        self.enc_conv3 = self.conv_block(in_channels=embedding_dim * 2, out_channels=embedding_dim * 4)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Bottleneck
        self.bottleneck_conv = self.conv_block(in_channels=embedding_dim * 4, out_channels=embedding_dim * 8)
        # Decoder
        self.upconv3 = self.upconv(in_channels=embedding_dim * 8, out_channels=embedding_dim * 4)
        self.dec_conv3 = self.conv_block(in_channels=embedding_dim * 8, out_channels=embedding_dim * 4)
        self.upconv2 = self.upconv(in_channels=embedding_dim * 4, out_channels=embedding_dim * 2)
        self.dec_conv2 = self.conv_block(in_channels=embedding_dim * 4, out_channels=embedding_dim * 2)
        self.upconv1 = self.upconv(in_channels=embedding_dim * 2, out_channels=embedding_dim)
        self.dec_conv1 = self.conv_block(in_channels=embedding_dim * 2, out_channels=embedding_dim)
        # Final convolution
        self.final_conv = nn.Conv2d(in_channels=embedding_dim, out_channels=n_channels, kernel_size=1)

    def forward(self, sensor_values: torch.Tensor) -> torch.Tensor:
        assert sensor_values.ndim == 5
        batch_size, in_timesteps, n_channels, H, W = sensor_values.shape
        assert n_channels == self.n_channels
        # Layer Norm
        reshaped_input: torch.Tensor = sensor_values.flatten(start_dim=0, end_dim=1)
        # Encoder
        enc1: torch.Tensor = self.enc_conv1(reshaped_input)
        enc2: torch.Tensor = self.enc_conv2(self.pool(enc1))
        enc3: torch.Tensor = self.enc_conv3(self.pool(enc2))
        # Bottleneck
        bottleneck: torch.Tensor = self.bottleneck_conv(self.pool(enc3))
        # Decoder
        dec3: torch.Tensor = self.upconv3(bottleneck)
        if dec3.shape[-2:] != enc3.shape[-2:]:  # due to input resolution not a power of 2
            dec3 = F.interpolate(dec3, size=enc3.shape[-2:], mode='bilinear', align_corners=False)
        dec3 = torch.cat(tensors=[dec3, enc3], dim=1)
        dec3 = self.dec_conv3(dec3)
        dec2: torch.Tensor = self.upconv2(dec3)
        dec2 = torch.cat(tensors=[dec2, enc2], dim=1)
        dec2 = self.dec_conv2(dec2)
        dec1: torch.Tensor = self.upconv1(dec2)
        dec1 = torch.cat(tensors=[dec1, enc1], dim=1)
        dec1 = self.dec_conv1(dec1)
        # Final output
        reshaped_output: torch.Tensor = self.final_conv(dec1)
        assert reshaped_output.shape == reshaped_input.shape
        return reshaped_output.reshape(batch_size, in_timesteps, self.n_channels, H, W)
    
    def conv_block(self, in_channels: int, out_channels: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
        )

    def upconv(self, in_channels: int, out_channels: int) -> nn.Module:
        return nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2)


class TransolverBranchNet(nn.Module):
    def __init__(
        self, 
        n_channels: int, 
        n_layers: int, 
        n_hidden: int, 
        n_head: int, 
        resolution: Tuple[int, int], 
        n_timeframes: int = 5,
        slice_num: int = 32,
        out_dim: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.resolution = resolution
        self.H, self.W = resolution
        
        # Grid coordinates (normalized to [0, 1])
        # We use unified_pos=False in TransolverModel, so we provide coordinates
        grid_x = torch.linspace(0, 1, self.H)
        grid_y = torch.linspace(0, 1, self.W)
        grid_x, grid_y = torch.meshgrid(grid_x, grid_y, indexing='ij')
        # (1, N, 2) 
        self.register_buffer('grid', torch.stack([grid_x, grid_y], dim=-1).reshape(1, self.H * self.W, 2))

        in_chans_total = n_timeframes * n_channels
        out_chans_total = n_timeframes * out_dim

        self.model = TransolverModel(
            space_dim=2,
            n_layers=n_layers,
            n_hidden=n_hidden,
            n_head=n_head,
            fun_dim=in_chans_total,
            out_dim=out_chans_total,
            slice_num=slice_num,
            H=self.H,
            W=self.W,
            dropout=dropout,
            unified_pos=False
        )

    def forward(self, sensor_values: torch.Tensor, out_resolution: Tuple[int, int] | None = None) -> torch.Tensor:
        # sensor_values: (B, T, C, H, W)
        batch_size, in_timeframes, n_channels, in_H, in_W = sensor_values.shape
        assert n_channels == self.n_channels
        
        # Flatten B and T
        # v: torch.Tensor = sensor_values.flatten(start_dim=0, end_dim=1) # (B*T, C, in_H, in_W)
        # Flatten T and C
        v: torch.Tensor = sensor_values.flatten(start_dim=1, end_dim=2) # (B, T*C, in_H, in_W)
        
        # Interpolate if input resolution is different from model resolution
        if (in_H, in_W) != self.resolution:
            v = F.interpolate(input=v, size=self.resolution, mode='bilinear', align_corners=False)
            curr_H, curr_W = self.resolution
        else:
            curr_H, curr_W = in_H, in_W
            
        # Reshape to (B*T, N, C) for Transolver
        # fx = v.permute(0, 2, 3, 1).reshape(batch_size * in_timeframes, curr_H * curr_W, n_channels)
        fx = v.permute(0, 2, 3, 1).reshape(batch_size, curr_H * curr_W, in_timeframes * n_channels)
        
        # Provide coordinates grid
        x = self.grid.repeat(batch_size * in_timeframes, 1, 1) # (B*T, N, 2)
        x = self.grid.repeat(batch_size, 1, 1) # (B*T, N, 2)
        
        # Transolver forward
        output = self.model(x, fx) # (B*T, N, out_dim)
        
        # Reshape back to (B*T, C, H, W)
        # output = output.reshape(batch_size * in_timeframes, curr_H, curr_W, -1).permute(0, 3, 1, 2)
        output = output.reshape(batch_size, curr_H, curr_W, -1).permute(0, 3, 1, 2)
        
        # Interpolate to out_resolution if requested
        if out_resolution is not None and out_resolution != (curr_H, curr_W):
            output = F.interpolate(input=output, size=out_resolution, mode='bilinear', align_corners=False)
            final_H, final_W = out_resolution
        else:
            final_H, final_W = curr_H, curr_W
            
        # Reshape back to (B, T, C, H, W)
        output = output.reshape(batch_size, in_timeframes, -1, final_H, final_W)
        return output


class MLPBranchNet(nn.Module):

    def __init__(self, n_channels: int, embedding_dim: int, n_sensors: int, resolution: Tuple[int, int]):
        super().__init__()
        self.n_channels: int = n_channels
        self.embedding_dim: int = embedding_dim
        self.n_sensors: int = n_sensors
        self.resolution: Tuple[int, int] = resolution
        self.H, self.W = resolution
        
        hidden_dim: int = embedding_dim * n_sensors
        block0 = nn.Sequential(
            nn.Linear(in_features=n_channels * n_sensors, out_features=hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )
        block1 = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )
        block2 = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )
        block3 = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
        )
        block4 = nn.Linear(in_features=hidden_dim, out_features=n_channels * self.H * self.W)
        self.blocks = nn.Sequential(block0, block1, block2, block3, block4)

    def forward(self, sensor_values: torch.Tensor):
        assert sensor_values.ndim == 4
        batch_size, in_timesteps, n_channels, n_sensors = sensor_values.shape
        assert n_channels == self.n_channels
        assert n_sensors == self.n_sensors
        
        output: torch.Tensor = sensor_values.flatten(start_dim=2, end_dim=3)
        assert output.shape == (batch_size, in_timesteps, n_channels * n_sensors)
        output = self.blocks(output)
        assert output.shape == (batch_size, in_timesteps, n_channels * self.H * self.W)
        return output.reshape(batch_size, in_timesteps, n_channels, self.H, self.W)
    

class SinusoidEmbedding(nn.Module):

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim: int = embedding_dim

        # Frequency scaling
        self.register_buffer('w', 1. / torch.pow(
            input=torch.tensor(10_000., dtype=torch.float),
            exponent=torch.arange(0, embedding_dim, 2, dtype=torch.float) / embedding_dim,
        ))

    def forward(self, timeframes: torch.Tensor) -> torch.Tensor:
        assert timeframes.ndim == 2
        batch_size, n_timeframes = timeframes.shape
        timeframes = timeframes.unsqueeze(-1)  # (batch_size, n_timeframes, 1)
        sinusoid = torch.zeros(*timeframes.shape[:-1], self.embedding_dim, device=timeframes.device)
        sinusoid[:, :, 0::2] = torch.sin(timeframes * self.w)
        sinusoid[:, :, 1::2] = torch.cos(timeframes * self.w)
        assert sinusoid.shape == (batch_size, n_timeframes, self.embedding_dim)
        return sinusoid


class TrunkNet(nn.Module):
    """
    Temporal Cross-Attention Trunk Network.

    The target (fullstate) timeframe embeddings serve as **Queries**,
    and the sensor timeframe embeddings serve as **Keys**.  The resulting
    attention weight matrix A of shape (B, T_f, T_s) is softmax-normalised
    over the Key (T_s) dimension, so for every target frame f the sensor
    weights sum to 1 — directly interpretable as a learned temporal
    interpolation kernel.

    The weight matrix A is then used in the DeepONet fusion step as:
        output[b,f] = Σ_s  A[b,f,s] * BranchNet_output[b,s]   (Cross-Attention × V)

    This is exactly standard Cross-Attention where:
        Q = query_mlp(sinusoid(fullstate_times))   shape (B, T_f, E)
        K = key_mlp(sinusoid(sensor_times))        shape (B, T_s, E)
        V = BranchNet spatial feature maps         shape (B, T_s, C, H, W)
    """
    def __init__(self, embedding_dim: int, n_outputs: int, is_cross_attn: bool = True):
        super().__init__()
        self.embedding_dim: int = embedding_dim
        self.n_outputs: int = n_outputs
        self.scale: float = 1.0 / math.sqrt(embedding_dim)
        self.is_cross_attn: bool = is_cross_attn

        # Per-stack independent Q and K projection MLPs
        self.query_mlps = nn.ModuleList(
            modules=[
                nn.Sequential(
                    nn.Linear(embedding_dim, embedding_dim),
                    nn.GELU(),
                    nn.Linear(embedding_dim, embedding_dim),
                )
                for _ in range(n_outputs)
            ]
        )
        self.key_mlps = nn.ModuleList(
            modules=[
                nn.Sequential(
                    nn.Linear(embedding_dim, embedding_dim),
                    nn.GELU(),
                    nn.Linear(embedding_dim, embedding_dim),
                )
                for _ in range(n_outputs)
            ]
        )
        self.mlps = nn.ModuleList(
            modules=[
                nn.Sequential(
                    nn.Linear(embedding_dim, embedding_dim),
                    nn.GELU(),
                    nn.Linear(embedding_dim, embedding_dim),
                )
                for _ in range(n_outputs)
            ]
        )

    def forward(
        self,
        fullstate_time_embeddings: torch.Tensor,  # (B, T_f, E)  — Queries
        sensor_time_embeddings: torch.Tensor,     # (B, T_s, E)  — Keys
    ) -> List[torch.Tensor]:
        assert fullstate_time_embeddings.ndim == sensor_time_embeddings.ndim == 3
        assert fullstate_time_embeddings.shape[0] == sensor_time_embeddings.shape[0]
        assert fullstate_time_embeddings.shape[2] == sensor_time_embeddings.shape[2] == self.embedding_dim
        batch_size: int = fullstate_time_embeddings.shape[0]
        n_fullstate_timeframes: int = fullstate_time_embeddings.shape[1]  # T_f
        n_sensor_timeframes: int = sensor_time_embeddings.shape[1]        # T_s

        outputs: List[torch.Tensor] = []
        for i in range(self.n_outputs):
            if self.is_cross_attn:
                # Q: (B, T_f, E)   K: (B, T_s, E)
                Q = self.query_mlps[i](fullstate_time_embeddings)  # (B, T_f, E)
                K = self.key_mlps[i](sensor_time_embeddings)      # (B, T_s, E)

                # Scaled dot-product: (B, T_f, E) × (B, E, T_s) → (B, T_f, T_s)
                # Softmax over T_s (dim=-1): each target frame's sensor weights sum to 1
                attn = torch.softmax(
                    torch.einsum('nfe,nse->nfs', Q, K) * self.scale,
                    dim=-1,
                )  # (B, T_f, T_s)
            else:
                mlp = self.mlps[i]
                attn = torch.einsum(
                    'nse,nfe->nfs',
                    mlp(sensor_time_embeddings),
                    mlp(fullstate_time_embeddings),
                )  # (B, T_f, T_s)

            assert attn.shape == (batch_size, n_fullstate_timeframes, n_sensor_timeframes)
            outputs.append(attn)
        return outputs


class _MeanFieldBase(nn.Module):
    """
    均值场网络公共基类：提供时间感知加权聚合（Δt → softmax → 加权求和）。
    子类实现 _spatial_process(x) 来定义空间处理器。

    数据流：
        sensor_values (B,T_s,C,H,W) × 时间权重 w(B,T_f,T_s)
            → 加权求和 x̃ (B,T_f,C,H,W)
            → reshape (B*T_f,C,H,W)
            → _spatial_process
            → μ(x,t_f) (B,T_f,C,H,W)
    """

    def __init__(self, n_channels: int, time_embedding_dim: int = 32):
        super().__init__()
        self.n_channels = n_channels
        self.time_embedding_dim = time_embedding_dim
        # Temporal weighting: sinusoid(Δt) → scalar → softmax over T_s
        self.time_embed = SinusoidEmbedding(embedding_dim=time_embedding_dim)
        self.temporal_proj = nn.Linear(time_embedding_dim, 1, bias=True)
        # Zero-init → uniform weights at start (≡ simple mean)
        nn.init.zeros_(self.temporal_proj.weight)
        nn.init.zeros_(self.temporal_proj.bias)

    def _temporal_weights(
        self,
        sensor_times: torch.Tensor,    # (B, T_s)
        fullstate_times: torch.Tensor, # (B, T_f)
    ) -> torch.Tensor:
        """Returns softmax weights (B, T_f, T_s). Δt = t_f - t_s (signed)."""
        dt = fullstate_times.unsqueeze(2) - sensor_times.unsqueeze(1)  # (B, T_f, T_s)
        B, T_f, T_s = dt.shape
        dt_flat = dt.reshape(B, T_f * T_s)
        e = self.time_embed(dt_flat).reshape(B, T_f, T_s, self.time_embedding_dim)
        logits = self.temporal_proj(e).squeeze(-1)                      # (B, T_f, T_s)
        return F.softmax(logits, dim=-1)

    def _spatial_process(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B*T_f, C, H, W)  →  (B*T_f, C, H, W). Override in subclass."""
        raise NotImplementedError

    def forward(
        self,
        sensor_values: torch.Tensor,    # (B, T_s, C, H, W)
        sensor_times: torch.Tensor,     # (B, T_s)
        fullstate_times: torch.Tensor,  # (B, T_f)
    ) -> torch.Tensor:
        """Returns μ(x, t_f): (B, T_f, C, H, W)"""
        assert sensor_values.ndim == 5
        B, T_s, C, H, W = sensor_values.shape
        T_f = fullstate_times.shape[1]
        # Step 1: learned temporal weights
        w = self._temporal_weights(sensor_times, fullstate_times)       # (B, T_f, T_s)
        # Step 2: time-weighted aggregation
        x = torch.einsum('bfs,bschw->bfchw', w, sensor_values)         # (B, T_f, C, H, W)
        # Step 3: spatial processing (per frame, shared weights)
        x = x.reshape(B * T_f, C, H, W)
        x = self._spatial_process(x)                                    # (B*T_f, C, H, W)
        return x.reshape(B, T_f, C, H, W)


# ---------------------------------------------------------------------------
# MeanFieldNet (CNN backbone) — default / FLRONetUNet
# ---------------------------------------------------------------------------
class MeanFieldNet(_MeanFieldBase):
    """
    均值场网络 — CNN 残差骨干（默认，用于 FLRONetUNet / 无特定骨干时）。
    空间处理：Stem(1×1) + 3×残差块(3×3 Conv+GN) + Head(1×1)
    """

    def __init__(self, n_channels: int, hidden_channels: int = 32, time_embedding_dim: int = 32):
        super().__init__(n_channels=n_channels, time_embedding_dim=time_embedding_dim)

        def _res_block(ch_in: int, ch_out: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(num_groups=min(8, ch_out), num_channels=ch_out),
                nn.GELU(),
                nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(num_groups=min(8, ch_out), num_channels=ch_out),
            )

        self.stem = nn.Conv2d(n_channels, hidden_channels, kernel_size=1, bias=True)
        self.res1 = _res_block(hidden_channels, hidden_channels)
        self.res2 = _res_block(hidden_channels, hidden_channels)
        self.res3 = _res_block(hidden_channels, hidden_channels)
        self.head = nn.Conv2d(hidden_channels, n_channels, kernel_size=1, bias=True)
        self.act  = nn.GELU()
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def _spatial_process(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.stem(x))
        x = self.act(self.res1(x) + x)
        x = self.act(self.res2(x) + x)
        x = self.act(self.res3(x) + x)
        return self.head(x)


# ---------------------------------------------------------------------------
# FNOMeanFieldNet — FNO 谱卷积骨干（用于 FLRONetFNO）
# ---------------------------------------------------------------------------
class FNOMeanFieldNet(_MeanFieldBase):
    """
    均值场网络 — FNO 谱卷积骨干（与 FNOBranchNet 空间结构一致）。

    空间处理流程（与 FNOBranchNet 后半段相同）：
        Linear lift (C → embedding_dim)
        → n_fno_layers × (SpectralConv2d + 1×1Conv 残差)
        → Linear proj (embedding_dim → C)
    """

    def __init__(
        self,
        n_channels: int,
        n_fno_layers: int,
        n_hmodes: int,
        n_wmodes: int,
        embedding_dim: int,
        time_embedding_dim: int = 32,
    ):
        super().__init__(n_channels=n_channels, time_embedding_dim=time_embedding_dim)
        self.n_fno_layers = n_fno_layers

        # Lift: C → embedding_dim (per pixel, applied in channel dim)
        self.lift = nn.Sequential(
            nn.Linear(n_channels, 128),  nn.GELU(),
            nn.Linear(128, 256),         nn.GELU(),
            nn.Linear(256, embedding_dim),
        )
        # Spectral conv layers (same as FNOBranchNet)
        self.spectral_convs = nn.ModuleList([
            SpectralConv2d(embedding_dim, n_hmodes, n_wmodes)
            for _ in range(n_fno_layers)
        ])
        self.Ws = nn.ModuleList([
            nn.Conv2d(embedding_dim, embedding_dim, kernel_size=1)
            for _ in range(n_fno_layers)
        ])
        # Project back to n_channels
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 128), nn.GELU(),
            nn.Linear(128, 128),           nn.GELU(),
            nn.Linear(128, n_channels),
        )

    def _spatial_process(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, H, W)  where N = B*T_f
        x = self.lift(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)      # (N, E, H, W)
        for i in range(self.n_fno_layers):
            x = self.spectral_convs[i](x) + self.Ws[i](x)
            if i < self.n_fno_layers - 1:
                x = F.gelu(x)
        x = self.decoder(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)   # (N, C, H, W)
        return x


# ---------------------------------------------------------------------------
# AFNOMeanFieldNet — AFNO 自适应谱骨干（用于 FLRONetAFNO）
# ---------------------------------------------------------------------------
class AFNOMeanFieldNet(_MeanFieldBase):
    """
    均值场网络 — AFNO 骨干（与 FNOBranchNet AFNO 路径结构一致）。

    空间处理流程：
        PatchEmbed (C → 768) → pos_embed → n_fno_layers × AFNOLayer → DePatchEmbed (768 → C)
    """

    def __init__(
        self,
        n_channels: int,
        n_fno_layers: int,
        resolution: Tuple[int, int],
        time_embedding_dim: int = 32,
    ):
        super().__init__(n_channels=n_channels, time_embedding_dim=time_embedding_dim)
        self.n_fno_layers = n_fno_layers
        patch_size = (8, 8) if resolution[0] % 8 == 0 and resolution[1] % 8 == 0 else resolution
        layer_embed_dim = 768

        self.patch_embed    = PatchEmbed(img_size=resolution, patch_size=patch_size,
                                         in_chans=n_channels, embed_dim=layer_embed_dim)
        self.pos_embed      = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, layer_embed_dim))
        self.afno_layers    = nn.ModuleList([
            AFNOLayer(embedding_dim=layer_embed_dim, img_size=resolution, patch_size=patch_size)
            for _ in range(n_fno_layers)
        ])
        self.de_patch_embed = DePatchEmbed(img_size=resolution, patch_size=patch_size,
                                           out_chans=n_channels, embed_dim=layer_embed_dim)

    def _spatial_process(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, H, W)
        x = self.patch_embed(x) + self.pos_embed                       # (N, P, 768)
        for layer in self.afno_layers:
            x = layer(x)
        return self.de_patch_embed(x)                                   # (N, C, H, W)


# ---------------------------------------------------------------------------
# VoronoiMeanFieldNet — identity spatial pass-through（用于 voronoi_output 模式）
# ---------------------------------------------------------------------------
class VoronoiMeanFieldNet(_MeanFieldBase):
    """
    均值场网络 — 无空间处理（voronoi_output 模式）。

    Voronoi 嵌入已在空间维度上插值为全场 (H, W)，因此时间加权聚合后
    直接输出，不再引入额外的空间参数。

    数据流：
        sensor_values (B,T_s,C,H,W) × 时间权重 w(B,T_f,T_s)
            → 加权求和 x̃ (B,T_f,C,H,W)  → 直接输出为 μ
    """

    def __init__(self, n_channels: int, time_embedding_dim: int = 32):
        super().__init__(n_channels=n_channels, time_embedding_dim=time_embedding_dim)
        # 无额外空间参数

    def _spatial_process(self, x: torch.Tensor) -> torch.Tensor:
        """Identity — Voronoi 已提供全场空间插值，无需再做空间变换。"""
        return x



class _BaseFLRONet(nn.Module):

    # Valid string values for use_mean_field
    _MEAN_FIELD_MODES = ('operator', 'voronoi_output', 'branch_output', 'none')

    def __init__(
        self,
        n_channels: int,
        embedding_dim: int,
        n_stacked_networks: int,
        is_cross_attn: bool = True,   # always True now; kept for checkpoint compat
        use_mean_field: str = 'operator',
        mean_field_hidden: int = 32,
        mean_field_time_embed_dim: int = 32,
    ):
        super().__init__()
        # Normalize legacy bool values from old checkpoints:
        #   True  → 'operator'   (was: use mean field with default backbone)
        #   False → 'none'       (was: no mean field, scalar bias; internal use by FLRONetMLP)
        if isinstance(use_mean_field, bool):
            _use_mean_field_str: str = 'operator' if use_mean_field else 'none'
        else:
            _use_mean_field_str = use_mean_field
        assert _use_mean_field_str in self._MEAN_FIELD_MODES, (
            f"use_mean_field must be one of {self._MEAN_FIELD_MODES}, got '{use_mean_field}'"
        )
        self.n_channels: int = n_channels
        self.embedding_dim: int = embedding_dim
        self.n_stacked_networks: int = n_stacked_networks
        self.use_mean_field: str = _use_mean_field_str
        self.mean_field_hidden: int = mean_field_hidden
        self.mean_field_time_embed_dim: int = mean_field_time_embed_dim
        self.is_cross_attn: bool = is_cross_attn
        # Trunk net (Cross-Attention: Q=fullstate times, K=sensor times, V=BranchNet outputs)
        self.sinusoid_embedding = SinusoidEmbedding(embedding_dim=embedding_dim)
        self.trunk_net = TrunkNet(
            embedding_dim=embedding_dim,
            n_outputs=n_stacked_networks,
            is_cross_attn=is_cross_attn,
        )
        # Dynamic bias: sensor-conditioned mean field μ(x) vs. global learnable bias.
        # FLRONetMLP operates on point sensors (no H/W), so it always uses the scalar bias.
        # For spatial models:
        #   'operator'       → MeanFieldNet (CNN backbone, default)
        #   'voronoi_output' → VoronoiMeanFieldNet (identity spatial pass-through)
        #   'branch_output'  → MeanFieldNet (same CNN backbone, input changed in forward)
        # Subclasses (FLRONetFNO, FLRONetAFNO) will override mean_field_net with their own backbone.
        # Legacy bool True (old checkpoints) is normalized to 'operator' above.
        _mode = self.use_mean_field
        if _mode in ('operator', 'branch_output'):
            self.mean_field_net = MeanFieldNet(
                n_channels=n_channels,
                hidden_channels=mean_field_hidden,
                time_embedding_dim=mean_field_time_embed_dim,
            )
            self.bias = None  # will not be used when mean_field_net is active
        elif _mode == 'voronoi_output':
            self.mean_field_net = VoronoiMeanFieldNet(
                n_channels=n_channels,
                time_embedding_dim=mean_field_time_embed_dim,
            )
            self.bias = None
        else:
            self.mean_field_net = None
            self.bias = nn.Parameter(data=torch.zeros(n_channels, 1, 1))


    def forward(
        self,
        sensor_timeframes: torch.Tensor,
        sensor_values: torch.Tensor,
        fullstate_timeframes: torch.Tensor,
        out_resolution: Tuple[int, int] | None = None,
    ) -> torch.Tensor:
        assert sensor_timeframes.ndim == fullstate_timeframes.ndim == 2
        assert sensor_timeframes.shape[0] == sensor_values.shape[0] == fullstate_timeframes.shape[0]
        batch_size, n_sensor_timeframes = sensor_timeframes.shape
        n_fullstate_timeframes: int = fullstate_timeframes.shape[1]
        if isinstance(self, FLRONetMLP):
            assert sensor_values.ndim == 4
            n_sensors: int = sensor_values.shape[-1]
            assert sensor_values.shape == (batch_size, n_sensor_timeframes, self.n_channels, n_sensors)
            in_H, in_W = self.resolution
        else:
            assert sensor_values.ndim == 5
            in_H, in_W = sensor_values.shape[-2:]
            assert sensor_values.shape == (batch_size, n_sensor_timeframes, self.n_channels, in_H, in_W)

        if isinstance(self, (FLRONetMLP, FLRONetUNet)):
            assert out_resolution is None, f'{self.__class__.__name__} cannot do super resolution'
            out_H, out_W = in_H, in_W
        elif out_resolution is None:
            out_H, out_W = in_H, in_W
        else:
            out_H, out_W = out_resolution

        # TrunkNet: compute cross-attention weights A[b,f,s] = softmax_s( Q_f · K_s / √E )
        # Q = sinusoid(fullstate_times), K = sinusoid(sensor_times)
        fullstate_time_embeddings: torch.Tensor = self.sinusoid_embedding(timeframes=fullstate_timeframes)
        sensor_time_embeddings: torch.Tensor = self.sinusoid_embedding(timeframes=sensor_timeframes)
        # trunk_outputs[i]: (B, T_f, T_s)  — attention weight matrices
        trunk_outputs: List[torch.Tensor] = self.trunk_net(
            fullstate_time_embeddings=fullstate_time_embeddings, sensor_time_embeddings=sensor_time_embeddings
        )
        assert len(trunk_outputs) == self.n_stacked_networks

        # BranchNet: V in the Cross-Attention, shape (B, T_s, C, H, W)
        branch_outputs: List[torch.Tensor] = []
        for i in range(self.n_stacked_networks):
            branch_net: FNOBranchNet | UNetBranchNet | MLPBranchNet | TransolverBranchNet = self.branch_nets[i]
            if isinstance(branch_net, (FNOBranchNet, TransolverBranchNet)):
                branch_output: torch.Tensor = branch_net(sensor_values=sensor_values, out_resolution=(out_H, out_W))
            else:
                branch_output: torch.Tensor = branch_net(sensor_values=sensor_values)
            assert branch_output.shape == (batch_size, n_sensor_timeframes, self.n_channels, out_H, out_W)
            branch_outputs.append(branch_output)

        # Cross-Attention Fusion:
        #   output[b,f,c,h,w] = Σ_s  A[b,f,s] * V[b,s,c,h,w]
        #   where A = trunk_outputs[i] (B, T_f, T_s), V = branch_outputs[i] (B, T_s, C, H, W)
        output: torch.Tensor = torch.zeros(
            batch_size, n_fullstate_timeframes, self.n_channels, out_H, out_W,
            device=sensor_values.device
        )
        for i in range(self.n_stacked_networks):
            output += torch.einsum('nfs,nschw->nfchw', trunk_outputs[i], branch_outputs[i])

        # Compute dynamic bias term (mean field μ)
        if self.mean_field_net is not None and sensor_values.ndim == 5:
            # Determine input to MeanFieldNet based on mode:
            #   'operator' / 'voronoi_output' → raw Voronoi sensor frames
            #   'branch_output'               → sum of all BranchNet outputs over T_s
            if self.use_mean_field == 'branch_output':
                # Σᵢ V[i]: (B, T_s, C, H, W) — combined BranchNet reconstruction at sensor times
                mean_field_input: torch.Tensor = sum(branch_outputs)
            else:
                # 'operator' or 'voronoi_output': use raw Voronoi embeddings
                mean_field_input = sensor_values

            # μ(x, t_f): (B, T_f, C, H, W) — per-target-frame mean field
            mu = self.mean_field_net(
                sensor_values=mean_field_input,
                sensor_times=sensor_timeframes,
                fullstate_times=fullstate_timeframes,
            )  # (B, T_f, C, H, W)
            if mu.shape[-2:] != (out_H, out_W):
                # Merge B,T_f for interpolate, then split back
                mu = F.interpolate(
                    mu.flatten(0, 1), size=(out_H, out_W), mode='bilinear', align_corners=False
                ).reshape(batch_size, n_fullstate_timeframes, self.n_channels, out_H, out_W)
            output = output + mu  # (B, T_f, C, H, W)
        else:
            # Fallback: global scalar bias (FLRONetMLP)
            output = output + self.bias

        return output

    def freeze_branchnets(self):
        for branch_net in self.branch_nets:
            for param in branch_net.parameters():
                param.requires_grad = False

    def freeze_trunknets(self):
        for trunk_net in self.trunk_nets:
            for param in trunk_net.parameters():
                param.requires_grad = False

    def freeze_bias(self):
        self.bias.requires_grad = False


class FLRONetFNO(_BaseFLRONet):

    def __init__(
        self,
        n_channels: int, n_fno_layers: int, n_hmodes: int, n_wmodes: int, 
        embedding_dim: int, n_stacked_networks: int, resolution: Tuple[int, int] = (48, 128),
        is_TC: bool = True, is_cross_attn: bool = False, use_mean_field: str = 'operator',
        mean_field_hidden: int = 32, mean_field_time_embed_dim: int = 32,
    ):
        super().__init__(
            n_channels=n_channels,
            embedding_dim=embedding_dim,
            n_stacked_networks=n_stacked_networks,
            is_cross_attn=is_cross_attn,
            use_mean_field=use_mean_field,
            mean_field_hidden=mean_field_hidden,
            mean_field_time_embed_dim=mean_field_time_embed_dim,
        )
        self.n_fno_layers: int = n_fno_layers
        self.n_hmodes: int = n_hmodes
        self.n_wmodes: int = n_wmodes
        self.resolution = resolution
        self.is_TC = is_TC

        self.branch_nets = nn.ModuleList(
            modules=[
                FNOBranchNet(
                    n_channels=n_channels, n_fno_layers=n_fno_layers, n_hmodes=n_hmodes, n_wmodes=n_wmodes, 
                    embedding_dim=embedding_dim, resolution=resolution, is_afno=False, is_TC=is_TC,
                )
                for _ in range(n_stacked_networks)
            ]
        )

        # Override parent's default MeanFieldNet with FNO-backbone variant
        if self.use_mean_field in ('operator', 'branch_output'):
            self.mean_field_net = FNOMeanFieldNet(
                n_channels=n_channels,
                n_fno_layers=n_fno_layers,
                n_hmodes=n_hmodes,
                n_wmodes=n_wmodes,
                embedding_dim=embedding_dim,
                time_embedding_dim=mean_field_time_embed_dim,
            )
        elif self.use_mean_field == 'voronoi_output':
            self.mean_field_net = VoronoiMeanFieldNet(
                n_channels=n_channels,
                time_embedding_dim=mean_field_time_embed_dim,
            )


class FLRONetAFNO(_BaseFLRONet):

    def __init__(
        self,
        n_channels: int,
        n_fno_layers: int,
        embedding_dim: int,
        n_stacked_networks: int,
        resolution: Tuple[int, int] = (48, 128),
        is_cross_attn: bool = False,
        use_mean_field: str = 'operator',
        mean_field_hidden: int = 32,
        mean_field_time_embed_dim: int = 32,
    ):
        super().__init__(
            n_channels=n_channels,
            embedding_dim=embedding_dim,
            n_stacked_networks=n_stacked_networks,
            is_cross_attn=is_cross_attn,
            use_mean_field=use_mean_field,
            mean_field_hidden=mean_field_hidden,
            mean_field_time_embed_dim=mean_field_time_embed_dim,
        )
        self.n_fno_layers: int = n_fno_layers
        self.resolution = resolution

        self.branch_nets = nn.ModuleList(
            modules=[
                FNOBranchNet(
                    n_channels=n_channels,
                    n_fno_layers=n_fno_layers,
                    n_hmodes=1,
                    n_wmodes=1,
                    embedding_dim=embedding_dim,
                    resolution=resolution,
                    is_afno=True,
                )
                for _ in range(n_stacked_networks)
            ]
        )

        # Override parent's default MeanFieldNet with AFNO-backbone variant
        if self.use_mean_field in ('operator', 'branch_output'):
            self.mean_field_net = AFNOMeanFieldNet(
                n_channels=n_channels,
                n_fno_layers=n_fno_layers,
                resolution=resolution,
                time_embedding_dim=mean_field_time_embed_dim,
            )
        elif self.use_mean_field == 'voronoi_output':
            self.mean_field_net = VoronoiMeanFieldNet(
                n_channels=n_channels,
                time_embedding_dim=mean_field_time_embed_dim,
            )


class FLRONetMLP(_BaseFLRONet):

    def __init__(
        self,
        n_channels: int,
        embedding_dim: int,
        n_sensors: int,
        resolution: int,
        n_stacked_networks: int,
        is_cross_attn: bool = False,
    ):
        # FLRONetMLP uses point sensors (ndim=4), so MeanFieldNet (which needs H/W) is disabled.
        super().__init__(
            n_channels=n_channels,
            embedding_dim=embedding_dim,
            n_stacked_networks=n_stacked_networks,
            is_cross_attn=is_cross_attn,
            use_mean_field=False,   # point-sensor inputs have no spatial H/W grid
        )
        self.n_sensors: int = n_sensors
        self.resolution: int = resolution

        self.branch_nets = nn.ModuleList(
            modules=[
                MLPBranchNet(n_channels=n_channels, embedding_dim=embedding_dim, n_sensors=n_sensors, resolution=resolution)
                for _ in range(n_stacked_networks)
            ]
        )


class FLRONetUNet(_BaseFLRONet):

    def __init__(
        self,
        n_channels: int,
        embedding_dim: int,
        n_stacked_networks: int,
        is_cross_attn: bool = False,
        use_mean_field: str = 'operator',
        mean_field_hidden: int = 32,
        mean_field_time_embed_dim: int = 32,
    ):
        super().__init__(
            n_channels=n_channels,
            embedding_dim=embedding_dim,
            n_stacked_networks=n_stacked_networks,
            is_cross_attn=is_cross_attn,
            use_mean_field=use_mean_field,
            mean_field_hidden=mean_field_hidden,
            mean_field_time_embed_dim=mean_field_time_embed_dim,
        )
        # FLRONetUNet uses the CNN MeanFieldNet from parent; VoronoiMeanFieldNet already
        # set by parent for 'voronoi_output' mode — no override needed here.

        self.branch_nets = nn.ModuleList(
            modules=[
                UNetBranchNet(n_channels=n_channels, embedding_dim=embedding_dim)
                for _ in range(n_stacked_networks)
            ]
        )


class FLRONetTransolver(_BaseFLRONet):

    def __init__(
        self,
        n_channels: int, n_layers: int, n_hidden: int, n_head: int,
        embedding_dim: int, n_stacked_networks: int, resolution: Tuple[int, int] = (48, 128), n_timeframes: int = 5,
        slice_num: int = 32, dropout: float = 0.0, is_cross_attn: bool = False,
        use_mean_field: str = 'operator', mean_field_hidden: int = 32, mean_field_time_embed_dim: int = 32,
    ):
        super().__init__(
            n_channels=n_channels,
            embedding_dim=embedding_dim,
            n_stacked_networks=n_stacked_networks,
            is_cross_attn=is_cross_attn,
            use_mean_field=use_mean_field,
            mean_field_hidden=mean_field_hidden,
            mean_field_time_embed_dim=mean_field_time_embed_dim,
        )
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_head = n_head
        self.resolution = resolution
        self.n_timeframes = n_timeframes
        self.slice_num = slice_num
        self.dropout = dropout

        self.branch_nets = nn.ModuleList(
            modules=[
                TransolverBranchNet(
                    n_channels=n_channels, n_layers=n_layers, n_hidden=n_hidden, n_head=n_head,
                    resolution=resolution, n_timeframes=n_timeframes, slice_num=slice_num, out_dim=n_channels, dropout=dropout
                )
                for _ in range(n_stacked_networks)
            ]
        )


class UNet(nn.Module):

    def __init__(
        self,
        n_channels: int,
        embedding_dim: int,
        n_timeframes: int = 5,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.embedding_dim = embedding_dim
        self.n_timeframes = n_timeframes
        self.branch_net = UNetBranchNet(n_channels=n_channels, embedding_dim=embedding_dim)

    def forward(
        self,
        sensor_timeframes: torch.Tensor,
        sensor_values: torch.Tensor,
        fullstate_timeframes: torch.Tensor,
        out_resolution: Tuple[int, int] | None = None,
    ) -> torch.Tensor:
        batch_size, n_sensor_timeframes, n_channels, in_H, in_W = sensor_values.shape
        assert n_sensor_timeframes == self.n_timeframes
        assert n_channels == self.n_channels

        reconstructed_frames = self.branch_net(sensor_values)

        if out_resolution is not None and out_resolution != (in_H, in_W):
            reconstructed_frames = F.interpolate(
                input=reconstructed_frames.flatten(0, 1),
                size=out_resolution,
                mode='bilinear',
                align_corners=False,
            ).reshape(batch_size, n_sensor_timeframes, n_channels, *out_resolution)
            out_H, out_W = out_resolution
        else:
            out_H, out_W = in_H, in_W

        n_fullstate_timeframes = fullstate_timeframes.shape[1]
        output = torch.zeros(
            batch_size, n_fullstate_timeframes, n_channels, out_H, out_W,
            device=sensor_values.device
        )

        for b in range(batch_size):
            times = sensor_timeframes[b]
            for t_idx in range(n_fullstate_timeframes):
                target_t = fullstate_timeframes[b, t_idx]
                idx = torch.searchsorted(times, target_t)
                if idx == 0:
                    output[b, t_idx] = reconstructed_frames[b, 0]
                elif idx == len(times):
                    t_prev, t_next = times[-2], times[-1]
                    w_next = (target_t - t_prev) / (t_next - t_prev)
                    output[b, t_idx] = (1.0 - w_next) * reconstructed_frames[b, -2] + w_next * reconstructed_frames[b, -1]
                else:
                    t_prev, t_next = times[idx - 1], times[idx]
                    w_next = (target_t - t_prev) / (t_next - t_prev)
                    output[b, t_idx] = (1.0 - w_next) * reconstructed_frames[b, idx - 1] + w_next * reconstructed_frames[b, idx]

        return output

class FNO(nn.Module):
    def __init__(
        self,
        n_channels: int, n_fno_layers: int, n_hmodes: int, n_wmodes: int, 
        embedding_dim: int, n_timeframes: int = 5,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_fno_layers = n_fno_layers
        self.n_hmodes = n_hmodes
        self.n_wmodes = n_wmodes
        self.embedding_dim = embedding_dim
        self.n_timeframes = n_timeframes

        in_chans_total = n_timeframes * n_channels

        self.embedding_layer = nn.Sequential(
            nn.Linear(in_features=in_chans_total, out_features=128),
            nn.GELU(),
            nn.Linear(in_features=128, out_features=256),
            nn.GELU(),
            nn.Linear(in_features=256, out_features=embedding_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=128),
            nn.GELU(),
            nn.Linear(in_features=128, out_features=128),
            nn.GELU(),
            nn.Linear(in_features=128, out_features=in_chans_total),
        ) 
        self.spectral_conv_layers = nn.ModuleList(
            modules=[SpectralConv2d(embedding_dim=embedding_dim, n_hmodes=n_hmodes, n_wmodes=n_wmodes) for _ in range(n_fno_layers)]
        )
        self.Ws = nn.ModuleList([
            nn.Conv2d(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=1)
            for _ in range(n_fno_layers)
        ])

    def forward(
        self,
        sensor_timeframes: torch.Tensor,
        sensor_values: torch.Tensor,
        fullstate_timeframes: torch.Tensor,
        out_resolution: Tuple[int, int] | None = None,
    ) -> torch.Tensor:
        # sensor_values: (B, T_in, C, H, W)
        batch_size, n_sensor_timeframes, n_channels, in_H, in_W = sensor_values.shape
        assert n_sensor_timeframes == self.n_timeframes
        assert n_channels == self.n_channels
        
        # Merge T and C (TC Merge), keep B unchanged
        x = sensor_values.flatten(start_dim=1, end_dim=2) # (B, T_in*C, H, W)
        
        # Encoder
        x = self.embedding_layer(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) # (B, embedding_dim, H, W)
        
        # Spectral layers
        for i in range(self.n_fno_layers):
            x1 = self.spectral_conv_layers[i](x)
            x2 = self.Ws[i](x)
            x = x1 + x2
            if i < self.n_fno_layers - 1:
                x = F.gelu(x)
        
        # Decoder
        x = self.decoder(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) # (B, T_in*C, H, W)
        
        # Reshape back to (B, T_in, C, H, W)
        reconstructed_frames = x.reshape(batch_size, n_sensor_timeframes, n_channels, in_H, in_W)

        # Interpolate if output resolution is different
        if out_resolution is not None and out_resolution != (in_H, in_W):
            reconstructed_frames = F.interpolate(
                input=reconstructed_frames.flatten(0, 1), 
                size=out_resolution, 
                mode='bilinear', 
                align_corners=False
            ).reshape(batch_size, n_sensor_timeframes, n_channels, *out_resolution)
            out_H, out_W = out_resolution
        else:
            out_H, out_W = in_H, in_W

        # Target Frame Interpolation
        n_fullstate_timeframes = fullstate_timeframes.shape[1]
        output = torch.zeros(
            batch_size, n_fullstate_timeframes, n_channels, out_H, out_W, 
            device=sensor_values.device
        )
        
        # sensor_timeframes: (B, T_in)
        # fullstate_timeframes: (B, T_out)
        for b in range(batch_size):
            times = sensor_timeframes[b]
            for t_idx in range(n_fullstate_timeframes):
                target_t = fullstate_timeframes[b, t_idx]
                
                # Find adjacent sensor frames
                idx = torch.searchsorted(times, target_t)
                
                if idx == 0:
                    t_prev, t_next = times[0], times[1]
                    w_next = (target_t - t_prev) / (t_next - t_prev)
                    output[b, t_idx] = (1.0 - w_next) * reconstructed_frames[b, 0] + w_next * reconstructed_frames[b, 1]
                elif idx == len(times):
                    t_prev, t_next = times[-2], times[-1]
                    w_next = (target_t - t_prev) / (t_next - t_prev)
                    output[b, t_idx] = (1.0 - w_next) * reconstructed_frames[b, -2] + w_next * reconstructed_frames[b, -1]
                else:
                    t_prev, t_next = times[idx-1], times[idx]
                    weight_next = (target_t - t_prev) / (t_next - t_prev)
                    weight_prev = 1.0 - weight_next
                    output[b, t_idx] = weight_prev * reconstructed_frames[b, idx-1] + weight_next * reconstructed_frames[b, idx]
        
        return output

class AFNO(nn.Module):
    def __init__(
        self,
        n_channels: int, n_fno_layers: int, embedding_dim: int, 
        resolution: Tuple[int, int] = (48, 128), n_timeframes: int = 5
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_fno_layers = n_fno_layers
        self.embedding_dim = embedding_dim
        self.resolution = resolution
        self.n_timeframes = n_timeframes

        # Determine patch size
        patch_size = (8, 8) if resolution[0] % 8 == 0 and resolution[1] % 8 == 0 else (resolution[0], resolution[1])

        # We use a large embed_dim for AFNOLayers (like in FNOBranchNet)
        layer_embed_dim = 768
        in_chans_total = n_timeframes * n_channels
        
        self.patch_embed = PatchEmbed(img_size=resolution, patch_size=patch_size, in_chans=in_chans_total, embed_dim=layer_embed_dim)
        self.num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, layer_embed_dim))
        
        self.spectral_conv_layers = nn.ModuleList(
            modules=[AFNOLayer(embedding_dim=layer_embed_dim, img_size=resolution, patch_size=patch_size) for _ in range(n_fno_layers)]
        )
        
        self.de_patch_embed = DePatchEmbed(img_size=resolution, patch_size=patch_size, out_chans=in_chans_total, embed_dim=layer_embed_dim)

    def forward(
        self,
        sensor_timeframes: torch.Tensor,
        sensor_values: torch.Tensor,
        fullstate_timeframes: torch.Tensor,
        out_resolution: Tuple[int, int] | None = None,
    ) -> torch.Tensor:
        batch_size, n_sensor_timeframes, n_channels, in_H, in_W = sensor_values.shape
        assert n_sensor_timeframes == self.n_timeframes
        
        # Merge T and C (TC Merge)
        x = sensor_values.flatten(start_dim=1, end_dim=2) # (B, T_in*C, H, W)
        
        # Interpolate if input resolution is different from model resolution
        if (in_H, in_W) != self.resolution:
            x = F.interpolate(input=x, size=self.resolution, mode='bilinear', align_corners=False)
            curr_H, curr_W = self.resolution
        else:
            curr_H, curr_W = in_H, in_W

        # AFNO forward
        x = self.patch_embed(x)
        x = x + self.pos_embed
        for layer in self.spectral_conv_layers:
            x = layer(x)
        x = self.de_patch_embed(x) # (B, T*C, H, W)

        # Reshape back to (B, T, C, H, W)
        reconstructed_frames = x.reshape(batch_size, n_sensor_timeframes, n_channels, curr_H, curr_W)

        # Interpolate if output resolution is different
        if out_resolution is not None and out_resolution != (curr_H, curr_W):
            reconstructed_frames = F.interpolate(
                input=reconstructed_frames.flatten(0, 1), 
                size=out_resolution, 
                mode='bilinear', 
                align_corners=False
            ).reshape(batch_size, n_sensor_timeframes, n_channels, *out_resolution)
            out_H, out_W = out_resolution
        else:
            out_H, out_W = curr_H, curr_W

        # Target Frame Interpolation
        n_fullstate_timeframes = fullstate_timeframes.shape[1]
        output = torch.zeros(batch_size, n_fullstate_timeframes, n_channels, out_H, out_W, device=sensor_values.device)
        
        for b in range(batch_size):
            times = sensor_timeframes[b]
            for t_idx in range(n_fullstate_timeframes):
                target_t = fullstate_timeframes[b, t_idx]
                idx = torch.searchsorted(times, target_t)
                if idx == 0:
                    t_prev, t_next = times[0], times[1]
                    w_next = (target_t - t_prev) / (t_next - t_prev)
                    output[b, t_idx] = (1.0 - w_next) * reconstructed_frames[b, 0] + w_next * reconstructed_frames[b, 1]
                elif idx == len(times):
                    t_prev, t_next = times[-2], times[-1]
                    w_next = (target_t - t_prev) / (t_next - t_prev)
                    output[b, t_idx] = (1.0 - w_next) * reconstructed_frames[b, -2] + w_next * reconstructed_frames[b, -1]
                else:
                    t_prev, t_next = times[idx-1], times[idx]
                    w_next = (target_t - t_prev) / (t_next - t_prev)
                    output[b, t_idx] = (1.0 - w_next) * reconstructed_frames[b, idx-1] + w_next * reconstructed_frames[b, idx]
        
        return output

class Transolver(nn.Module):
    def __init__(
        self, 
        n_channels: int, n_layers: int, n_hidden: int, n_head: int, 
        resolution: Tuple[int, int], n_timeframes: int = 5,
        slice_num: int = 32, dropout: float = 0.0,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_head = n_head
        self.resolution = resolution
        self.H, self.W = resolution
        self.n_timeframes = n_timeframes
        self.slice_num = slice_num
        self.dropout = dropout
        
        grid_x = torch.linspace(0, 1, self.H)
        grid_y = torch.linspace(0, 1, self.W)
        grid_x, grid_y = torch.meshgrid(grid_x, grid_y, indexing='ij')
        self.register_buffer('grid', torch.stack([grid_x, grid_y], dim=-1).reshape(1, self.H * self.W, 2))

        in_chans_total = n_timeframes * n_channels

        self.model = TransolverModel(
            space_dim=2, n_layers=n_layers, n_hidden=n_hidden, n_head=n_head,
            fun_dim=in_chans_total, out_dim=in_chans_total, slice_num=slice_num,
            H=self.H, W=self.W, dropout=dropout, unified_pos=False
        )

    def forward(
        self,
        sensor_timeframes: torch.Tensor,
        sensor_values: torch.Tensor,
        fullstate_timeframes: torch.Tensor,
        out_resolution: Tuple[int, int] | None = None,
    ) -> torch.Tensor:
        batch_size, n_sensor_timeframes, n_channels, in_H, in_W = sensor_values.shape
        assert n_sensor_timeframes == self.n_timeframes
        
        # Merge T and C (TC Merge)
        v = sensor_values.flatten(start_dim=1, end_dim=2) # (B, T*C, H, W)
        
        if (in_H, in_W) != self.resolution:
            v = F.interpolate(input=v, size=self.resolution, mode='bilinear', align_corners=False)
            curr_H, curr_W = self.resolution
        else:
            curr_H, curr_W = in_H, in_W
            
        fx = v.permute(0, 2, 3, 1).reshape(batch_size, curr_H * curr_W, -1) # (B, N, T*C)
        x_coord = self.grid.repeat(batch_size, 1, 1)
        
        output = self.model(x_coord, fx) # (B, N, T*C)
        output = output.reshape(batch_size, curr_H, curr_W, -1).permute(0, 3, 1, 2) # (B, T*C, H, W)
        
        reconstructed_frames = output.reshape(batch_size, n_sensor_timeframes, n_channels, curr_H, curr_W)

        if out_resolution is not None and out_resolution != (curr_H, curr_W):
            reconstructed_frames = F.interpolate(
                input=reconstructed_frames.flatten(0, 1), 
                size=out_resolution, mode='bilinear', align_corners=False
            ).reshape(batch_size, n_sensor_timeframes, n_channels, *out_resolution)
            out_H, out_W = out_resolution
        else:
            out_H, out_W = curr_H, curr_W

        # Target Frame Interpolation
        n_fullstate_timeframes = fullstate_timeframes.shape[1]
        final_output = torch.zeros(batch_size, n_fullstate_timeframes, n_channels, out_H, out_W, device=sensor_values.device)
        
        for b in range(batch_size):
            times = sensor_timeframes[b]
            for t_idx in range(n_fullstate_timeframes):
                target_t = fullstate_timeframes[b, t_idx]
                idx = torch.searchsorted(times, target_t)
                if idx == 0:
                    t_prev, t_next = times[0], times[1]
                    w_next = (target_t - t_prev) / (t_next - t_prev)
                    final_output[b, t_idx] = (1.0 - w_next) * reconstructed_frames[b, 0] + w_next * reconstructed_frames[b, 1]
                elif idx == len(times):
                    t_prev, t_next = times[-2], times[-1]
                    w_next = (target_t - t_prev) / (t_next - t_prev)
                    final_output[b, t_idx] = (1.0 - w_next) * reconstructed_frames[b, -2] + w_next * reconstructed_frames[b, -1]
                else:
                    t_prev, t_next = times[idx-1], times[idx]
                    w_next = (target_t - t_prev) / (t_next - t_prev)
                    final_output[b, t_idx] = (1.0 - w_next) * reconstructed_frames[b, idx-1] + w_next * reconstructed_frames[b, idx]
        
        return final_output
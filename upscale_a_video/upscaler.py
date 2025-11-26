"""
Main upscaler class for Upscale-A-Video.
"""

import os
import math
import warnings
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from PIL import Image

# Suppress warnings during import
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)

from models_video.RAFT.raft_bi import RAFT_bi
from models_video.propagation_module import Propagation
from models_video.autoencoder_kl_cond_video import AutoencoderKLVideo
from models_video.unet_video import UNetVideoModel
from models_video.pipeline_upscale_a_video import VideoUpscalePipeline
from models_video.scheduling_ddim import DDIMScheduler
from models_video.color_correction import wavelet_reconstruction, adaptive_instance_normalization


class UpscaleAVideo:
    """
    Upscale-A-Video: Temporal-Consistent Diffusion Model for Real-World Video Super-Resolution.
    
    This class provides a simple interface to upscale video frames using the Upscale-A-Video model.
    
    Args:
        pretrained_path: Path to the pretrained model directory containing unet, vae, scheduler, etc.
        device: Device to run the model on (e.g., "cuda:0", "cuda:1", "cpu").
        use_video_vae: Whether to use the video VAE (default: False uses 3D VAE).
        use_propagation: Whether to enable optical flow propagation for temporal consistency.
        
    Example:
        >>> upscaler = UpscaleAVideo("./pretrained_models/upscale_a_video", device="cuda:0")
        >>> upscaled = upscaler.upscale_frames(frames, noise_level=120)
    """
    
    def __init__(
        self,
        pretrained_path: str = "./pretrained_models/upscale_a_video",
        device: str = "cuda:0",
        use_video_vae: bool = False,
        use_propagation: bool = False,
    ):
        self.device = device
        self.pretrained_path = Path(pretrained_path)
        self.use_video_vae = use_video_vae
        self.use_propagation = use_propagation
        
        self._pipeline = None
        self._raft = None
        self._loaded = False
        
    def load_models(self) -> None:
        """Load all models into memory. Called automatically on first use if not called explicitly."""
        if self._loaded:
            return
            
        pretrained_path = self.pretrained_path
        
        # Load the pipeline (includes text_encoder, tokenizer, low_res_scheduler)
        self._pipeline = VideoUpscalePipeline.from_pretrained(
            str(pretrained_path), 
            torch_dtype=torch.float16
        )
        
        # Load VAE
        if self.use_video_vae:
            vae_config = pretrained_path / "vae" / "vae_video_config.json"
            vae_weights = pretrained_path / "vae" / "vae_video.bin"
            self._pipeline.vae = AutoencoderKLVideo.from_config(str(vae_config))
        else:
            vae_config = pretrained_path / "vae" / "vae_3d_config.json"
            vae_weights = pretrained_path / "vae" / "vae_3d.bin"
            self._pipeline.vae = AutoencoderKLVideo.from_config(str(vae_config))
        
        self._pipeline.vae.load_state_dict(
            torch.load(str(vae_weights), map_location="cpu")
        )
        
        # Load UNet
        unet_config = pretrained_path / "unet" / "unet_video_config.json"
        unet_weights = pretrained_path / "unet" / "unet_video.bin"
        self._pipeline.unet = UNetVideoModel.from_config(str(unet_config))
        self._pipeline.unet.load_state_dict(
            torch.load(str(unet_weights), map_location="cpu"),
            strict=True
        )
        self._pipeline.unet = self._pipeline.unet.half()
        self._pipeline.unet.eval()
        
        # Load scheduler
        scheduler_config = pretrained_path / "scheduler" / "scheduler_config.json"
        self._pipeline.scheduler = DDIMScheduler.from_config(str(scheduler_config))
        
        # Load propagator (RAFT) if needed
        if self.use_propagation:
            raft_weights = pretrained_path / "propagator" / "raft-things.pth"
            self._raft = RAFT_bi(str(raft_weights))
            propagator = Propagation(4, learnable=False)
            self._pipeline.propagator = propagator
        else:
            self._pipeline.propagator = None
        
        # Move to device
        self._pipeline = self._pipeline.to(self.device)
        
        self._loaded = True
        
    def unload_models(self) -> None:
        """Unload models from memory to free up GPU resources."""
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
        if self._raft is not None:
            del self._raft
            self._raft = None
        self._loaded = False
        torch.cuda.empty_cache()
        
    def _preprocess_frames(
        self, 
        frames: Union[List[np.ndarray], List[Image.Image], np.ndarray, torch.Tensor]
    ) -> torch.Tensor:
        """
        Convert input frames to the format expected by the model.
        
        Args:
            frames: Input frames in various formats:
                - List of numpy arrays (H, W, C) in RGB, uint8 [0-255]
                - List of PIL Images
                - Numpy array (T, H, W, C) in RGB, uint8 [0-255]
                - Torch tensor (T, C, H, W) in [0, 1] or [-1, 1]
                
        Returns:
            Torch tensor of shape (1, C, T, H, W) in range [-1, 1]
        """
        if isinstance(frames, torch.Tensor):
            # Assume (T, C, H, W) format
            if frames.dim() == 4:
                vframes = frames
            else:
                raise ValueError(f"Expected 4D tensor (T, C, H, W), got {frames.dim()}D")
            
            # Normalize to [-1, 1] if needed
            if vframes.max() > 1.0:
                vframes = vframes / 255.0
            if vframes.min() >= 0:  # [0, 1] range
                vframes = (vframes - 0.5) * 2  # Convert to [-1, 1]
                
        elif isinstance(frames, np.ndarray):
            # Assume (T, H, W, C) format, uint8
            if frames.ndim == 4:
                vframes = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
                vframes = (vframes / 255.0 - 0.5) * 2  # Convert to [-1, 1]
            else:
                raise ValueError(f"Expected 4D numpy array (T, H, W, C), got {frames.ndim}D")
                
        elif isinstance(frames, list):
            if isinstance(frames[0], np.ndarray):
                # List of (H, W, C) numpy arrays
                stacked = np.stack(frames, axis=0)  # (T, H, W, C)
                vframes = torch.from_numpy(stacked).permute(0, 3, 1, 2).float()
                vframes = (vframes / 255.0 - 0.5) * 2
            elif isinstance(frames[0], Image.Image):
                # List of PIL Images
                arrays = [np.array(img) for img in frames]
                stacked = np.stack(arrays, axis=0)
                vframes = torch.from_numpy(stacked).permute(0, 3, 1, 2).float()
                vframes = (vframes / 255.0 - 0.5) * 2
            elif isinstance(frames[0], torch.Tensor):
                # List of tensors
                vframes = torch.stack(frames, dim=0)
                if vframes.max() > 1.0:
                    vframes = vframes / 255.0
                if vframes.min() >= 0:
                    vframes = (vframes - 0.5) * 2
            else:
                raise ValueError(f"Unsupported frame type: {type(frames[0])}")
        else:
            raise ValueError(f"Unsupported frames type: {type(frames)}")
        
        # Move to device
        vframes = vframes.to(self.device)
        
        # Reshape to (1, C, T, H, W)
        vframes = vframes.unsqueeze(0)  # (1, T, C, H, W)
        vframes = rearrange(vframes, 'b t c h w -> b c t h w').contiguous()
        
        return vframes
    
    def _postprocess_frames(
        self, 
        output: torch.Tensor,
        output_format: str = "numpy"
    ) -> Union[List[np.ndarray], List[Image.Image], np.ndarray, torch.Tensor]:
        """
        Convert model output to the requested format.
        
        Args:
            output: Model output tensor (T, C, H, W) in range [-1, 1]
            output_format: One of "numpy", "pil", "torch", "numpy_list", "pil_list"
            
        Returns:
            Frames in the requested format
        """
        # Convert from [-1, 1] to [0, 1]
        output = (output / 2 + 0.5).clamp(0, 1)
        
        if output_format == "torch":
            return output  # (T, C, H, W) in [0, 1]
            
        # Convert to numpy (T, H, W, C) uint8
        output_np = output.permute(0, 2, 3, 1).cpu().numpy()
        output_np = (output_np * 255).astype(np.uint8)
        
        if output_format == "numpy":
            return output_np
        elif output_format == "numpy_list":
            return [output_np[i] for i in range(output_np.shape[0])]
        elif output_format == "pil":
            return [Image.fromarray(output_np[i]) for i in range(output_np.shape[0])]
        elif output_format == "pil_list":
            return [Image.fromarray(output_np[i]) for i in range(output_np.shape[0])]
        else:
            raise ValueError(f"Unknown output_format: {output_format}")
    
    def upscale_frames(
        self,
        frames: Union[List[np.ndarray], List[Image.Image], np.ndarray, torch.Tensor],
        prompt: str = "",
        noise_level: int = 120,
        guidance_scale: float = 6.0,
        inference_steps: int = 30,
        propagation_steps: Optional[List[int]] = None,
        positive_prompt: str = "best quality, extremely detailed",
        negative_prompt: str = "blur, worst quality",
        color_fix: str = "None",
        tile_size: int = 256,
        seed: int = 10,
        output_format: str = "numpy",
    ) -> Union[List[np.ndarray], List[Image.Image], np.ndarray, torch.Tensor]:
        """
        Upscale video frames by 4x.
        
        Args:
            frames: Input frames. Accepts:
                - List of numpy arrays (H, W, C) in RGB, uint8 [0-255]
                - List of PIL Images
                - Numpy array (T, H, W, C) in RGB, uint8 [0-255]
                - Torch tensor (T, C, H, W)
            prompt: Text prompt to guide upscaling (optional, can be auto-generated with LLaVA)
            noise_level: Noise level [0-200]. Higher = better quality but lower fidelity. Default: 120
            guidance_scale: Classifier-free guidance scale. Higher = more details. Default: 6.0
            inference_steps: Number of denoising steps. More = higher quality. Default: 30
            propagation_steps: Steps at which to apply temporal propagation (e.g., [24, 26, 28])
            positive_prompt: Positive prompt suffix. Default: "best quality, extremely detailed"
            negative_prompt: Negative prompt. Default: "blur, worst quality"
            color_fix: Color correction method: "None", "AdaIn", or "Wavelet"
            tile_size: Tile size for processing large frames. Default: 256
            seed: Random seed for reproducibility. Default: 10
            output_format: Output format: "numpy", "numpy_list", "pil", "pil_list", "torch"
            
        Returns:
            Upscaled frames in the requested format (4x the input resolution)
        """
        # Ensure models are loaded
        self.load_models()
        
        # Preprocess frames
        vframes = self._preprocess_frames(frames)
        
        h, w = vframes.shape[-2:]
        
        # Downscale very large inputs
        if h >= 1280 and w >= 1280:
            vframes = F.interpolate(
                vframes.view(-1, vframes.shape[2], h, w),
                (int(h // 4), int(w // 4)),
                mode='area'
            )
            vframes = vframes.view(1, -1, vframes.shape[1], vframes.shape[2], vframes.shape[3])
            vframes = rearrange(vframes, 'b t c h w -> b c t h w')
            h, w = vframes.shape[-2:]
        
        # Calculate optical flow if propagation is enabled
        flows_bi = None
        if self.use_propagation and self._raft is not None and propagation_steps:
            flows_forward, flows_backward = self._raft.forward_slicing(vframes)
            flows_bi = [flows_forward, flows_backward]
        
        b, c, t, h, w = vframes.shape
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Build full prompt
        full_prompt = prompt + " " + positive_prompt if prompt else positive_prompt
        
        # Determine if tiling is needed
        perform_tile = h * w >= 384 * 384
        
        if perform_tile:
            output = self._upscale_with_tiles(
                vframes, flows_bi, full_prompt, negative_prompt,
                noise_level, guidance_scale, inference_steps,
                propagation_steps or [], tile_size, generator
            )
        else:
            with torch.no_grad():
                output = self._pipeline(
                    full_prompt,
                    image=vframes,
                    flows_bi=flows_bi,
                    generator=generator,
                    num_inference_steps=inference_steps,
                    guidance_scale=guidance_scale,
                    noise_level=noise_level,
                    negative_prompt=negative_prompt,
                    propagation_steps=propagation_steps or [],
                ).images
        
        # Apply color correction if needed
        if color_fix in ['AdaIn', 'Wavelet']:
            vframes_orig = rearrange(vframes.squeeze(0), 'c t h w -> t c h w').contiguous()
            output = rearrange(output.squeeze(0), 'c t h w -> t c h w').contiguous()
            vframes_orig = F.interpolate(vframes_orig, scale_factor=4, mode='bicubic')
            if color_fix == 'AdaIn':
                output = adaptive_instance_normalization(output, vframes_orig)
            elif color_fix == 'Wavelet':
                output = wavelet_reconstruction(output, vframes_orig)
        else:
            output = rearrange(output.squeeze(0), 'c t h w -> t c h w').contiguous()
        
        output = output.cpu()
        
        return self._postprocess_frames(output, output_format)
    
    def _upscale_with_tiles(
        self,
        vframes: torch.Tensor,
        flows_bi: Optional[list],
        prompt: str,
        negative_prompt: str,
        noise_level: int,
        guidance_scale: float,
        inference_steps: int,
        propagation_steps: List[int],
        tile_size: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """Process large frames using tiling."""
        b, c, t, h, w = vframes.shape
        tile_height = tile_width = tile_size
        tile_overlap_height = tile_overlap_width = 64
        
        output_h = h * 4
        output_w = w * 4
        output_shape = (b, c, t, output_h, output_w)
        output = vframes.new_zeros(output_shape)
        
        tiles_x = math.ceil(w / tile_width)
        tiles_y = math.ceil(h / tile_height)
        
        rm_end_pad_w, rm_end_pad_h = True, True
        if (tiles_x - 1) * tile_width + tile_overlap_width >= w:
            tiles_x = tiles_x - 1
            rm_end_pad_w = False
        if (tiles_y - 1) * tile_height + tile_overlap_height >= h:
            tiles_y = tiles_y - 1
            rm_end_pad_h = False
        
        for y in range(tiles_y):
            for x in range(tiles_x):
                ofs_x = x * tile_width
                ofs_y = y * tile_height
                
                input_start_x = ofs_x
                input_end_x = min(ofs_x + tile_width, w)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + tile_height, h)
                
                input_start_x_pad = max(input_start_x - tile_overlap_width, 0)
                input_end_x_pad = min(input_end_x + tile_overlap_width, w)
                input_start_y_pad = max(input_start_y - tile_overlap_height, 0)
                input_end_y_pad = min(input_end_y + tile_overlap_height, h)
                
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                
                input_tile = vframes[:, :, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]
                
                flows_bi_tile = None
                if flows_bi is not None:
                    flows_bi_tile = [
                        flows_bi[0][:, :, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad],
                        flows_bi[1][:, :, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]
                    ]
                
                with torch.no_grad():
                    output_tile = self._pipeline(
                        prompt,
                        image=input_tile,
                        flows_bi=flows_bi_tile,
                        generator=generator,
                        num_inference_steps=inference_steps,
                        guidance_scale=guidance_scale,
                        noise_level=noise_level,
                        negative_prompt=negative_prompt,
                        propagation_steps=propagation_steps,
                    ).images
                
                output_start_x = input_start_x * 4
                output_end_x = output_w if (x == tiles_x - 1 and not rm_end_pad_w) else input_end_x * 4
                output_start_y = input_start_y * 4
                output_end_y = output_h if (y == tiles_y - 1 and not rm_end_pad_h) else input_end_y * 4
                
                output_start_x_tile = (input_start_x - input_start_x_pad) * 4
                output_end_x_tile = output_start_x_tile + (output_w - output_start_x if (x == tiles_x - 1 and not rm_end_pad_w) else input_tile_width * 4)
                output_start_y_tile = (input_start_y - input_start_y_pad) * 4
                output_end_y_tile = output_start_y_tile + (output_h - output_start_y if (y == tiles_y - 1 and not rm_end_pad_h) else input_tile_height * 4)
                
                output[:, :, :, output_start_y:output_end_y, output_start_x:output_end_x] = \
                    output_tile[:, :, :, output_start_y_tile:output_end_y_tile, output_start_x_tile:output_end_x_tile]
        
        return output
    
    def __enter__(self):
        """Context manager entry - loads models."""
        self.load_models()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - unloads models."""
        self.unload_models()
        return False

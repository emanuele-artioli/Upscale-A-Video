"""
Upscale-A-Video: Temporal-Consistent Diffusion Model for Real-World Video Super-Resolution

This package provides a simple API to upscale video frames using the Upscale-A-Video model.

Example usage:
    from upscale_a_video import UpscaleAVideo
    
    # Initialize the upscaler
    upscaler = UpscaleAVideo(
        pretrained_path="./pretrained_models/upscale_a_video",
        device="cuda:0"
    )
    
    # Upscale frames (list of numpy arrays or torch tensors)
    upscaled_frames = upscaler.upscale_frames(
        frames,
        noise_level=120,
        guidance_scale=6,
        inference_steps=30
    )
"""

from .upscaler import UpscaleAVideo

__version__ = "0.1.0"
__all__ = ["UpscaleAVideo"]

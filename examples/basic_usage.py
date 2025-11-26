"""
Example usage of the upscale_a_video package.

This script demonstrates how to upscale video frames using the Upscale-A-Video model.
"""

import numpy as np
from PIL import Image
from upscale_a_video import UpscaleAVideo


def example_with_numpy_array():
    """Example using numpy arrays as input."""
    # Create sample frames (replace with your actual frames)
    # Frames should be (T, H, W, C) in RGB, uint8 [0-255]
    frames = np.random.randint(0, 255, (10, 128, 128, 3), dtype=np.uint8)
    
    # Initialize upscaler
    upscaler = UpscaleAVideo(
        pretrained_path="./pretrained_models/upscale_a_video",
        device="cuda:0",
        use_video_vae=False,
        use_propagation=False,
    )
    
    # Upscale frames
    upscaled = upscaler.upscale_frames(
        frames,
        noise_level=120,
        guidance_scale=6.0,
        inference_steps=30,
        output_format="numpy"  # Returns (T, H, W, C) numpy array
    )
    
    print(f"Input shape: {frames.shape}")
    print(f"Output shape: {upscaled.shape}")  # Should be (10, 512, 512, 3)
    
    return upscaled


def example_with_pil_images():
    """Example using PIL Images as input."""
    # Create sample PIL images (replace with your actual images)
    frames = [Image.new('RGB', (128, 128), color='red') for _ in range(10)]
    
    # Initialize upscaler
    upscaler = UpscaleAVideo(
        pretrained_path="./pretrained_models/upscale_a_video",
        device="cuda:0"
    )
    
    # Upscale frames
    upscaled = upscaler.upscale_frames(
        frames,
        noise_level=120,
        guidance_scale=6.0,
        inference_steps=30,
        output_format="pil"  # Returns list of PIL Images
    )
    
    print(f"Input: {len(frames)} PIL images of size {frames[0].size}")
    print(f"Output: {len(upscaled)} PIL images of size {upscaled[0].size}")
    
    return upscaled


def example_with_context_manager():
    """Example using context manager for automatic model cleanup."""
    import torch
    
    # Create sample frames
    frames = torch.randn(10, 3, 128, 128)  # (T, C, H, W)
    
    # Use context manager for automatic cleanup
    with UpscaleAVideo(
        pretrained_path="./pretrained_models/upscale_a_video",
        device="cuda:0"
    ) as upscaler:
        upscaled = upscaler.upscale_frames(
            frames,
            noise_level=120,
            output_format="torch"  # Returns (T, C, H, W) tensor
        )
    
    # Models are automatically unloaded after the with block
    print(f"Input shape: {frames.shape}")
    print(f"Output shape: {upscaled.shape}")
    
    return upscaled


def example_with_propagation():
    """Example using temporal propagation for better consistency."""
    frames = np.random.randint(0, 255, (10, 128, 128, 3), dtype=np.uint8)
    
    # Enable propagation for temporal consistency
    upscaler = UpscaleAVideo(
        pretrained_path="./pretrained_models/upscale_a_video",
        device="cuda:0",
        use_propagation=True,  # Enable RAFT optical flow
    )
    
    upscaled = upscaler.upscale_frames(
        frames,
        noise_level=150,
        guidance_scale=6.0,
        inference_steps=30,
        propagation_steps=[24, 26, 28],  # Steps at which to apply propagation
        output_format="numpy"
    )
    
    return upscaled


def example_with_prompt():
    """Example using a text prompt to guide upscaling."""
    frames = np.random.randint(0, 255, (10, 128, 128, 3), dtype=np.uint8)
    
    upscaler = UpscaleAVideo(
        pretrained_path="./pretrained_models/upscale_a_video",
        device="cuda:0"
    )
    
    # Use a prompt to guide the upscaling (e.g., from LLaVA)
    upscaled = upscaler.upscale_frames(
        frames,
        prompt="A beautiful landscape with mountains and a lake",
        noise_level=120,
        guidance_scale=9.0,  # Higher guidance for stronger prompt following
        inference_steps=30,
        positive_prompt="best quality, extremely detailed, 4k",
        negative_prompt="blur, worst quality, artifacts",
        output_format="numpy"
    )
    
    return upscaled


if __name__ == "__main__":
    # Run the example that best fits your use case
    # example_with_numpy_array()
    # example_with_pil_images()
    # example_with_context_manager()
    # example_with_propagation()
    # example_with_prompt()
    
    print("Examples available:")
    print("  - example_with_numpy_array()")
    print("  - example_with_pil_images()")
    print("  - example_with_context_manager()")
    print("  - example_with_propagation()")
    print("  - example_with_prompt()")

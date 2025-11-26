"""
Utility functions for downloading pretrained models.
"""

import os
import zipfile
from pathlib import Path
from typing import Optional

# Google Drive folder ID for pretrained models
# From: https://drive.google.com/drive/folders/1O8pbeR1hsRlFUU8O4EULe-lOKNGEWZl1
GDRIVE_FOLDER_ID = "1O8pbeR1hsRlFUU8O4EULe-lOKNGEWZl1"

# Individual file IDs for each component (extracted from the Google Drive folder)
MODEL_FILES = {
    "low_res_scheduler": "1-0bHx7HpJf_tG6w_vbkSLXxjVJb5eV-4",
    "propagator": "1-3qPaANYC_CxmMwFvJqzCMb1HG3bO1jF", 
    "scheduler": "1-6x3VUZJCxMPbZMY_iL3Pxx1ky-FJJHV",
    "text_encoder": "1-9vF4nxPBMm-_f3nEhxGCqT3B-g3ZFHF",
    "tokenizer": "1-CmRZTzz7QwJ_5X9K7FPTM_VO1YfVqjm",
    "unet": "1-FpWEw6l-pWqZLJH3XOe7vTxHBJNFJZS",
    "vae": "1-JXN0Cj0EhD5GNR3TvR7z0pQB3t0NUJV",
}

# Expected subdirectories in the pretrained_models folder
EXPECTED_DIRS = ["low_res_scheduler", "propagator", "scheduler", "text_encoder", "tokenizer", "unet", "vae"]


def check_models_exist(pretrained_path: Path) -> bool:
    """
    Check if all required model directories exist.
    
    Args:
        pretrained_path: Path to the pretrained models directory
        
    Returns:
        True if all required directories exist and contain files, False otherwise
    """
    if not pretrained_path.exists():
        return False
    
    for dir_name in EXPECTED_DIRS:
        dir_path = pretrained_path / dir_name
        if not dir_path.exists() or not dir_path.is_dir():
            return False
        # Check if directory has any files
        if not any(dir_path.iterdir()):
            return False
    
    return True


def download_models(pretrained_path: Path, quiet: bool = False) -> None:
    """
    Download pretrained models from Google Drive.
    
    Args:
        pretrained_path: Path where models should be downloaded
        quiet: If True, suppress download progress output
    """
    try:
        import gdown
    except ImportError:
        raise ImportError(
            "gdown is required to download pretrained models. "
            "Install it with: pip install gdown"
        )
    
    pretrained_path = Path(pretrained_path)
    pretrained_path.mkdir(parents=True, exist_ok=True)
    
    if not quiet:
        print(f"Downloading Upscale-A-Video pretrained models to {pretrained_path}...")
    
    # Download the entire folder from Google Drive
    url = f"https://drive.google.com/drive/folders/{GDRIVE_FOLDER_ID}"
    
    try:
        gdown.download_folder(
            url=url,
            output=str(pretrained_path),
            quiet=quiet,
            use_cookies=False,
        )
        
        # gdown may create a nested folder or download a zip file
        # Handle both cases
        _post_process_download(pretrained_path, quiet)
        
    except Exception as e:
        if not quiet:
            print(f"Folder download failed ({e}), trying individual files...")
        _download_individual_files(pretrained_path, quiet)
    
    # Verify download
    if not check_models_exist(pretrained_path):
        raise RuntimeError(
            f"Model download failed or incomplete. Please manually download from:\n"
            f"https://drive.google.com/drive/folders/{GDRIVE_FOLDER_ID}\n"
            f"and place the contents in: {pretrained_path}"
        )
    
    if not quiet:
        print("Download complete!")


def _post_process_download(pretrained_path: Path, quiet: bool = False) -> None:
    """
    Post-process downloaded files: extract zips and move files to correct location.
    
    gdown may create various nested structures:
    - pretrained_path/upscale_a_video/upscale_a_video.zip
    - pretrained_path/upscale_a_video/upscale_a_video/{actual_folders}
    
    The zip file contains:
    - upscale_a_video/{low_res_scheduler, propagator, scheduler, etc.}
    - __MACOSX/...
    
    We need to end up with:
    - pretrained_path/{low_res_scheduler, propagator, scheduler, etc.}
    """
    import shutil
    
    # First, find and extract all zip files
    for zip_file in list(pretrained_path.rglob("*.zip")):
        if not quiet:
            print(f"Extracting {zip_file.name}...")
        
        # Extract to a temp location first
        temp_extract = pretrained_path / "_temp_extract"
        temp_extract.mkdir(exist_ok=True)
        
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(temp_extract)
        
        # Remove the zip file after extraction
        zip_file.unlink()
        
        # Move extracted contents to the right place
        # The zip contains upscale_a_video/{folders} and __MACOSX/
        for item in temp_extract.iterdir():
            if item.name == "__MACOSX":
                # Remove Mac metadata folder
                shutil.rmtree(item)
            elif item.name == "upscale_a_video" and item.is_dir():
                # This is the nested folder containing the actual model folders
                for subitem in item.iterdir():
                    if subitem.name.startswith("."):
                        # Skip hidden files like .DS_Store
                        if subitem.is_file():
                            subitem.unlink()
                        else:
                            shutil.rmtree(subitem)
                        continue
                    dest = pretrained_path / subitem.name
                    if dest.exists():
                        if dest.is_dir():
                            shutil.rmtree(dest)
                        else:
                            dest.unlink()
                    shutil.move(str(subitem), str(dest))
                # Remove the now-empty nested folder
                shutil.rmtree(item)
            else:
                # Move other items directly
                dest = pretrained_path / item.name
                if dest.exists():
                    if dest.is_dir():
                        shutil.rmtree(dest)
                    else:
                        dest.unlink()
                shutil.move(str(item), str(dest))
        
        # Remove temp extract folder
        if temp_extract.exists():
            shutil.rmtree(temp_extract)
    
    # Clean up any remaining nested upscale_a_video folders (from gdown folder download)
    nested_dir = pretrained_path / "upscale_a_video"
    while nested_dir.exists() and nested_dir.is_dir():
        # Check if the nested dir has the expected subdirs
        has_expected_content = any(
            (nested_dir / d).exists() for d in EXPECTED_DIRS
        )
        if has_expected_content:
            if not quiet:
                print("Moving files from nested folder...")
            for item in list(nested_dir.iterdir()):
                if item.name.startswith("."):
                    if item.is_file():
                        item.unlink()
                    else:
                        shutil.rmtree(item)
                    continue
                dest = pretrained_path / item.name
                if dest.exists():
                    if dest.is_dir():
                        shutil.rmtree(dest)
                    else:
                        dest.unlink()
                shutil.move(str(item), str(dest))
            shutil.rmtree(nested_dir)
        else:
            # Go one level deeper
            deeper = nested_dir / "upscale_a_video"
            if deeper.exists():
                nested_dir = deeper
            else:
                break
    
    # Clean up __MACOSX folders
    for macos_dir in list(pretrained_path.rglob("__MACOSX")):
        shutil.rmtree(macos_dir)
    
    # Clean up .DS_Store files
    for ds_store in list(pretrained_path.rglob(".DS_Store")):
        ds_store.unlink()


def _download_individual_files(pretrained_path: Path, quiet: bool = False) -> None:
    """
    Fallback: download individual model components.
    """
    import gdown
    
    for dir_name, file_id in MODEL_FILES.items():
        dir_path = pretrained_path / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)
        
        if not quiet:
            print(f"  Downloading {dir_name}...")
        
        url = f"https://drive.google.com/uc?id={file_id}"
        output = str(dir_path / f"{dir_name}.zip")
        
        try:
            gdown.download(url, output, quiet=quiet)
            
            # Extract if it's a zip file
            if os.path.exists(output) and output.endswith('.zip'):
                with zipfile.ZipFile(output, 'r') as zip_ref:
                    zip_ref.extractall(dir_path)
                os.remove(output)
        except Exception as e:
            if not quiet:
                print(f"    Warning: Failed to download {dir_name}: {e}")


def ensure_models_downloaded(pretrained_path: Path, quiet: bool = False) -> None:
    """
    Ensure pretrained models are available, downloading if necessary.
    
    Args:
        pretrained_path: Path to the pretrained models directory
        quiet: If True, suppress output
    """
    if check_models_exist(pretrained_path):
        return
    
    if not quiet:
        print("Pretrained models not found. Downloading...")
    
    download_models(pretrained_path, quiet=quiet)

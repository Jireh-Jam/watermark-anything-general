#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Watermark Generation Script
This script provides a complete solution for embedding and detecting watermarks in images.
"""

import os
import sys
import argparse
import json
import urllib.request
from pathlib import Path
from typing import Optional, Dict, Tuple, List

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.utils import save_image

# Import watermark_anything modules
from watermark_anything.data.metrics import msg_predict_inference
from notebooks.inference_utils import (
    load_model_from_checkpoint,
    default_transform,
    unnormalize_img,
    create_random_mask,
    msg2str,
    str2msg
)


class WatermarkGenerator:
    """
    A class to handle watermark embedding and detection operations.
    """
    
    def __init__(self, checkpoint_dir: str = "checkpoints", device: Optional[str] = None):
        """
        Initialize the watermark generator.
        
        Args:
            checkpoint_dir: Directory containing model weights and params.json
            device: Device to run inference on ('cuda' or 'cpu'). Auto-detect if None.
        """
        self.checkpoint_dir = checkpoint_dir
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.model_loaded = False
        
    def download_model_weights(self, force_download: bool = False):
        """
        Download the MIT-licensed model weights if they don't exist.
        
        Args:
            force_download: If True, download even if weights exist
        """
        weights_path = os.path.join(self.checkpoint_dir, 'checkpoint.pth')
        
        if os.path.exists(weights_path) and not force_download:
            print(f"Model weights already exist at {weights_path}")
            return
        
        print("Downloading model weights (MIT license)...")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        url = "https://dl.fbaipublicfiles.com/watermark_anything/wam_mit.pth"
        
        try:
            def show_progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                percent = min(downloaded * 100.0 / total_size, 100)
                sys.stdout.write(f"\rDownloading: {percent:.1f}% ({downloaded}/{total_size} bytes)")
                sys.stdout.flush()
            
            urllib.request.urlretrieve(url, weights_path, show_progress)
            print("\nModel weights downloaded successfully!")
        except Exception as e:
            print(f"\nError downloading model weights: {e}")
            print(f"Please manually download from {url} to {weights_path}")
            sys.exit(1)
    
    def load_model(self):
        """
        Load the watermark model from checkpoint.
        """
        if self.model_loaded:
            return
        
        json_path = os.path.join(self.checkpoint_dir, "params.json")
        ckpt_path = os.path.join(self.checkpoint_dir, "checkpoint.pth")
        
        # Check if files exist
        if not os.path.exists(json_path):
            print(f"Error: params.json not found at {json_path}")
            sys.exit(1)
        
        if not os.path.exists(ckpt_path):
            print(f"Model weights not found. Attempting to download...")
            self.download_model_weights()
        
        print(f"Loading model on {self.device}...")
        try:
            self.model = load_model_from_checkpoint(json_path, ckpt_path)
            self.model = self.model.to(self.device).eval()
            self.model_loaded = True
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)
    
    def generate_random_message(self, num_bits: int = 32) -> torch.Tensor:
        """
        Generate a random binary message.
        
        Args:
            num_bits: Number of bits in the message
            
        Returns:
            Random binary message tensor
        """
        return torch.randint(0, 2, (num_bits,)).float().to(self.device)
    
    def embed_watermark(
        self,
        image_path: str,
        output_path: str,
        message: Optional[torch.Tensor] = None,
        mask_percentage: float = 1.0,
        save_mask: bool = True
    ) -> Dict:
        """
        Embed a watermark into an image.
        
        Args:
            image_path: Path to input image
            output_path: Path to save watermarked image
            message: Binary message to embed (auto-generated if None)
            mask_percentage: Percentage of image to watermark (0.0 to 1.0)
            save_mask: Whether to save the watermark mask
            
        Returns:
            Dictionary containing embedding information
        """
        self.load_model()
        
        # Load and preprocess image
        try:
            img = Image.open(image_path).convert("RGB")
            img_pt = default_transform(img).unsqueeze(0).to(self.device)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
        
        # Generate or use provided message
        if message is None:
            message = self.generate_random_message()
        else:
            message = message.to(self.device)
        
        # Embed watermark
        with torch.no_grad():
            outputs = self.model.embed(img_pt, message)
        
        # Create mask
        if mask_percentage >= 0.999:
            mask = torch.ones((1, 1, img_pt.shape[2], img_pt.shape[3])).to(self.device)
        else:
            mask = create_random_mask(img_pt, num_masks=1, mask_percentage=mask_percentage)
        
        # Apply watermark with mask
        img_w = outputs['imgs_w'] * mask + img_pt * (1 - mask)
        
        # Save watermarked image
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        save_image(unnormalize_img(img_w), output_path)
        
        # Optionally save mask
        if save_mask:
            mask_path = output_path.replace('.png', '_mask.png').replace('.jpg', '_mask.png')
            save_image(mask, mask_path)
        
        # Calculate PSNR
        psnr = self._calculate_psnr(img_pt, img_w)
        
        result = {
            'input_path': image_path,
            'output_path': output_path,
            'message': message.cpu().numpy(),
            'message_str': msg2str(message.cpu()),
            'mask_percentage': mask_percentage,
            'psnr': psnr
        }
        
        return result
    
    def detect_watermark(
        self,
        image_path: str,
        output_dir: Optional[str] = None,
        save_detection_mask: bool = True
    ) -> Dict:
        """
        Detect and extract watermark from an image.
        
        Args:
            image_path: Path to watermarked image
            output_dir: Directory to save detection results
            save_detection_mask: Whether to save detection mask
            
        Returns:
            Dictionary containing detection results
        """
        self.load_model()
        
        # Load and preprocess image
        try:
            img = Image.open(image_path).convert("RGB")
            img_pt = default_transform(img).unsqueeze(0).to(self.device)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
        
        # Detect watermark
        with torch.no_grad():
            preds = self.model.detect(img_pt)["preds"]
        
        # Extract mask and bits
        mask_preds = F.sigmoid(preds[:, 0, :, :])
        bit_preds = preds[:, 1:, :, :]
        
        # Predict message
        pred_message = msg_predict_inference(bit_preds, mask_preds).cpu().float()
        
        # Save detection mask if requested
        if save_detection_mask and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            mask_preds_res = F.interpolate(
                mask_preds.unsqueeze(1),
                size=(img_pt.shape[-2], img_pt.shape[-1]),
                mode="bilinear",
                align_corners=False
            )
            base_name = os.path.basename(image_path)
            mask_output_path = os.path.join(output_dir, f"{base_name}_detected_mask.png")
            save_image(mask_preds_res, mask_output_path)
        
        result = {
            'input_path': image_path,
            'detected_message': pred_message[0].numpy(),
            'message_str': msg2str(pred_message[0]),
            'mask_coverage': mask_preds.mean().item()
        }
        
        return result
    
    def verify_watermark(
        self,
        image_path: str,
        original_message: torch.Tensor
    ) -> Dict:
        """
        Verify if a specific watermark exists in an image.
        
        Args:
            image_path: Path to watermarked image
            original_message: The original embedded message
            
        Returns:
            Dictionary containing verification results
        """
        detection_result = self.detect_watermark(image_path, save_detection_mask=False)
        
        if detection_result is None:
            return None
        
        detected_msg = torch.tensor(detection_result['detected_message'])
        original_msg = original_message.cpu()
        
        # Calculate bit accuracy
        bit_acc = (detected_msg == original_msg).float().mean().item()
        hamming_distance = int((detected_msg != original_msg).sum().item())
        
        result = {
            'input_path': image_path,
            'original_message': msg2str(original_msg),
            'detected_message': detection_result['message_str'],
            'bit_accuracy': bit_acc,
            'hamming_distance': hamming_distance,
            'total_bits': len(original_msg),
            'match': bit_acc > 0.9  # Consider a match if >90% bits correct
        }
        
        return result
    
    def batch_embed(
        self,
        input_dir: str,
        output_dir: str,
        message: Optional[torch.Tensor] = None,
        mask_percentage: float = 1.0,
        use_same_message: bool = True
    ) -> List[Dict]:
        """
        Embed watermarks in all images in a directory.
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save watermarked images
            message: Message to embed (auto-generated if None)
            mask_percentage: Percentage of image to watermark
            use_same_message: Use same message for all images
            
        Returns:
            List of embedding results
        """
        self.load_model()
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = [
            f for f in os.listdir(input_dir)
            if os.path.splitext(f.lower())[1] in image_extensions
        ]
        
        if not image_files:
            print(f"No images found in {input_dir}")
            return []
        
        results = []
        base_message = message if message is not None else self.generate_random_message()
        
        print(f"Processing {len(image_files)} images...")
        
        for i, img_file in enumerate(image_files):
            input_path = os.path.join(input_dir, img_file)
            output_path = os.path.join(output_dir, f"watermarked_{img_file}")
            
            # Use same or different message
            current_message = base_message if use_same_message else self.generate_random_message()
            
            print(f"[{i+1}/{len(image_files)}] Processing {img_file}...")
            
            result = self.embed_watermark(
                input_path,
                output_path,
                message=current_message,
                mask_percentage=mask_percentage,
                save_mask=True
            )
            
            if result:
                results.append(result)
                print(f"  ✓ Saved to {output_path}")
                print(f"  Message: {result['message_str']}")
                print(f"  PSNR: {result['psnr']:.2f} dB")
        
        return results
    
    def _calculate_psnr(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """Calculate PSNR between two images."""
        mse = F.mse_loss(img1, img2)
        if mse == 0:
            return float('inf')
        return 20 * torch.log10(1.0 / torch.sqrt(mse)).item()


def main():
    parser = argparse.ArgumentParser(
        description="Watermark Anything - Embed and detect watermarks in images"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Embed command
    embed_parser = subparsers.add_parser('embed', help='Embed watermark in image(s)')
    embed_parser.add_argument('input', help='Input image or directory')
    embed_parser.add_argument('output', help='Output image or directory')
    embed_parser.add_argument('--message', help='32-bit binary message (e.g., "10101010101010101010101010101010")')
    embed_parser.add_argument('--mask-percentage', type=float, default=1.0,
                             help='Percentage of image to watermark (0.0-1.0, default: 1.0)')
    embed_parser.add_argument('--batch', action='store_true',
                             help='Process all images in input directory')
    embed_parser.add_argument('--different-messages', action='store_true',
                             help='Use different random messages for each image in batch mode')
    
    # Detect command
    detect_parser = subparsers.add_parser('detect', help='Detect watermark in image')
    detect_parser.add_argument('input', help='Input watermarked image')
    detect_parser.add_argument('--output-dir', help='Directory to save detection results')
    
    # Verify command
    verify_parser = subparsers.add_parser('verify', help='Verify watermark in image')
    verify_parser.add_argument('input', help='Input watermarked image')
    verify_parser.add_argument('message', help='Original 32-bit binary message')
    
    # Download command
    download_parser = subparsers.add_parser('download', help='Download model weights')
    download_parser.add_argument('--force', action='store_true',
                               help='Force download even if weights exist')
    
    # Common arguments
    parser.add_argument('--checkpoint-dir', default='checkpoints',
                       help='Directory containing model weights (default: checkpoints)')
    parser.add_argument('--device', choices=['cuda', 'cpu'],
                       help='Device to use (default: auto-detect)')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # Initialize generator
    generator = WatermarkGenerator(
        checkpoint_dir=args.checkpoint_dir,
        device=args.device
    )
    
    # Execute command
    if args.command == 'download':
        generator.download_model_weights(force_download=args.force)
    
    elif args.command == 'embed':
        # Parse message if provided
        message = None
        if args.message:
            try:
                message = torch.tensor([float(b) for b in args.message]).float()
                if len(message) != 32:
                    print("Error: Message must be exactly 32 bits")
                    return
            except Exception as e:
                print(f"Error parsing message: {e}")
                return
        
        if args.batch or os.path.isdir(args.input):
            # Batch processing
            results = generator.batch_embed(
                args.input,
                args.output,
                message=message,
                mask_percentage=args.mask_percentage,
                use_same_message=not args.different_messages
            )
            print(f"\n✓ Successfully processed {len(results)} images")
        else:
            # Single image
            result = generator.embed_watermark(
                args.input,
                args.output,
                message=message,
                mask_percentage=args.mask_percentage
            )
            if result:
                print("\n✓ Watermark embedded successfully!")
                print(f"  Output: {result['output_path']}")
                print(f"  Message: {result['message_str']}")
                print(f"  PSNR: {result['psnr']:.2f} dB")
    
    elif args.command == 'detect':
        result = generator.detect_watermark(
            args.input,
            output_dir=args.output_dir
        )
        if result:
            print("\n✓ Watermark detected!")
            print(f"  Detected message: {result['message_str']}")
            print(f"  Mask coverage: {result['mask_coverage']*100:.2f}%")
    
    elif args.command == 'verify':
        try:
            original_message = torch.tensor([float(b) for b in args.message]).float()
            if len(original_message) != 32:
                print("Error: Message must be exactly 32 bits")
                return
        except Exception as e:
            print(f"Error parsing message: {e}")
            return
        
        result = generator.verify_watermark(args.input, original_message)
        if result:
            print("\n✓ Verification complete!")
            print(f"  Original message:  {result['original_message']}")
            print(f"  Detected message:  {result['detected_message']}")
            print(f"  Bit accuracy: {result['bit_accuracy']*100:.1f}%")
            print(f"  Hamming distance: {result['hamming_distance']}/{result['total_bits']}")
            print(f"  Match: {'YES' if result['match'] else 'NO'}")


if __name__ == "__main__":
    main()

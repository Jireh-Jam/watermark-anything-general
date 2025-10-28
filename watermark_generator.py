#!/usr/bin/env python3
"""
Watermark Generator for the Watermark Anything Repository

This module provides a comprehensive watermarking solution that builds upon
the existing Watermark Anything infrastructure. It supports both the advanced
deep learning-based watermarking and traditional watermarking methods.

Features:
- Deep learning-based invisible watermarks using the WAM model
- Traditional visible text and image watermarks
- Batch processing capabilities
- Multiple output formats
- Robust error handling

Author: Generated for Watermark Anything Repository
License: MIT (for new code), respects existing licenses
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Union, List, Optional, Tuple, Dict, Any
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision import transforms

# Import existing watermark anything modules
try:
    from watermark_anything.data.metrics import msg_predict_inference
    from notebooks.inference_utils import (
        load_model_from_checkpoint, default_transform, unnormalize_img,
        create_random_mask, msg2str, multiwm_dbscan
    )
    WAM_AVAILABLE = True
except ImportError as e:
    print(f"Warning: WAM modules not available: {e}")
    print("Traditional watermarking will still work.")
    WAM_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WatermarkGenerator:
    """
    A comprehensive watermarking system that supports both advanced AI-based
    invisible watermarks and traditional visible watermarks.
    """
    
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize the watermark generator.
        
        Args:
            model_path: Path to the WAM model checkpoint directory
            device: Device to use ('cuda', 'cpu', or None for auto-detection)
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize WAM model if available
        self.wam_model = None
        if WAM_AVAILABLE and model_path and os.path.exists(model_path):
            self._load_wam_model(model_path)
        elif WAM_AVAILABLE:
            # Try to load from default checkpoint location
            default_path = "checkpoints"
            if os.path.exists(os.path.join(default_path, "params.json")):
                self._load_wam_model(default_path)
        
        # Default settings
        self.default_font_size = 36
        self.default_opacity = 0.5
        self.default_position = "bottom-right"
        
    def _load_wam_model(self, model_path: str):
        """Load the WAM (Watermark Anything Model) for invisible watermarking."""
        try:
            json_path = os.path.join(model_path, "params.json")
            
            # Look for available checkpoint files
            checkpoint_files = [
                "checkpoint.pth", "wam_mit.pth", "wam_coco.pth"
            ]
            
            ckpt_path = None
            for ckpt_file in checkpoint_files:
                potential_path = os.path.join(model_path, ckpt_file)
                if os.path.exists(potential_path):
                    ckpt_path = potential_path
                    break
            
            if not ckpt_path:
                logger.warning(f"No checkpoint file found in {model_path}")
                return
                
            if not os.path.exists(json_path):
                logger.warning(f"No params.json found in {model_path}")
                return
                
            self.wam_model = load_model_from_checkpoint(json_path, ckpt_path)
            self.wam_model = self.wam_model.to(self.device).eval()
            logger.info(f"WAM model loaded successfully from {ckpt_path}")
            
        except Exception as e:
            logger.error(f"Failed to load WAM model: {e}")
            self.wam_model = None
    
    def embed_invisible_watermark(
        self, 
        image_path: str, 
        message: Union[str, List[int], torch.Tensor],
        output_path: str,
        mask_percentage: float = 0.5,
        scaling_factor: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Embed an invisible watermark using the WAM model.
        
        Args:
            image_path: Path to the input image
            message: Watermark message (string, list of bits, or tensor)
            output_path: Path to save the watermarked image
            mask_percentage: Percentage of image to watermark (0.0 to 1.0)
            scaling_factor: Watermark strength (None for default)
            
        Returns:
            Dictionary with embedding results and metadata
        """
        if not self.wam_model:
            raise RuntimeError("WAM model not available. Please ensure model is loaded.")
        
        # Load and preprocess image
        img = Image.open(image_path).convert("RGB")
        img_pt = default_transform(img).unsqueeze(0).to(self.device)
        
        # Convert message to tensor format
        wm_msg = self._prepare_message(message)
        
        # Set scaling factor if provided
        if scaling_factor is not None:
            original_scaling = self.wam_model.scaling_w
            self.wam_model.scaling_w = scaling_factor
        
        try:
            # Embed watermark
            outputs = self.wam_model.embed(img_pt, wm_msg)
            
            # Create mask for localized watermarking
            mask = create_random_mask(
                img_pt, 
                num_masks=1, 
                mask_percentage=mask_percentage
            )
            
            # Apply watermark only to masked regions
            img_w = outputs['imgs_w'] * mask + img_pt * (1 - mask)
            
            # Save watermarked image
            save_image(unnormalize_img(img_w), output_path)
            
            # Calculate PSNR for quality assessment
            psnr = self._calculate_psnr(img_pt, img_w)
            
            result = {
                'success': True,
                'output_path': output_path,
                'message': msg2str(wm_msg.cpu()),
                'psnr': psnr,
                'mask_percentage': mask_percentage,
                'scaling_factor': scaling_factor or self.wam_model.scaling_w
            }
            
        finally:
            # Restore original scaling factor
            if scaling_factor is not None:
                self.wam_model.scaling_w = original_scaling
        
        return result
    
    def detect_invisible_watermark(
        self, 
        image_path: str,
        expected_message: Optional[Union[str, List[int], torch.Tensor]] = None
    ) -> Dict[str, Any]:
        """
        Detect and decode invisible watermarks from an image.
        
        Args:
            image_path: Path to the watermarked image
            expected_message: Expected message for accuracy calculation
            
        Returns:
            Dictionary with detection results
        """
        if not self.wam_model:
            raise RuntimeError("WAM model not available. Please ensure model is loaded.")
        
        # Load and preprocess image
        img = Image.open(image_path).convert("RGB")
        img_pt = default_transform(img).unsqueeze(0).to(self.device)
        
        # Detect watermark
        preds = self.wam_model.detect(img_pt)["preds"]
        mask_preds = F.sigmoid(preds[:, 0, :, :])
        bit_preds = preds[:, 1:, :, :]
        
        # Predict message
        pred_message = msg_predict_inference(bit_preds, mask_preds).cpu().float()
        pred_message_str = msg2str(pred_message[0])
        
        result = {
            'success': True,
            'detected_message': pred_message_str,
            'confidence_mask': mask_preds.cpu().numpy(),
            'watermark_detected': torch.sum(mask_preds > 0.5).item() > 100  # Threshold for detection
        }
        
        # Calculate accuracy if expected message is provided
        if expected_message is not None:
            expected_tensor = self._prepare_message(expected_message)
            bit_acc = (pred_message == expected_tensor.cpu()).float().mean().item()
            result['bit_accuracy'] = bit_acc
            result['hamming_distance'] = int(torch.sum(pred_message != expected_tensor.cpu()).item())
        
        return result
    
    def detect_multiple_watermarks(
        self,
        image_path: str,
        epsilon: float = 1.0,
        min_samples: int = 500
    ) -> Dict[str, Any]:
        """
        Detect multiple watermarks in a single image using DBSCAN clustering.
        
        Args:
            image_path: Path to the image
            epsilon: DBSCAN epsilon parameter
            min_samples: DBSCAN min_samples parameter
            
        Returns:
            Dictionary with multiple watermark detection results
        """
        if not self.wam_model:
            raise RuntimeError("WAM model not available. Please ensure model is loaded.")
        
        # Load and preprocess image
        img = Image.open(image_path).convert("RGB")
        img_pt = default_transform(img).unsqueeze(0).to(self.device)
        
        # Detect watermarks
        preds = self.wam_model.detect(img_pt)["preds"]
        mask_preds = F.sigmoid(preds[:, 0, :, :])
        bit_preds = preds[:, 1:, :, :]
        
        # Use DBSCAN to find multiple watermarks
        centroids, positions = multiwm_dbscan(
            bit_preds, mask_preds, 
            epsilon=epsilon, min_samples=min_samples
        )
        
        detected_messages = []
        for centroid in centroids.values():
            detected_messages.append(msg2str(centroid))
        
        return {
            'success': True,
            'num_watermarks_detected': len(centroids),
            'detected_messages': detected_messages,
            'cluster_positions': positions.cpu().numpy() if positions is not None else None
        }
    
    def add_text_watermark(
        self,
        image_path: str,
        text: str,
        output_path: str,
        position: str = "bottom-right",
        font_size: int = None,
        opacity: float = None,
        color: Tuple[int, int, int] = (255, 255, 255),
        font_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Add a visible text watermark to an image.
        
        Args:
            image_path: Path to input image
            text: Watermark text
            output_path: Path to save watermarked image
            position: Position ("top-left", "top-right", "bottom-left", "bottom-right", "center")
            font_size: Font size (None for default)
            opacity: Opacity (0.0 to 1.0, None for default)
            color: Text color as RGB tuple
            font_path: Path to custom font file
            
        Returns:
            Dictionary with watermarking results
        """
        try:
            # Load image
            img = Image.open(image_path).convert("RGBA")
            
            # Create transparent overlay
            overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            
            # Setup font
            font_size = font_size or self.default_font_size
            try:
                if font_path and os.path.exists(font_path):
                    font = ImageFont.truetype(font_path, font_size)
                else:
                    # Try to use a default system font
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
            except (OSError, IOError):
                # Fallback to default font
                font = ImageFont.load_default()
            
            # Calculate text position
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            x, y = self._calculate_text_position(
                position, img.size, (text_width, text_height)
            )
            
            # Apply opacity to color
            opacity = opacity or self.default_opacity
            alpha = int(255 * opacity)
            text_color = (*color, alpha)
            
            # Draw text
            draw.text((x, y), text, font=font, fill=text_color)
            
            # Combine with original image
            watermarked = Image.alpha_composite(img, overlay)
            watermarked = watermarked.convert("RGB")
            
            # Save result
            watermarked.save(output_path, quality=95)
            
            return {
                'success': True,
                'output_path': output_path,
                'text': text,
                'position': position,
                'font_size': font_size,
                'opacity': opacity,
                'color': color
            }
            
        except Exception as e:
            logger.error(f"Error adding text watermark: {e}")
            return {'success': False, 'error': str(e)}
    
    def add_image_watermark(
        self,
        image_path: str,
        watermark_path: str,
        output_path: str,
        position: str = "bottom-right",
        scale: float = 0.1,
        opacity: float = None
    ) -> Dict[str, Any]:
        """
        Add a visible image watermark to an image.
        
        Args:
            image_path: Path to input image
            watermark_path: Path to watermark image
            output_path: Path to save watermarked image
            position: Position of watermark
            scale: Scale factor for watermark size (0.0 to 1.0)
            opacity: Opacity (0.0 to 1.0, None for default)
            
        Returns:
            Dictionary with watermarking results
        """
        try:
            # Load images
            base_img = Image.open(image_path).convert("RGBA")
            watermark_img = Image.open(watermark_path).convert("RGBA")
            
            # Scale watermark
            base_width, base_height = base_img.size
            wm_width = int(base_width * scale)
            wm_height = int(watermark_img.height * (wm_width / watermark_img.width))
            watermark_img = watermark_img.resize((wm_width, wm_height), Image.Resampling.LANCZOS)
            
            # Apply opacity
            opacity = opacity or self.default_opacity
            if opacity < 1.0:
                # Create alpha mask
                alpha = watermark_img.split()[-1]
                alpha = ImageEnhance.Brightness(alpha).enhance(opacity)
                watermark_img.putalpha(alpha)
            
            # Calculate position
            x, y = self._calculate_text_position(
                position, base_img.size, watermark_img.size
            )
            
            # Paste watermark
            base_img.paste(watermark_img, (x, y), watermark_img)
            
            # Convert back to RGB and save
            result_img = base_img.convert("RGB")
            result_img.save(output_path, quality=95)
            
            return {
                'success': True,
                'output_path': output_path,
                'watermark_path': watermark_path,
                'position': position,
                'scale': scale,
                'opacity': opacity
            }
            
        except Exception as e:
            logger.error(f"Error adding image watermark: {e}")
            return {'success': False, 'error': str(e)}
    
    def batch_watermark(
        self,
        input_dir: str,
        output_dir: str,
        watermark_type: str = "invisible",
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Apply watermarks to all images in a directory.
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save watermarked images
            watermark_type: Type of watermark ("invisible", "text", "image")
            **kwargs: Additional arguments for specific watermark types
            
        Returns:
            List of results for each processed image
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Supported image extensions
        supported_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
        # Find all image files
        image_files = []
        for ext in supported_exts:
            image_files.extend(Path(input_dir).glob(f"*{ext}"))
            image_files.extend(Path(input_dir).glob(f"*{ext.upper()}"))
        
        results = []
        
        for img_path in image_files:
            try:
                output_path = os.path.join(output_dir, f"watermarked_{img_path.name}")
                
                if watermark_type == "invisible":
                    result = self.embed_invisible_watermark(
                        str(img_path), 
                        kwargs.get('message', 'DEFAULT_MESSAGE'),
                        output_path,
                        **{k: v for k, v in kwargs.items() if k != 'message'}
                    )
                elif watermark_type == "text":
                    result = self.add_text_watermark(
                        str(img_path),
                        kwargs.get('text', 'WATERMARK'),
                        output_path,
                        **{k: v for k, v in kwargs.items() if k != 'text'}
                    )
                elif watermark_type == "image":
                    if 'watermark_path' not in kwargs:
                        raise ValueError("watermark_path required for image watermarking")
                    result = self.add_image_watermark(
                        str(img_path),
                        kwargs['watermark_path'],
                        output_path,
                        **{k: v for k, v in kwargs.items() if k != 'watermark_path'}
                    )
                else:
                    raise ValueError(f"Unknown watermark type: {watermark_type}")
                
                result['input_path'] = str(img_path)
                results.append(result)
                
                if result['success']:
                    logger.info(f"Successfully processed: {img_path.name}")
                else:
                    logger.error(f"Failed to process: {img_path.name}")
                    
            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")
                results.append({
                    'success': False,
                    'input_path': str(img_path),
                    'error': str(e)
                })
        
        return results
    
    def _prepare_message(self, message: Union[str, List[int], torch.Tensor]) -> torch.Tensor:
        """Convert message to tensor format expected by WAM model."""
        if isinstance(message, str):
            # Convert string to binary (32 bits)
            if len(message) > 4:
                message = message[:4]  # Truncate to 4 characters
            
            # Convert to bytes and then to bits
            message_bytes = message.encode('utf-8')
            bits = []
            for byte in message_bytes:
                bits.extend([int(b) for b in format(byte, '08b')])
            
            # Pad or truncate to 32 bits
            if len(bits) < 32:
                bits.extend([0] * (32 - len(bits)))
            else:
                bits = bits[:32]
            
            return torch.tensor(bits, dtype=torch.float32).to(self.device)
            
        elif isinstance(message, list):
            return torch.tensor(message, dtype=torch.float32).to(self.device)
            
        elif isinstance(message, torch.Tensor):
            return message.to(self.device)
            
        else:
            raise ValueError(f"Unsupported message type: {type(message)}")
    
    def _calculate_text_position(
        self, 
        position: str, 
        img_size: Tuple[int, int], 
        text_size: Tuple[int, int]
    ) -> Tuple[int, int]:
        """Calculate text position based on position string."""
        img_width, img_height = img_size
        text_width, text_height = text_size
        margin = 20
        
        if position == "top-left":
            return (margin, margin)
        elif position == "top-right":
            return (img_width - text_width - margin, margin)
        elif position == "bottom-left":
            return (margin, img_height - text_height - margin)
        elif position == "bottom-right":
            return (img_width - text_width - margin, img_height - text_height - margin)
        elif position == "center":
            return ((img_width - text_width) // 2, (img_height - text_height) // 2)
        else:
            # Default to bottom-right
            return (img_width - text_width - margin, img_height - text_height - margin)
    
    def _calculate_psnr(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """Calculate Peak Signal-to-Noise Ratio between two images."""
        mse = torch.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * torch.log10(1.0 / torch.sqrt(mse)).item()


def main():
    """Command-line interface for the watermark generator."""
    parser = argparse.ArgumentParser(description="Watermark Generator for Images")
    
    # Common arguments
    parser.add_argument("--input", "-i", required=True, help="Input image or directory")
    parser.add_argument("--output", "-o", required=True, help="Output path or directory")
    parser.add_argument("--type", "-t", choices=["invisible", "text", "image"], 
                       default="invisible", help="Type of watermark")
    parser.add_argument("--model-path", help="Path to WAM model checkpoint directory")
    parser.add_argument("--device", choices=["cpu", "cuda"], help="Device to use")
    parser.add_argument("--batch", action="store_true", help="Process directory of images")
    
    # Invisible watermark arguments
    parser.add_argument("--message", help="Message to embed (invisible watermark)")
    parser.add_argument("--mask-percentage", type=float, default=0.5,
                       help="Percentage of image to watermark (0.0-1.0)")
    parser.add_argument("--scaling-factor", type=float, help="Watermark strength")
    parser.add_argument("--detect", action="store_true", help="Detect watermark instead of embedding")
    parser.add_argument("--detect-multiple", action="store_true", help="Detect multiple watermarks")
    
    # Text watermark arguments
    parser.add_argument("--text", help="Text for watermark")
    parser.add_argument("--font-size", type=int, help="Font size for text watermark")
    parser.add_argument("--position", choices=["top-left", "top-right", "bottom-left", 
                       "bottom-right", "center"], default="bottom-right",
                       help="Position of watermark")
    parser.add_argument("--opacity", type=float, help="Opacity (0.0-1.0)")
    parser.add_argument("--color", nargs=3, type=int, default=[255, 255, 255],
                       help="Text color as RGB values")
    parser.add_argument("--font-path", help="Path to custom font file")
    
    # Image watermark arguments
    parser.add_argument("--watermark-image", help="Path to watermark image")
    parser.add_argument("--scale", type=float, default=0.1, help="Scale factor for image watermark")
    
    args = parser.parse_args()
    
    # Initialize watermark generator
    wm_gen = WatermarkGenerator(model_path=args.model_path, device=args.device)
    
    try:
        if args.type == "invisible":
            if args.detect or args.detect_multiple:
                # Detection mode
                if args.detect_multiple:
                    result = wm_gen.detect_multiple_watermarks(args.input)
                    print("Multiple Watermark Detection Results:")
                    print(f"Number of watermarks detected: {result['num_watermarks_detected']}")
                    for i, msg in enumerate(result['detected_messages']):
                        print(f"Watermark {i+1}: {msg}")
                else:
                    result = wm_gen.detect_invisible_watermark(args.input, args.message)
                    print("Watermark Detection Results:")
                    print(f"Detected message: {result['detected_message']}")
                    print(f"Watermark detected: {result['watermark_detected']}")
                    if 'bit_accuracy' in result:
                        print(f"Bit accuracy: {result['bit_accuracy']:.3f}")
            else:
                # Embedding mode
                if args.batch:
                    results = wm_gen.batch_watermark(
                        args.input, args.output, "invisible",
                        message=args.message or "DEFAULT",
                        mask_percentage=args.mask_percentage,
                        scaling_factor=args.scaling_factor
                    )
                    successful = sum(1 for r in results if r['success'])
                    print(f"Processed {len(results)} images, {successful} successful")
                else:
                    result = wm_gen.embed_invisible_watermark(
                        args.input, 
                        args.message or "DEFAULT",
                        args.output,
                        mask_percentage=args.mask_percentage,
                        scaling_factor=args.scaling_factor
                    )
                    if result['success']:
                        print(f"Invisible watermark embedded successfully!")
                        print(f"Output: {result['output_path']}")
                        print(f"PSNR: {result['psnr']:.2f} dB")
                    else:
                        print(f"Failed to embed watermark: {result.get('error', 'Unknown error')}")
        
        elif args.type == "text":
            if not args.text:
                args.text = "WATERMARK"
            
            if args.batch:
                results = wm_gen.batch_watermark(
                    args.input, args.output, "text",
                    text=args.text,
                    position=args.position,
                    font_size=args.font_size,
                    opacity=args.opacity,
                    color=tuple(args.color),
                    font_path=args.font_path
                )
                successful = sum(1 for r in results if r['success'])
                print(f"Processed {len(results)} images, {successful} successful")
            else:
                result = wm_gen.add_text_watermark(
                    args.input,
                    args.text,
                    args.output,
                    position=args.position,
                    font_size=args.font_size,
                    opacity=args.opacity,
                    color=tuple(args.color),
                    font_path=args.font_path
                )
                if result['success']:
                    print(f"Text watermark added successfully!")
                    print(f"Output: {result['output_path']}")
                else:
                    print(f"Failed to add watermark: {result.get('error', 'Unknown error')}")
        
        elif args.type == "image":
            if not args.watermark_image:
                print("Error: --watermark-image is required for image watermarks")
                sys.exit(1)
            
            if args.batch:
                results = wm_gen.batch_watermark(
                    args.input, args.output, "image",
                    watermark_path=args.watermark_image,
                    position=args.position,
                    scale=args.scale,
                    opacity=args.opacity
                )
                successful = sum(1 for r in results if r['success'])
                print(f"Processed {len(results)} images, {successful} successful")
            else:
                result = wm_gen.add_image_watermark(
                    args.input,
                    args.watermark_image,
                    args.output,
                    position=args.position,
                    scale=args.scale,
                    opacity=args.opacity
                )
                if result['success']:
                    print(f"Image watermark added successfully!")
                    print(f"Output: {result['output_path']}")
                else:
                    print(f"Failed to add watermark: {result.get('error', 'Unknown error')}")
    
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
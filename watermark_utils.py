"""
Watermark utilities for the Watermark Anything tool
Provides core functionality for embedding and detecting watermarks
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.utils import save_image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from sklearn.cluster import DBSCAN
import json

from notebooks.inference_utils import (
    load_model_from_checkpoint,
    create_random_mask,
    multiwm_dbscan,
    msg2str
)
from watermark_anything.data.metrics import msg_predict_inference
from watermark_anything.data.transforms import default_transform, normalize_img, unnormalize_img


class WatermarkProcessor:
    """Main processor for watermark operations"""
    
    def __init__(self, checkpoint_dir="checkpoints", device=None):
        """Initialize the watermark processor
        
        Args:
            checkpoint_dir: Directory containing model checkpoints
            device: Device to run on (cuda/cpu), auto-detected if None
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_dir = Path(checkpoint_dir)
        
        # Load model
        json_path = self.checkpoint_dir / "params.json"
        ckpt_path = self.checkpoint_dir / "checkpoint.pth"
        
        if not json_path.exists() or not ckpt_path.exists():
            raise FileNotFoundError(
                f"Model files not found in {checkpoint_dir}. "
                "Please download the model weights first."
            )
        
        print(f"Loading model from {ckpt_path}...")
        self.model = load_model_from_checkpoint(str(json_path), str(ckpt_path))
        self.model = self.model.to(self.device).eval()
        print(f"✓ Model loaded successfully on {self.device}")
        
        # Default transform
        self.transform = default_transform
        
    def embed_watermark(self, image, message, mask_proportion=1.0, 
                       custom_mask=None, watermark_strength=None):
        """Embed a watermark into an image
        
        Args:
            image: PIL Image or path to image
            message: 32-bit message tensor or binary string
            mask_proportion: Proportion of image to watermark (0-1)
            custom_mask: Optional custom mask tensor
            watermark_strength: Strength of watermark (default from model)
            
        Returns:
            Dictionary with results including watermarked image
        """
        # Load image if path provided
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        
        # Convert image to tensor
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Ensure message is proper format
        if isinstance(message, (list, np.ndarray)):
            message = torch.tensor(message).float()
        if len(message) != 32:
            raise ValueError(f"Message must be 32 bits, got {len(message)}")
        message = message.to(self.device)
        
        # Set watermark strength if provided
        if watermark_strength is not None:
            original_strength = self.model.scaling_w
            self.model.scaling_w = watermark_strength
        
        # Create or use mask
        if custom_mask is not None:
            mask = custom_mask.to(self.device)
            if mask.shape[-2:] != img_tensor.shape[-2:]:
                mask = F.interpolate(mask, size=img_tensor.shape[-2:], 
                                   mode='bilinear', align_corners=False)
        else:
            mask = create_random_mask(img_tensor, num_masks=1, 
                                    mask_percentage=mask_proportion)
        
        # Embed watermark
        with torch.no_grad():
            outputs = self.model.embed(img_tensor, message)
            img_w = outputs['imgs_w'] * mask + img_tensor * (1 - mask)
        
        # Calculate metrics
        psnr = self._calculate_psnr(img_tensor, img_w)
        ssim = self._calculate_ssim(img_tensor, img_w)
        
        # Restore original strength
        if watermark_strength is not None:
            self.model.scaling_w = original_strength
        
        return {
            'watermarked_tensor': unnormalize_img(img_w),
            'watermarked_image': self._tensor_to_pil(img_w),
            'original_tensor': unnormalize_img(img_tensor),
            'mask': mask,
            'message': message,
            'psnr': psnr,
            'ssim': ssim
        }
    
    def detect_watermark(self, image, confidence_threshold=0.5):
        """Detect watermark in an image
        
        Args:
            image: PIL Image or path to image
            confidence_threshold: Minimum confidence to consider detection positive
            
        Returns:
            Dictionary with detection results
        """
        # Load image if path provided
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        
        # Convert image to tensor
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Detect watermark
        with torch.no_grad():
            preds = self.model.detect(img_tensor)["preds"]
            mask_preds = F.sigmoid(preds[:, 0, :, :])
            bit_preds = preds[:, 1:, :, :]
        
        # Predict message
        pred_message = msg_predict_inference(bit_preds, mask_preds).cpu().float()
        
        # Calculate confidence
        mask_confidence = mask_preds.mean().item()
        detected = mask_confidence > confidence_threshold
        
        # Calculate bit accuracy (assuming we don't know the original message)
        # We use the consistency of predictions across the masked region
        if detected:
            masked_bits = bit_preds * mask_preds.unsqueeze(1)
            bit_means = masked_bits.mean(dim=[2, 3])
            bit_stds = masked_bits.std(dim=[2, 3])
            bit_accuracy = (1 - bit_stds.mean()).item()
        else:
            bit_accuracy = 0.0
        
        # Resize mask prediction to original size
        mask_preds_resized = F.interpolate(
            mask_preds.unsqueeze(1), 
            size=img_tensor.shape[-2:],
            mode="bilinear", 
            align_corners=False
        )
        
        return {
            'detected': detected,
            'confidence': mask_confidence if detected else 0.0,
            'bit_accuracy': bit_accuracy,
            'message_tensor': pred_message[0],
            'message_binary': msg2str(pred_message[0] > 0.5),
            'mask_pred': mask_preds_resized,
            'watermarked_tensor': unnormalize_img(img_tensor)
        }
    
    def embed_multiple_watermarks(self, image, messages, mask_proportion=0.1,
                                watermark_strength=None):
        """Embed multiple watermarks in different regions of an image
        
        Args:
            image: PIL Image or path to image
            messages: List of 32-bit message tensors
            mask_proportion: Max proportion per watermark
            watermark_strength: Strength of watermarks
            
        Returns:
            Dictionary with results
        """
        # Load image if path provided
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        
        # Convert image to tensor
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Set watermark strength if provided
        if watermark_strength is not None:
            original_strength = self.model.scaling_w
            self.model.scaling_w = watermark_strength
        
        # Create masks for each watermark
        masks = create_random_mask(img_tensor, num_masks=len(messages),
                                 mask_percentage=mask_proportion)
        
        # Embed each watermark
        multi_wm_img = img_tensor.clone()
        with torch.no_grad():
            for i, (msg, mask) in enumerate(zip(messages, masks)):
                msg = msg.to(self.device).unsqueeze(0)
                outputs = self.model.embed(img_tensor, msg)
                multi_wm_img = outputs['imgs_w'] * mask + multi_wm_img * (1 - mask)
        
        # Calculate metrics
        psnr = self._calculate_psnr(img_tensor, multi_wm_img)
        ssim = self._calculate_ssim(img_tensor, multi_wm_img)
        
        # Combine masks
        combined_mask = torch.max(torch.stack(masks), dim=0)[0]
        
        # Restore original strength
        if watermark_strength is not None:
            self.model.scaling_w = original_strength
        
        return {
            'watermarked_tensor': unnormalize_img(multi_wm_img),
            'watermarked_image': self._tensor_to_pil(multi_wm_img),
            'original_tensor': unnormalize_img(img_tensor),
            'masks': masks,
            'combined_mask': combined_mask,
            'messages': messages,
            'psnr': psnr,
            'ssim': ssim
        }
    
    def detect_multiple_watermarks(self, image, epsilon=1, min_samples=500):
        """Detect multiple watermarks using DBSCAN clustering
        
        Args:
            image: PIL Image or path to image
            epsilon: DBSCAN epsilon parameter
            min_samples: DBSCAN min_samples parameter
            
        Returns:
            Dictionary with detection results for each watermark
        """
        # Load image if path provided
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        
        # Convert image to tensor
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Detect watermark
        with torch.no_grad():
            preds = self.model.detect(img_tensor)["preds"]
            mask_preds = F.sigmoid(preds[:, 0, :, :])
            bit_preds = preds[:, 1:, :, :]
        
        # Use DBSCAN to find multiple watermarks
        centroids, positions = multiwm_dbscan(
            bit_preds, mask_preds, 
            epsilon=epsilon, 
            min_samples=min_samples
        )
        
        # Format results
        messages = []
        confidences = []
        
        for label, centroid in centroids.items():
            messages.append(msg2str(centroid > 0.5))
            # Calculate confidence for this cluster
            cluster_mask = (positions == label).float()
            confidence = (mask_preds * cluster_mask.unsqueeze(0)).sum() / cluster_mask.sum()
            confidences.append(confidence.item())
        
        # Resize mask prediction
        mask_preds_resized = F.interpolate(
            mask_preds.unsqueeze(1),
            size=img_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False
        )
        
        return {
            'num_watermarks': len(messages),
            'messages': messages,
            'confidences': confidences,
            'positions': positions,
            'mask_pred': mask_preds_resized,
            'watermarked_tensor': unnormalize_img(img_tensor)
        }
    
    def _calculate_psnr(self, original, watermarked):
        """Calculate PSNR between two image tensors"""
        orig_np = self._tensor_to_numpy(original)
        wm_np = self._tensor_to_numpy(watermarked)
        return peak_signal_noise_ratio(orig_np, wm_np)
    
    def _calculate_ssim(self, original, watermarked):
        """Calculate SSIM between two image tensors"""
        orig_np = self._tensor_to_numpy(original)
        wm_np = self._tensor_to_numpy(watermarked)
        return structural_similarity(orig_np, wm_np, channel_axis=2)
    
    def _tensor_to_numpy(self, tensor):
        """Convert image tensor to numpy array"""
        tensor = unnormalize_img(tensor).clamp(0, 1)
        tensor = tensor.squeeze().permute(1, 2, 0).cpu()
        return tensor.numpy()
    
    def _tensor_to_pil(self, tensor):
        """Convert image tensor to PIL Image"""
        np_array = self._tensor_to_numpy(tensor)
        return Image.fromarray((np_array * 255).astype(np.uint8))


def generate_random_message(bits=32):
    """Generate a random binary message
    
    Args:
        bits: Number of bits in the message
        
    Returns:
        Tensor of random bits
    """
    return torch.randint(0, 2, (bits,)).float()


def encode_text_to_binary(text, bits=32):
    """Encode text to binary message
    
    Args:
        text: Text string to encode
        bits: Total number of bits (will pad or truncate)
        
    Returns:
        Tensor of binary values
    """
    # Convert text to binary
    binary_str = ''.join(format(ord(char), '08b') for char in text)
    
    # Convert to tensor
    binary_list = [int(b) for b in binary_str]
    binary_tensor = torch.tensor(binary_list).float()
    
    # Pad or truncate to desired length
    if len(binary_tensor) < bits:
        binary_tensor = F.pad(binary_tensor, (0, bits - len(binary_tensor)))
    elif len(binary_tensor) > bits:
        binary_tensor = binary_tensor[:bits]
    
    return binary_tensor


def decode_binary_to_text(binary_tensor):
    """Decode binary message to text
    
    Args:
        binary_tensor: Tensor of binary values
        
    Returns:
        Decoded text string
    """
    # Convert tensor to binary string
    binary_str = ''.join(str(int(b.item())) for b in (binary_tensor > 0.5))
    
    # Decode to text
    text = ''
    for i in range(0, len(binary_str), 8):
        byte = binary_str[i:i+8]
        if len(byte) == 8:
            char_code = int(byte, 2)
            if 32 <= char_code <= 126:  # Printable ASCII
                text += chr(char_code)
            else:
                break
    
    return text


def calculate_metrics(original_image, watermarked_image):
    """Calculate quality metrics between images
    
    Args:
        original_image: Original PIL Image
        watermarked_image: Watermarked PIL Image
        
    Returns:
        Dictionary with PSNR, SSIM, and MSE
    """
    # Convert to numpy arrays
    orig_np = np.array(original_image).astype(np.float32) / 255.0
    wm_np = np.array(watermarked_image).astype(np.float32) / 255.0
    
    # Calculate metrics
    psnr = peak_signal_noise_ratio(orig_np, wm_np)
    ssim = structural_similarity(orig_np, wm_np, channel_axis=2)
    mse = np.mean((orig_np - wm_np) ** 2)
    
    return {
        'psnr': psnr,
        'ssim': ssim,
        'mse': mse
    }


def visualize_results(original_tensor, watermarked_tensor, mask, mask_pred, save_path=None):
    """Visualize watermarking results
    
    Args:
        original_tensor: Original image tensor (can be None)
        watermarked_tensor: Watermarked image tensor
        mask: Ground truth mask (can be None)
        mask_pred: Predicted mask (can be None)
        save_path: Path to save visualization
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.ravel()
    
    # Helper function to convert tensor to displayable format
    def tensor_to_display(tensor):
        if tensor is None:
            return None
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        if tensor.shape[0] == 3:
            tensor = tensor.permute(1, 2, 0)
        tensor = tensor.cpu().numpy()
        return np.clip(tensor, 0, 1)
    
    # Plot original image
    if original_tensor is not None:
        axes[0].imshow(tensor_to_display(original_tensor))
        axes[0].set_title('Original Image')
        axes[0].axis('off')
    else:
        axes[0].axis('off')
    
    # Plot watermarked image
    axes[1].imshow(tensor_to_display(watermarked_tensor))
    axes[1].set_title('Watermarked Image')
    axes[1].axis('off')
    
    # Plot ground truth mask
    if mask is not None:
        mask_display = tensor_to_display(mask)
        if mask_display.ndim == 3:
            mask_display = mask_display[:, :, 0]
        axes[2].imshow(mask_display, cmap='gray')
        axes[2].set_title('Watermark Region')
        axes[2].axis('off')
    else:
        axes[2].axis('off')
    
    # Plot predicted mask
    if mask_pred is not None:
        mask_pred_display = tensor_to_display(mask_pred)
        if mask_pred_display.ndim == 3:
            mask_pred_display = mask_pred_display[:, :, 0]
        axes[3].imshow(mask_pred_display, cmap='gray')
        axes[3].set_title('Detected Watermark Region')
        axes[3].axis('off')
    else:
        axes[3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
    else:
        plt.show()


def create_custom_mask(shape, mask_type='center', **kwargs):
    """Create custom mask patterns
    
    Args:
        shape: Shape of the mask (H, W)
        mask_type: Type of mask ('center', 'corner', 'ring', 'random')
        **kwargs: Additional parameters for specific mask types
        
    Returns:
        Binary mask tensor
    """
    H, W = shape
    mask = torch.zeros((1, 1, H, W))
    
    if mask_type == 'center':
        size = kwargs.get('size', 0.5)
        h_start = int(H * (1 - size) / 2)
        h_end = int(H * (1 + size) / 2)
        w_start = int(W * (1 - size) / 2)
        w_end = int(W * (1 + size) / 2)
        mask[:, :, h_start:h_end, w_start:w_end] = 1
        
    elif mask_type == 'corner':
        corner = kwargs.get('corner', 'top_left')
        size = kwargs.get('size', 0.25)
        h_size = int(H * size)
        w_size = int(W * size)
        
        if corner == 'top_left':
            mask[:, :, :h_size, :w_size] = 1
        elif corner == 'top_right':
            mask[:, :, :h_size, -w_size:] = 1
        elif corner == 'bottom_left':
            mask[:, :, -h_size:, :w_size] = 1
        elif corner == 'bottom_right':
            mask[:, :, -h_size:, -w_size:] = 1
            
    elif mask_type == 'ring':
        outer_radius = kwargs.get('outer_radius', 0.4)
        inner_radius = kwargs.get('inner_radius', 0.2)
        center_h, center_w = H // 2, W // 2
        
        y, x = torch.meshgrid(torch.arange(H), torch.arange(W))
        dist = torch.sqrt((x - center_w) ** 2 + (y - center_h) ** 2)
        mask[:, :, :, :] = ((dist > inner_radius * min(H, W)) & 
                           (dist < outer_radius * min(H, W))).float()
                           
    elif mask_type == 'random':
        coverage = kwargs.get('coverage', 0.5)
        mask = (torch.rand((1, 1, H, W)) < coverage).float()
    
    return mask
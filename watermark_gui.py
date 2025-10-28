"""
Gradio GUI interface for the Watermark Anything tool
Provides an easy-to-use web interface for watermarking operations
"""

import gradio as gr
import torch
import numpy as np
from PIL import Image
import tempfile
from pathlib import Path
import json
from datetime import datetime

from watermark_utils import (
    WatermarkProcessor,
    generate_random_message,
    encode_text_to_binary,
    decode_binary_to_text,
    calculate_metrics,
    create_custom_mask
)


# Global processor instance
processor = None


def initialize_processor():
    """Initialize the watermark processor"""
    global processor
    if processor is None:
        try:
            processor = WatermarkProcessor()
            return True, "✓ Model loaded successfully"
        except Exception as e:
            return False, f"Error loading model: {str(e)}"
    return True, "✓ Model already loaded"


def embed_single_watermark(image, message_type, message_text, binary_message,
                         mask_type, mask_proportion, custom_mask_image,
                         watermark_strength):
    """Embed a single watermark in an image"""
    if image is None:
        return None, None, "Please upload an image first"
    
    # Initialize processor if needed
    success, status = initialize_processor()
    if not success:
        return None, None, status
    
    try:
        # Prepare message
        if message_type == "Random":
            message = generate_random_message()
            message_display = "Random: " + ''.join(str(int(b)) for b in message)
        elif message_type == "Text":
            if not message_text:
                return None, None, "Please enter text message"
            message = encode_text_to_binary(message_text)
            message_display = f"Text: {message_text}"
        else:  # Binary
            if not binary_message or not all(c in '01' for c in binary_message):
                return None, None, "Please enter valid binary message (0s and 1s only)"
            message = torch.tensor([int(b) for b in binary_message]).float()
            if len(message) < 32:
                message = torch.nn.functional.pad(message, (0, 32 - len(message)))
            elif len(message) > 32:
                message = message[:32]
            message_display = f"Binary: {binary_message}"
        
        # Prepare mask
        custom_mask = None
        if mask_type == "Full Image":
            mask_proportion = 1.0
        elif mask_type == "Custom Mask" and custom_mask_image is not None:
            custom_mask = torch.tensor(np.array(custom_mask_image.convert('L')) / 255.0)
            custom_mask = custom_mask.unsqueeze(0).unsqueeze(0).float()
        elif mask_type == "Center":
            h, w = image.size[1], image.size[0]
            custom_mask = create_custom_mask((h, w), 'center', size=0.5)
        elif mask_type == "Corner":
            h, w = image.size[1], image.size[0]
            custom_mask = create_custom_mask((h, w), 'corner', corner='bottom_right', size=0.25)
        
        # Embed watermark
        result = processor.embed_watermark(
            image,
            message,
            mask_proportion=mask_proportion if mask_type == "Random Area" else 1.0,
            custom_mask=custom_mask,
            watermark_strength=watermark_strength
        )
        
        # Create info text
        info = f"""✓ Watermark embedded successfully!
Message: {message_display}
PSNR: {result['psnr']:.2f} dB
SSIM: {result['ssim']:.4f}
Strength: {watermark_strength}"""
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original
        axes[0].imshow(image)
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        # Watermarked
        axes[1].imshow(result['watermarked_image'])
        axes[1].set_title('Watermarked')
        axes[1].axis('off')
        
        # Difference (amplified)
        diff = np.array(result['watermarked_image']).astype(float) - np.array(image).astype(float)
        diff_amplified = np.clip(np.abs(diff) * 10, 0, 255).astype(np.uint8)
        axes[2].imshow(diff_amplified)
        axes[2].set_title('Difference (10x)')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # Convert plot to image
        import io
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        viz_image = Image.open(buf)
        plt.close()
        
        return result['watermarked_image'], viz_image, info
        
    except Exception as e:
        return None, None, f"Error: {str(e)}"


def detect_single_watermark(image, decode_as_text):
    """Detect watermark in an image"""
    if image is None:
        return None, "Please upload an image first"
    
    # Initialize processor if needed
    success, status = initialize_processor()
    if not success:
        return None, status
    
    try:
        # Detect watermark
        result = processor.detect_watermark(image)
        
        if not result['detected']:
            return None, "No watermark detected in this image"
        
        # Prepare info text
        info = f"""✓ Watermark detected!
Confidence: {result['confidence']:.2%}
Bit Accuracy: {result['bit_accuracy']:.2%}
Binary Message: {result['message_binary']}"""
        
        if decode_as_text:
            try:
                decoded_text = decode_binary_to_text(result['message_tensor'])
                info += f"\nDecoded Text: {decoded_text}"
            except:
                info += "\nDecoded Text: [Unable to decode as text]"
        
        # Create visualization
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        # Watermarked image
        axes[0].imshow(image)
        axes[0].set_title('Watermarked Image')
        axes[0].axis('off')
        
        # Detected mask
        mask_np = result['mask_pred'].squeeze().cpu().numpy()
        axes[1].imshow(mask_np, cmap='hot')
        axes[1].set_title('Detected Watermark Region')
        axes[1].axis('off')
        
        plt.tight_layout()
        
        # Convert plot to image
        import io
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        viz_image = Image.open(buf)
        plt.close()
        
        return viz_image, info
        
    except Exception as e:
        return None, f"Error: {str(e)}"


def embed_multiple_watermarks(image, messages_text, proportion_per_wm, watermark_strength):
    """Embed multiple watermarks in an image"""
    if image is None:
        return None, None, "Please upload an image first"
    
    if not messages_text.strip():
        return None, None, "Please enter messages (one per line)"
    
    # Initialize processor if needed
    success, status = initialize_processor()
    if not success:
        return None, None, status
    
    try:
        # Parse messages
        message_lines = [line.strip() for line in messages_text.strip().split('\n') if line.strip()]
        messages = []
        
        for line in message_lines:
            if all(c in '01' for c in line):
                # Binary message
                msg = torch.tensor([int(b) for b in line]).float()
                if len(msg) < 32:
                    msg = torch.nn.functional.pad(msg, (0, 32 - len(msg)))
                elif len(msg) > 32:
                    msg = msg[:32]
            else:
                # Text message
                msg = encode_text_to_binary(line)
            messages.append(msg)
        
        # Embed watermarks
        result = processor.embed_multiple_watermarks(
            image,
            messages,
            mask_proportion=proportion_per_wm,
            watermark_strength=watermark_strength
        )
        
        # Detect the watermarks
        detection = processor.detect_multiple_watermarks(result['watermarked_image'])
        
        # Create info text
        info = f"""✓ {len(messages)} watermarks embedded successfully!
PSNR: {result['psnr']:.2f} dB
SSIM: {result['ssim']:.4f}
Strength: {watermark_strength}

Detected {detection['num_watermarks']} watermarks:"""
        
        for i, (msg, conf) in enumerate(zip(detection['messages'], detection['confidences'])):
            info += f"\n  Watermark {i+1}: Confidence {conf:.2%}"
            # Try to decode as text if original was text
            if i < len(message_lines) and not all(c in '01' for c in message_lines[i]):
                try:
                    decoded = decode_binary_to_text(torch.tensor([int(b) for b in msg]))
                    info += f" - '{decoded}'"
                except:
                    pass
        
        # Create visualization
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original
        axes[0].imshow(image)
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        # Watermarked
        axes[1].imshow(result['watermarked_image'])
        axes[1].set_title(f'Watermarked ({len(messages)} watermarks)')
        axes[1].axis('off')
        
        # Detection clusters
        if detection['positions'] is not None:
            positions_np = detection['positions'].cpu().numpy()
            unique_labels = np.unique(positions_np[positions_np >= 0])
            
            # Create color map
            colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
            colored_mask = np.zeros((*positions_np.shape, 3))
            
            for label, color in zip(unique_labels, colors):
                mask = positions_np == label
                colored_mask[mask] = color[:3]
            
            axes[2].imshow(colored_mask)
            axes[2].set_title('Detected Watermark Regions')
            axes[2].axis('off')
        else:
            axes[2].axis('off')
        
        plt.tight_layout()
        
        # Convert plot to image
        import io
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        viz_image = Image.open(buf)
        plt.close()
        
        return result['watermarked_image'], viz_image, info
        
    except Exception as e:
        return None, None, f"Error: {str(e)}"


def compare_images(original, watermarked):
    """Compare original and watermarked images"""
    if original is None or watermarked is None:
        return None, "Please upload both images"
    
    try:
        # Calculate metrics
        metrics = calculate_metrics(original, watermarked)
        
        # Detect watermark
        success, status = initialize_processor()
        if not success:
            detection_info = status
        else:
            result = processor.detect_watermark(watermarked)
            if result['detected']:
                detection_info = f"Watermark detected with {result['confidence']:.2%} confidence"
            else:
                detection_info = "No watermark detected"
        
        # Create info text
        info = f"""Image Comparison Results:
PSNR: {metrics['psnr']:.2f} dB
SSIM: {metrics['ssim']:.4f}
MSE: {metrics['mse']:.6f}

{detection_info}"""
        
        # Create visualization
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        
        # Original
        axes[0, 0].imshow(original)
        axes[0, 0].set_title('Original')
        axes[0, 0].axis('off')
        
        # Watermarked
        axes[0, 1].imshow(watermarked)
        axes[0, 1].set_title('Watermarked')
        axes[0, 1].axis('off')
        
        # Difference
        diff = np.array(watermarked).astype(float) - np.array(original).astype(float)
        axes[1, 0].imshow(np.abs(diff).astype(np.uint8))
        axes[1, 0].set_title('Absolute Difference')
        axes[1, 0].axis('off')
        
        # Difference amplified
        diff_amplified = np.clip(np.abs(diff) * 10, 0, 255).astype(np.uint8)
        axes[1, 1].imshow(diff_amplified)
        axes[1, 1].set_title('Difference (10x amplified)')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        # Convert plot to image
        import io
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        viz_image = Image.open(buf)
        plt.close()
        
        return viz_image, info
        
    except Exception as e:
        return None, f"Error: {str(e)}"


def create_interface():
    """Create the Gradio interface"""
    
    # Import matplotlib here to avoid issues
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    with gr.Blocks(title="Watermark Anything Tool", theme=gr.themes.Soft()) as interface:
        gr.Markdown(
            """
            # 🐤 Watermark Anything Tool
            
            A powerful tool for embedding and detecting invisible watermarks in images.
            Based on the [Watermark Anything](https://github.com/facebookresearch/watermark-anything) model.
            """
        )
        
        with gr.Tab("Embed Single Watermark"):
            with gr.Row():
                with gr.Column():
                    embed_image = gr.Image(label="Upload Image", type="pil")
                    
                    message_type = gr.Radio(
                        ["Random", "Text", "Binary"],
                        value="Random",
                        label="Message Type"
                    )
                    
                    message_text = gr.Textbox(
                        label="Text Message",
                        placeholder="Enter your message here",
                        visible=False
                    )
                    
                    binary_message = gr.Textbox(
                        label="Binary Message (32 bits)",
                        placeholder="e.g., 01010101010101010101010101010101",
                        visible=False
                    )
                    
                    mask_type = gr.Radio(
                        ["Full Image", "Random Area", "Center", "Corner", "Custom Mask"],
                        value="Full Image",
                        label="Watermark Region"
                    )
                    
                    mask_proportion = gr.Slider(
                        0.1, 1.0, value=0.5, step=0.1,
                        label="Proportion of Image",
                        visible=False
                    )
                    
                    custom_mask_image = gr.Image(
                        label="Upload Mask (white = watermark area)",
                        type="pil",
                        visible=False
                    )
                    
                    watermark_strength = gr.Slider(
                        0.5, 5.0, value=2.0, step=0.1,
                        label="Watermark Strength (higher = more robust but less invisible)"
                    )
                    
                    embed_btn = gr.Button("Embed Watermark", variant="primary")
                
                with gr.Column():
                    watermarked_image = gr.Image(label="Watermarked Image", type="pil")
                    visualization = gr.Image(label="Visualization", type="pil")
                    embed_info = gr.Textbox(label="Results", lines=6)
            
            # Update visibility based on selections
            def update_message_inputs(choice):
                return (
                    gr.update(visible=choice == "Text"),
                    gr.update(visible=choice == "Binary")
                )
            
            message_type.change(
                update_message_inputs,
                inputs=[message_type],
                outputs=[message_text, binary_message]
            )
            
            def update_mask_inputs(choice):
                return (
                    gr.update(visible=choice == "Random Area"),
                    gr.update(visible=choice == "Custom Mask")
                )
            
            mask_type.change(
                update_mask_inputs,
                inputs=[mask_type],
                outputs=[mask_proportion, custom_mask_image]
            )
            
            embed_btn.click(
                embed_single_watermark,
                inputs=[
                    embed_image, message_type, message_text, binary_message,
                    mask_type, mask_proportion, custom_mask_image, watermark_strength
                ],
                outputs=[watermarked_image, visualization, embed_info]
            )
        
        with gr.Tab("Detect Watermark"):
            with gr.Row():
                with gr.Column():
                    detect_image = gr.Image(label="Upload Watermarked Image", type="pil")
                    decode_text = gr.Checkbox(label="Try to decode as text", value=True)
                    detect_btn = gr.Button("Detect Watermark", variant="primary")
                
                with gr.Column():
                    detection_viz = gr.Image(label="Detection Visualization", type="pil")
                    detection_info = gr.Textbox(label="Detection Results", lines=8)
            
            detect_btn.click(
                detect_single_watermark,
                inputs=[detect_image, decode_text],
                outputs=[detection_viz, detection_info]
            )
        
        with gr.Tab("Multiple Watermarks"):
            with gr.Row():
                with gr.Column():
                    multi_image = gr.Image(label="Upload Image", type="pil")
                    
                    messages_input = gr.Textbox(
                        label="Messages (one per line)",
                        placeholder="Message 1\nMessage 2\n01010101010101010101010101010101",
                        lines=5
                    )
                    
                    proportion_per = gr.Slider(
                        0.05, 0.5, value=0.1, step=0.05,
                        label="Max proportion per watermark"
                    )
                    
                    multi_strength = gr.Slider(
                        0.5, 5.0, value=2.0, step=0.1,
                        label="Watermark Strength"
                    )
                    
                    multi_embed_btn = gr.Button("Embed Multiple Watermarks", variant="primary")
                
                with gr.Column():
                    multi_watermarked = gr.Image(label="Watermarked Image", type="pil")
                    multi_viz = gr.Image(label="Detection Visualization", type="pil")
                    multi_info = gr.Textbox(label="Results", lines=10)
            
            multi_embed_btn.click(
                embed_multiple_watermarks,
                inputs=[multi_image, messages_input, proportion_per, multi_strength],
                outputs=[multi_watermarked, multi_viz, multi_info]
            )
        
        with gr.Tab("Compare Images"):
            with gr.Row():
                with gr.Column():
                    original_compare = gr.Image(label="Original Image", type="pil")
                    watermarked_compare = gr.Image(label="Watermarked Image", type="pil")
                    compare_btn = gr.Button("Compare Images", variant="primary")
                
                with gr.Column():
                    comparison_viz = gr.Image(label="Comparison Visualization", type="pil")
                    comparison_info = gr.Textbox(label="Comparison Results", lines=8)
            
            compare_btn.click(
                compare_images,
                inputs=[original_compare, watermarked_compare],
                outputs=[comparison_viz, comparison_info]
            )
        
        gr.Markdown(
            """
            ## Tips:
            - **Watermark Strength**: Higher values make watermarks more robust but less invisible
            - **Text Messages**: Limited to ~4 characters due to 32-bit constraint
            - **Binary Messages**: Must be exactly 32 bits (will be padded/truncated)
            - **Multiple Watermarks**: Each watermark uses a different region of the image
            
            ## About:
            This tool is based on the Watermark Anything model from Meta Research.
            The watermarks are invisible to the human eye but can be detected and decoded.
            """
        )
    
    return interface


if __name__ == "__main__":
    # For testing the GUI directly
    interface = create_interface()
    interface.launch()
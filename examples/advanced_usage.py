#!/usr/bin/env python3
"""
Advanced Usage Examples for Watermark Generator

This script demonstrates advanced features and use cases
for the WatermarkGenerator class.
"""

import os
import sys
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
sys.path.append('..')

from watermark_generator import WatermarkGenerator


def create_sample_logo():
    """Create a sample logo for testing."""
    # Create a simple logo
    logo = Image.new('RGBA', (200, 100), (0, 0, 0, 0))
    draw = ImageDraw.Draw(logo)
    
    # Draw a simple company logo
    draw.rectangle([10, 10, 190, 90], fill=(0, 100, 200, 200), outline=(255, 255, 255, 255), width=3)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except:
        font = ImageFont.load_default()
    
    # Add text
    draw.text((100, 50), "LOGO", font=font, fill=(255, 255, 255, 255), anchor="mm")
    
    logo.save("sample_logo.png")
    return "sample_logo.png"


def example_custom_message_encoding():
    """Example: Custom message encoding for invisible watermarks."""
    print("🔐 Custom Message Encoding Example")
    print("=" * 40)
    
    wm_gen = WatermarkGenerator()
    
    if not wm_gen.wam_model:
        print("❌ WAM model not available for this example.")
        return
    
    input_image = "../assets/images/alpaca.jpg"
    if not os.path.exists(input_image):
        print(f"❌ Sample image not found: {input_image}")
        return
    
    # Example 1: Binary message
    print("📝 Using binary message...")
    binary_message = [1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0,
                     0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0]
    
    result = wm_gen.embed_invisible_watermark(
        image_path=input_image,
        message=binary_message,
        output_path="watermarked_binary.jpg",
        mask_percentage=0.4
    )
    
    if result['success']:
        print(f"✅ Binary watermark embedded: {result['message']}")
        
        # Detect it back
        detect_result = wm_gen.detect_invisible_watermark(
            "watermarked_binary.jpg", 
            binary_message
        )
        print(f"🎯 Detection accuracy: {detect_result.get('bit_accuracy', 'N/A')}")
    
    # Example 2: Torch tensor message
    print("\n📝 Using torch tensor message...")
    tensor_message = torch.randint(0, 2, (32,)).float()
    
    result = wm_gen.embed_invisible_watermark(
        image_path=input_image,
        message=tensor_message,
        output_path="watermarked_tensor.jpg",
        mask_percentage=0.4
    )
    
    if result['success']:
        print(f"✅ Tensor watermark embedded: {result['message']}")
    
    print()


def example_robustness_testing():
    """Example: Test watermark robustness against various attacks."""
    print("🛡️ Robustness Testing Example")
    print("=" * 40)
    
    wm_gen = WatermarkGenerator()
    
    if not wm_gen.wam_model:
        print("❌ WAM model not available for this example.")
        return
    
    input_image = "../assets/images/ducks.jpg"
    if not os.path.exists(input_image):
        print(f"❌ Sample image not found: {input_image}")
        return
    
    # Embed a watermark with high robustness
    print("📝 Embedding robust watermark...")
    message = "ROBUST"
    
    result = wm_gen.embed_invisible_watermark(
        image_path=input_image,
        message=message,
        output_path="watermarked_robust.jpg",
        mask_percentage=0.6,
        scaling_factor=3.0  # Higher scaling for more robustness
    )
    
    if not result['success']:
        print(f"❌ Failed to embed watermark: {result.get('error')}")
        return
    
    print(f"✅ Robust watermark embedded (PSNR: {result['psnr']:.2f} dB)")
    
    # Test detection on original watermarked image
    print("\n🔍 Testing detection on original watermarked image...")
    detect_result = wm_gen.detect_invisible_watermark("watermarked_robust.jpg", message)
    print(f"🎯 Original accuracy: {detect_result.get('bit_accuracy', 'N/A'):.3f}")
    
    # Simulate compression attack
    print("\n📦 Simulating JPEG compression attack...")
    try:
        img = Image.open("watermarked_robust.jpg")
        img.save("watermarked_compressed.jpg", "JPEG", quality=50)  # Heavy compression
        
        detect_result = wm_gen.detect_invisible_watermark("watermarked_compressed.jpg", message)
        print(f"🎯 After compression accuracy: {detect_result.get('bit_accuracy', 'N/A'):.3f}")
    except Exception as e:
        print(f"❌ Compression test failed: {e}")
    
    print()


def example_different_positions():
    """Example: Text watermarks at different positions."""
    print("📍 Position Variations Example")
    print("=" * 40)
    
    wm_gen = WatermarkGenerator()
    
    input_image = "../assets/images/seabackground.jpg"
    if not os.path.exists(input_image):
        print(f"❌ Sample image not found: {input_image}")
        return
    
    positions = ["top-left", "top-right", "bottom-left", "bottom-right", "center"]
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    
    print("📝 Creating watermarks at different positions...")
    
    for i, (position, color) in enumerate(zip(positions, colors)):
        output_path = f"watermarked_position_{position.replace('-', '_')}.jpg"
        
        result = wm_gen.add_text_watermark(
            image_path=input_image,
            text=f"POS: {position.upper()}",
            output_path=output_path,
            position=position,
            font_size=28,
            opacity=0.8,
            color=color
        )
        
        if result['success']:
            print(f"✅ {position}: {output_path}")
        else:
            print(f"❌ {position}: Failed")
    
    print()


def example_opacity_variations():
    """Example: Different opacity levels for watermarks."""
    print("🎨 Opacity Variations Example")
    print("=" * 40)
    
    wm_gen = WatermarkGenerator()
    
    input_image = "../assets/images/trex_bike.jpg"
    if not os.path.exists(input_image):
        print(f"❌ Sample image not found: {input_image}")
        return
    
    # Create sample logo
    logo_path = create_sample_logo()
    
    opacities = [0.2, 0.4, 0.6, 0.8, 1.0]
    
    print("🖼️ Creating logo watermarks with different opacities...")
    
    for opacity in opacities:
        output_path = f"watermarked_opacity_{int(opacity*100)}.jpg"
        
        result = wm_gen.add_image_watermark(
            image_path=input_image,
            watermark_path=logo_path,
            output_path=output_path,
            position="top-right",
            scale=0.2,
            opacity=opacity
        )
        
        if result['success']:
            print(f"✅ Opacity {opacity:.1f}: {output_path}")
        else:
            print(f"❌ Opacity {opacity:.1f}: Failed")
    
    print()


def example_scaling_variations():
    """Example: Different scaling factors for invisible watermarks."""
    print("⚖️ Scaling Factor Variations Example")
    print("=" * 40)
    
    wm_gen = WatermarkGenerator()
    
    if not wm_gen.wam_model:
        print("❌ WAM model not available for this example.")
        return
    
    input_image = "../assets/images/gauguin_256.jpg"
    if not os.path.exists(input_image):
        print(f"❌ Sample image not found: {input_image}")
        return
    
    scaling_factors = [1.0, 2.0, 3.0, 4.0]
    message = "SCALE"
    
    print("📝 Testing different scaling factors...")
    
    for scale in scaling_factors:
        output_path = f"watermarked_scale_{scale:.1f}.jpg"
        
        result = wm_gen.embed_invisible_watermark(
            image_path=input_image,
            message=message,
            output_path=output_path,
            mask_percentage=0.5,
            scaling_factor=scale
        )
        
        if result['success']:
            print(f"✅ Scale {scale:.1f}: PSNR {result['psnr']:.2f} dB")
            
            # Test detection
            detect_result = wm_gen.detect_invisible_watermark(output_path, message)
            accuracy = detect_result.get('bit_accuracy', 0)
            print(f"   🎯 Detection accuracy: {accuracy:.3f}")
        else:
            print(f"❌ Scale {scale:.1f}: Failed")
    
    print()


def example_mask_coverage_variations():
    """Example: Different mask coverage percentages."""
    print("🎭 Mask Coverage Variations Example")
    print("=" * 40)
    
    wm_gen = WatermarkGenerator()
    
    if not wm_gen.wam_model:
        print("❌ WAM model not available for this example.")
        return
    
    input_image = "../assets/images/alpaca.jpg"
    if not os.path.exists(input_image):
        print(f"❌ Sample image not found: {input_image}")
        return
    
    coverages = [0.1, 0.25, 0.5, 0.75, 0.9]
    message = "COVER"
    
    print("📝 Testing different mask coverage percentages...")
    
    for coverage in coverages:
        output_path = f"watermarked_coverage_{int(coverage*100)}.jpg"
        
        result = wm_gen.embed_invisible_watermark(
            image_path=input_image,
            message=message,
            output_path=output_path,
            mask_percentage=coverage,
            scaling_factor=2.0
        )
        
        if result['success']:
            print(f"✅ Coverage {coverage*100:2.0f}%: PSNR {result['psnr']:.2f} dB")
            
            # Test detection
            detect_result = wm_gen.detect_invisible_watermark(output_path, message)
            accuracy = detect_result.get('bit_accuracy', 0)
            print(f"   🎯 Detection accuracy: {accuracy:.3f}")
        else:
            print(f"❌ Coverage {coverage*100:2.0f}%: Failed")
    
    print()


def example_error_handling():
    """Example: Error handling and edge cases."""
    print("⚠️ Error Handling Example")
    print("=" * 40)
    
    wm_gen = WatermarkGenerator()
    
    # Test with non-existent image
    print("📝 Testing with non-existent image...")
    result = wm_gen.add_text_watermark(
        image_path="non_existent.jpg",
        text="Test",
        output_path="output.jpg"
    )
    print(f"Result: {'✅ Handled gracefully' if not result['success'] else '❌ Should have failed'}")
    
    # Test with invalid parameters
    print("\n📝 Testing with invalid opacity...")
    if os.path.exists("../assets/images/alpaca.jpg"):
        result = wm_gen.add_text_watermark(
            image_path="../assets/images/alpaca.jpg",
            text="Test",
            output_path="test_invalid.jpg",
            opacity=1.5  # Invalid opacity > 1.0
        )
        print(f"Result: {'✅ Handled gracefully' if result['success'] else '❌ Failed as expected'}")
    
    # Test with very long message
    if wm_gen.wam_model and os.path.exists("../assets/images/alpaca.jpg"):
        print("\n📝 Testing with very long message...")
        result = wm_gen.embed_invisible_watermark(
            image_path="../assets/images/alpaca.jpg",
            message="THIS_IS_A_VERY_LONG_MESSAGE_THAT_EXCEEDS_NORMAL_LIMITS",
            output_path="test_long_message.jpg"
        )
        print(f"Result: {'✅ Handled gracefully' if result['success'] else '❌ Failed'}")
    
    print()


def main():
    """Run all advanced examples."""
    print("🌊 Advanced Watermark Generator Examples")
    print("=" * 50)
    print()
    
    # Create examples directory if it doesn't exist
    os.makedirs(".", exist_ok=True)
    
    # Run examples
    example_different_positions()
    example_opacity_variations()
    example_scaling_variations()
    example_mask_coverage_variations()
    example_error_handling()
    
    # Only run invisible watermark examples if WAM model is available
    wm_gen = WatermarkGenerator()
    if wm_gen.wam_model:
        example_custom_message_encoding()
        example_robustness_testing()
    else:
        print("ℹ️ Some advanced examples skipped (WAM model not available)")
        print("   To use all features, ensure the model checkpoint is available.")
        print()
    
    print("🎉 All advanced examples completed!")
    print("📁 Check the current directory for output files.")
    
    # Cleanup
    if os.path.exists("sample_logo.png"):
        os.remove("sample_logo.png")


if __name__ == "__main__":
    main()
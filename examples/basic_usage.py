#!/usr/bin/env python3
"""
Basic Usage Examples for Watermark Generator

This script demonstrates how to use the WatermarkGenerator class
for various watermarking tasks.
"""

import os
import sys
sys.path.append('..')

from watermark_generator import WatermarkGenerator


def example_invisible_watermark():
    """Example: Embed and detect invisible watermarks."""
    print("🔒 Invisible Watermark Example")
    print("=" * 40)
    
    # Initialize the watermark generator
    wm_gen = WatermarkGenerator()
    
    # Check if we have sample images
    input_image = "../assets/images/alpaca.jpg"
    if not os.path.exists(input_image):
        print(f"❌ Sample image not found: {input_image}")
        return
    
    # Embed invisible watermark
    print("📝 Embedding invisible watermark...")
    result = wm_gen.embed_invisible_watermark(
        image_path=input_image,
        message="SECRET123",
        output_path="watermarked_invisible.jpg",
        mask_percentage=0.3,  # Watermark 30% of the image
    )
    
    if result['success']:
        print(f"✅ Watermark embedded successfully!")
        print(f"📁 Output: {result['output_path']}")
        print(f"📊 PSNR: {result['psnr']:.2f} dB")
        print(f"💬 Message: {result['message']}")
        
        # Now detect the watermark
        print("\n🔍 Detecting watermark...")
        detect_result = wm_gen.detect_invisible_watermark(
            image_path="watermarked_invisible.jpg",
            expected_message="SECRET123"
        )
        
        if detect_result['success']:
            print(f"📱 Detected message: {detect_result['detected_message']}")
            print(f"✅ Watermark detected: {'Yes' if detect_result['watermark_detected'] else 'No'}")
            if 'bit_accuracy' in detect_result:
                print(f"🎯 Bit accuracy: {detect_result['bit_accuracy']:.3f}")
        else:
            print(f"❌ Detection failed: {detect_result.get('error', 'Unknown error')}")
    else:
        print(f"❌ Embedding failed: {result.get('error', 'Unknown error')}")
    
    print()


def example_text_watermark():
    """Example: Add text watermarks."""
    print("📝 Text Watermark Example")
    print("=" * 40)
    
    # Initialize the watermark generator
    wm_gen = WatermarkGenerator()
    
    # Check if we have sample images
    input_image = "../assets/images/ducks.jpg"
    if not os.path.exists(input_image):
        print(f"❌ Sample image not found: {input_image}")
        return
    
    # Add text watermark
    print("📝 Adding text watermark...")
    result = wm_gen.add_text_watermark(
        image_path=input_image,
        text="© 2024 Sample Company",
        output_path="watermarked_text.jpg",
        position="bottom-right",
        font_size=24,
        opacity=0.7,
        color=(255, 255, 255)  # White text
    )
    
    if result['success']:
        print(f"✅ Text watermark added successfully!")
        print(f"📁 Output: {result['output_path']}")
        print(f"💬 Text: '{result['text']}'")
        print(f"📍 Position: {result['position']}")
    else:
        print(f"❌ Failed: {result.get('error', 'Unknown error')}")
    
    print()


def example_image_watermark():
    """Example: Add image/logo watermarks."""
    print("🖼️ Image Watermark Example")
    print("=" * 40)
    
    # Initialize the watermark generator
    wm_gen = WatermarkGenerator()
    
    # Check if we have sample images
    input_image = "../assets/images/seabackground.jpg"
    watermark_image = "../assets/images/gauguin_256.jpg"  # Use another image as watermark
    
    if not os.path.exists(input_image):
        print(f"❌ Sample image not found: {input_image}")
        return
    
    if not os.path.exists(watermark_image):
        print(f"❌ Watermark image not found: {watermark_image}")
        return
    
    # Add image watermark
    print("🖼️ Adding image watermark...")
    result = wm_gen.add_image_watermark(
        image_path=input_image,
        watermark_path=watermark_image,
        output_path="watermarked_image.jpg",
        position="top-left",
        scale=0.15,  # 15% of the original image size
        opacity=0.6
    )
    
    if result['success']:
        print(f"✅ Image watermark added successfully!")
        print(f"📁 Output: {result['output_path']}")
        print(f"🖼️ Watermark: {result['watermark_path']}")
        print(f"📍 Position: {result['position']}")
        print(f"📏 Scale: {result['scale']:.2f}")
    else:
        print(f"❌ Failed: {result.get('error', 'Unknown error')}")
    
    print()


def example_batch_processing():
    """Example: Batch process multiple images."""
    print("📦 Batch Processing Example")
    print("=" * 40)
    
    # Initialize the watermark generator
    wm_gen = WatermarkGenerator()
    
    # Check if we have sample images directory
    input_dir = "../assets/images"
    if not os.path.exists(input_dir):
        print(f"❌ Sample images directory not found: {input_dir}")
        return
    
    # Create output directory
    output_dir = "batch_watermarked"
    os.makedirs(output_dir, exist_ok=True)
    
    # Batch process with text watermarks
    print("📝 Batch processing with text watermarks...")
    results = wm_gen.batch_watermark(
        input_dir=input_dir,
        output_dir=output_dir,
        watermark_type="text",
        text="© 2024 Batch Processed",
        position="bottom-left",
        font_size=20,
        opacity=0.8,
        color=(255, 255, 0)  # Yellow text
    )
    
    successful = sum(1 for r in results if r['success'])
    total = len(results)
    
    print(f"📊 Batch Processing Results:")
    print(f"✅ Successful: {successful}/{total}")
    print(f"❌ Failed: {total - successful}/{total}")
    
    if successful > 0:
        print(f"📁 Output directory: {output_dir}")
    
    print()


def example_multiple_watermarks():
    """Example: Detect multiple watermarks (requires WAM model)."""
    print("🔍 Multiple Watermarks Detection Example")
    print("=" * 40)
    
    # Initialize the watermark generator
    wm_gen = WatermarkGenerator()
    
    if not wm_gen.wam_model:
        print("❌ WAM model not available. Please ensure model is loaded for this example.")
        return
    
    # This example would require an image with multiple embedded watermarks
    # For demonstration, we'll show how to call the function
    input_image = "../assets/images/trex_bike.jpg"
    if not os.path.exists(input_image):
        print(f"❌ Sample image not found: {input_image}")
        return
    
    print("🔍 Detecting multiple watermarks...")
    result = wm_gen.detect_multiple_watermarks(
        image_path=input_image,
        epsilon=1.0,
        min_samples=500
    )
    
    if result['success']:
        print(f"📊 Number of watermarks detected: {result['num_watermarks_detected']}")
        
        if result['detected_messages']:
            print(f"📱 Detected messages:")
            for i, msg in enumerate(result['detected_messages'], 1):
                print(f"   {i}. {msg}")
        else:
            print("ℹ️ No watermarks detected in this image")
    else:
        print(f"❌ Detection failed: {result.get('error', 'Unknown error')}")
    
    print()


def main():
    """Run all examples."""
    print("🌊 Watermark Generator Examples")
    print("=" * 50)
    print()
    
    # Create examples directory if it doesn't exist
    os.makedirs(".", exist_ok=True)
    
    # Run examples
    example_text_watermark()
    example_image_watermark()
    example_batch_processing()
    
    # Only run invisible watermark examples if WAM model is available
    wm_gen = WatermarkGenerator()
    if wm_gen.wam_model:
        example_invisible_watermark()
        example_multiple_watermarks()
    else:
        print("ℹ️ Invisible watermark examples skipped (WAM model not available)")
        print("   To use invisible watermarks, ensure the model checkpoint is available.")
        print()
    
    print("🎉 All examples completed!")
    print("📁 Check the current directory for output files.")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Simple examples demonstrating how to use the WatermarkGenerator class.
"""

import torch
from generate_watermark import WatermarkGenerator


def example_1_basic_embedding():
    """Example 1: Embed a watermark in a single image."""
    print("=" * 60)
    print("Example 1: Basic Watermark Embedding")
    print("=" * 60)
    
    generator = WatermarkGenerator()
    
    # Embed watermark with auto-generated message
    result = generator.embed_watermark(
        image_path="assets/images/alpaca.jpg",
        output_path="outputs/alpaca_watermarked.png",
        mask_percentage=1.0  # Watermark entire image
    )
    
    if result:
        print(f"✓ Watermark embedded!")
        print(f"  Message: {result['message_str']}")
        print(f"  PSNR: {result['psnr']:.2f} dB")
        print(f"  Output: {result['output_path']}")
    
    return result


def example_2_custom_message():
    """Example 2: Embed a custom message."""
    print("\n" + "=" * 60)
    print("Example 2: Custom Message Embedding")
    print("=" * 60)
    
    generator = WatermarkGenerator()
    
    # Create custom 32-bit message
    custom_message = torch.tensor([
        1, 0, 1, 0, 1, 0, 1, 0,  # Pattern
        1, 1, 0, 0, 1, 1, 0, 0,
        0, 0, 1, 1, 0, 0, 1, 1,
        1, 1, 1, 1, 0, 0, 0, 0
    ]).float()
    
    result = generator.embed_watermark(
        image_path="assets/images/ducks.jpg",
        output_path="outputs/ducks_custom_watermark.png",
        message=custom_message,
        mask_percentage=1.0
    )
    
    if result:
        print(f"✓ Custom watermark embedded!")
        print(f"  Message: {result['message_str']}")
        print(f"  PSNR: {result['psnr']:.2f} dB")
    
    return result


def example_3_partial_watermark():
    """Example 3: Watermark only part of the image."""
    print("\n" + "=" * 60)
    print("Example 3: Partial Watermarking")
    print("=" * 60)
    
    generator = WatermarkGenerator()
    
    # Watermark only 30% of the image
    result = generator.embed_watermark(
        image_path="assets/images/trex_bike.jpg",
        output_path="outputs/trex_partial_watermark.png",
        mask_percentage=0.3
    )
    
    if result:
        print(f"✓ Partial watermark embedded!")
        print(f"  Watermarked area: {result['mask_percentage']*100}%")
        print(f"  Message: {result['message_str']}")
        print(f"  PSNR: {result['psnr']:.2f} dB")
    
    return result


def example_4_detect_watermark():
    """Example 4: Detect watermark in an image."""
    print("\n" + "=" * 60)
    print("Example 4: Watermark Detection")
    print("=" * 60)
    
    generator = WatermarkGenerator()
    
    # First embed a watermark
    embed_result = generator.embed_watermark(
        image_path="assets/images/seabackground.jpg",
        output_path="outputs/seabackground_watermarked.png",
        mask_percentage=1.0
    )
    
    if embed_result:
        print(f"✓ Embedded message: {embed_result['message_str']}")
        
        # Now detect it
        detect_result = generator.detect_watermark(
            image_path="outputs/seabackground_watermarked.png",
            output_dir="outputs"
        )
        
        if detect_result:
            print(f"✓ Detected message: {detect_result['message_str']}")
            print(f"  Mask coverage: {detect_result['mask_coverage']*100:.2f}%")
    
    return embed_result, detect_result


def example_5_verify_watermark():
    """Example 5: Verify a specific watermark."""
    print("\n" + "=" * 60)
    print("Example 5: Watermark Verification")
    print("=" * 60)
    
    generator = WatermarkGenerator()
    
    # Create and embed a known message
    known_message = torch.tensor([
        1, 1, 1, 1, 0, 0, 0, 0,
        1, 0, 1, 0, 1, 0, 1, 0,
        0, 1, 0, 1, 0, 1, 0, 1,
        1, 1, 0, 0, 1, 1, 0, 0
    ]).float()
    
    embed_result = generator.embed_watermark(
        image_path="assets/images/gauguin_256.jpg",
        output_path="outputs/gauguin_verified.png",
        message=known_message,
        mask_percentage=1.0
    )
    
    if embed_result:
        # Verify the watermark
        verify_result = generator.verify_watermark(
            image_path="outputs/gauguin_verified.png",
            original_message=known_message
        )
        
        if verify_result:
            print(f"✓ Original:  {verify_result['original_message']}")
            print(f"  Detected:  {verify_result['detected_message']}")
            print(f"  Accuracy:  {verify_result['bit_accuracy']*100:.1f}%")
            print(f"  Hamming:   {verify_result['hamming_distance']}/{verify_result['total_bits']}")
            print(f"  Match:     {'YES ✓' if verify_result['match'] else 'NO ✗'}")
    
    return verify_result


def example_6_batch_processing():
    """Example 6: Batch process multiple images."""
    print("\n" + "=" * 60)
    print("Example 6: Batch Processing")
    print("=" * 60)
    
    generator = WatermarkGenerator()
    
    # Process all images in a directory
    results = generator.batch_embed(
        input_dir="assets/images",
        output_dir="outputs/batch",
        mask_percentage=1.0,
        use_same_message=True  # Use same message for all images
    )
    
    print(f"\n✓ Batch processed {len(results)} images")
    if results:
        print(f"  Shared message: {results[0]['message_str']}")
        avg_psnr = sum(r['psnr'] for r in results) / len(results)
        print(f"  Average PSNR: {avg_psnr:.2f} dB")
    
    return results


def example_7_programmatic_usage():
    """Example 7: Advanced programmatic usage."""
    print("\n" + "=" * 60)
    print("Example 7: Advanced Programmatic Usage")
    print("=" * 60)
    
    generator = WatermarkGenerator()
    generator.load_model()  # Pre-load model for faster processing
    
    # Process multiple images with different settings
    test_cases = [
        ("assets/images/alpaca.jpg", 1.0),
        ("assets/images/ducks.jpg", 0.7),
        ("assets/images/trex_bike.jpg", 0.5),
    ]
    
    results = []
    shared_message = generator.generate_random_message()
    
    from notebooks.inference_utils import msg2str
    print(f"Shared message: {msg2str(shared_message)}")
    print()
    
    for img_path, mask_pct in test_cases:
        img_name = img_path.split('/')[-1].split('.')[0]
        output_path = f"outputs/advanced_{img_name}.png"
        
        result = generator.embed_watermark(
            image_path=img_path,
            output_path=output_path,
            message=shared_message,
            mask_percentage=mask_pct,
            save_mask=True
        )
        
        if result:
            results.append(result)
            print(f"✓ {img_name}: PSNR={result['psnr']:.2f}dB, coverage={mask_pct*100}%")
    
    return results


def main():
    """Run all examples."""
    print("Watermark Anything - Usage Examples")
    print("=" * 60)
    print()
    
    # Create output directory
    import os
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("outputs/batch", exist_ok=True)
    
    try:
        # Run examples
        example_1_basic_embedding()
        example_2_custom_message()
        example_3_partial_watermark()
        example_4_detect_watermark()
        example_5_verify_watermark()
        example_6_batch_processing()
        example_7_programmatic_usage()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
        print("\nCheck the 'outputs/' directory for results.")
        
    except Exception as e:
        print(f"\n✗ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

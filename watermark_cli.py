#!/usr/bin/env python3
"""
Command Line Interface for Watermark Generator

A user-friendly CLI wrapper around the WatermarkGenerator class that provides
easy-to-use commands for different watermarking operations.

Usage Examples:
    # Invisible watermark
    python watermark_cli.py embed --input image.jpg --output watermarked.jpg --message "SECRET"
    python watermark_cli.py detect --input watermarked.jpg
    
    # Text watermark
    python watermark_cli.py text --input image.jpg --output watermarked.jpg --text "© 2024"
    
    # Image watermark
    python watermark_cli.py logo --input image.jpg --output watermarked.jpg --logo logo.png
    
    # Batch processing
    python watermark_cli.py batch-text --input-dir ./images --output-dir ./watermarked --text "© Company"
"""

import argparse
import sys
import os
from pathlib import Path
from watermark_generator import WatermarkGenerator


def setup_common_args(parser):
    """Add common arguments to a parser."""
    parser.add_argument("--input", "-i", required=True, help="Input image path")
    parser.add_argument("--output", "-o", required=True, help="Output image path")
    parser.add_argument("--model-path", help="Path to WAM model checkpoint directory")
    parser.add_argument("--device", choices=["cpu", "cuda"], help="Device to use")


def setup_batch_args(parser):
    """Add batch processing arguments to a parser."""
    parser.add_argument("--input-dir", required=True, help="Input directory containing images")
    parser.add_argument("--output-dir", required=True, help="Output directory for watermarked images")
    parser.add_argument("--model-path", help="Path to WAM model checkpoint directory")
    parser.add_argument("--device", choices=["cpu", "cuda"], help="Device to use")


def cmd_embed(args):
    """Embed invisible watermark command."""
    wm_gen = WatermarkGenerator(model_path=args.model_path, device=args.device)
    
    result = wm_gen.embed_invisible_watermark(
        args.input,
        args.message,
        args.output,
        mask_percentage=args.mask_percentage,
        scaling_factor=args.scaling_factor
    )
    
    if result['success']:
        print(f"✅ Invisible watermark embedded successfully!")
        print(f"📁 Output: {result['output_path']}")
        print(f"📊 PSNR: {result['psnr']:.2f} dB")
        print(f"💬 Message: {result['message']}")
        print(f"🎯 Coverage: {result['mask_percentage']*100:.1f}% of image")
    else:
        print(f"❌ Failed to embed watermark: {result.get('error', 'Unknown error')}")
        sys.exit(1)


def cmd_detect(args):
    """Detect invisible watermark command."""
    wm_gen = WatermarkGenerator(model_path=args.model_path, device=args.device)
    
    result = wm_gen.detect_invisible_watermark(args.input, args.expected_message)
    
    if result['success']:
        print(f"🔍 Watermark Detection Results:")
        print(f"📱 Detected message: {result['detected_message']}")
        print(f"✅ Watermark detected: {'Yes' if result['watermark_detected'] else 'No'}")
        
        if 'bit_accuracy' in result:
            print(f"🎯 Bit accuracy: {result['bit_accuracy']:.3f}")
            print(f"📏 Hamming distance: {result['hamming_distance']}/32")
    else:
        print(f"❌ Failed to detect watermark: {result.get('error', 'Unknown error')}")
        sys.exit(1)


def cmd_detect_multiple(args):
    """Detect multiple invisible watermarks command."""
    wm_gen = WatermarkGenerator(model_path=args.model_path, device=args.device)
    
    result = wm_gen.detect_multiple_watermarks(
        args.input,
        epsilon=args.epsilon,
        min_samples=args.min_samples
    )
    
    if result['success']:
        print(f"🔍 Multiple Watermark Detection Results:")
        print(f"📊 Number of watermarks detected: {result['num_watermarks_detected']}")
        
        if result['detected_messages']:
            print(f"📱 Detected messages:")
            for i, msg in enumerate(result['detected_messages'], 1):
                print(f"   {i}. {msg}")
        else:
            print("❌ No watermarks detected")
    else:
        print(f"❌ Failed to detect watermarks: {result.get('error', 'Unknown error')}")
        sys.exit(1)


def cmd_text(args):
    """Add text watermark command."""
    wm_gen = WatermarkGenerator(model_path=args.model_path, device=args.device)
    
    result = wm_gen.add_text_watermark(
        args.input,
        args.text,
        args.output,
        position=args.position,
        font_size=args.font_size,
        opacity=args.opacity,
        color=tuple(args.color) if args.color else (255, 255, 255),
        font_path=args.font_path
    )
    
    if result['success']:
        print(f"✅ Text watermark added successfully!")
        print(f"📁 Output: {result['output_path']}")
        print(f"💬 Text: '{result['text']}'")
        print(f"📍 Position: {result['position']}")
        print(f"🎨 Opacity: {result['opacity']:.2f}")
    else:
        print(f"❌ Failed to add text watermark: {result.get('error', 'Unknown error')}")
        sys.exit(1)


def cmd_logo(args):
    """Add image/logo watermark command."""
    wm_gen = WatermarkGenerator(model_path=args.model_path, device=args.device)
    
    result = wm_gen.add_image_watermark(
        args.input,
        args.logo,
        args.output,
        position=args.position,
        scale=args.scale,
        opacity=args.opacity
    )
    
    if result['success']:
        print(f"✅ Logo watermark added successfully!")
        print(f"📁 Output: {result['output_path']}")
        print(f"🖼️ Logo: {result['watermark_path']}")
        print(f"📍 Position: {result['position']}")
        print(f"📏 Scale: {result['scale']:.2f}")
        print(f"🎨 Opacity: {result['opacity']:.2f}")
    else:
        print(f"❌ Failed to add logo watermark: {result.get('error', 'Unknown error')}")
        sys.exit(1)


def cmd_batch_invisible(args):
    """Batch invisible watermark command."""
    wm_gen = WatermarkGenerator(model_path=args.model_path, device=args.device)
    
    results = wm_gen.batch_watermark(
        args.input_dir,
        args.output_dir,
        "invisible",
        message=args.message,
        mask_percentage=args.mask_percentage,
        scaling_factor=args.scaling_factor
    )
    
    successful = sum(1 for r in results if r['success'])
    total = len(results)
    
    print(f"📊 Batch Processing Results:")
    print(f"✅ Successful: {successful}/{total}")
    print(f"❌ Failed: {total - successful}/{total}")
    
    if successful > 0:
        print(f"📁 Output directory: {args.output_dir}")


def cmd_batch_text(args):
    """Batch text watermark command."""
    wm_gen = WatermarkGenerator(model_path=args.model_path, device=args.device)
    
    results = wm_gen.batch_watermark(
        args.input_dir,
        args.output_dir,
        "text",
        text=args.text,
        position=args.position,
        font_size=args.font_size,
        opacity=args.opacity,
        color=tuple(args.color) if args.color else (255, 255, 255),
        font_path=args.font_path
    )
    
    successful = sum(1 for r in results if r['success'])
    total = len(results)
    
    print(f"📊 Batch Processing Results:")
    print(f"✅ Successful: {successful}/{total}")
    print(f"❌ Failed: {total - successful}/{total}")
    
    if successful > 0:
        print(f"📁 Output directory: {args.output_dir}")


def cmd_batch_logo(args):
    """Batch logo watermark command."""
    wm_gen = WatermarkGenerator(model_path=args.model_path, device=args.device)
    
    results = wm_gen.batch_watermark(
        args.input_dir,
        args.output_dir,
        "image",
        watermark_path=args.logo,
        position=args.position,
        scale=args.scale,
        opacity=args.opacity
    )
    
    successful = sum(1 for r in results if r['success'])
    total = len(results)
    
    print(f"📊 Batch Processing Results:")
    print(f"✅ Successful: {successful}/{total}")
    print(f"❌ Failed: {total - successful}/{total}")
    
    if successful > 0:
        print(f"📁 Output directory: {args.output_dir}")


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="🌊 Watermark Generator CLI - Add watermarks to your images!",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Embed invisible watermark
  python watermark_cli.py embed -i photo.jpg -o watermarked.jpg --message "SECRET123"
  
  # Detect invisible watermark
  python watermark_cli.py detect -i watermarked.jpg --expected-message "SECRET123"
  
  # Add text watermark
  python watermark_cli.py text -i photo.jpg -o watermarked.jpg --text "© 2024 Company"
  
  # Add logo watermark
  python watermark_cli.py logo -i photo.jpg -o watermarked.jpg --logo company_logo.png
  
  # Batch process with text watermarks
  python watermark_cli.py batch-text --input-dir ./photos --output-dir ./watermarked --text "© 2024"
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Embed invisible watermark
    embed_parser = subparsers.add_parser("embed", help="Embed invisible watermark")
    setup_common_args(embed_parser)
    embed_parser.add_argument("--message", required=True, help="Message to embed")
    embed_parser.add_argument("--mask-percentage", type=float, default=0.5,
                             help="Percentage of image to watermark (0.0-1.0)")
    embed_parser.add_argument("--scaling-factor", type=float,
                             help="Watermark strength (higher = more robust but visible)")
    embed_parser.set_defaults(func=cmd_embed)
    
    # Detect invisible watermark
    detect_parser = subparsers.add_parser("detect", help="Detect invisible watermark")
    detect_parser.add_argument("--input", "-i", required=True, help="Input image path")
    detect_parser.add_argument("--expected-message", help="Expected message for accuracy calculation")
    detect_parser.add_argument("--model-path", help="Path to WAM model checkpoint directory")
    detect_parser.add_argument("--device", choices=["cpu", "cuda"], help="Device to use")
    detect_parser.set_defaults(func=cmd_detect)
    
    # Detect multiple invisible watermarks
    detect_multi_parser = subparsers.add_parser("detect-multiple", help="Detect multiple invisible watermarks")
    detect_multi_parser.add_argument("--input", "-i", required=True, help="Input image path")
    detect_multi_parser.add_argument("--epsilon", type=float, default=1.0, help="DBSCAN epsilon parameter")
    detect_multi_parser.add_argument("--min-samples", type=int, default=500, help="DBSCAN min_samples parameter")
    detect_multi_parser.add_argument("--model-path", help="Path to WAM model checkpoint directory")
    detect_multi_parser.add_argument("--device", choices=["cpu", "cuda"], help="Device to use")
    detect_multi_parser.set_defaults(func=cmd_detect_multiple)
    
    # Text watermark
    text_parser = subparsers.add_parser("text", help="Add text watermark")
    setup_common_args(text_parser)
    text_parser.add_argument("--text", required=True, help="Text to add as watermark")
    text_parser.add_argument("--position", choices=["top-left", "top-right", "bottom-left", 
                            "bottom-right", "center"], default="bottom-right",
                            help="Position of watermark")
    text_parser.add_argument("--font-size", type=int, help="Font size")
    text_parser.add_argument("--opacity", type=float, help="Opacity (0.0-1.0)")
    text_parser.add_argument("--color", nargs=3, type=int, help="Text color as RGB values (e.g., 255 255 255)")
    text_parser.add_argument("--font-path", help="Path to custom font file")
    text_parser.set_defaults(func=cmd_text)
    
    # Logo/Image watermark
    logo_parser = subparsers.add_parser("logo", help="Add logo/image watermark")
    setup_common_args(logo_parser)
    logo_parser.add_argument("--logo", required=True, help="Path to logo/watermark image")
    logo_parser.add_argument("--position", choices=["top-left", "top-right", "bottom-left", 
                            "bottom-right", "center"], default="bottom-right",
                            help="Position of watermark")
    logo_parser.add_argument("--scale", type=float, default=0.1, help="Scale factor for logo size")
    logo_parser.add_argument("--opacity", type=float, help="Opacity (0.0-1.0)")
    logo_parser.set_defaults(func=cmd_logo)
    
    # Batch invisible watermark
    batch_invisible_parser = subparsers.add_parser("batch-invisible", help="Batch process invisible watermarks")
    setup_batch_args(batch_invisible_parser)
    batch_invisible_parser.add_argument("--message", required=True, help="Message to embed")
    batch_invisible_parser.add_argument("--mask-percentage", type=float, default=0.5,
                                       help="Percentage of image to watermark (0.0-1.0)")
    batch_invisible_parser.add_argument("--scaling-factor", type=float,
                                       help="Watermark strength")
    batch_invisible_parser.set_defaults(func=cmd_batch_invisible)
    
    # Batch text watermark
    batch_text_parser = subparsers.add_parser("batch-text", help="Batch process text watermarks")
    setup_batch_args(batch_text_parser)
    batch_text_parser.add_argument("--text", required=True, help="Text to add as watermark")
    batch_text_parser.add_argument("--position", choices=["top-left", "top-right", "bottom-left", 
                                  "bottom-right", "center"], default="bottom-right",
                                  help="Position of watermark")
    batch_text_parser.add_argument("--font-size", type=int, help="Font size")
    batch_text_parser.add_argument("--opacity", type=float, help="Opacity (0.0-1.0)")
    batch_text_parser.add_argument("--color", nargs=3, type=int, help="Text color as RGB values")
    batch_text_parser.add_argument("--font-path", help="Path to custom font file")
    batch_text_parser.set_defaults(func=cmd_batch_text)
    
    # Batch logo watermark
    batch_logo_parser = subparsers.add_parser("batch-logo", help="Batch process logo watermarks")
    setup_batch_args(batch_logo_parser)
    batch_logo_parser.add_argument("--logo", required=True, help="Path to logo/watermark image")
    batch_logo_parser.add_argument("--position", choices=["top-left", "top-right", "bottom-left", 
                                  "bottom-right", "center"], default="bottom-right",
                                  help="Position of watermark")
    batch_logo_parser.add_argument("--scale", type=float, default=0.1, help="Scale factor for logo size")
    batch_logo_parser.add_argument("--opacity", type=float, help="Opacity (0.0-1.0)")
    batch_logo_parser.set_defaults(func=cmd_batch_logo)
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Validate paths
    if hasattr(args, 'input') and not os.path.exists(args.input):
        print(f"❌ Error: Input file '{args.input}' does not exist")
        sys.exit(1)
    
    if hasattr(args, 'input_dir') and not os.path.exists(args.input_dir):
        print(f"❌ Error: Input directory '{args.input_dir}' does not exist")
        sys.exit(1)
    
    if hasattr(args, 'logo') and not os.path.exists(args.logo):
        print(f"❌ Error: Logo file '{args.logo}' does not exist")
        sys.exit(1)
    
    # Create output directory if needed
    if hasattr(args, 'output'):
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
    
    if hasattr(args, 'output_dir') and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Execute command
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\n⚠️ Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
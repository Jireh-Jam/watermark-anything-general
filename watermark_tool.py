#!/usr/bin/env python3
"""
Watermark Anything Tool - A comprehensive watermarking solution
Supports single/multiple watermarks, batch processing, and various output formats
"""

import os
import argparse
import sys
import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
import json
from datetime import datetime
import numpy as np
from torchvision.utils import save_image

# Add the workspace to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from watermark_utils import (
    WatermarkProcessor,
    generate_random_message,
    encode_text_to_binary,
    decode_binary_to_text,
    calculate_metrics,
    visualize_results
)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Watermark Anything Tool - Embed and detect watermarks in images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Embed a random watermark in a single image
  python watermark_tool.py embed -i image.jpg -o watermarked.jpg
  
  # Embed a text message as watermark
  python watermark_tool.py embed -i image.jpg -o watermarked.jpg -m "Hello World"
  
  # Embed watermark with custom mask proportion
  python watermark_tool.py embed -i image.jpg -o watermarked.jpg -p 0.3
  
  # Detect watermark in an image
  python watermark_tool.py detect -i watermarked.jpg
  
  # Batch process multiple images
  python watermark_tool.py batch-embed -i input_dir/ -o output_dir/ -m "Copyright 2024"
  
  # Run interactive GUI
  python watermark_tool.py gui
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Embed command
    embed_parser = subparsers.add_parser('embed', help='Embed watermark in image')
    embed_parser.add_argument('-i', '--input', required=True, help='Input image path')
    embed_parser.add_argument('-o', '--output', required=True, help='Output watermarked image path')
    embed_parser.add_argument('-m', '--message', help='Message to embed (text or binary string)')
    embed_parser.add_argument('-p', '--proportion', type=float, default=1.0, 
                            help='Proportion of image to watermark (0-1, default: 1.0)')
    embed_parser.add_argument('--mask', help='Custom mask image path')
    embed_parser.add_argument('--strength', type=float, default=2.0,
                            help='Watermark strength (higher = more robust, default: 2.0)')
    embed_parser.add_argument('--visualize', action='store_true', help='Save visualization plots')
    
    # Detect command
    detect_parser = subparsers.add_parser('detect', help='Detect watermark in image')
    detect_parser.add_argument('-i', '--input', required=True, help='Input watermarked image path')
    detect_parser.add_argument('-o', '--output', help='Output directory for results')
    detect_parser.add_argument('--decode-text', action='store_true', 
                             help='Try to decode message as text')
    detect_parser.add_argument('--visualize', action='store_true', help='Save visualization plots')
    
    # Batch embed command
    batch_embed_parser = subparsers.add_parser('batch-embed', 
                                              help='Batch embed watermarks in multiple images')
    batch_embed_parser.add_argument('-i', '--input', required=True, help='Input directory')
    batch_embed_parser.add_argument('-o', '--output', required=True, help='Output directory')
    batch_embed_parser.add_argument('-m', '--message', help='Message to embed')
    batch_embed_parser.add_argument('-p', '--proportion', type=float, default=1.0,
                                  help='Proportion of image to watermark')
    batch_embed_parser.add_argument('--strength', type=float, default=2.0,
                                  help='Watermark strength')
    batch_embed_parser.add_argument('--recursive', action='store_true', 
                                  help='Process subdirectories recursively')
    
    # Batch detect command
    batch_detect_parser = subparsers.add_parser('batch-detect',
                                               help='Batch detect watermarks in multiple images')
    batch_detect_parser.add_argument('-i', '--input', required=True, help='Input directory')
    batch_detect_parser.add_argument('-o', '--output', required=True, help='Output directory')
    batch_detect_parser.add_argument('--decode-text', action='store_true',
                                   help='Try to decode messages as text')
    batch_detect_parser.add_argument('--recursive', action='store_true',
                                   help='Process subdirectories recursively')
    
    # Multi-watermark command
    multi_parser = subparsers.add_parser('multi-embed', 
                                        help='Embed multiple watermarks in single image')
    multi_parser.add_argument('-i', '--input', required=True, help='Input image path')
    multi_parser.add_argument('-o', '--output', required=True, help='Output watermarked image path')
    multi_parser.add_argument('-m', '--messages', nargs='+', required=True,
                            help='Messages to embed (space-separated)')
    multi_parser.add_argument('-p', '--proportion', type=float, default=0.1,
                            help='Max proportion per watermark (default: 0.1)')
    multi_parser.add_argument('--strength', type=float, default=2.0,
                            help='Watermark strength')
    multi_parser.add_argument('--visualize', action='store_true', help='Save visualization plots')
    
    # GUI command
    gui_parser = subparsers.add_parser('gui', help='Launch interactive GUI')
    gui_parser.add_argument('--port', type=int, default=7860, help='Port for web interface')
    gui_parser.add_argument('--share', action='store_true', help='Create public link')
    
    # Verify command
    verify_parser = subparsers.add_parser('verify', help='Verify watermark integrity')
    verify_parser.add_argument('-o', '--original', required=True, help='Original image path')
    verify_parser.add_argument('-w', '--watermarked', required=True, help='Watermarked image path')
    verify_parser.add_argument('-m', '--message', help='Expected message to verify')
    
    return parser.parse_args()


def embed_watermark(args, processor):
    """Embed watermark in a single image"""
    print(f"Embedding watermark in {args.input}...")
    
    # Load image
    img = Image.open(args.input).convert("RGB")
    
    # Generate or parse message
    if args.message:
        if args.message.startswith('0b') or all(c in '01' for c in args.message):
            # Binary message
            message = torch.tensor([int(b) for b in args.message.replace('0b', '')]).float()
            if len(message) < 32:
                message = F.pad(message, (0, 32 - len(message)))
            elif len(message) > 32:
                message = message[:32]
        else:
            # Text message
            message = encode_text_to_binary(args.message)
    else:
        # Random message
        message = generate_random_message()
    
    # Load custom mask if provided
    mask = None
    if args.mask:
        mask_img = Image.open(args.mask).convert('L')
        mask = torch.tensor(np.array(mask_img) / 255.0).float().unsqueeze(0).unsqueeze(0)
    
    # Process watermark
    result = processor.embed_watermark(
        img, 
        message,
        mask_proportion=args.proportion,
        custom_mask=mask,
        watermark_strength=args.strength
    )
    
    # Save watermarked image
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(result['watermarked_tensor'], str(output_path))
    
    # Save metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'original_image': args.input,
        'message': message.tolist(),
        'proportion': args.proportion,
        'strength': args.strength,
        'psnr': result['psnr'],
        'ssim': result['ssim']
    }
    
    metadata_path = output_path.with_suffix('.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Watermarked image saved to: {output_path}")
    print(f"  PSNR: {result['psnr']:.2f} dB")
    print(f"  SSIM: {result['ssim']:.4f}")
    
    if args.visualize:
        viz_path = output_path.with_suffix('.viz.png')
        visualize_results(
            result['original_tensor'],
            result['watermarked_tensor'],
            result['mask'],
            None,
            save_path=str(viz_path)
        )
        print(f"✓ Visualization saved to: {viz_path}")
    
    return result


def detect_watermark(args, processor):
    """Detect watermark in a single image"""
    print(f"Detecting watermark in {args.input}...")
    
    # Load image
    img = Image.open(args.input).convert("RGB")
    
    # Detect watermark
    result = processor.detect_watermark(img)
    
    # Print results
    print(f"\n✓ Watermark Detection Results:")
    print(f"  Detected: {'Yes' if result['detected'] else 'No'}")
    
    if result['detected']:
        print(f"  Confidence: {result['confidence']:.2%}")
        print(f"  Bit Accuracy: {result['bit_accuracy']:.2%}")
        print(f"  Message (binary): {result['message_binary']}")
        
        if args.decode_text:
            try:
                text = decode_binary_to_text(result['message_tensor'])
                print(f"  Message (text): {text}")
            except:
                print("  Message (text): [Unable to decode as text]")
    
    # Save results if output directory specified
    if args.output:
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save detection results
        results_path = output_path / f"{Path(args.input).stem}_detection.json"
        with open(results_path, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'input_image': args.input,
                'detected': result['detected'],
                'confidence': float(result['confidence']) if result['detected'] else 0,
                'bit_accuracy': float(result['bit_accuracy']) if result['detected'] else 0,
                'message': result['message_tensor'].tolist() if result['detected'] else None
            }, f, indent=2)
        
        # Save mask prediction
        if result['detected']:
            mask_path = output_path / f"{Path(args.input).stem}_mask.png"
            save_image(result['mask_pred'], str(mask_path))
            print(f"\n✓ Results saved to: {output_path}")
    
    if args.visualize and result['detected']:
        viz_path = Path(args.output) / f"{Path(args.input).stem}_viz.png" if args.output else "detection_viz.png"
        visualize_results(
            None,
            result['watermarked_tensor'],
            None,
            result['mask_pred'],
            save_path=str(viz_path)
        )
        print(f"✓ Visualization saved to: {viz_path}")
    
    return result


def batch_embed(args, processor):
    """Batch embed watermarks in multiple images"""
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all image files
    if args.recursive:
        image_files = list(input_path.rglob("*.jpg")) + list(input_path.rglob("*.png")) + \
                     list(input_path.rglob("*.jpeg")) + list(input_path.rglob("*.bmp"))
    else:
        image_files = list(input_path.glob("*.jpg")) + list(input_path.glob("*.png")) + \
                     list(input_path.glob("*.jpeg")) + list(input_path.glob("*.bmp"))
    
    print(f"Found {len(image_files)} images to process")
    
    # Generate message
    if args.message:
        if all(c in '01' for c in args.message):
            message = torch.tensor([int(b) for b in args.message]).float()
        else:
            message = encode_text_to_binary(args.message)
    else:
        message = generate_random_message()
    
    results = []
    for i, img_path in enumerate(image_files, 1):
        print(f"\nProcessing [{i}/{len(image_files)}]: {img_path.name}")
        
        try:
            # Load and process image
            img = Image.open(img_path).convert("RGB")
            result = processor.embed_watermark(
                img,
                message,
                mask_proportion=args.proportion,
                watermark_strength=args.strength
            )
            
            # Save watermarked image
            relative_path = img_path.relative_to(input_path)
            output_file = output_path / relative_path
            output_file.parent.mkdir(parents=True, exist_ok=True)
            save_image(result['watermarked_tensor'], str(output_file))
            
            results.append({
                'input': str(img_path),
                'output': str(output_file),
                'psnr': result['psnr'],
                'ssim': result['ssim'],
                'success': True
            })
            
            print(f"  ✓ Saved to: {output_file}")
            print(f"    PSNR: {result['psnr']:.2f} dB, SSIM: {result['ssim']:.4f}")
            
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            results.append({
                'input': str(img_path),
                'error': str(e),
                'success': False
            })
    
    # Save batch results summary
    summary_path = output_path / "batch_results.json"
    with open(summary_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'total_images': len(image_files),
            'successful': sum(1 for r in results if r['success']),
            'failed': sum(1 for r in results if not r['success']),
            'message': message.tolist(),
            'results': results
        }, f, indent=2)
    
    print(f"\n✓ Batch processing complete!")
    print(f"  Total: {len(image_files)}")
    print(f"  Success: {sum(1 for r in results if r['success'])}")
    print(f"  Failed: {sum(1 for r in results if not r['success'])}")
    print(f"  Summary saved to: {summary_path}")


def batch_detect(args, processor):
    """Batch detect watermarks in multiple images"""
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all image files
    if args.recursive:
        image_files = list(input_path.rglob("*.jpg")) + list(input_path.rglob("*.png")) + \
                     list(input_path.rglob("*.jpeg")) + list(input_path.rglob("*.bmp"))
    else:
        image_files = list(input_path.glob("*.jpg")) + list(input_path.glob("*.png")) + \
                     list(input_path.glob("*.jpeg")) + list(input_path.glob("*.bmp"))
    
    print(f"Found {len(image_files)} images to analyze")
    
    results = []
    detected_count = 0
    
    for i, img_path in enumerate(image_files, 1):
        print(f"\nAnalyzing [{i}/{len(image_files)}]: {img_path.name}")
        
        try:
            # Load and detect
            img = Image.open(img_path).convert("RGB")
            result = processor.detect_watermark(img)
            
            detection_result = {
                'input': str(img_path),
                'detected': result['detected'],
                'success': True
            }
            
            if result['detected']:
                detected_count += 1
                detection_result.update({
                    'confidence': float(result['confidence']),
                    'bit_accuracy': float(result['bit_accuracy']),
                    'message': result['message_tensor'].tolist()
                })
                
                print(f"  ✓ Watermark detected!")
                print(f"    Confidence: {result['confidence']:.2%}")
                print(f"    Bit Accuracy: {result['bit_accuracy']:.2%}")
                
                if args.decode_text:
                    try:
                        text = decode_binary_to_text(result['message_tensor'])
                        detection_result['decoded_text'] = text
                        print(f"    Decoded text: {text}")
                    except:
                        print("    Decoded text: [Unable to decode]")
                
                # Save mask
                mask_path = output_path / f"{img_path.stem}_mask.png"
                save_image(result['mask_pred'], str(mask_path))
            else:
                print(f"  ✗ No watermark detected")
            
            results.append(detection_result)
            
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            results.append({
                'input': str(img_path),
                'error': str(e),
                'success': False
            })
    
    # Save batch results summary
    summary_path = output_path / "detection_results.json"
    with open(summary_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'total_images': len(image_files),
            'detected': detected_count,
            'not_detected': len(image_files) - detected_count - sum(1 for r in results if not r['success']),
            'failed': sum(1 for r in results if not r['success']),
            'results': results
        }, f, indent=2)
    
    print(f"\n✓ Batch detection complete!")
    print(f"  Total: {len(image_files)}")
    print(f"  Detected: {detected_count}")
    print(f"  Not detected: {len(image_files) - detected_count - sum(1 for r in results if not r['success'])}")
    print(f"  Failed: {sum(1 for r in results if not r['success'])}")
    print(f"  Summary saved to: {summary_path}")


def multi_embed(args, processor):
    """Embed multiple watermarks in a single image"""
    print(f"Embedding {len(args.messages)} watermarks in {args.input}...")
    
    # Load image
    img = Image.open(args.input).convert("RGB")
    
    # Parse messages
    messages = []
    for msg in args.messages:
        if all(c in '01' for c in msg):
            message = torch.tensor([int(b) for b in msg]).float()
        else:
            message = encode_text_to_binary(msg)
        messages.append(message)
    
    # Embed multiple watermarks
    result = processor.embed_multiple_watermarks(
        img,
        messages,
        mask_proportion=args.proportion,
        watermark_strength=args.strength
    )
    
    # Save watermarked image
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(result['watermarked_tensor'], str(output_path))
    
    # Save metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'original_image': args.input,
        'messages': [msg.tolist() for msg in messages],
        'proportion_per_watermark': args.proportion,
        'strength': args.strength,
        'psnr': result['psnr'],
        'ssim': result['ssim']
    }
    
    metadata_path = output_path.with_suffix('.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Multi-watermarked image saved to: {output_path}")
    print(f"  PSNR: {result['psnr']:.2f} dB")
    print(f"  SSIM: {result['ssim']:.4f}")
    
    # Detect the multiple watermarks
    print("\nDetecting embedded watermarks...")
    detection_result = processor.detect_multiple_watermarks(
        Image.open(output_path).convert("RGB")
    )
    
    print(f"✓ Detected {len(detection_result['messages'])} watermarks:")
    for i, (msg, confidence) in enumerate(zip(detection_result['messages'], 
                                              detection_result['confidences'])):
        print(f"  Watermark {i+1}:")
        print(f"    Confidence: {confidence:.2%}")
        print(f"    Message: {msg}")
        if not all(c in '01' for c in args.messages[i]):
            try:
                text = decode_binary_to_text(torch.tensor([int(b) for b in msg]))
                print(f"    Decoded: {text}")
            except:
                pass
    
    if args.visualize:
        viz_path = output_path.with_suffix('.viz.png')
        visualize_results(
            result['original_tensor'],
            result['watermarked_tensor'],
            result['combined_mask'],
            detection_result['mask_pred'] if detection_result['messages'] else None,
            save_path=str(viz_path)
        )
        print(f"✓ Visualization saved to: {viz_path}")


def verify_watermark(args, processor):
    """Verify watermark integrity between original and watermarked images"""
    print(f"Verifying watermark integrity...")
    
    # Load images
    original = Image.open(args.original).convert("RGB")
    watermarked = Image.open(args.watermarked).convert("RGB")
    
    # Calculate metrics
    metrics = calculate_metrics(original, watermarked)
    
    print(f"\n✓ Image Quality Metrics:")
    print(f"  PSNR: {metrics['psnr']:.2f} dB")
    print(f"  SSIM: {metrics['ssim']:.4f}")
    print(f"  MSE: {metrics['mse']:.6f}")
    
    # Detect watermark
    result = processor.detect_watermark(watermarked)
    
    print(f"\n✓ Watermark Detection:")
    print(f"  Detected: {'Yes' if result['detected'] else 'No'}")
    
    if result['detected']:
        print(f"  Confidence: {result['confidence']:.2%}")
        print(f"  Bit Accuracy: {result['bit_accuracy']:.2%}")
        
        if args.message:
            # Verify against expected message
            if all(c in '01' for c in args.message):
                expected = torch.tensor([int(b) for b in args.message]).float()
            else:
                expected = encode_text_to_binary(args.message)
            
            match_accuracy = (result['message_tensor'] == expected).float().mean().item()
            print(f"\n✓ Message Verification:")
            print(f"  Match Accuracy: {match_accuracy:.2%}")
            print(f"  Expected: {args.message}")
            print(f"  Detected: {result['message_binary']}")


def run_gui(args):
    """Launch the Gradio GUI interface"""
    try:
        from watermark_gui import create_interface
        interface = create_interface()
        interface.launch(server_port=args.port, share=args.share)
    except ImportError:
        print("Error: GUI dependencies not installed. Please install gradio:")
        print("  pip install gradio")
        sys.exit(1)


def main():
    """Main entry point"""
    args = parse_arguments()
    
    if not args.command:
        print("Error: No command specified. Use -h for help.")
        sys.exit(1)
    
    # Check if model weights exist
    checkpoint_path = Path("checkpoints/checkpoint.pth")
    if not checkpoint_path.exists():
        print("Model weights not found. Downloading...")
        os.makedirs("checkpoints", exist_ok=True)
        
        # Try to download the model
        import subprocess
        try:
            # Try MIT licensed model first
            subprocess.run([
                "wget", 
                "https://dl.fbaipublicfiles.com/watermark_anything/wam_mit.pth",
                "-O", "checkpoints/checkpoint.pth"
            ], check=True)
            print("✓ Model downloaded successfully")
        except:
            print("Error: Failed to download model weights")
            print("Please download manually from:")
            print("  https://dl.fbaipublicfiles.com/watermark_anything/wam_mit.pth")
            print("And save to: checkpoints/checkpoint.pth")
            sys.exit(1)
    
    # Initialize processor
    try:
        processor = WatermarkProcessor()
    except Exception as e:
        print(f"Error initializing watermark processor: {e}")
        sys.exit(1)
    
    # Execute command
    try:
        if args.command == 'embed':
            embed_watermark(args, processor)
        elif args.command == 'detect':
            detect_watermark(args, processor)
        elif args.command == 'batch-embed':
            batch_embed(args, processor)
        elif args.command == 'batch-detect':
            batch_detect(args, processor)
        elif args.command == 'multi-embed':
            multi_embed(args, processor)
        elif args.command == 'verify':
            verify_watermark(args, processor)
        elif args.command == 'gui':
            run_gui(args)
        else:
            print(f"Unknown command: {args.command}")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
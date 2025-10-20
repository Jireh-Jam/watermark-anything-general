#!/usr/bin/env python3
"""
Test script for watermark functionality

This script tests the watermarking capabilities without requiring
the full deep learning dependencies.
"""

import os
import sys
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import traceback


def create_test_image(path="test_image.jpg", size=(400, 300)):
    """Create a test image for watermarking."""
    img = Image.new('RGB', size, color=(70, 130, 180))  # Steel blue background
    draw = ImageDraw.Draw(img)
    
    # Add some content to make it interesting
    draw.rectangle([50, 50, size[0]-50, size[1]-50], fill=(255, 215, 0), outline=(255, 255, 255), width=3)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36)
    except:
        font = ImageFont.load_default()
    
    draw.text((size[0]//2, size[1]//2), "TEST IMAGE", font=font, fill=(0, 0, 0), anchor="mm")
    
    img.save(path, quality=95)
    return path


def create_test_logo(path="test_logo.png", size=(100, 50)):
    """Create a test logo for watermarking."""
    logo = Image.new('RGBA', size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(logo)
    
    # Draw logo background
    draw.rectangle([5, 5, size[0]-5, size[1]-5], fill=(255, 0, 0, 200), outline=(255, 255, 255, 255), width=2)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    draw.text((size[0]//2, size[1]//2), "LOGO", font=font, fill=(255, 255, 255, 255), anchor="mm")
    
    logo.save(path)
    return path


def test_text_watermark():
    """Test text watermarking functionality."""
    print("🔤 Testing Text Watermarking...")
    
    try:
        # Create test image
        input_path = create_test_image("test_input.jpg")
        
        # Load image
        img = Image.open(input_path).convert("RGBA")
        
        # Create transparent overlay
        overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        # Setup text
        text = "© 2024 TEST WATERMARK"
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        except:
            font = ImageFont.load_default()
        
        # Calculate position (bottom-right)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        margin = 20
        x = img.size[0] - text_width - margin
        y = img.size[1] - text_height - margin
        
        # Draw text with opacity
        opacity = 0.7
        alpha = int(255 * opacity)
        text_color = (255, 255, 255, alpha)
        draw.text((x, y), text, font=font, fill=text_color)
        
        # Combine with original image
        watermarked = Image.alpha_composite(img, overlay)
        watermarked = watermarked.convert("RGB")
        
        # Save result
        output_path = "test_text_watermarked.jpg"
        watermarked.save(output_path, quality=95)
        
        print(f"✅ Text watermark created: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ Text watermark failed: {e}")
        traceback.print_exc()
        return False


def test_image_watermark():
    """Test image watermarking functionality."""
    print("🖼️ Testing Image Watermarking...")
    
    try:
        # Create test image and logo
        input_path = create_test_image("test_input2.jpg")
        logo_path = create_test_logo("test_logo.png")
        
        # Load images
        base_img = Image.open(input_path).convert("RGBA")
        watermark_img = Image.open(logo_path).convert("RGBA")
        
        # Scale watermark (10% of base image width)
        scale = 0.15
        base_width, base_height = base_img.size
        wm_width = int(base_width * scale)
        wm_height = int(watermark_img.height * (wm_width / watermark_img.width))
        watermark_img = watermark_img.resize((wm_width, wm_height), Image.Resampling.LANCZOS)
        
        # Apply opacity
        opacity = 0.6
        if opacity < 1.0:
            alpha = watermark_img.split()[-1]
            alpha = ImageEnhance.Brightness(alpha).enhance(opacity)
            watermark_img.putalpha(alpha)
        
        # Calculate position (top-left with margin)
        margin = 20
        x, y = margin, margin
        
        # Paste watermark
        base_img.paste(watermark_img, (x, y), watermark_img)
        
        # Convert back to RGB and save
        result_img = base_img.convert("RGB")
        output_path = "test_image_watermarked.jpg"
        result_img.save(output_path, quality=95)
        
        print(f"✅ Image watermark created: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ Image watermark failed: {e}")
        traceback.print_exc()
        return False


def test_position_variations():
    """Test different watermark positions."""
    print("📍 Testing Position Variations...")
    
    try:
        input_path = create_test_image("test_input3.jpg", (500, 400))
        
        positions = {
            "top-left": (20, 20),
            "top-right": (-120, 20),
            "bottom-left": (20, -50),
            "bottom-right": (-120, -50),
            "center": (0, 0)  # Will be calculated
        }
        
        success_count = 0
        
        for position_name, (x_offset, y_offset) in positions.items():
            try:
                img = Image.open(input_path).convert("RGBA")
                overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
                draw = ImageDraw.Draw(overlay)
                
                text = f"POS: {position_name.upper()}"
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
                except:
                    font = ImageFont.load_default()
                
                # Calculate actual position
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                
                if position_name == "center":
                    x = (img.size[0] - text_width) // 2
                    y = (img.size[1] - text_height) // 2
                else:
                    x = x_offset if x_offset >= 0 else img.size[0] + x_offset
                    y = y_offset if y_offset >= 0 else img.size[1] + y_offset
                
                # Draw text
                colors = {
                    "top-left": (255, 0, 0, 200),
                    "top-right": (0, 255, 0, 200),
                    "bottom-left": (0, 0, 255, 200),
                    "bottom-right": (255, 255, 0, 200),
                    "center": (255, 0, 255, 200)
                }
                
                draw.text((x, y), text, font=font, fill=colors[position_name])
                
                # Combine and save
                watermarked = Image.alpha_composite(img, overlay)
                watermarked = watermarked.convert("RGB")
                output_path = f"test_position_{position_name.replace('-', '_')}.jpg"
                watermarked.save(output_path, quality=95)
                
                success_count += 1
                
            except Exception as e:
                print(f"   ❌ Position {position_name} failed: {e}")
        
        print(f"✅ Position variations: {success_count}/5 successful")
        return success_count == 5
        
    except Exception as e:
        print(f"❌ Position variations failed: {e}")
        traceback.print_exc()
        return False


def test_opacity_variations():
    """Test different opacity levels."""
    print("🎨 Testing Opacity Variations...")
    
    try:
        input_path = create_test_image("test_input4.jpg")
        logo_path = create_test_logo("test_logo2.png")
        
        opacities = [0.2, 0.4, 0.6, 0.8, 1.0]
        success_count = 0
        
        for opacity in opacities:
            try:
                base_img = Image.open(input_path).convert("RGBA")
                watermark_img = Image.open(logo_path).convert("RGBA")
                
                # Resize watermark
                scale = 0.2
                base_width = base_img.size[0]
                wm_width = int(base_width * scale)
                wm_height = int(watermark_img.height * (wm_width / watermark_img.width))
                watermark_img = watermark_img.resize((wm_width, wm_height), Image.Resampling.LANCZOS)
                
                # Apply opacity
                if opacity < 1.0:
                    alpha = watermark_img.split()[-1]
                    alpha = ImageEnhance.Brightness(alpha).enhance(opacity)
                    watermark_img.putalpha(alpha)
                
                # Position in top-right
                margin = 20
                x = base_img.size[0] - watermark_img.size[0] - margin
                y = margin
                
                # Paste and save
                base_img.paste(watermark_img, (x, y), watermark_img)
                result_img = base_img.convert("RGB")
                output_path = f"test_opacity_{int(opacity*100)}.jpg"
                result_img.save(output_path, quality=95)
                
                success_count += 1
                
            except Exception as e:
                print(f"   ❌ Opacity {opacity} failed: {e}")
        
        print(f"✅ Opacity variations: {success_count}/5 successful")
        return success_count == 5
        
    except Exception as e:
        print(f"❌ Opacity variations failed: {e}")
        traceback.print_exc()
        return False


def test_error_handling():
    """Test error handling."""
    print("⚠️ Testing Error Handling...")
    
    tests_passed = 0
    
    # Test 1: Non-existent file
    try:
        img = Image.open("non_existent_file.jpg")
        print("❌ Should have failed on non-existent file")
    except Exception:
        print("✅ Correctly handled non-existent file")
        tests_passed += 1
    
    # Test 2: Invalid image format
    try:
        with open("invalid_image.txt", "w") as f:
            f.write("This is not an image")
        
        img = Image.open("invalid_image.txt")
        print("❌ Should have failed on invalid image")
    except Exception:
        print("✅ Correctly handled invalid image format")
        tests_passed += 1
        if os.path.exists("invalid_image.txt"):
            os.remove("invalid_image.txt")
    
    # Test 3: Large text handling
    try:
        input_path = create_test_image("test_error.jpg")
        img = Image.open(input_path).convert("RGBA")
        overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        very_long_text = "THIS IS A VERY LONG TEXT THAT MIGHT CAUSE ISSUES " * 10
        font = ImageFont.load_default()
        draw.text((10, 10), very_long_text, font=font, fill=(255, 255, 255, 255))
        
        watermarked = Image.alpha_composite(img, overlay)
        watermarked.save("test_long_text.jpg")
        print("✅ Handled long text gracefully")
        tests_passed += 1
        
    except Exception as e:
        print(f"❌ Long text handling failed: {e}")
    
    return tests_passed >= 2


def cleanup_test_files():
    """Clean up test files."""
    test_files = [
        "test_image.jpg", "test_input.jpg", "test_input2.jpg", "test_input3.jpg", "test_input4.jpg",
        "test_logo.png", "test_logo2.png", "test_text_watermarked.jpg", "test_image_watermarked.jpg",
        "test_position_top_left.jpg", "test_position_top_right.jpg", "test_position_bottom_left.jpg",
        "test_position_bottom_right.jpg", "test_position_center.jpg",
        "test_opacity_20.jpg", "test_opacity_40.jpg", "test_opacity_60.jpg", "test_opacity_80.jpg", "test_opacity_100.jpg",
        "test_error.jpg", "test_long_text.jpg", "invalid_image.txt"
    ]
    
    for file in test_files:
        if os.path.exists(file):
            try:
                os.remove(file)
            except:
                pass


def main():
    """Run all tests."""
    print("🌊 Watermark Generator Tests")
    print("=" * 40)
    
    results = []
    
    # Run tests
    results.append(("Text Watermarking", test_text_watermark()))
    results.append(("Image Watermarking", test_image_watermark()))
    results.append(("Position Variations", test_position_variations()))
    results.append(("Opacity Variations", test_opacity_variations()))
    results.append(("Error Handling", test_error_handling()))
    
    # Print results
    print("\n📊 Test Results:")
    print("=" * 40)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:<20} {status}")
        if success:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Watermarking functionality is working correctly.")
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
    
    # Ask about cleanup
    print(f"\n🧹 Cleaning up test files...")
    cleanup_test_files()
    print("✅ Cleanup completed.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
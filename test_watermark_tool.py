#!/usr/bin/env python3
"""
Test script for the Watermark Anything Tool
Verifies all functionalities work correctly
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import tempfile
import shutil
import json

# Add workspace to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from watermark_utils import (
    WatermarkProcessor,
    generate_random_message,
    encode_text_to_binary,
    decode_binary_to_text,
    calculate_metrics,
    create_custom_mask
)


class TestWatermarkTool:
    """Test suite for watermark tool"""
    
    def __init__(self):
        self.test_dir = Path("test_outputs")
        self.test_dir.mkdir(exist_ok=True)
        self.processor = None
        self.test_image = None
        self.results = {
            "passed": 0,
            "failed": 0,
            "tests": []
        }
    
    def setup(self):
        """Setup test environment"""
        print("Setting up test environment...")
        
        # Check if model exists
        if not Path("checkpoints/checkpoint.pth").exists():
            print("❌ Model weights not found. Please download them first.")
            return False
        
        # Initialize processor
        try:
            self.processor = WatermarkProcessor()
            print("✓ Processor initialized")
        except Exception as e:
            print(f"❌ Failed to initialize processor: {e}")
            return False
        
        # Create test image
        self.test_image = self._create_test_image()
        print("✓ Test image created")
        
        return True
    
    def _create_test_image(self, size=(512, 512)):
        """Create a test image with gradient"""
        width, height = size
        array = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Create gradient
        for y in range(height):
            for x in range(width):
                array[y, x] = [
                    int(255 * x / width),
                    int(255 * y / height),
                    int(255 * (1 - x / width) * (1 - y / height))
                ]
        
        return Image.fromarray(array)
    
    def _record_test(self, name, passed, details=""):
        """Record test result"""
        result = {
            "name": name,
            "passed": passed,
            "details": details
        }
        self.results["tests"].append(result)
        
        if passed:
            self.results["passed"] += 1
            print(f"✓ {name}")
            if details:
                print(f"  {details}")
        else:
            self.results["failed"] += 1
            print(f"❌ {name}")
            if details:
                print(f"  Error: {details}")
    
    def test_random_message_generation(self):
        """Test random message generation"""
        try:
            msg1 = generate_random_message()
            msg2 = generate_random_message()
            
            assert len(msg1) == 32, f"Expected 32 bits, got {len(msg1)}"
            assert len(msg2) == 32, f"Expected 32 bits, got {len(msg2)}"
            assert not torch.equal(msg1, msg2), "Random messages should be different"
            assert all(b in [0, 1] for b in msg1), "Message should contain only 0s and 1s"
            
            self._record_test("Random message generation", True, 
                            f"Generated two different 32-bit messages")
        except Exception as e:
            self._record_test("Random message generation", False, str(e))
    
    def test_text_encoding_decoding(self):
        """Test text to binary encoding and decoding"""
        try:
            test_texts = ["Hi", "Test", "WAM!", "1234"]
            
            for text in test_texts:
                # Encode
                encoded = encode_text_to_binary(text)
                assert len(encoded) == 32, f"Encoded message should be 32 bits"
                
                # Decode
                decoded = decode_binary_to_text(encoded)
                assert decoded.startswith(text), f"Decoded '{decoded}' should start with '{text}'"
            
            self._record_test("Text encoding/decoding", True, 
                            f"Successfully encoded and decoded {len(test_texts)} texts")
        except Exception as e:
            self._record_test("Text encoding/decoding", False, str(e))
    
    def test_single_watermark_embedding(self):
        """Test embedding a single watermark"""
        try:
            # Test with random message
            message = generate_random_message()
            result = self.processor.embed_watermark(
                self.test_image,
                message,
                mask_proportion=1.0,
                watermark_strength=2.0
            )
            
            # Verify results
            assert 'watermarked_image' in result
            assert 'psnr' in result and result['psnr'] > 20
            assert 'ssim' in result and result['ssim'] > 0.8
            
            # Save watermarked image
            wm_path = self.test_dir / "single_watermark.png"
            result['watermarked_image'].save(wm_path)
            
            self._record_test("Single watermark embedding", True,
                            f"PSNR: {result['psnr']:.2f} dB, SSIM: {result['ssim']:.4f}")
        except Exception as e:
            self._record_test("Single watermark embedding", False, str(e))
    
    def test_watermark_detection(self):
        """Test watermark detection"""
        try:
            # First embed a watermark
            message = encode_text_to_binary("TEST")
            embed_result = self.processor.embed_watermark(
                self.test_image,
                message,
                mask_proportion=1.0
            )
            
            # Then detect it
            detect_result = self.processor.detect_watermark(
                embed_result['watermarked_image']
            )
            
            # Verify detection
            assert detect_result['detected'], "Watermark should be detected"
            assert detect_result['confidence'] > 0.5, f"Low confidence: {detect_result['confidence']}"
            assert detect_result['bit_accuracy'] > 0.8, f"Low bit accuracy: {detect_result['bit_accuracy']}"
            
            # Check message accuracy
            detected_msg = detect_result['message_tensor']
            accuracy = (detected_msg == message).float().mean().item()
            
            self._record_test("Watermark detection", True,
                            f"Confidence: {detect_result['confidence']:.2%}, "
                            f"Message accuracy: {accuracy:.2%}")
        except Exception as e:
            self._record_test("Watermark detection", False, str(e))
    
    def test_custom_masks(self):
        """Test different mask types"""
        try:
            mask_configs = [
                ("center", {"size": 0.5}),
                ("corner", {"corner": "top_left", "size": 0.25}),
                ("ring", {"outer_radius": 0.4, "inner_radius": 0.2}),
                ("random", {"coverage": 0.3})
            ]
            
            h, w = self.test_image.size[1], self.test_image.size[0]
            
            for mask_type, kwargs in mask_configs:
                mask = create_custom_mask((h, w), mask_type, **kwargs)
                assert mask.shape == (1, 1, h, w), f"Wrong mask shape for {mask_type}"
                assert 0 <= mask.min() <= mask.max() <= 1, f"Mask values out of range for {mask_type}"
            
            self._record_test("Custom mask creation", True,
                            f"Successfully created {len(mask_configs)} mask types")
        except Exception as e:
            self._record_test("Custom mask creation", False, str(e))
    
    def test_multiple_watermarks(self):
        """Test embedding multiple watermarks"""
        try:
            # Create messages
            messages = [
                encode_text_to_binary("ONE"),
                encode_text_to_binary("TWO"),
                generate_random_message()
            ]
            
            # Embed multiple watermarks
            result = self.processor.embed_multiple_watermarks(
                self.test_image,
                messages,
                mask_proportion=0.2,
                watermark_strength=2.5
            )
            
            # Verify results
            assert 'watermarked_image' in result
            assert result['psnr'] > 20
            
            # Save result
            wm_path = self.test_dir / "multiple_watermarks.png"
            result['watermarked_image'].save(wm_path)
            
            # Detect multiple watermarks
            detect_result = self.processor.detect_multiple_watermarks(
                result['watermarked_image']
            )
            
            self._record_test("Multiple watermarks", True,
                            f"Embedded {len(messages)} watermarks, "
                            f"detected {detect_result['num_watermarks']}")
        except Exception as e:
            self._record_test("Multiple watermarks", False, str(e))
    
    def test_robustness(self):
        """Test watermark robustness to transformations"""
        try:
            # Embed watermark
            message = encode_text_to_binary("ROBUST")
            embed_result = self.processor.embed_watermark(
                self.test_image,
                message,
                watermark_strength=3.0  # Higher strength for robustness
            )
            
            wm_image = embed_result['watermarked_image']
            
            # Test various transformations
            transformations = [
                ("JPEG compression", lambda img: self._jpeg_compress(img, quality=85)),
                ("Resize", lambda img: img.resize((256, 256)).resize((512, 512))),
                ("Slight rotation", lambda img: img.rotate(2, expand=False)),
                ("Brightness", lambda img: Image.eval(img, lambda x: min(255, x + 20))),
                ("Contrast", lambda img: Image.eval(img, lambda x: min(255, int(x * 1.2))))
            ]
            
            results = []
            for name, transform in transformations:
                transformed = transform(wm_image)
                detect_result = self.processor.detect_watermark(transformed)
                
                if detect_result['detected']:
                    detected_msg = detect_result['message_tensor']
                    accuracy = (detected_msg == message).float().mean().item()
                    results.append(f"{name}: {accuracy:.2%}")
                else:
                    results.append(f"{name}: Not detected")
            
            # At least some transformations should preserve the watermark
            detected_count = sum(1 for r in results if "Not detected" not in r)
            success = detected_count >= 3  # At least 3 out of 5
            
            self._record_test("Robustness test", success,
                            f"Detected in {detected_count}/5 transformations: " + 
                            ", ".join(results))
        except Exception as e:
            self._record_test("Robustness test", False, str(e))
    
    def _jpeg_compress(self, image, quality=85):
        """Compress image using JPEG"""
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            image.save(tmp.name, 'JPEG', quality=quality)
            compressed = Image.open(tmp.name).convert('RGB')
        os.unlink(tmp.name)
        return compressed
    
    def test_batch_processing(self):
        """Test batch processing capabilities"""
        try:
            # Create test batch directory
            batch_dir = self.test_dir / "batch_test"
            batch_dir.mkdir(exist_ok=True)
            
            # Create multiple test images
            num_images = 5
            for i in range(num_images):
                img = self._create_test_image((256 + i * 50, 256 + i * 50))
                img.save(batch_dir / f"test_{i}.png")
            
            # Test batch embedding (would use CLI in real scenario)
            embedded_count = 0
            message = encode_text_to_binary("BATCH")
            
            for img_path in batch_dir.glob("*.png"):
                try:
                    img = Image.open(img_path)
                    result = self.processor.embed_watermark(img, message)
                    result['watermarked_image'].save(
                        batch_dir / f"wm_{img_path.name}"
                    )
                    embedded_count += 1
                except:
                    pass
            
            # Test batch detection
            detected_count = 0
            for wm_path in batch_dir.glob("wm_*.png"):
                try:
                    img = Image.open(wm_path)
                    result = self.processor.detect_watermark(img)
                    if result['detected']:
                        detected_count += 1
                except:
                    pass
            
            success = embedded_count == num_images and detected_count == num_images
            
            self._record_test("Batch processing", success,
                            f"Embedded: {embedded_count}/{num_images}, "
                            f"Detected: {detected_count}/{num_images}")
            
            # Cleanup
            shutil.rmtree(batch_dir)
            
        except Exception as e:
            self._record_test("Batch processing", False, str(e))
    
    def test_metrics_calculation(self):
        """Test image quality metrics calculation"""
        try:
            # Create two slightly different images
            img1 = self.test_image
            img2 = Image.eval(img1, lambda x: min(255, x + 5))  # Slight brightness increase
            
            metrics = calculate_metrics(img1, img2)
            
            # Verify metrics are in reasonable ranges
            assert 'psnr' in metrics and 20 < metrics['psnr'] < 50
            assert 'ssim' in metrics and 0.9 < metrics['ssim'] < 1.0
            assert 'mse' in metrics and metrics['mse'] > 0
            
            self._record_test("Metrics calculation", True,
                            f"PSNR: {metrics['psnr']:.2f}, "
                            f"SSIM: {metrics['ssim']:.4f}, "
                            f"MSE: {metrics['mse']:.6f}")
        except Exception as e:
            self._record_test("Metrics calculation", False, str(e))
    
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        try:
            # Test with very small image
            small_img = self._create_test_image((64, 64))
            result = self.processor.embed_watermark(
                small_img,
                generate_random_message()
            )
            assert result['watermarked_image'] is not None
            
            # Test with non-square image
            rect_img = self._create_test_image((512, 256))
            result = self.processor.embed_watermark(
                rect_img,
                generate_random_message()
            )
            assert result['watermarked_image'] is not None
            
            # Test with grayscale image
            gray_img = self.test_image.convert('L').convert('RGB')
            result = self.processor.embed_watermark(
                gray_img,
                generate_random_message()
            )
            assert result['watermarked_image'] is not None
            
            self._record_test("Edge cases", True,
                            "Handled small, rectangular, and grayscale images")
        except Exception as e:
            self._record_test("Edge cases", False, str(e))
    
    def run_all_tests(self):
        """Run all tests"""
        print("\n" + "="*50)
        print("Running Watermark Tool Tests")
        print("="*50 + "\n")
        
        if not self.setup():
            print("\nTest setup failed. Exiting.")
            return False
        
        # Run all test methods
        test_methods = [
            self.test_random_message_generation,
            self.test_text_encoding_decoding,
            self.test_single_watermark_embedding,
            self.test_watermark_detection,
            self.test_custom_masks,
            self.test_multiple_watermarks,
            self.test_robustness,
            self.test_batch_processing,
            self.test_metrics_calculation,
            self.test_edge_cases
        ]
        
        for test_method in test_methods:
            print(f"\n--- {test_method.__name__} ---")
            test_method()
        
        # Print summary
        print("\n" + "="*50)
        print("Test Summary")
        print("="*50)
        print(f"Total tests: {len(self.results['tests'])}")
        print(f"Passed: {self.results['passed']}")
        print(f"Failed: {self.results['failed']}")
        print(f"Success rate: {self.results['passed']/len(self.results['tests'])*100:.1f}%")
        
        # Save results
        results_path = self.test_dir / "test_results.json"
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nDetailed results saved to: {results_path}")
        
        # Save example outputs info
        print(f"\nExample outputs saved in: {self.test_dir}/")
        
        return self.results['failed'] == 0
    
    def cleanup(self):
        """Clean up test outputs"""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
            print("\nTest outputs cleaned up.")


def main():
    """Main test runner"""
    tester = TestWatermarkTool()
    
    try:
        success = tester.run_all_tests()
        
        # Ask if user wants to keep test outputs
        if success:
            response = input("\nAll tests passed! Keep test outputs? (y/n): ")
            if response.lower() != 'y':
                tester.cleanup()
        else:
            print("\nSome tests failed. Test outputs preserved for debugging.")
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user.")
        tester.cleanup()
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
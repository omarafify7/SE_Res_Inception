"""
SE-Res-Inception API Test Client
================================

Simple test script to verify the inference API is working correctly.
Sends a test image to the /predict endpoint and displays the results.

Usage:
    python test_api.py
    python test_api.py --image path/to/your/image.jpg
    python test_api.py --url http://your-server:8000
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import requests
except ImportError:
    print("Error: 'requests' library is required.")
    print("Install it with: pip install requests")
    sys.exit(1)


def test_health_check(base_url: str) -> bool:
    """
    Test the health check endpoint.
    
    Args:
        base_url: Base URL of the API server
        
    Returns:
        True if health check passes, False otherwise
    """
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            print("✓ Health check passed")
            health_data = response.json()
            print(f"  Device: {health_data.get('device', 'unknown')}")
            if 'gpu' in health_data:
                print(f"  GPU: {health_data['gpu'].get('name', 'unknown')}")
            return True
        else:
            print(f"✗ Health check failed with status: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"✗ Connection error: Cannot connect to {base_url}")
        print("  Make sure the server is running!")
        return False
    except Exception as e:
        print(f"✗ Health check error: {e}")
        return False


def test_prediction(base_url: str, image_path: str) -> None:
    """
    Send an image to the prediction endpoint and display results.
    
    Args:
        base_url: Base URL of the API server
        image_path: Path to the test image file
    """
    # Validate image file exists
    image_file = Path(image_path)
    if not image_file.exists():
        print(f"✗ Error: Image file not found: {image_path}")
        sys.exit(1)
    
    if not image_file.is_file():
        print(f"✗ Error: Path is not a file: {image_path}")
        sys.exit(1)
    
    # Determine MIME type
    suffix = image_file.suffix.lower()
    mime_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.bmp': 'image/bmp',
        '.webp': 'image/webp'
    }
    
    mime_type = mime_types.get(suffix, 'image/jpeg')
    
    print(f"\n📤 Sending image: {image_path}")
    print(f"   File size: {image_file.stat().st_size / 1024:.1f} KB")
    
    try:
        # Open and send the image file
        with open(image_path, 'rb') as f:
            files = {'file': (image_file.name, f, mime_type)}
            response = requests.post(
                f"{base_url}/predict",
                files=files,
                timeout=30
            )
        
        # Handle response
        if response.status_code == 200:
            result = response.json()
            
            print("\n" + "=" * 50)
            print("🎯 PREDICTION RESULTS")
            print("=" * 50)
            
            print("\n📊 Top 5 Predictions:")
            print("-" * 40)
            
            for i, pred in enumerate(result['top_5_predictions'], 1):
                class_name = pred['class_name']
                confidence = pred['confidence_percent']
                
                # Create visual bar
                bar_length = int(confidence / 5)  # Scale to max 20 chars
                bar = "█" * bar_length + "░" * (20 - bar_length)
                
                print(f"  {i}. {class_name:<15} {bar} {confidence:>6.2f}%")
            
            print("-" * 40)
            print(f"⏱️  Inference Time: {result['inference_time_ms']:.2f} ms")
            print("=" * 50)
            
            # Pretty print full JSON for debugging
            print("\n📝 Raw JSON Response:")
            print(json.dumps(result, indent=2))
            
        else:
            print(f"\n✗ Prediction failed with status: {response.status_code}")
            print(f"  Response: {response.text}")
            
    except requests.exceptions.Timeout:
        print("✗ Request timed out. The server might be under heavy load.")
    except requests.exceptions.ConnectionError:
        print(f"✗ Connection error: Cannot connect to {base_url}")
    except Exception as e:
        print(f"✗ Request error: {e}")


def create_test_image(output_path: str) -> str:
    """
    Create a simple test image if none exists.
    
    Args:
        output_path: Path to save the test image
        
    Returns:
        Path to the created/existing test image
    """
    try:
        from PIL import Image
        import random
        
        # Create a simple 224x224 RGB image with random colors
        img = Image.new('RGB', (224, 224))
        pixels = img.load()
        
        # Fill with some colored patterns
        for i in range(224):
            for j in range(224):
                r = int(128 + 127 * ((i + j) % 256) / 255)
                g = int(128 + 127 * ((i * 2) % 256) / 255)
                b = int(128 + 127 * ((j * 2) % 256) / 255)
                pixels[i, j] = (r, g, b)
        
        img.save(output_path, 'JPEG', quality=95)
        print(f"✓ Created test image: {output_path}")
        return output_path
        
    except ImportError:
        print("Warning: Pillow not installed, cannot create test image")
        return None


def main():
    """Main entry point for the test script."""
    parser = argparse.ArgumentParser(
        description="Test the SE-Res-Inception inference API"
    )
    parser.add_argument(
        '--url',
        type=str,
        default='http://localhost:8000',
        help='Base URL of the API server (default: http://localhost:8000)'
    )
    parser.add_argument(
        '--image',
        type=str,
        default='test.jpg',
        help='Path to the test image (default: test.jpg)'
    )
    parser.add_argument(
        '--create-test-image',
        action='store_true',
        help='Create a simple test image if it does not exist'
    )
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("SE-Res-Inception API Test Client")
    print("=" * 50)
    print(f"Server URL: {args.url}")
    print(f"Test Image: {args.image}")
    print("=" * 50)
    
    # Health check first
    if not test_health_check(args.url):
        print("\nServer health check failed. Exiting.")
        sys.exit(1)
    
    # Create test image if requested and it doesn't exist
    image_path = args.image
    if args.create_test_image and not Path(image_path).exists():
        created_path = create_test_image(image_path)
        if created_path:
            image_path = created_path
    
    # Check if image exists
    if not Path(image_path).exists():
        print(f"\n✗ Test image not found: {image_path}")
        print("  Please provide an image with --image or use --create-test-image")
        sys.exit(1)
    
    # Run prediction test
    test_prediction(args.url, image_path)
    
    print("\n✓ Test completed successfully!")


if __name__ == "__main__":
    main()

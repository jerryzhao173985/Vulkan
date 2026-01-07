#!/usr/bin/env python3
"""
Download and validate ML models for new architectures
Supports MobileNet V3, EfficientNet, and basic transformer models
"""

import hashlib
import json
import os
import sys
import urllib.request
import urllib.error


class ModelRegistry:
    """Registry of available ML models for download"""

    # Model definitions with download URLs and validation info
    MODELS = {
        # Classification models
        "mobilenet_v3_small": {
            "url": "https://tfhub.dev/google/lite-model/imagenet/mobilenet_v3_small_100_224/classification/5/default/1?lite-format=tflite",
            "filename": "mobilenet_v3_small.tflite",
            "category": "classification",
            "description": "MobileNet V3 Small - Efficient image classification",
            "input_shape": [1, 224, 224, 3],
            "expected_size_mb": 2.4,
            "architecture": "mobilenet_v3"
        },
        "mobilenet_v3_large": {
            "url": "https://tfhub.dev/google/lite-model/imagenet/mobilenet_v3_large_100_224/classification/5/default/1?lite-format=tflite",
            "filename": "mobilenet_v3_large.tflite",
            "category": "classification",
            "description": "MobileNet V3 Large - High accuracy image classification",
            "input_shape": [1, 224, 224, 3],
            "expected_size_mb": 5.4,
            "architecture": "mobilenet_v3"
        },
        "efficientnet_lite0": {
            "url": "https://tfhub.dev/tensorflow/lite-model/efficientnet/lite0/int8/2?lite-format=tflite",
            "filename": "efficientnet_lite0.tflite",
            "category": "classification",
            "description": "EfficientNet-Lite0 - Balanced efficiency and accuracy",
            "input_shape": [1, 224, 224, 3],
            "expected_size_mb": 4.7,
            "architecture": "efficientnet"
        },
        "efficientnet_lite4": {
            "url": "https://tfhub.dev/tensorflow/lite-model/efficientnet/lite4/int8/2?lite-format=tflite",
            "filename": "efficientnet_lite4.tflite",
            "category": "classification",
            "description": "EfficientNet-Lite4 - High accuracy classification",
            "input_shape": [1, 300, 300, 3],
            "expected_size_mb": 12.9,
            "architecture": "efficientnet"
        },
        # Detection models
        "ssd_mobilenet_v2": {
            "url": "https://tfhub.dev/tensorflow/lite-model/ssd_mobilenet_v1/1/metadata/2?lite-format=tflite",
            "filename": "ssd_mobilenet_v2.tflite",
            "category": "detection",
            "description": "SSD MobileNet V2 - Object detection",
            "input_shape": [1, 300, 300, 3],
            "expected_size_mb": 4.0,
            "architecture": "ssd"
        },
        # Transformer-based models (BERT variants)
        "mobilebert": {
            "url": "https://tfhub.dev/tensorflow/lite-model/mobilebert/1/metadata/1?lite-format=tflite",
            "filename": "mobilebert.tflite",
            "category": "transformer",
            "description": "MobileBERT - Efficient transformer for NLP",
            "input_shape": [1, 384],
            "expected_size_mb": 100.0,
            "architecture": "transformer"
        },
        "albert_lite": {
            "url": "https://tfhub.dev/tensorflow/lite-model/albert_lite_base/1/metadata/1?lite-format=tflite",
            "filename": "albert_lite.tflite",
            "category": "transformer",
            "description": "ALBERT Lite - Compact transformer model",
            "input_shape": [1, 512],
            "expected_size_mb": 45.0,
            "architecture": "transformer"
        },
        # Segmentation models
        "deeplabv3": {
            "url": "https://tfhub.dev/tensorflow/lite-model/deeplabv3/1/metadata/2?lite-format=tflite",
            "filename": "deeplabv3.tflite",
            "category": "segmentation",
            "description": "DeepLab V3 - Semantic segmentation",
            "input_shape": [1, 257, 257, 3],
            "expected_size_mb": 2.7,
            "architecture": "deeplabv3"
        }
    }

    @classmethod
    def list_models(cls, category=None):
        """List available models, optionally filtered by category"""
        models = cls.MODELS
        if category:
            models = {k: v for k, v in models.items() if v["category"] == category}
        return models

    @classmethod
    def get_model(cls, name):
        """Get model info by name"""
        return cls.MODELS.get(name)

    @classmethod
    def get_categories(cls):
        """Get all unique model categories"""
        return list(set(m["category"] for m in cls.MODELS.values()))


class ModelDownloader:
    """Download and validate ML models"""

    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.download_results = []

        # Ensure output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def download_model(self, model_name, force=False):
        """Download a single model by name"""
        model_info = ModelRegistry.get_model(model_name)
        if not model_info:
            return {
                "model": model_name,
                "success": False,
                "error": f"Unknown model: {model_name}"
            }

        output_path = os.path.join(self.output_dir, model_info["filename"])

        # Check if already exists
        if os.path.exists(output_path) and not force:
            result = self._validate_model(output_path, model_info)
            if result["valid"]:
                return {
                    "model": model_name,
                    "success": True,
                    "path": output_path,
                    "skipped": True,
                    "message": "Model already exists and is valid"
                }

        print(f"\nDownloading {model_name}...")
        print(f"  URL: {model_info['url']}")
        print(f"  Destination: {output_path}")

        try:
            # Download with progress reporting
            self._download_with_progress(model_info["url"], output_path)

            # Validate downloaded model
            validation = self._validate_model(output_path, model_info)

            result = {
                "model": model_name,
                "success": validation["valid"],
                "path": output_path,
                "size_mb": os.path.getsize(output_path) / 1024 / 1024,
                "validation": validation
            }

            if validation["valid"]:
                print(f"  ✓ Download successful: {result['size_mb']:.2f} MB")
            else:
                print(f"  ✗ Validation failed: {validation.get('error', 'Unknown error')}")

        except urllib.error.URLError as e:
            result = {
                "model": model_name,
                "success": False,
                "error": f"Download failed: {str(e)}"
            }
            print(f"  ✗ Download failed: {e}")
        except Exception as e:
            result = {
                "model": model_name,
                "success": False,
                "error": f"Error: {str(e)}"
            }
            print(f"  ✗ Error: {e}")

        self.download_results.append(result)
        return result

    def _download_with_progress(self, url, output_path):
        """Download file with progress indicator"""
        # Create custom opener with headers
        opener = urllib.request.build_opener()
        opener.addheaders = [
            ('User-Agent', 'ARM-ML-SDK-Vulkan/1.0'),
            ('Accept', '*/*')
        ]
        urllib.request.install_opener(opener)

        # Download to temporary file first
        temp_path = output_path + ".tmp"

        try:
            response = urllib.request.urlopen(url, timeout=60)
            total_size = int(response.headers.get('content-length', 0))

            downloaded = 0
            block_size = 8192

            with open(temp_path, 'wb') as f:
                while True:
                    chunk = response.read(block_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

                    # Progress indicator
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        bar_len = 40
                        filled = int(bar_len * downloaded / total_size)
                        bar = '=' * filled + '-' * (bar_len - filled)
                        sys.stdout.write(f'\r  Progress: [{bar}] {percent:.1f}%')
                        sys.stdout.flush()

            print()  # New line after progress

            # Move to final location
            os.rename(temp_path, output_path)

        finally:
            # Cleanup temp file if it exists
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def _validate_model(self, model_path, model_info):
        """Validate a downloaded model"""
        validation = {"valid": True, "checks": []}

        # Check file exists
        if not os.path.exists(model_path):
            return {"valid": False, "error": "File does not exist"}

        # Check file size
        actual_size_mb = os.path.getsize(model_path) / 1024 / 1024
        expected_size = model_info.get("expected_size_mb", 0)

        # Allow 50% tolerance on size (models may vary)
        if expected_size > 0:
            size_ratio = actual_size_mb / expected_size
            if size_ratio < 0.5 or size_ratio > 1.5:
                validation["checks"].append({
                    "check": "size",
                    "expected_mb": expected_size,
                    "actual_mb": actual_size_mb,
                    "warning": True
                })
            else:
                validation["checks"].append({
                    "check": "size",
                    "passed": True,
                    "actual_mb": actual_size_mb
                })

        # Validate TFLite format
        tflite_check = self._validate_tflite_format(model_path)
        validation["checks"].append(tflite_check)

        if not tflite_check.get("passed", False):
            validation["valid"] = False
            validation["error"] = tflite_check.get("error", "Invalid TFLite format")

        return validation

    def _validate_tflite_format(self, model_path):
        """Validate TFLite model file format"""
        try:
            with open(model_path, 'rb') as f:
                data = f.read(16)  # Read header

                # TFLite uses FlatBuffers format
                # Check for TFLite identifier at offset 4
                if len(data) >= 8:
                    identifier = data[4:8]
                    if identifier == b'TFL3':
                        return {
                            "check": "tflite_format",
                            "passed": True,
                            "version": "TFLite v3"
                        }
                    elif identifier in [b'TFL1', b'TFL2']:
                        return {
                            "check": "tflite_format",
                            "passed": True,
                            "version": f"TFLite v{chr(identifier[3])}"
                        }

                # Check for FlatBuffer signature
                # FlatBuffers don't have a strict signature but we can check structure
                if len(data) >= 4:
                    # First 4 bytes are root offset in FlatBuffers
                    root_offset = int.from_bytes(data[0:4], 'little')
                    if root_offset > 0 and root_offset < os.path.getsize(model_path):
                        return {
                            "check": "tflite_format",
                            "passed": True,
                            "version": "TFLite (FlatBuffer)"
                        }

                return {
                    "check": "tflite_format",
                    "passed": False,
                    "error": "Unknown model format"
                }

        except Exception as e:
            return {
                "check": "tflite_format",
                "passed": False,
                "error": str(e)
            }

    def download_category(self, category, force=False):
        """Download all models in a category"""
        models = ModelRegistry.list_models(category)

        if not models:
            print(f"No models found in category: {category}")
            return []

        print(f"\n=== Downloading {category} models ===")
        print(f"Found {len(models)} models")

        results = []
        for model_name in models:
            result = self.download_model(model_name, force)
            results.append(result)

        return results

    def download_all(self, force=False):
        """Download all available models"""
        print("\n=== Downloading all models ===")
        models = ModelRegistry.list_models()
        print(f"Total models: {len(models)}")

        results = []
        for model_name in models:
            result = self.download_model(model_name, force)
            results.append(result)

        return results

    def generate_report(self, output_file=None):
        """Generate download and validation report"""
        print("\n=== Download Report ===")

        successful = sum(1 for r in self.download_results if r.get("success", False))
        skipped = sum(1 for r in self.download_results if r.get("skipped", False))
        failed = sum(1 for r in self.download_results if not r.get("success", False))

        print(f"Total: {len(self.download_results)}")
        print(f"Successful: {successful}")
        print(f"Skipped (already exists): {skipped}")
        print(f"Failed: {failed}")

        # Print failures
        if failed > 0:
            print("\nFailed downloads:")
            for r in self.download_results:
                if not r.get("success", False):
                    print(f"  - {r['model']}: {r.get('error', 'Unknown error')}")

        # Save report
        if output_file:
            report = {
                "summary": {
                    "total": len(self.download_results),
                    "successful": successful,
                    "skipped": skipped,
                    "failed": failed
                },
                "results": self.download_results
            }
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"\nReport saved to: {output_file}")

        return self.download_results


class ModelValidator:
    """Validate existing models in the models directory"""

    def __init__(self, models_dir):
        self.models_dir = models_dir
        self.validation_results = []

    def validate_all(self):
        """Validate all TFLite models in the directory"""
        print("\n=== Validating Existing Models ===")
        print(f"Directory: {self.models_dir}")

        if not os.path.exists(self.models_dir):
            print("Error: Models directory does not exist")
            return []

        # Find all .tflite files
        tflite_files = [f for f in os.listdir(self.models_dir) if f.endswith('.tflite')]

        if not tflite_files:
            print("No TFLite models found")
            return []

        print(f"Found {len(tflite_files)} TFLite models")

        for filename in tflite_files:
            model_path = os.path.join(self.models_dir, filename)
            result = self._validate_model(model_path)
            self.validation_results.append(result)

            status = "✓" if result["valid"] else "✗"
            print(f"  {status} {filename}: {result['size_mb']:.2f} MB - {result['format']}")

        return self.validation_results

    def _validate_model(self, model_path):
        """Validate a single model file"""
        result = {
            "path": model_path,
            "filename": os.path.basename(model_path),
            "valid": False,
            "size_mb": 0,
            "format": "unknown"
        }

        if not os.path.exists(model_path):
            result["error"] = "File does not exist"
            return result

        result["size_mb"] = os.path.getsize(model_path) / 1024 / 1024

        try:
            with open(model_path, 'rb') as f:
                data = f.read(16)

                if len(data) >= 8:
                    identifier = data[4:8]
                    if identifier == b'TFL3':
                        result["valid"] = True
                        result["format"] = "TFLite v3"
                    elif identifier in [b'TFL1', b'TFL2']:
                        result["valid"] = True
                        result["format"] = f"TFLite v{chr(identifier[3])}"
                    else:
                        # Check for valid FlatBuffer structure
                        root_offset = int.from_bytes(data[0:4], 'little')
                        if root_offset > 0 and root_offset < os.path.getsize(model_path):
                            result["valid"] = True
                            result["format"] = "TFLite (FlatBuffer)"

        except Exception as e:
            result["error"] = str(e)

        return result

    def generate_report(self):
        """Generate validation summary"""
        valid_count = sum(1 for r in self.validation_results if r.get("valid", False))
        total = len(self.validation_results)
        total_size = sum(r.get("size_mb", 0) for r in self.validation_results)

        print(f"\nValidation Summary:")
        print(f"  Valid models: {valid_count}/{total}")
        print(f"  Total size: {total_size:.2f} MB")

        return {
            "valid_count": valid_count,
            "total": total,
            "total_size_mb": total_size,
            "results": self.validation_results
        }


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Download and validate ML models for ARM ML SDK"
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # List command
    list_parser = subparsers.add_parser("list", help="List available models")
    list_parser.add_argument(
        "--category",
        choices=ModelRegistry.get_categories() + [None],
        help="Filter by category"
    )

    # Download command
    download_parser = subparsers.add_parser("download", help="Download models")
    download_parser.add_argument(
        "models",
        nargs="*",
        help="Model names to download (or 'all' for all models)"
    )
    download_parser.add_argument(
        "--category",
        help="Download all models in category"
    )
    download_parser.add_argument(
        "--output-dir",
        default="../models",
        help="Output directory for models"
    )
    download_parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if exists"
    )
    download_parser.add_argument(
        "--report",
        help="Save download report to file"
    )

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate existing models")
    validate_parser.add_argument(
        "--models-dir",
        default="../models",
        help="Directory containing models"
    )

    args = parser.parse_args()

    # Resolve relative paths based on script location
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if args.command == "list":
        print("=== Available Models ===\n")
        models = ModelRegistry.list_models(args.category)

        # Group by category
        categories = {}
        for name, info in models.items():
            cat = info["category"]
            if cat not in categories:
                categories[cat] = []
            categories[cat].append((name, info))

        for category, model_list in sorted(categories.items()):
            print(f"{category.upper()}:")
            for name, info in model_list:
                print(f"  {name}")
                print(f"    Description: {info['description']}")
                print(f"    Architecture: {info['architecture']}")
                print(f"    Input shape: {info['input_shape']}")
                print(f"    Expected size: {info['expected_size_mb']:.1f} MB")
            print()

        print(f"Total: {len(models)} models")
        return 0

    elif args.command == "download":
        output_dir = os.path.join(script_dir, args.output_dir)
        downloader = ModelDownloader(output_dir)

        if args.category:
            downloader.download_category(args.category, args.force)
        elif args.models:
            if "all" in args.models:
                downloader.download_all(args.force)
            else:
                for model_name in args.models:
                    downloader.download_model(model_name, args.force)
        else:
            print("Error: Specify model names, --category, or 'all'")
            return 1

        report_file = None
        if args.report:
            report_file = os.path.join(script_dir, args.report)
        downloader.generate_report(report_file)
        return 0

    elif args.command == "validate":
        models_dir = os.path.join(script_dir, args.models_dir)
        validator = ModelValidator(models_dir)
        validator.validate_all()
        validator.generate_report()
        return 0

    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())

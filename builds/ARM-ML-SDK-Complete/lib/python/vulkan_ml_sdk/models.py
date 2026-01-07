#!/usr/bin/env python3
"""
Model loading, validation, and registry for Vulkan ML SDK.

This module provides model management functionality including:
- Model discovery and enumeration
- TFLite format validation
- Model metadata extraction
- Model loading and caching with version tracking
- Warm-start support for fast model loading
"""

import os
import json
import hashlib
import shutil
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field, asdict


class ModelError(Exception):
    """Base exception for model-related errors."""
    pass


class ModelNotFoundError(ModelError):
    """Raised when a requested model cannot be found."""
    pass


class ModelValidationError(ModelError):
    """Raised when model validation fails."""
    pass


class ModelInfo:
    """
    Information about a loaded model.

    Attributes:
        name: Model name (without extension).
        path: Full path to the model file.
        size: File size in bytes.
        format: Model format (e.g., 'tflite').
        metadata: Additional model metadata.
    """

    def __init__(self, path: Path):
        """
        Initialize ModelInfo from a model file path.

        Args:
            path: Path to the model file.
        """
        self.path = path
        self.name = path.stem
        self.size = path.stat().st_size
        self.format = self._detect_format()
        self.metadata = self._extract_metadata()

    def _detect_format(self) -> str:
        """Detect model format from file extension and magic bytes."""
        extension = self.path.suffix.lower()
        if extension == ".tflite":
            return "tflite"
        elif extension == ".onnx":
            return "onnx"
        elif extension == ".pb":
            return "tensorflow"
        else:
            return "unknown"

    def _extract_metadata(self) -> Dict[str, Any]:
        """Extract metadata from model file."""
        metadata = {
            "size_mb": round(self.size / 1024 / 1024, 2),
            "extension": self.path.suffix,
        }

        # Attempt to extract TFLite-specific metadata
        if self.format == "tflite":
            metadata.update(self._extract_tflite_metadata())

        return metadata

    def _extract_tflite_metadata(self) -> Dict[str, Any]:
        """Extract TFLite-specific metadata."""
        tflite_meta = {
            "version": None,
            "valid": False,
        }

        try:
            with open(self.path, 'rb') as f:
                data = f.read(8)
                # TFLite uses FlatBuffers format with identifier at bytes 4-8
                if len(data) >= 8:
                    identifier = data[4:8]
                    if identifier == b'TFL3':
                        tflite_meta["version"] = "3"
                        tflite_meta["valid"] = True
                    elif identifier.startswith(b'TFL'):
                        tflite_meta["version"] = identifier[3:].decode('ascii', errors='ignore')
                        tflite_meta["valid"] = True
        except Exception:
            pass

        return tflite_meta

    @property
    def is_valid(self) -> bool:
        """Check if model passed validation."""
        return self.metadata.get("valid", False)

    @property
    def size_mb(self) -> float:
        """Get model size in megabytes."""
        return self.metadata.get("size_mb", 0.0)

    def to_dict(self) -> Dict[str, Any]:
        """Convert model info to dictionary."""
        return {
            "name": self.name,
            "path": str(self.path),
            "size": self.size,
            "size_mb": self.size_mb,
            "format": self.format,
            "valid": self.is_valid,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        return f"ModelInfo(name={self.name!r}, format={self.format!r}, size_mb={self.size_mb})"


class ModelRegistry:
    """
    Registry for discovering and managing ML models.

    Provides model discovery, validation, and loading functionality
    for the Vulkan ML SDK.

    Example:
        registry = ModelRegistry("/path/to/sdk")
        models = registry.list_models()
        model_info = registry.get_model("mobilenet_v2")
    """

    # Supported model formats and their extensions
    SUPPORTED_FORMATS = {
        ".tflite": "tflite",
        ".onnx": "onnx",
        ".pb": "tensorflow",
    }

    def __init__(self, sdk_root: Union[str, Path]):
        """
        Initialize the ModelRegistry.

        Args:
            sdk_root: Path to the SDK root directory.
        """
        self._sdk_root = Path(sdk_root).resolve()
        self._models_dir = self._sdk_root / "models"
        self._cache: Dict[str, ModelInfo] = {}
        self._scan_models()

    def _scan_models(self) -> None:
        """Scan the models directory and populate the cache."""
        self._cache.clear()

        if not self._models_dir.exists():
            return

        for model_file in self._models_dir.iterdir():
            if model_file.is_file() and model_file.suffix.lower() in self.SUPPORTED_FORMATS:
                try:
                    model_info = ModelInfo(model_file)
                    self._cache[model_info.name] = model_info
                except Exception:
                    # Skip models that fail to load
                    pass

    @property
    def sdk_root(self) -> Path:
        """Path to the SDK root directory."""
        return self._sdk_root

    @property
    def models_dir(self) -> Path:
        """Path to the models directory."""
        return self._models_dir

    def list_models(self) -> List[str]:
        """
        List all available model names.

        Returns:
            Sorted list of model names (without extensions).
        """
        return sorted(self._cache.keys())

    def list_models_by_format(self, format_type: str) -> List[str]:
        """
        List models filtered by format.

        Args:
            format_type: Format to filter by (e.g., 'tflite').

        Returns:
            List of model names matching the format.
        """
        return sorted([
            name for name, info in self._cache.items()
            if info.format == format_type
        ])

    def get_model(self, name: str) -> ModelInfo:
        """
        Get model information by name.

        Args:
            name: Model name (with or without extension).

        Returns:
            ModelInfo object for the model.

        Raises:
            ModelNotFoundError: If model is not found.
        """
        # Strip extension if provided
        clean_name = Path(name).stem

        if clean_name in self._cache:
            return self._cache[clean_name]

        raise ModelNotFoundError(f"Model not found: {name}")

    def get_model_path(self, name: str) -> Path:
        """
        Get the full path to a model file.

        Args:
            name: Model name (with or without extension).

        Returns:
            Path to the model file.

        Raises:
            ModelNotFoundError: If model is not found.
        """
        return self.get_model(name).path

    def has_model(self, name: str) -> bool:
        """
        Check if a model exists in the registry.

        Args:
            name: Model name to check.

        Returns:
            True if model exists, False otherwise.
        """
        clean_name = Path(name).stem
        return clean_name in self._cache

    def validate_model(self, name: str) -> Dict[str, Any]:
        """
        Validate a model and return validation results.

        Args:
            name: Model name to validate.

        Returns:
            Dict with validation results.

        Raises:
            ModelNotFoundError: If model is not found.
        """
        model_info = self.get_model(name)

        results = {
            "name": model_info.name,
            "path": str(model_info.path),
            "exists": model_info.path.exists(),
            "readable": os.access(model_info.path, os.R_OK),
            "format_valid": model_info.is_valid,
            "size_ok": model_info.size > 0,
            "format": model_info.format,
        }

        results["valid"] = all([
            results["exists"],
            results["readable"],
            results["format_valid"],
            results["size_ok"],
        ])

        return results

    def get_all_models(self) -> List[ModelInfo]:
        """
        Get information for all models.

        Returns:
            List of ModelInfo objects for all registered models.
        """
        return [self._cache[name] for name in self.list_models()]

    def get_total_size(self) -> int:
        """
        Get total size of all models in bytes.

        Returns:
            Total size in bytes.
        """
        return sum(info.size for info in self._cache.values())

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the model registry.

        Returns:
            Dict with registry statistics.
        """
        models = self.get_all_models()

        format_counts = {}
        for model in models:
            fmt = model.format
            format_counts[fmt] = format_counts.get(fmt, 0) + 1

        return {
            "models_dir": str(self._models_dir),
            "total_models": len(models),
            "total_size_mb": round(self.get_total_size() / 1024 / 1024, 2),
            "formats": format_counts,
            "models": [m.to_dict() for m in models],
        }

    def refresh(self) -> None:
        """Rescan the models directory to update the cache."""
        self._scan_models()

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, name: str) -> bool:
        return self.has_model(name)

    def __iter__(self):
        return iter(self.list_models())

    def __repr__(self) -> str:
        return f"ModelRegistry(models_dir={self._models_dir!r}, count={len(self)})"


def load_model(sdk_root: Union[str, Path], model_name: str) -> ModelInfo:
    """
    Convenience function to load a model.

    Args:
        sdk_root: Path to SDK root directory.
        model_name: Name of the model to load.

    Returns:
        ModelInfo for the requested model.
    """
    registry = ModelRegistry(sdk_root)
    return registry.get_model(model_name)


def validate_tflite(model_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Validate a TFLite model file.

    Args:
        model_path: Path to the TFLite model file.

    Returns:
        Dict with validation results.

    Raises:
        ModelValidationError: If file does not exist.
    """
    path = Path(model_path)

    if not path.exists():
        raise ModelValidationError(f"Model file not found: {model_path}")

    results = {
        "path": str(path),
        "exists": True,
        "size": path.stat().st_size,
        "size_mb": round(path.stat().st_size / 1024 / 1024, 2),
        "valid_format": False,
        "version": None,
        "errors": [],
    }

    # Check file extension
    if path.suffix.lower() != ".tflite":
        results["errors"].append(f"Expected .tflite extension, got {path.suffix}")

    # Check TFLite magic bytes
    try:
        with open(path, 'rb') as f:
            data = f.read(8)
            if len(data) >= 8:
                identifier = data[4:8]
                if identifier == b'TFL3':
                    results["valid_format"] = True
                    results["version"] = "3"
                elif identifier.startswith(b'TFL'):
                    results["valid_format"] = True
                    results["version"] = identifier[3:].decode('ascii', errors='ignore')
                else:
                    results["errors"].append("Invalid TFLite identifier in file header")
            else:
                results["errors"].append("File too small to be valid TFLite model")
    except IOError as e:
        results["errors"].append(f"Failed to read file: {e}")

    results["valid"] = results["valid_format"] and len(results["errors"]) == 0

    return results


# SDK version for cache compatibility checking
_SDK_CACHE_VERSION = "1.0.0"


@dataclass
class CacheEntry:
    """
    Represents a cached model entry.

    Attributes:
        model_name: Name of the cached model.
        original_path: Original path to the model file.
        cached_path: Path to the cached copy.
        file_hash: SHA-256 hash of the model file.
        size: File size in bytes.
        sdk_version: SDK version that created the cache entry.
        created_at: Unix timestamp of cache creation.
        last_accessed: Unix timestamp of last access.
        access_count: Number of times this entry has been accessed.
        metadata: Additional model metadata.
    """
    model_name: str
    original_path: str
    cached_path: str
    file_hash: str
    size: int
    sdk_version: str
    created_at: float
    last_accessed: float
    access_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert cache entry to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CacheEntry":
        """Create cache entry from dictionary."""
        return cls(**data)


class CacheCorruptionError(ModelError):
    """Raised when cache integrity check fails."""
    pass


class ModelCache:
    """
    Model caching with version tracking and warm-start support.

    Provides efficient model caching with LRU eviction, integrity validation,
    and warm-start functionality for fast model loading.

    Features:
    - Persistent disk cache with version tracking
    - SHA-256 integrity validation
    - LRU eviction policy with configurable max size
    - Warm-start preloading for frequently used models
    - Thread-safe operations

    Example:
        cache = ModelCache()

        # Cache a model
        cache.put("mobilenet_v2", "/path/to/model.tflite")

        # Get cached model path (fast warm-start)
        cached_path = cache.get("mobilenet_v2")

        # Preload models for warm-start
        cache.warm_start(["mobilenet_v2", "style_transfer"])
    """

    # Default cache directory
    DEFAULT_CACHE_DIR = Path.home() / ".vulkan_ml_sdk" / "cache"

    # Default maximum cache size (500 MB)
    DEFAULT_MAX_SIZE_MB = 500

    # Manifest file name
    MANIFEST_FILE = "cache_manifest.json"

    def __init__(
        self,
        cache_dir: Optional[Union[str, Path]] = None,
        max_size_mb: int = DEFAULT_MAX_SIZE_MB,
        auto_evict: bool = True
    ):
        """
        Initialize the ModelCache.

        Args:
            cache_dir: Directory for cached models. Defaults to ~/.vulkan_ml_sdk/cache/
            max_size_mb: Maximum cache size in megabytes. Defaults to 500MB.
            auto_evict: If True, automatically evict old entries when cache is full.
        """
        self._cache_dir = Path(cache_dir) if cache_dir else self.DEFAULT_CACHE_DIR
        self._max_size_bytes = max_size_mb * 1024 * 1024
        self._auto_evict = auto_evict
        self._lock = threading.RLock()
        self._entries: Dict[str, CacheEntry] = {}
        self._warm_models: Dict[str, bytes] = {}  # In-memory warm cache

        # Ensure cache directory exists
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        # Load existing manifest
        self._load_manifest()

    @property
    def cache_dir(self) -> Path:
        """Path to the cache directory."""
        return self._cache_dir

    @property
    def max_size_bytes(self) -> int:
        """Maximum cache size in bytes."""
        return self._max_size_bytes

    @property
    def current_size(self) -> int:
        """Current cache size in bytes."""
        with self._lock:
            return sum(entry.size for entry in self._entries.values())

    @property
    def entry_count(self) -> int:
        """Number of cached entries."""
        with self._lock:
            return len(self._entries)

    def _manifest_path(self) -> Path:
        """Path to the cache manifest file."""
        return self._cache_dir / self.MANIFEST_FILE

    def _load_manifest(self) -> None:
        """Load cache manifest from disk."""
        manifest_path = self._manifest_path()
        if not manifest_path.exists():
            return

        try:
            with open(manifest_path, 'r') as f:
                data = json.load(f)

            # Validate manifest version
            manifest_version = data.get("sdk_version", "")
            if not manifest_version.startswith(_SDK_CACHE_VERSION.split('.')[0]):
                # Major version mismatch - invalidate cache
                self._clear_all()
                return

            # Load entries
            entries = data.get("entries", {})
            for name, entry_data in entries.items():
                try:
                    entry = CacheEntry.from_dict(entry_data)
                    # Verify cached file exists
                    if Path(entry.cached_path).exists():
                        self._entries[name] = entry
                except (KeyError, TypeError):
                    # Skip corrupted entries
                    pass

        except (json.JSONDecodeError, IOError):
            # Corrupted manifest - start fresh
            self._clear_all()

    def _save_manifest(self) -> None:
        """Save cache manifest to disk."""
        manifest_path = self._manifest_path()

        manifest = {
            "sdk_version": _SDK_CACHE_VERSION,
            "created_at": time.time(),
            "entries": {
                name: entry.to_dict()
                for name, entry in self._entries.items()
            }
        }

        try:
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
        except IOError:
            pass  # Non-critical failure

    def _compute_hash(self, file_path: Path) -> str:
        """Compute SHA-256 hash of a file."""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _cached_file_path(self, model_name: str, file_hash: str) -> Path:
        """Generate cached file path for a model."""
        # Use hash prefix for subdirectory to avoid too many files in one dir
        subdir = file_hash[:2]
        return self._cache_dir / subdir / f"{model_name}_{file_hash[:12]}.tflite"

    def _evict_lru(self, required_space: int) -> bool:
        """
        Evict least recently used entries to free space.

        Args:
            required_space: Space needed in bytes.

        Returns:
            True if enough space was freed, False otherwise.
        """
        with self._lock:
            # Sort entries by last_accessed (oldest first)
            sorted_entries = sorted(
                self._entries.items(),
                key=lambda x: x[1].last_accessed
            )

            freed = 0
            entries_to_remove = []

            for name, entry in sorted_entries:
                if self.current_size - freed + required_space <= self._max_size_bytes:
                    break
                entries_to_remove.append(name)
                freed += entry.size

            # Remove evicted entries
            for name in entries_to_remove:
                self._remove_entry(name)

            return self.current_size + required_space <= self._max_size_bytes

    def _remove_entry(self, model_name: str) -> None:
        """Remove a cache entry and its file."""
        if model_name not in self._entries:
            return

        entry = self._entries[model_name]

        # Remove cached file
        cached_path = Path(entry.cached_path)
        if cached_path.exists():
            try:
                cached_path.unlink()
                # Remove empty parent directory
                if cached_path.parent.exists() and not any(cached_path.parent.iterdir()):
                    cached_path.parent.rmdir()
            except IOError:
                pass

        # Remove from memory cache
        if model_name in self._warm_models:
            del self._warm_models[model_name]

        # Remove entry
        del self._entries[model_name]

    def _clear_all(self) -> None:
        """Clear all cache entries and files."""
        with self._lock:
            # Remove all cached files
            for entry in self._entries.values():
                cached_path = Path(entry.cached_path)
                if cached_path.exists():
                    try:
                        cached_path.unlink()
                    except IOError:
                        pass

            self._entries.clear()
            self._warm_models.clear()
            self._save_manifest()

    def put(
        self,
        model_name: str,
        model_path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None
    ) -> CacheEntry:
        """
        Add a model to the cache.

        Args:
            model_name: Name to identify the cached model.
            model_path: Path to the source model file.
            metadata: Optional additional metadata.

        Returns:
            CacheEntry for the cached model.

        Raises:
            ModelNotFoundError: If source file doesn't exist.
            ModelError: If caching fails.
        """
        model_path = Path(model_path)

        if not model_path.exists():
            raise ModelNotFoundError(f"Model file not found: {model_path}")

        with self._lock:
            # Compute file hash
            file_hash = self._compute_hash(model_path)
            file_size = model_path.stat().st_size

            # Check if already cached with same hash
            if model_name in self._entries:
                existing = self._entries[model_name]
                if existing.file_hash == file_hash:
                    # Update access time and return existing
                    existing.last_accessed = time.time()
                    existing.access_count += 1
                    self._save_manifest()
                    return existing
                else:
                    # Remove old version
                    self._remove_entry(model_name)

            # Check space and evict if needed
            if self.current_size + file_size > self._max_size_bytes:
                if self._auto_evict:
                    if not self._evict_lru(file_size):
                        raise ModelError(
                            f"Not enough cache space for model ({file_size} bytes). "
                            f"Current: {self.current_size}, Max: {self._max_size_bytes}"
                        )
                else:
                    raise ModelError("Cache is full and auto-eviction is disabled")

            # Copy file to cache
            cached_path = self._cached_file_path(model_name, file_hash)
            cached_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                shutil.copy2(model_path, cached_path)
            except IOError as e:
                raise ModelError(f"Failed to cache model: {e}")

            # Create entry
            now = time.time()
            entry = CacheEntry(
                model_name=model_name,
                original_path=str(model_path),
                cached_path=str(cached_path),
                file_hash=file_hash,
                size=file_size,
                sdk_version=_SDK_CACHE_VERSION,
                created_at=now,
                last_accessed=now,
                access_count=1,
                metadata=metadata or {}
            )

            self._entries[model_name] = entry
            self._save_manifest()

            return entry

    def get(self, model_name: str, verify_integrity: bool = False) -> Optional[Path]:
        """
        Get the cached path for a model.

        Args:
            model_name: Name of the cached model.
            verify_integrity: If True, verify file hash before returning.

        Returns:
            Path to cached model file, or None if not cached.

        Raises:
            CacheCorruptionError: If integrity check fails (when verify_integrity=True).
        """
        with self._lock:
            if model_name not in self._entries:
                return None

            entry = self._entries[model_name]
            cached_path = Path(entry.cached_path)

            # Check file exists
            if not cached_path.exists():
                self._remove_entry(model_name)
                self._save_manifest()
                return None

            # Verify integrity if requested
            if verify_integrity:
                current_hash = self._compute_hash(cached_path)
                if current_hash != entry.file_hash:
                    self._remove_entry(model_name)
                    self._save_manifest()
                    raise CacheCorruptionError(
                        f"Cache integrity check failed for {model_name}"
                    )

            # Update access time
            entry.last_accessed = time.time()
            entry.access_count += 1
            self._save_manifest()

            return cached_path

    def get_entry(self, model_name: str) -> Optional[CacheEntry]:
        """
        Get the cache entry for a model.

        Args:
            model_name: Name of the cached model.

        Returns:
            CacheEntry or None if not cached.
        """
        with self._lock:
            return self._entries.get(model_name)

    def has(self, model_name: str) -> bool:
        """
        Check if a model is cached.

        Args:
            model_name: Name of the model.

        Returns:
            True if model is cached, False otherwise.
        """
        with self._lock:
            if model_name not in self._entries:
                return False
            # Also verify file exists
            return Path(self._entries[model_name].cached_path).exists()

    def remove(self, model_name: str) -> bool:
        """
        Remove a model from the cache.

        Args:
            model_name: Name of the model to remove.

        Returns:
            True if model was removed, False if not found.
        """
        with self._lock:
            if model_name not in self._entries:
                return False
            self._remove_entry(model_name)
            self._save_manifest()
            return True

    def clear(self) -> None:
        """Clear all cached models."""
        with self._lock:
            self._clear_all()

    def warm_start(
        self,
        model_names: List[str],
        progress_callback: Optional[Callable[[str, int, int], None]] = None
    ) -> Dict[str, bool]:
        """
        Preload models into memory for fast warm-start access.

        Args:
            model_names: List of model names to preload.
            progress_callback: Optional callback(model_name, current, total).

        Returns:
            Dict mapping model names to success status.
        """
        results = {}
        total = len(model_names)

        for i, name in enumerate(model_names):
            if progress_callback:
                progress_callback(name, i + 1, total)

            cached_path = self.get(name)
            if cached_path and cached_path.exists():
                try:
                    # Load into memory cache
                    with open(cached_path, 'rb') as f:
                        self._warm_models[name] = f.read()
                    results[name] = True
                except IOError:
                    results[name] = False
            else:
                results[name] = False

        return results

    def get_warm(self, model_name: str) -> Optional[bytes]:
        """
        Get model data from warm cache (memory).

        Args:
            model_name: Name of the model.

        Returns:
            Model bytes if in warm cache, None otherwise.
        """
        return self._warm_models.get(model_name)

    def is_warm(self, model_name: str) -> bool:
        """
        Check if a model is in the warm cache.

        Args:
            model_name: Name of the model.

        Returns:
            True if model is in warm cache, False otherwise.
        """
        return model_name in self._warm_models

    def evict_warm(self, model_name: str) -> bool:
        """
        Remove a model from warm cache (memory only).

        Args:
            model_name: Name of the model.

        Returns:
            True if model was evicted, False if not in warm cache.
        """
        if model_name in self._warm_models:
            del self._warm_models[model_name]
            return True
        return False

    def clear_warm(self) -> None:
        """Clear all models from warm cache (memory only)."""
        self._warm_models.clear()

    def list_models(self) -> List[str]:
        """
        List all cached model names.

        Returns:
            Sorted list of cached model names.
        """
        with self._lock:
            return sorted(self._entries.keys())

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dict with cache statistics.
        """
        with self._lock:
            entries_list = list(self._entries.values())

            if entries_list:
                oldest = min(e.created_at for e in entries_list)
                newest = max(e.created_at for e in entries_list)
                total_accesses = sum(e.access_count for e in entries_list)
            else:
                oldest = newest = total_accesses = 0

            return {
                "cache_dir": str(self._cache_dir),
                "max_size_mb": self._max_size_bytes / (1024 * 1024),
                "current_size_mb": self.current_size / (1024 * 1024),
                "usage_percent": (self.current_size / self._max_size_bytes * 100)
                    if self._max_size_bytes > 0 else 0,
                "entry_count": len(self._entries),
                "warm_count": len(self._warm_models),
                "warm_size_mb": sum(len(d) for d in self._warm_models.values()) / (1024 * 1024),
                "total_accesses": total_accesses,
                "oldest_entry": oldest,
                "newest_entry": newest,
                "sdk_version": _SDK_CACHE_VERSION,
            }

    def validate_integrity(self) -> Dict[str, Any]:
        """
        Validate integrity of all cached models.

        Returns:
            Dict with validation results for each model.
        """
        results = {"valid": [], "corrupted": [], "missing": []}

        with self._lock:
            for name, entry in list(self._entries.items()):
                cached_path = Path(entry.cached_path)

                if not cached_path.exists():
                    results["missing"].append(name)
                    self._remove_entry(name)
                    continue

                try:
                    current_hash = self._compute_hash(cached_path)
                    if current_hash == entry.file_hash:
                        results["valid"].append(name)
                    else:
                        results["corrupted"].append(name)
                        self._remove_entry(name)
                except IOError:
                    results["corrupted"].append(name)
                    self._remove_entry(name)

            self._save_manifest()

        results["summary"] = {
            "total": len(results["valid"]) + len(results["corrupted"]) + len(results["missing"]),
            "valid_count": len(results["valid"]),
            "corrupted_count": len(results["corrupted"]),
            "missing_count": len(results["missing"]),
            "all_valid": len(results["corrupted"]) == 0 and len(results["missing"]) == 0
        }

        return results

    def __len__(self) -> int:
        return self.entry_count

    def __contains__(self, model_name: str) -> bool:
        return self.has(model_name)

    def __repr__(self) -> str:
        return f"ModelCache(cache_dir={self._cache_dir!r}, entries={self.entry_count})"

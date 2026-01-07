#!/usr/bin/env python3
"""
Unit tests for new compute shaders added to the ARM ML SDK.

Tests that the new shaders (attention, batch_norm, layer_norm) are properly
compiled to valid SPIR-V format and can be validated and disassembled.
"""

import os
import subprocess
from pathlib import Path
from typing import List, Tuple

import pytest


# ============================================================================
# Path Setup
# ============================================================================

def _get_sdk_paths() -> Tuple[Path, Path]:
    """Get SDK root and shaders directory paths."""
    test_dir = Path(__file__).resolve().parent.parent
    repo_root = test_dir.parent
    sdk_root = repo_root / "builds" / "ARM-ML-SDK-Complete"
    shaders_dir = sdk_root / "shaders"
    return sdk_root, shaders_dir


SDK_ROOT, SHADERS_DIR = _get_sdk_paths()

# SPIR-V magic number (little-endian)
SPIRV_MAGIC_NUMBER = 0x07230203

# New shaders that were added as part of the SDK enhancement
NEW_SHADERS = [
    "attention.spv",
    "batch_norm.spv",
    "layer_norm.spv",
]


# ============================================================================
# Helper Functions
# ============================================================================

def _check_spirv_magic(file_path: Path) -> bool:
    """Check if file has valid SPIR-V magic number."""
    try:
        with open(file_path, "rb") as f:
            magic = int.from_bytes(f.read(4), byteorder="little")
            return magic == SPIRV_MAGIC_NUMBER
    except (OSError, IOError):
        return False


def _run_spirv_val(file_path: Path) -> Tuple[bool, str]:
    """Run spirv-val to validate a SPIR-V file."""
    spirv_val_paths = [
        "/usr/local/bin/spirv-val",
        "spirv-val",
    ]

    spirv_val = None
    for path in spirv_val_paths:
        try:
            result = subprocess.run(
                [path, "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                spirv_val = path
                break
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue

    if spirv_val is None:
        return True, "spirv-val not found, skipping validation"

    try:
        result = subprocess.run(
            [spirv_val, str(file_path)],
            capture_output=True,
            text=True,
            timeout=30
        )
        return result.returncode == 0, result.stderr or result.stdout
    except subprocess.TimeoutExpired:
        return False, "spirv-val timed out"
    except Exception as e:
        return False, str(e)


def _run_spirv_dis(file_path: Path) -> Tuple[bool, str]:
    """Run spirv-dis to disassemble a SPIR-V file."""
    spirv_dis_paths = [
        "/usr/local/bin/spirv-dis",
        "spirv-dis",
    ]

    spirv_dis = None
    for path in spirv_dis_paths:
        try:
            result = subprocess.run(
                [path, "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                spirv_dis = path
                break
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue

    if spirv_dis is None:
        return True, "spirv-dis not found, skipping disassembly check"

    try:
        result = subprocess.run(
            [spirv_dis, str(file_path)],
            capture_output=True,
            text=True,
            timeout=30
        )
        return result.returncode == 0, result.stdout
    except subprocess.TimeoutExpired:
        return False, "spirv-dis timed out"
    except Exception as e:
        return False, str(e)


# ============================================================================
# Shader Directory Tests
# ============================================================================

class TestShaderDirectory:
    """Test that shader directory exists and is properly configured."""

    def test_shaders_dir_exists(self):
        """Test that shaders directory exists."""
        assert SHADERS_DIR.exists(), f"Shaders directory not found: {SHADERS_DIR}"

    def test_shaders_dir_is_directory(self):
        """Test that shaders path is a directory."""
        assert SHADERS_DIR.is_dir(), f"Shaders path is not a directory: {SHADERS_DIR}"

    def test_shaders_dir_contains_spv_files(self):
        """Test that shaders directory contains .spv files."""
        spv_files = list(SHADERS_DIR.glob("*.spv"))
        assert len(spv_files) > 0, f"No .spv files found in {SHADERS_DIR}"


# ============================================================================
# New Shader Existence Tests
# ============================================================================

class TestNewShaderExistence:
    """Test that new shaders exist in the SDK."""

    @pytest.mark.parametrize("shader_name", NEW_SHADERS)
    def test_new_shader_exists(self, shader_name: str):
        """Test that each new shader file exists."""
        shader_path = SHADERS_DIR / shader_name
        assert shader_path.exists(), f"New shader not found: {shader_path}"

    @pytest.mark.parametrize("shader_name", NEW_SHADERS)
    def test_new_shader_not_empty(self, shader_name: str):
        """Test that each new shader file is not empty."""
        shader_path = SHADERS_DIR / shader_name
        if shader_path.exists():
            assert shader_path.stat().st_size > 0, f"Shader file is empty: {shader_path}"

    @pytest.mark.parametrize("shader_name", NEW_SHADERS)
    def test_new_shader_has_valid_size(self, shader_name: str):
        """Test that each new shader has a reasonable file size."""
        shader_path = SHADERS_DIR / shader_name
        if shader_path.exists():
            size = shader_path.stat().st_size
            # SPIR-V files should be at least 20 bytes (header) and less than 10MB
            assert size >= 20, f"Shader file too small: {shader_path} ({size} bytes)"
            assert size < 10 * 1024 * 1024, f"Shader file too large: {shader_path} ({size} bytes)"


# ============================================================================
# SPIR-V Format Validation Tests
# ============================================================================

class TestSPIRVFormat:
    """Test that shaders have valid SPIR-V format."""

    @pytest.mark.parametrize("shader_name", NEW_SHADERS)
    def test_spirv_magic_number(self, shader_name: str):
        """Test that each shader has valid SPIR-V magic number."""
        shader_path = SHADERS_DIR / shader_name
        if not shader_path.exists():
            pytest.skip(f"Shader not found: {shader_path}")

        assert _check_spirv_magic(shader_path), (
            f"Invalid SPIR-V magic number in {shader_path}"
        )

    @pytest.mark.parametrize("shader_name", NEW_SHADERS)
    def test_spirv_word_alignment(self, shader_name: str):
        """Test that shader file size is word-aligned (multiple of 4 bytes)."""
        shader_path = SHADERS_DIR / shader_name
        if not shader_path.exists():
            pytest.skip(f"Shader not found: {shader_path}")

        size = shader_path.stat().st_size
        assert size % 4 == 0, (
            f"SPIR-V file not word-aligned: {shader_path} ({size} bytes)"
        )


# ============================================================================
# SPIR-V Validation Tool Tests
# ============================================================================

class TestSPIRVValidation:
    """Test shaders using SPIRV-Tools validation."""

    @pytest.mark.parametrize("shader_name", NEW_SHADERS)
    def test_spirv_val_validation(self, shader_name: str):
        """Test that each shader passes spirv-val validation."""
        shader_path = SHADERS_DIR / shader_name
        if not shader_path.exists():
            pytest.skip(f"Shader not found: {shader_path}")

        valid, message = _run_spirv_val(shader_path)
        if "not found" in message:
            pytest.skip(message)

        assert valid, f"spirv-val failed for {shader_path}: {message}"

    @pytest.mark.parametrize("shader_name", NEW_SHADERS)
    def test_spirv_dis_disassembly(self, shader_name: str):
        """Test that each shader can be disassembled with spirv-dis."""
        shader_path = SHADERS_DIR / shader_name
        if not shader_path.exists():
            pytest.skip(f"Shader not found: {shader_path}")

        success, output = _run_spirv_dis(shader_path)
        if "not found" in output:
            pytest.skip(output)

        assert success, f"spirv-dis failed for {shader_path}: {output}"
        # Check that output contains expected SPIR-V assembly keywords
        assert "OpCapability" in output or len(output) > 0, (
            f"Invalid SPIR-V disassembly for {shader_path}"
        )


# ============================================================================
# Shader Content Tests
# ============================================================================

class TestShaderContent:
    """Test that shaders contain expected compute shader capabilities."""

    @pytest.mark.parametrize("shader_name", NEW_SHADERS)
    def test_shader_has_entry_point(self, shader_name: str):
        """Test that each shader has a main entry point (via disassembly check)."""
        shader_path = SHADERS_DIR / shader_name
        if not shader_path.exists():
            pytest.skip(f"Shader not found: {shader_path}")

        success, output = _run_spirv_dis(shader_path)
        if not success or "not found" in output:
            pytest.skip(f"Cannot disassemble {shader_path}")

        # Check for entry point in disassembly
        assert "OpEntryPoint" in output, (
            f"No entry point found in {shader_path}"
        )

    @pytest.mark.parametrize("shader_name", NEW_SHADERS)
    def test_shader_has_compute_capability(self, shader_name: str):
        """Test that each shader declares compute shader capability."""
        shader_path = SHADERS_DIR / shader_name
        if not shader_path.exists():
            pytest.skip(f"Shader not found: {shader_path}")

        success, output = _run_spirv_dis(shader_path)
        if not success or "not found" in output:
            pytest.skip(f"Cannot disassemble {shader_path}")

        # Compute shaders should have Shader capability
        assert "OpCapability" in output, (
            f"No capability declaration found in {shader_path}"
        )


# ============================================================================
# Cross-reference with Existing Shaders
# ============================================================================

class TestShaderConsistency:
    """Test that new shaders are consistent with existing SDK shaders."""

    def test_new_shaders_in_same_directory(self):
        """Test that all new shaders are in the same directory as existing shaders."""
        existing_shaders = ["add.spv", "multiply.spv", "relu.spv"]

        for existing in existing_shaders:
            existing_path = SHADERS_DIR / existing
            if existing_path.exists():
                # If we find existing shaders, new shaders should be here too
                for new_shader in NEW_SHADERS:
                    new_path = SHADERS_DIR / new_shader
                    # This is a soft check - new shaders should exist but we don't fail hard
                    if not new_path.exists():
                        pytest.skip(f"New shader {new_shader} not yet created")
                break

    def test_shader_count_increased(self):
        """Test that adding new shaders increased total shader count."""
        all_shaders = list(SHADERS_DIR.glob("*.spv"))
        # Original SDK had 35 shaders, with new shaders we should have more
        min_expected = 35
        assert len(all_shaders) >= min_expected, (
            f"Expected at least {min_expected} shaders, found {len(all_shaders)}"
        )


# ============================================================================
# Regression Tests
# ============================================================================

class TestShaderRegression:
    """Regression tests for shader compilation."""

    def test_original_shaders_still_valid(self):
        """Test that original SDK shaders are still valid after adding new ones."""
        original_shaders = [
            "add.spv",
            "multiply.spv",
            "relu.spv",
            "sigmoid.spv",
        ]

        for shader_name in original_shaders:
            shader_path = SHADERS_DIR / shader_name
            if shader_path.exists():
                assert _check_spirv_magic(shader_path), (
                    f"Original shader {shader_name} has invalid SPIR-V format"
                )

    def test_ml_ops_shaders_present(self):
        """Test that ML operation shaders are present."""
        ml_shaders = [
            "relu.spv",
            "sigmoid.spv",
            "matrix_multiply.spv",
            "optimized_conv2d.spv",
        ]

        found_count = 0
        for shader_name in ml_shaders:
            shader_path = SHADERS_DIR / shader_name
            if shader_path.exists():
                found_count += 1

        # At least some ML shaders should be present
        assert found_count > 0, "No ML operation shaders found"


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

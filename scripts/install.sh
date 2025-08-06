#!/bin/bash
# Install vulkan-ml-sdk tool

echo "=== ARM ML SDK Tool Installer ==="
echo ""

# Check if already in PATH
if command -v vulkan-ml-sdk &> /dev/null; then
    echo "✓ vulkan-ml-sdk is already installed"
    exit 0
fi

# Create symlink in /usr/local/bin
echo "Installing vulkan-ml-sdk to /usr/local/bin..."
sudo ln -sf "$PWD/vulkan-ml-sdk" /usr/local/bin/vulkan-ml-sdk

if [ $? -eq 0 ]; then
    echo "✓ Installation complete!"
    echo ""
    echo "You can now use 'vulkan-ml-sdk' from anywhere:"
    echo "  vulkan-ml-sdk status"
    echo "  vulkan-ml-sdk help"
else
    echo "✗ Installation failed"
    echo "You can still use ./vulkan-ml-sdk from this directory"
fi
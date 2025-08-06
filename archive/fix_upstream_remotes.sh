#!/bin/bash
# Fix upstream remotes to point to correct ARM repositories

echo "=== Fixing Upstream Remotes ==="
echo ""

# The correct ARM GitHub organization is "ARM-software"
REPOS=(
    "ai-ml-sdk-manifest"
    "ai-ml-sdk-for-vulkan"
    "ai-ml-sdk-model-converter"
    "ai-ml-sdk-scenario-runner"
    "ai-ml-sdk-vgf-library"
    "ai-ml-emulation-layer-for-vulkan"
)

for repo in "${REPOS[@]}"; do
    echo "Fixing upstream for $repo..."
    cd "$repo" 2>/dev/null || { echo "Error: Cannot enter $repo"; continue; }
    
    # Remove old upstream
    git remote remove upstream 2>/dev/null || true
    
    # Add correct upstream
    git remote add upstream "https://github.com/ARM-software/$repo.git"
    
    echo "Remotes for $repo:"
    git remote -v
    echo ""
    
    cd ..
done

echo "Testing fetch from upstream..."
echo ""

# Test one repository
cd ai-ml-sdk-for-vulkan
echo "Testing fetch for ai-ml-sdk-for-vulkan..."
if git fetch upstream; then
    echo "✓ Successfully fetched from upstream"
    
    # Show any differences
    echo ""
    echo "Comparing with upstream/main:"
    git rev-list --left-right --count origin/main...upstream/main 2>/dev/null || echo "Cannot compare (branch might not exist)"
else
    echo "✗ Failed to fetch from upstream"
fi

cd ..

echo ""
echo "Upstream remotes fixed. You can now sync with ARM repositories."
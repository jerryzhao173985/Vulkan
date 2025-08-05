#!/bin/bash
# Setup Git workflow with forks and upstream remotes

set -e

echo "=== Setting up Git Workflow for ARM ML SDK Repos ==="
echo ""

# Define repositories (macOS compatible)
REPOS=(
    "ai-ml-sdk-manifest"
    "ai-ml-sdk-for-vulkan"
    "ai-ml-sdk-model-converter"
    "ai-ml-sdk-scenario-runner"
    "ai-ml-sdk-vgf-library"
    "ai-ml-emulation-layer-for-vulkan"
)

GITHUB_USER="jerryzhao173985"
ARM_ORG="ARM-software"
VULKAN_ROOT="/Users/jerry/Vulkan"

# Function to setup remotes for a repository
setup_repo_remotes() {
    local repo_name=$1
    local repo_path="$VULKAN_ROOT/$repo_name"
    
    echo "=== Setting up $repo_name ==="
    
    if [ ! -d "$repo_path" ]; then
        echo "Repository $repo_name not found at $repo_path"
        return 1
    fi
    
    cd "$repo_path"
    
    # Check if it's a git repository
    if [ ! -d ".git" ]; then
        echo "Error: $repo_name is not a git repository"
        return 1
    fi
    
    # Show current remotes
    echo "Current remotes:"
    git remote -v
    
    # Get current branch
    current_branch=$(git branch --show-current)
    echo "Current branch: $current_branch"
    
    echo ""
}

# Step 1: Verify remotes for each repository
echo "Step 1: Verifying remotes for each repository"
echo "============================================="

for repo in "${REPOS[@]}"; do
    setup_repo_remotes "$repo"
done

# Step 2: Setup parent repository
echo ""
echo "Step 2: Setting up parent Vulkan repository"
echo "==========================================="

cd "$VULKAN_ROOT"

# Add remote for parent repo if not exists
if ! git remote get-url origin &>/dev/null; then
    git remote add origin "https://github.com/$GITHUB_USER/Vulkan.git"
    echo "Added origin remote for parent repository"
fi

# Create initial commit if needed
if ! git rev-parse HEAD &>/dev/null; then
    git add README.md .gitignore *.sh *.md
    git commit -m "Initial commit - Vulkan ML SDK collection for macOS ARM64"
fi

# Step 3: Add submodules
echo ""
echo "Step 3: Adding repositories as submodules"
echo "========================================"

# Remove any existing submodule configuration
rm -f .gitmodules
git config --file .gitmodules --unset-all . 2>/dev/null || true

# Add each repository as a submodule
for repo in "${REPOS[@]}"; do
    echo "Adding submodule: $repo"
    
    # Remove existing submodule entry if exists
    git config --file .git/config --remove-section submodule.$repo 2>/dev/null || true
    rm -rf ".git/modules/$repo" 2>/dev/null || true
    
    # Add as submodule
    if [ -d "$repo/.git" ]; then
        git submodule add "https://github.com/$GITHUB_USER/$repo.git" "$repo" || echo "Submodule $repo already exists"
    fi
done

# Step 4: Update .gitmodules
echo ""
echo "Step 4: Configuring submodules"
echo "=============================="

cat > .gitmodules << EOF
[submodule "ai-ml-sdk-manifest"]
	path = ai-ml-sdk-manifest
	url = https://github.com/jerryzhao173985/ai-ml-sdk-manifest.git
	branch = main
[submodule "ai-ml-sdk-for-vulkan"]
	path = ai-ml-sdk-for-vulkan
	url = https://github.com/jerryzhao173985/ai-ml-sdk-for-vulkan.git
	branch = main
[submodule "ai-ml-sdk-model-converter"]
	path = ai-ml-sdk-model-converter
	url = https://github.com/jerryzhao173985/ai-ml-sdk-model-converter.git
	branch = main
[submodule "ai-ml-sdk-scenario-runner"]
	path = ai-ml-sdk-scenario-runner
	url = https://github.com/jerryzhao173985/ai-ml-sdk-scenario-runner.git
	branch = main
[submodule "ai-ml-sdk-vgf-library"]
	path = ai-ml-sdk-vgf-library
	url = https://github.com/jerryzhao173985/ai-ml-sdk-vgf-library.git
	branch = main
[submodule "ai-ml-emulation-layer-for-vulkan"]
	path = ai-ml-emulation-layer-for-vulkan
	url = https://github.com/jerryzhao173985/ai-ml-emulation-layer-for-vulkan.git
	branch = main
EOF

# Sync submodule configuration
git submodule sync

# Step 5: Commit submodule configuration
echo ""
echo "Step 5: Committing configuration"
echo "================================"

git add .gitmodules
git add .gitignore README.md
git add -A
git commit -m "Configure submodules pointing to forks

- Added all 6 ARM ML SDK repositories as submodules
- Submodules point to jerryzhao173985 forks
- Each fork contains macOS ARM64 fixes and optimizations
- Ready for development workflow" || echo "Nothing to commit"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Your workflow is now configured with:"
echo "- Your forks as submodule origins"
echo "- Parent Vulkan repository ready"
echo ""
echo "Next step: Push the parent repository to GitHub:"
echo "  git push -u origin main"
echo ""
echo "To work with the setup:"
echo "  source git_workflow_helpers.sh"
echo "  check_all_status"
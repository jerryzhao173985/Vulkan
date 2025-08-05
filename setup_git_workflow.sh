#!/bin/bash
# Setup Git workflow with forks and upstream remotes

set -e

echo "=== Setting up Git Workflow for ARM ML SDK Repos ==="
echo ""

# Define repositories
declare -A REPOS=(
    ["ai-ml-sdk-manifest"]="ai-ml-sdk-manifest"
    ["ai-ml-sdk-for-vulkan"]="ai-ml-sdk-for-vulkan"
    ["ai-ml-sdk-model-converter"]="ai-ml-sdk-model-converter"
    ["ai-ml-sdk-scenario-runner"]="ai-ml-sdk-scenario-runner"
    ["ai-ml-sdk-vgf-library"]="ai-ml-sdk-vgf-library"
    ["ai-ml-emulation-layer-for-vulkan"]="ai-ml-emulation-layer-for-vulkan"
)

GITHUB_USER="jerryzhao173985"
ARM_ORG="arm"
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
    
    # Remove existing remotes if they exist
    git remote remove origin 2>/dev/null || true
    git remote remove upstream 2>/dev/null || true
    
    # Add your fork as origin
    echo "Adding fork as origin..."
    git remote add origin "https://github.com/$GITHUB_USER/$repo_name.git"
    
    # Add ARM repo as upstream
    echo "Adding ARM repo as upstream..."
    git remote add upstream "https://github.com/$ARM_ORG/$repo_name.git"
    
    # Verify remotes
    echo "Updated remotes:"
    git remote -v
    
    # Fetch from both remotes
    echo "Fetching from remotes..."
    git fetch origin || echo "Warning: Could not fetch from origin"
    git fetch upstream || echo "Warning: Could not fetch from upstream"
    
    # Get current branch
    current_branch=$(git branch --show-current)
    echo "Current branch: $current_branch"
    
    # Set upstream for current branch
    if [ -n "$current_branch" ]; then
        git branch --set-upstream-to=origin/$current_branch $current_branch 2>/dev/null || \
        echo "Note: Could not set upstream for $current_branch"
    fi
    
    echo ""
}

# Step 1: Setup remotes for each repository
echo "Step 1: Setting up remotes for each repository"
echo "============================================="

for repo in "${!REPOS[@]}"; do
    setup_repo_remotes "$repo"
done

# Step 2: Commit and push local changes
echo ""
echo "Step 2: Checking for local changes"
echo "=================================="

for repo in "${!REPOS[@]}"; do
    repo_path="$VULKAN_ROOT/$repo"
    if [ -d "$repo_path/.git" ]; then
        cd "$repo_path"
        echo ""
        echo "=== Checking $repo ==="
        
        # Check for uncommitted changes
        if ! git diff-index --quiet HEAD 2>/dev/null; then
            echo "Found uncommitted changes in $repo"
            git status --short
            
            # Offer to commit changes
            echo "Creating commit for all changes..."
            git add -A
            git commit -m "Save local development work - ARM ML SDK macOS port

- Fixed build issues for macOS ARM64
- Added compatibility layers
- Created unified SDK structure
- Optimized for Apple Silicon" || echo "No changes to commit"
        else
            echo "No uncommitted changes in $repo"
        fi
        
        # Check if there are commits to push
        if [ -n "$(git log origin/$(git branch --show-current)..HEAD 2>/dev/null)" ]; then
            echo "Found unpushed commits in $repo"
            echo "Pushing to fork..."
            git push -u origin $(git branch --show-current) || echo "Could not push (might need to create fork first)"
        else
            echo "No unpushed commits in $repo"
        fi
    fi
done

# Step 3: Create parent Vulkan repository structure
echo ""
echo "Step 3: Setting up parent Vulkan repository"
echo "==========================================="

cd "$VULKAN_ROOT"

# Initialize git repo if not already
if [ ! -d ".git" ]; then
    echo "Initializing Vulkan parent repository..."
    git init
    
    # Create initial commit
    echo "# Vulkan ML SDK Collection" > README.md
    echo "" >> README.md
    echo "This repository contains ARM ML SDK components for Vulkan, adapted for macOS ARM64." >> README.md
    echo "" >> README.md
    echo "## Submodules" >> README.md
    echo "" >> README.md
    echo "- ai-ml-sdk-manifest" >> README.md
    echo "- ai-ml-sdk-for-vulkan" >> README.md
    echo "- ai-ml-sdk-model-converter" >> README.md
    echo "- ai-ml-sdk-scenario-runner" >> README.md
    echo "- ai-ml-sdk-vgf-library" >> README.md
    echo "- ai-ml-emulation-layer-for-vulkan" >> README.md
    
    git add README.md
    git commit -m "Initial commit - Vulkan ML SDK collection"
fi

# Add remote for parent repo
git remote remove origin 2>/dev/null || true
git remote add origin "https://github.com/$GITHUB_USER/Vulkan.git"

# Remove existing submodules if any
echo "Cleaning up existing submodules..."
for repo in "${!REPOS[@]}"; do
    git submodule deinit -f "$repo" 2>/dev/null || true
    git rm -rf "$repo" 2>/dev/null || true
    rm -rf ".git/modules/$repo" 2>/dev/null || true
done

# Add each repository as a submodule pointing to YOUR fork
echo ""
echo "Adding repositories as submodules..."
for repo in "${!REPOS[@]}"; do
    echo "Adding submodule: $repo"
    git submodule add "https://github.com/$GITHUB_USER/$repo.git" "$repo"
done

# Create .gitmodules with proper configuration
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

# Create comprehensive .gitignore
cat > .gitignore << 'EOF'
# Build artifacts
build/
build-*/
*.o
*.a
*.so
*.dylib
*.exe

# IDE files
.vscode/
.idea/
*.swp
*.swo
*~

# OS files
.DS_Store
Thumbs.db

# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.env

# Logs
*.log

# Temporary files
*.tmp
*.temp
/tmp/

# Other repos that aren't submodules
ComputeLibrary/
ML-examples/
MoltenVK/
dependencies/
ml-sdk-*/
EOF

git add .gitmodules .gitignore
git commit -m "Configure submodules and gitignore"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "1. Create forks on GitHub if they don't exist:"
echo "   Go to each ARM repo and click 'Fork' to create under your account"
echo ""
echo "2. Push the parent Vulkan repo to GitHub:"
echo "   git push -u origin main"
echo ""
echo "3. For each submodule, push your changes:"
echo "   git submodule foreach 'git push -u origin main || true'"
echo ""
echo "To update from upstream ARM repos later:"
echo "   git submodule foreach 'git fetch upstream && git merge upstream/main'"
echo ""
echo "Your workflow is now set up with:"
echo "- Your forks as 'origin' (for your changes)"
echo "- ARM repos as 'upstream' (for updates)"
echo "- Everything organized under your Vulkan parent repo"
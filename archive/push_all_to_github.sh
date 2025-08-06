#!/bin/bash
# Push all repositories to GitHub forks

set -e

echo "=== Pushing All Repositories to GitHub ==="
echo ""

REPOS=(
    "ai-ml-sdk-manifest"
    "ai-ml-sdk-for-vulkan"
    "ai-ml-sdk-model-converter"
    "ai-ml-sdk-scenario-runner"
    "ai-ml-sdk-vgf-library"
    "ai-ml-emulation-layer-for-vulkan"
)

GITHUB_USER="jerryzhao173985"
VULKAN_ROOT="/Users/jerry/Vulkan"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m'

echo "This script will push all your local changes to your GitHub forks."
echo ""
echo "Prerequisites:"
echo "1. You must have forked all ARM repos on GitHub"
echo "2. You must have GitHub access configured (SSH key or token)"
echo ""
echo "Your forks should be at:"
for repo in "${REPOS[@]}"; do
    echo "  https://github.com/$GITHUB_USER/$repo"
done
echo ""
read -p "Have you created all the forks on GitHub? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Please create the forks first:${NC}"
    echo "1. Go to https://github.com/ARM-software/<repo-name>"
    echo "2. Click 'Fork' button"
    echo "3. Run this script again"
    exit 1
fi

# Function to push repository
push_repo() {
    local repo_name=$1
    local repo_path="$VULKAN_ROOT/$repo_name"
    
    echo -e "\n${GREEN}=== Pushing $repo_name ===${NC}"
    
    if [ ! -d "$repo_path/.git" ]; then
        echo -e "${RED}Error: $repo_name is not a git repository${NC}"
        return 1
    fi
    
    cd "$repo_path"
    
    # Check remote
    if ! git remote get-url origin &>/dev/null; then
        echo "Setting up remote..."
        git remote add origin "https://github.com/$GITHUB_USER/$repo_name.git"
    fi
    
    # Get current branch
    branch=$(git branch --show-current || echo "main")
    
    # Push to GitHub
    echo "Pushing to origin/$branch..."
    if git push -u origin "$branch" --force; then
        echo -e "${GREEN}✓ Successfully pushed $repo_name${NC}"
        
        # Push tags if any
        git push origin --tags 2>/dev/null || true
    else
        echo -e "${RED}✗ Failed to push $repo_name${NC}"
        echo "  Try: git push -u origin $branch --force"
        return 1
    fi
}

# Push each repository
failed_repos=()
for repo in "${REPOS[@]}"; do
    if ! push_repo "$repo"; then
        failed_repos+=("$repo")
    fi
done

# Summary
echo -e "\n${GREEN}=== Push Summary ===${NC}"
echo ""

if [ ${#failed_repos[@]} -eq 0 ]; then
    echo -e "${GREEN}✓ All repositories pushed successfully!${NC}"
    echo ""
    echo "Your development work is now on GitHub!"
    echo ""
    echo "Next steps:"
    echo "1. Verify the repos on GitHub:"
    for repo in "${REPOS[@]}"; do
        echo "   https://github.com/$GITHUB_USER/$repo"
    done
    echo ""
    echo "2. Run the workflow setup:"
    echo "   ./setup_git_workflow.sh"
else
    echo -e "${RED}Failed to push the following repositories:${NC}"
    for repo in "${failed_repos[@]}"; do
        echo "  - $repo"
    done
    echo ""
    echo "Troubleshooting:"
    echo "1. Check if the fork exists on GitHub"
    echo "2. Verify your GitHub credentials"
    echo "3. Try pushing manually:"
    echo "   cd $VULKAN_ROOT/<repo-name>"
    echo "   git push -u origin main --force"
fi

# Create a push status file
cat > "$VULKAN_ROOT/.push_status" << EOF
Push Status - $(date)
========================

Successful:
EOF

for repo in "${REPOS[@]}"; do
    if [[ ! " ${failed_repos[@]} " =~ " $repo " ]]; then
        echo "✓ $repo" >> "$VULKAN_ROOT/.push_status"
    fi
done

if [ ${#failed_repos[@]} -gt 0 ]; then
    echo -e "\nFailed:" >> "$VULKAN_ROOT/.push_status"
    for repo in "${failed_repos[@]}"; do
        echo "✗ $repo" >> "$VULKAN_ROOT/.push_status"
    done
fi

echo ""
echo "Push status saved to .push_status"
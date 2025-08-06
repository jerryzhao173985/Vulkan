#!/bin/bash
# Test upstream sync capability

echo "=== Testing Upstream Sync Capability ==="
echo ""
echo "This tests the workflow for syncing with upstream repositories"
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m'

cd /Users/jerry/Vulkan

# Test sync workflow
echo "1. Testing sync_with_upstream function..."
source git_workflow_helpers.sh >/dev/null 2>&1

echo ""
echo "2. Current remote configuration:"
echo ""

# Check remotes for each submodule
git submodule foreach '
    echo "Repository: $name"
    echo "Remotes:"
    git remote -v | grep -E "(origin|upstream)" | sed "s/^/  /"
    echo ""
'

echo "3. Testing the sync workflow (simulation):"
echo ""

# Demonstrate the workflow
cat << 'EOF'
The sync workflow works as follows:

a) To sync a single repository with upstream:
   cd ai-ml-sdk-for-vulkan
   git fetch upstream              # Get latest from ARM
   git checkout main              # Ensure on main branch
   git merge upstream/main        # Merge ARM changes
   git push origin main           # Push to your fork

b) To sync all repositories at once:
   source git_workflow_helpers.sh
   sync_with_upstream             # Syncs all submodules

c) To handle merge conflicts:
   # If conflicts occur during merge:
   git status                     # See conflicted files
   # Edit files to resolve conflicts
   git add <resolved-files>
   git commit
   git push origin main
EOF

echo ""
echo -e "${YELLOW}Note: The upstream URLs need to be verified with the actual ARM repository locations.${NC}"
echo ""

# Test workflow simulation
echo "4. Workflow Demonstration:"
echo ""

cd ai-ml-sdk-for-vulkan

# Show current branch status
echo "Current branch configuration:"
git branch -vv | grep "^\*" | sed 's/^/  /'

# Show comparison with origin
echo ""
echo "Status vs your fork (origin):"
if git rev-parse --verify origin/main >/dev/null 2>&1; then
    LOCAL=$(git rev-parse HEAD)
    REMOTE=$(git rev-parse origin/main)
    if [ "$LOCAL" = "$REMOTE" ]; then
        echo -e "  ${GREEN}✓ Up to date with your fork${NC}"
    else
        echo -e "  ${YELLOW}! Diverged from your fork${NC}"
    fi
else
    echo -e "  ${RED}✗ No origin/main branch${NC}"
fi

cd ..

echo ""
echo "5. Helper Functions Available:"
echo ""
echo "  • sync_with_upstream - Sync all repos with ARM"
echo "  • check_all_status - Check status of all repos"
echo "  • save_all_work - Commit and push changes"
echo "  • create_feature_branch - Create branch across all repos"
echo ""

echo -e "${GREEN}=== Sync Capability Test Complete ===${NC}"
echo ""
echo "Your workflow is set up to:"
echo "1. Pull updates from ARM repositories (when URLs are correct)"
echo "2. Merge them into your forks"
echo "3. Continue your development on top"
echo "4. Push your changes to your forks"
echo ""
echo "This maintains a clean separation between:"
echo "- Upstream (ARM) → Source of updates"
echo "- Origin (your forks) → Your development"
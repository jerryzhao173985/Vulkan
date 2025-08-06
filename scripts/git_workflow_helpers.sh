#!/bin/bash
# Helper functions for Git workflow with ARM ML SDK

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to sync all submodules with upstream
sync_with_upstream() {
    echo -e "${BLUE}=== Syncing all submodules with upstream ===${NC}"
    
    git submodule foreach '
        echo -e "\n${GREEN}Syncing $name...${NC}"
        git fetch upstream
        git checkout main
        git merge upstream/main --no-edit || echo -e "${YELLOW}Merge conflicts may need resolution${NC}"
        git push origin main || echo -e "${YELLOW}Push to fork failed${NC}"
    '
}

# Function to push all changes to forks
push_all_changes() {
    echo -e "${BLUE}=== Pushing all changes to forks ===${NC}"
    
    # First, push submodule changes
    git submodule foreach '
        echo -e "\n${GREEN}Checking $name...${NC}"
        if ! git diff-index --quiet HEAD 2>/dev/null; then
            echo "Uncommitted changes found, committing..."
            git add -A
            git commit -m "Update: Local development changes" || true
        fi
        git push -u origin $(git branch --show-current) || echo -e "${YELLOW}Push failed${NC}"
    '
    
    # Then push parent repo
    echo -e "\n${GREEN}Pushing parent Vulkan repo...${NC}"
    git add -A
    git commit -m "Update submodules" || true
    git push -u origin main || true
}

# Function to check status of all repos
check_all_status() {
    echo -e "${BLUE}=== Status of all repositories ===${NC}"
    
    # Parent repo status
    echo -e "\n${GREEN}Parent Vulkan repo:${NC}"
    git status --short
    
    # Submodule status
    git submodule foreach '
        echo -e "\n${GREEN}$name:${NC}"
        git status --short
        
        # Check if ahead/behind origin
        LOCAL=$(git rev-parse @)
        REMOTE=$(git rev-parse @{u} 2>/dev/null || echo "none")
        BASE=$(git merge-base @ @{u} 2>/dev/null || echo "none")
        
        if [ "$REMOTE" = "none" ]; then
            echo -e "${YELLOW}No upstream branch set${NC}"
        elif [ "$LOCAL" = "$REMOTE" ]; then
            echo "Up to date with origin"
        elif [ "$LOCAL" = "$BASE" ]; then
            echo -e "${YELLOW}Behind origin - pull needed${NC}"
        elif [ "$REMOTE" = "$BASE" ]; then
            echo -e "${GREEN}Ahead of origin - push needed${NC}"
        else
            echo -e "${RED}Diverged from origin${NC}"
        fi
    '
}

# Function to create a new feature branch across all repos
create_feature_branch() {
    local branch_name=$1
    
    if [ -z "$branch_name" ]; then
        echo -e "${RED}Error: Please provide a branch name${NC}"
        echo "Usage: create_feature_branch <branch-name>"
        return 1
    fi
    
    echo -e "${BLUE}=== Creating feature branch '$branch_name' ===${NC}"
    
    # Create in parent repo
    git checkout -b "$branch_name"
    
    # Create in all submodules
    git submodule foreach "git checkout -b $branch_name"
    
    echo -e "${GREEN}Feature branch '$branch_name' created in all repos${NC}"
}

# Function to update submodule URLs to use SSH (for push access)
use_ssh_urls() {
    echo -e "${BLUE}=== Switching to SSH URLs for push access ===${NC}"
    
    git submodule foreach '
        echo -e "\n${GREEN}Updating $name...${NC}"
        git remote set-url origin git@github.com:jerryzhao173985/$name.git
        git remote -v
    '
    
    # Update parent repo
    git remote set-url origin git@github.com:jerryzhao173985/Vulkan.git
    
    echo -e "${GREEN}All remotes updated to use SSH${NC}"
}

# Function to save all work
save_all_work() {
    local commit_message="${1:-Save development work}"
    
    echo -e "${BLUE}=== Saving all work ===${NC}"
    
    # Save in each submodule
    git submodule foreach "
        if ! git diff-index --quiet HEAD 2>/dev/null; then
            git add -A
            git commit -m '$commit_message'
            git push -u origin \$(git branch --show-current)
        fi
    "
    
    # Update parent repo
    git add -A
    git commit -m "Update submodules: $commit_message" || true
    git push -u origin $(git branch --show-current)
    
    echo -e "${GREEN}All work saved and pushed${NC}"
}

# Function to clone fresh setup
clone_fresh_setup() {
    local target_dir="${1:-Vulkan-fresh}"
    
    echo -e "${BLUE}=== Cloning fresh setup to $target_dir ===${NC}"
    
    git clone --recursive https://github.com/jerryzhao173985/Vulkan.git "$target_dir"
    
    cd "$target_dir"
    
    # Setup remotes in all submodules
    git submodule foreach '
        git remote add upstream https://github.com/arm/$name.git
        git fetch upstream
    '
    
    echo -e "${GREEN}Fresh setup cloned to $target_dir${NC}"
}

# Print available commands
show_help() {
    echo -e "${BLUE}=== Git Workflow Helper Commands ===${NC}"
    echo ""
    echo "Available functions:"
    echo "  sync_with_upstream    - Sync all repos with upstream ARM repos"
    echo "  push_all_changes      - Push all changes to your forks"
    echo "  check_all_status      - Check status of all repositories"
    echo "  create_feature_branch - Create a feature branch across all repos"
    echo "  use_ssh_urls         - Switch to SSH URLs for push access"
    echo "  save_all_work        - Commit and push all changes"
    echo "  clone_fresh_setup    - Clone a fresh copy of the entire setup"
    echo ""
    echo "Usage: source git_workflow_helpers.sh && <function_name>"
}

# Show help when sourced
echo -e "${GREEN}Git workflow helpers loaded!${NC}"
echo "Run 'show_help' to see available commands"
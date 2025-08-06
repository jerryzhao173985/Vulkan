#!/bin/bash
# Comprehensive check of the entire Git workflow setup

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}=== Comprehensive Git Workflow Check ===${NC}"
echo ""

GITHUB_USER="jerryzhao173985"
REPOS=(
    "ai-ml-sdk-manifest"
    "ai-ml-sdk-for-vulkan"
    "ai-ml-sdk-model-converter"
    "ai-ml-sdk-scenario-runner"
    "ai-ml-sdk-vgf-library"
    "ai-ml-emulation-layer-for-vulkan"
)

PASS_COUNT=0
FAIL_COUNT=0
WARN_COUNT=0

# Function to check a condition
check() {
    local description=$1
    local command=$2
    
    if eval "$command" &>/dev/null; then
        echo -e "${GREEN}✓${NC} $description"
        ((PASS_COUNT++))
        return 0
    else
        echo -e "${RED}✗${NC} $description"
        ((FAIL_COUNT++))
        return 1
    fi
}

# Function to warn
warn() {
    local description=$1
    echo -e "${YELLOW}!${NC} $description"
    ((WARN_COUNT++))
}

# 1. Check Git configuration
echo -e "${CYAN}1. Git Configuration${NC}"
check "Git user configured" "git config --global user.name"
check "Git email configured" "git config --global user.email"
echo ""

# 2. Check directory structure
echo -e "${CYAN}2. Directory Structure${NC}"
check "In Vulkan directory" "[ $(basename $PWD) = 'Vulkan' ]"
for repo in "${REPOS[@]}"; do
    check "Repository exists: $repo" "[ -d $repo ]"
done
echo ""

# 3. Check Git repositories
echo -e "${CYAN}3. Git Repository Status${NC}"
for repo in "${REPOS[@]}"; do
    if [ -d "$repo/.git" ]; then
        check "$repo is a git repository" "[ -d $repo/.git ]"
        
        # Check remotes
        cd "$repo" 2>/dev/null
        if git remote get-url origin &>/dev/null; then
            origin_url=$(git remote get-url origin)
            if [[ "$origin_url" == *"$GITHUB_USER"* ]]; then
                check "$repo origin points to your fork" "true"
            else
                check "$repo origin points to your fork" "false"
            fi
        else
            check "$repo has origin remote" "false"
        fi
        
        if git remote get-url upstream &>/dev/null; then
            upstream_url=$(git remote get-url upstream)
            if [[ "$upstream_url" == *"ARM-software"* ]] || [[ "$upstream_url" == *"arm"* ]]; then
                check "$repo upstream points to ARM" "true"
            else
                check "$repo upstream points to ARM" "false"
            fi
        else
            warn "$repo missing upstream remote"
        fi
        
        cd - &>/dev/null
    else
        check "$repo is a git repository" "false"
    fi
done
echo ""

# 4. Check parent repository
echo -e "${CYAN}4. Parent Repository${NC}"
check "Parent .git directory exists" "[ -d .git ]"
if [ -d .git ]; then
    check "On main branch" "[ $(git branch --show-current 2>/dev/null) = 'main' ]"
    
    # Check if README exists
    if [ -f README.md ]; then
        check "README.md exists" "true"
    else
        warn "README.md missing (will be created by setup script)"
    fi
    
    # Check .gitignore
    check ".gitignore exists" "[ -f .gitignore ]"
    
    # Check .gitmodules
    if [ -f .gitmodules ]; then
        check ".gitmodules exists" "true"
        
        # Verify submodule URLs
        for repo in "${REPOS[@]}"; do
            if grep -q "url = https://github.com/$GITHUB_USER/$repo.git" .gitmodules 2>/dev/null; then
                check "Submodule $repo points to your fork" "true"
            else
                warn "Submodule $repo not configured correctly"
            fi
        done
    else
        warn ".gitmodules missing (will be created by setup script)"
    fi
fi
echo ""

# 5. Check scripts
echo -e "${CYAN}5. Setup Scripts${NC}"
scripts=(
    "setup_git_workflow.sh"
    "push_all_to_github.sh"
    "git_workflow_helpers.sh"
    "prepare_repos_for_github.sh"
    "github_setup_wizard.sh"
)

for script in "${scripts[@]}"; do
    check "$script exists" "[ -f $script ]"
    check "$script is executable" "[ -x $script ]"
done
echo ""

# 6. Check documentation
echo -e "${CYAN}6. Documentation${NC}"
docs=(
    "SETUP_INSTRUCTIONS.md"
    "GIT_WORKFLOW_GUIDE.md"
    "COMPLETE_GITHUB_SETUP.md"
)

for doc in "${docs[@]}"; do
    check "$doc exists" "[ -f $doc ]"
done
echo ""

# 7. Test helper functions
echo -e "${CYAN}7. Helper Functions${NC}"
if [ -f git_workflow_helpers.sh ]; then
    source git_workflow_helpers.sh &>/dev/null
    check "Helper functions load" "type check_all_status &>/dev/null"
    check "show_help function exists" "type show_help &>/dev/null"
    check "save_all_work function exists" "type save_all_work &>/dev/null"
    check "sync_with_upstream function exists" "type sync_with_upstream &>/dev/null"
else
    check "Helper functions available" "false"
fi
echo ""

# 8. Check for important files in main SDK
echo -e "${CYAN}8. Main SDK Repository Files${NC}"
if [ -d "ai-ml-sdk-for-vulkan" ]; then
    cd ai-ml-sdk-for-vulkan
    check "Build instructions exist" "[ -f BUILD_MACOS.md ]"
    check "Journey log exists" "[ -f COMPLETE_JOURNEY_LOG.md ]"
    check "Unified SDK created" "[ -d unified-ml-sdk ]"
    check "Production package exists" "[ -f arm-ml-sdk-vulkan-macos-v1.0.0-production.tar.gz ]"
    cd - &>/dev/null
fi
echo ""

# 9. Summary
echo -e "${BLUE}=== Summary ===${NC}"
echo -e "Passed: ${GREEN}$PASS_COUNT${NC}"
echo -e "Failed: ${RED}$FAIL_COUNT${NC}"
echo -e "Warnings: ${YELLOW}$WARN_COUNT${NC}"
echo ""

if [ $FAIL_COUNT -eq 0 ]; then
    if [ $WARN_COUNT -eq 0 ]; then
        echo -e "${GREEN}✓ Everything looks perfect!${NC}"
    else
        echo -e "${GREEN}✓ Setup is functional with minor warnings.${NC}"
    fi
    echo ""
    echo "Ready to proceed with GitHub upload:"
    echo "1. Create forks on GitHub (if not done)"
    echo "2. Run: ./github_setup_wizard.sh"
else
    echo -e "${RED}✗ Some checks failed. Please fix the issues above.${NC}"
    echo ""
    echo "Common fixes:"
    echo "- Run: ./prepare_repos_for_github.sh"
    echo "- Ensure all repos have .git directories"
    echo "- Check file permissions"
fi

# 10. Quick status of each repo
echo ""
echo -e "${CYAN}Repository Status:${NC}"
for repo in "${REPOS[@]}"; do
    if [ -d "$repo/.git" ]; then
        cd "$repo"
        branch=$(git branch --show-current 2>/dev/null || echo "unknown")
        if git diff-index --quiet HEAD 2>/dev/null; then
            status="clean"
        else
            status="uncommitted changes"
        fi
        echo "  $repo: branch=$branch, status=$status"
        cd - &>/dev/null
    else
        echo "  $repo: not a git repository"
    fi
done
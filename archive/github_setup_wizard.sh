#!/bin/bash
# Interactive GitHub setup wizard for ARM ML SDK

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${BLUE}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║       ARM ML SDK GitHub Setup Wizard                      ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════╝${NC}"
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

# Step 1: Check prerequisites
echo -e "${CYAN}Step 1: Checking prerequisites...${NC}"
echo ""

# Check if GitHub CLI is installed (optional but helpful)
if command -v gh &> /dev/null; then
    echo -e "${GREEN}✓ GitHub CLI installed${NC}"
    GH_AVAILABLE=true
else
    echo -e "${YELLOW}○ GitHub CLI not installed (optional)${NC}"
    GH_AVAILABLE=false
fi

# Check git configuration
GIT_USER=$(git config --global user.name || echo "")
GIT_EMAIL=$(git config --global user.email || echo "")

if [ -n "$GIT_USER" ] && [ -n "$GIT_EMAIL" ]; then
    echo -e "${GREEN}✓ Git configured: $GIT_USER <$GIT_EMAIL>${NC}"
else
    echo -e "${RED}✗ Git not configured${NC}"
    echo "  Please run:"
    echo "  git config --global user.name \"Your Name\""
    echo "  git config --global user.email \"your.email@example.com\""
fi

echo ""

# Step 2: Check and create forks
echo -e "${CYAN}Step 2: GitHub Forks Status${NC}"
echo ""

echo "You need to fork these repositories on GitHub:"
echo ""

for i in "${!REPOS[@]}"; do
    repo="${REPOS[$i]}"
    echo -e "${YELLOW}$((i+1)). ${repo}${NC}"
    echo "   Original: https://github.com/ARM-software/$repo"
    echo "   Your fork: https://github.com/$GITHUB_USER/$repo"
    echo ""
done

echo -e "${MAGENTA}Please open GitHub and fork each repository listed above.${NC}"
echo -e "${MAGENTA}Click each 'Original' link and press the Fork button.${NC}"
echo ""

read -p "Have you created all 6 forks? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Please create the forks first, then run this script again.${NC}"
    exit 0
fi

# Step 3: Create parent repository
echo ""
echo -e "${CYAN}Step 3: Parent Repository${NC}"
echo ""

echo "You need to create a parent repository on GitHub:"
echo -e "${YELLOW}Repository name: Vulkan${NC}"
echo -e "${YELLOW}URL: https://github.com/$GITHUB_USER/Vulkan${NC}"
echo ""
echo -e "${RED}IMPORTANT: Do NOT initialize with README${NC}"
echo ""

read -p "Have you created the parent 'Vulkan' repository? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Please create the repository at: https://github.com/new${NC}"
    echo "Then run this script again."
    exit 0
fi

# Step 4: Run setup process
echo ""
echo -e "${CYAN}Step 4: Running Setup Process${NC}"
echo ""

# First, check if we can authenticate to GitHub
echo "Testing GitHub authentication..."
if git ls-remote https://github.com/$GITHUB_USER/Vulkan.git &>/dev/null; then
    echo -e "${GREEN}✓ GitHub authentication working${NC}"
else
    echo -e "${RED}✗ Cannot access your GitHub repository${NC}"
    echo ""
    echo "Please ensure you have set up GitHub authentication:"
    echo "1. Personal Access Token: https://github.com/settings/tokens"
    echo "2. Or SSH Key: https://github.com/settings/keys"
    echo ""
    read -p "Do you want to continue anyway? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Step 5: Execute setup
echo ""
echo -e "${CYAN}Step 5: Executing Setup${NC}"
echo ""

echo "This will:"
echo "1. Push all your local work to your GitHub forks"
echo "2. Set up the submodule structure"
echo "3. Configure remotes for syncing with ARM"
echo ""

read -p "Ready to proceed? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Setup cancelled."
    exit 0
fi

# Run the setup scripts
echo -e "${GREEN}Pushing to GitHub...${NC}"
if ./push_all_to_github.sh; then
    echo -e "${GREEN}✓ Successfully pushed to GitHub${NC}"
else
    echo -e "${RED}✗ Failed to push some repositories${NC}"
    echo "Please check the errors above and try again."
    exit 1
fi

echo ""
echo -e "${GREEN}Setting up workflow...${NC}"
if ./setup_git_workflow.sh; then
    echo -e "${GREEN}✓ Workflow setup complete${NC}"
else
    echo -e "${RED}✗ Workflow setup failed${NC}"
    exit 1
fi

# Step 6: Final verification
echo ""
echo -e "${CYAN}Step 6: Final Verification${NC}"
echo ""

# Try to push parent repo
echo "Pushing parent repository..."
if git push -u origin main 2>/dev/null; then
    echo -e "${GREEN}✓ Parent repository pushed successfully${NC}"
else
    echo -e "${YELLOW}! Could not push parent repo (may already be pushed)${NC}"
fi

# Summary
echo ""
echo -e "${GREEN}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║              Setup Complete! 🎉                           ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""

echo "Your repositories are now on GitHub:"
echo ""
echo -e "${CYAN}Parent Repository:${NC}"
echo "  https://github.com/$GITHUB_USER/Vulkan"
echo ""
echo -e "${CYAN}Your Forks:${NC}"
for repo in "${REPOS[@]}"; do
    echo "  https://github.com/$GITHUB_USER/$repo"
done

echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo "1. Load helper functions:"
echo "   ${GREEN}source git_workflow_helpers.sh${NC}"
echo ""
echo "2. Check status:"
echo "   ${GREEN}check_all_status${NC}"
echo ""
echo "3. To clone fresh copy elsewhere:"
echo "   ${GREEN}git clone --recursive https://github.com/$GITHUB_USER/Vulkan.git${NC}"
echo ""
echo "Happy coding! 🚀"
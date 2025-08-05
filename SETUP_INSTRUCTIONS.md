# Setup Instructions for ARM ML SDK GitHub Workflow

## Overview

This guide will help you set up a complete GitHub workflow for the ARM ML SDK development, with your forks as the primary development repositories and the ability to sync with upstream ARM repositories.

## Prerequisites

1. **GitHub Account**: Ensure you're logged into GitHub as `jerryzhao173985`
2. **Git Configuration**: Your Git should be configured with your GitHub credentials
3. **SSH Key (Optional)**: For easier pushing without passwords

## Step-by-Step Setup

### Step 1: Create GitHub Forks

First, you need to fork all ARM repositories to your GitHub account.

Go to each of these ARM repositories and click the "Fork" button:

1. [ARM-software/ai-ml-sdk-manifest](https://github.com/ARM-software/ai-ml-sdk-manifest) → Fork
2. [ARM-software/ai-ml-sdk-for-vulkan](https://github.com/ARM-software/ai-ml-sdk-for-vulkan) → Fork
3. [ARM-software/ai-ml-sdk-model-converter](https://github.com/ARM-software/ai-ml-sdk-model-converter) → Fork
4. [ARM-software/ai-ml-sdk-scenario-runner](https://github.com/ARM-software/ai-ml-sdk-scenario-runner) → Fork
5. [ARM-software/ai-ml-sdk-vgf-library](https://github.com/ARM-software/ai-ml-sdk-vgf-library) → Fork
6. [ARM-software/ai-ml-emulation-layer-for-vulkan](https://github.com/ARM-software/ai-ml-emulation-layer-for-vulkan) → Fork

### Step 2: Create Parent Vulkan Repository

Create a new repository on GitHub:

1. Go to https://github.com/new
2. Repository name: `Vulkan`
3. Description: "ARM ML SDK for Vulkan - macOS ARM64 Port"
4. Make it public or private as desired
5. Do NOT initialize with README (we'll push our own)
6. Click "Create repository"

### Step 3: Prepare Local Repositories

```bash
cd /Users/jerry/Vulkan

# Make scripts executable
chmod +x prepare_repos_for_github.sh
chmod +x push_all_to_github.sh
chmod +x setup_git_workflow.sh

# Prepare all repositories with proper commits
./prepare_repos_for_github.sh
```

### Step 4: Push to Your Forks

```bash
# Push all your work to GitHub
./push_all_to_github.sh
```

This will push all your local development work to your GitHub forks.

### Step 5: Setup Complete Workflow

```bash
# Setup the submodule structure and remotes
./setup_git_workflow.sh

# Push the parent Vulkan repo
git push -u origin main
```

### Step 6: Load Helper Functions

For convenient workflow management:

```bash
# Load helper functions
source git_workflow_helpers.sh

# Check status of everything
check_all_status
```

## Verification

After setup, verify everything is working:

1. **Check GitHub**: Visit your repositories to ensure code is uploaded
   - https://github.com/jerryzhao173985/Vulkan
   - https://github.com/jerryzhao173985/ai-ml-sdk-for-vulkan
   - etc.

2. **Clone Test**: Try cloning your setup to a new location
   ```bash
   git clone --recursive https://github.com/jerryzhao173985/Vulkan.git ~/test-vulkan
   ```

3. **Check Remotes**: Verify remotes are set up correctly
   ```bash
   cd ai-ml-sdk-for-vulkan
   git remote -v
   # Should show:
   # origin: your fork (for pushing)
   # upstream: ARM repo (for syncing)
   ```

## Daily Workflow

Once set up, your daily workflow will be:

```bash
# Start work
cd /Users/jerry/Vulkan
source git_workflow_helpers.sh

# Check status
check_all_status

# Make changes...

# Save work
save_all_work "Description of today's changes"

# Sync with upstream when needed
sync_with_upstream
```

## Troubleshooting

### Push Permission Denied

If you get permission denied when pushing:

1. **Check GitHub login**: Make sure you're authenticated
   ```bash
   git config --global user.name "jerryzhao173985"
   git config --global user.email "your-email@example.com"
   ```

2. **Use SSH**: Switch to SSH URLs for easier access
   ```bash
   source git_workflow_helpers.sh
   use_ssh_urls
   ```

3. **Use Personal Access Token**: For HTTPS, create a token at https://github.com/settings/tokens

### Fork Not Found

If push fails with "repository not found":
1. Verify you created the fork on GitHub
2. Check the remote URL is correct
3. Try creating the fork again

### Submodule Issues

If submodules aren't working:
```bash
# Reinitialize submodules
git submodule deinit --all -f
git submodule update --init --recursive
```

## Success Indicators

You'll know the setup is successful when:

1. ✅ All 6 ARM SDK repos are forked under your GitHub account
2. ✅ The parent Vulkan repo contains all 6 as submodules
3. ✅ Each submodule has two remotes: origin (your fork) and upstream (ARM)
4. ✅ You can push changes to your forks
5. ✅ You can pull updates from ARM repos
6. ✅ The helper functions work correctly

## Next Steps

After successful setup:

1. **Document Your Work**: Update READMEs in your forks
2. **Create Issues**: Track future work in GitHub Issues
3. **Use Branches**: Create feature branches for new development
4. **Consider PRs**: When ready, submit improvements back to ARM

## Questions?

If you encounter any issues:

1. Check the Git Workflow Guide: `GIT_WORKFLOW_GUIDE.md`
2. Review the helper functions: `source git_workflow_helpers.sh && show_help`
3. Check push status: `cat .push_status`

Good luck with your ARM ML SDK development!
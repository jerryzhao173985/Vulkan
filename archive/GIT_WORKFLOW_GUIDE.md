# Git Workflow Guide for ARM ML SDK Development

## Overview

This guide explains how to use the Git workflow setup for developing the ARM ML SDK on macOS. The setup includes:

- Your GitHub forks as `origin` (for your changes)
- ARM's repos as `upstream` (for syncing updates)
- Everything organized as submodules under your main Vulkan repository

## Repository Structure

```
Vulkan/ (parent repo - jerryzhao173985/Vulkan)
├── ai-ml-sdk-manifest/          (submodule → your fork)
├── ai-ml-sdk-for-vulkan/        (submodule → your fork)
├── ai-ml-sdk-model-converter/   (submodule → your fork)
├── ai-ml-sdk-scenario-runner/   (submodule → your fork)
├── ai-ml-sdk-vgf-library/       (submodule → your fork)
└── ai-ml-emulation-layer-for-vulkan/ (submodule → your fork)
```

## Initial Setup

### 1. Fork ARM Repositories

First, ensure you have forked all ARM repos on GitHub:

1. Go to each ARM repository:
   - https://github.com/ARM-software/ai-ml-sdk-manifest
   - https://github.com/ARM-software/ai-ml-sdk-for-vulkan
   - etc.

2. Click "Fork" to create your own copy under `jerryzhao173985`

### 2. Run Setup Script

```bash
cd /Users/jerry/Vulkan
chmod +x setup_git_workflow.sh
./setup_git_workflow.sh
```

### 3. Load Helper Functions

```bash
source git_workflow_helpers.sh
```

## Daily Workflow

### Checking Status

```bash
# Check status of all repos
check_all_status
```

### Saving Your Work

```bash
# Quick save all changes
save_all_work "Description of changes"

# Or manually for specific repo
cd ai-ml-sdk-for-vulkan
git add -A
git commit -m "Fixed macOS build issues"
git push origin main
```

### Syncing with Upstream

```bash
# Sync all repos with ARM upstream
sync_with_upstream

# Or manually for specific repo
cd ai-ml-sdk-for-vulkan
git fetch upstream
git merge upstream/main
git push origin main
```

## Common Tasks

### 1. Start New Feature

```bash
# Create feature branch across all repos
create_feature_branch feature/macos-optimization

# Work on your feature...

# Push feature branch
push_all_changes
```

### 2. Update Submodules

```bash
# Pull latest from your forks
git submodule update --remote --merge

# Or reinitialize if needed
git submodule update --init --recursive
```

### 3. Submit Changes Upstream

When ready to contribute back to ARM:

```bash
# 1. Push to your fork
cd ai-ml-sdk-for-vulkan
git push origin main

# 2. Go to GitHub and create Pull Request
# From: jerryzhao173985/ai-ml-sdk-for-vulkan:main
# To: ARM-software/ai-ml-sdk-for-vulkan:main
```

### 4. Clone Fresh Copy

```bash
# Clone entire setup to new location
git clone --recursive https://github.com/jerryzhao173985/Vulkan.git ~/Vulkan-new

# Or use helper
clone_fresh_setup ~/Vulkan-new
```

## Advanced Operations

### Switch to SSH URLs

For easier pushing without passwords:

```bash
use_ssh_urls
```

### Handle Merge Conflicts

```bash
# If conflicts during upstream sync
cd ai-ml-sdk-for-vulkan
git status  # See conflicts
# Edit files to resolve
git add resolved-file.cpp
git merge --continue
git push origin main
```

### Update Parent Repo Only

```bash
# Update submodule references in parent
git add -A
git commit -m "Update submodule references"
git push origin main
```

## Best Practices

1. **Commit Often**: Make small, focused commits
2. **Write Good Messages**: Describe what and why, not how
3. **Sync Regularly**: Fetch upstream changes frequently
4. **Branch for Features**: Use branches for new features
5. **Test Before Push**: Ensure builds work before pushing

## Troubleshooting

### Submodule Issues

```bash
# Reset submodule to clean state
git submodule deinit -f path/to/submodule
git submodule update --init

# Fix detached HEAD in submodule
cd submodule-name
git checkout main
git pull origin main
```

### Push Rejected

```bash
# If push rejected (not fast-forward)
git pull --rebase origin main
git push origin main
```

### Lost Changes

```bash
# View recent commits across all repos
git submodule foreach 'git log --oneline -5'

# Recover lost commit
git reflog
git checkout <commit-hash>
```

## Quick Reference

```bash
# Status
git status                          # Current repo
git submodule status               # All submodules
check_all_status                   # Detailed status

# Save Work
git add -A && git commit -m "msg"  # Commit changes
git push origin main               # Push to fork
save_all_work "msg"               # Save everything

# Sync
git fetch upstream                 # Get upstream changes
git merge upstream/main           # Merge changes
sync_with_upstream                # Sync all repos

# Submodules
git submodule update --init       # Initialize
git submodule update --remote     # Update all
git submodule foreach 'command'   # Run in each
```

## Repository URLs

### Your Forks (origin)
- https://github.com/jerryzhao173985/Vulkan (parent)
- https://github.com/jerryzhao173985/ai-ml-sdk-manifest
- https://github.com/jerryzhao173985/ai-ml-sdk-for-vulkan
- https://github.com/jerryzhao173985/ai-ml-sdk-model-converter
- https://github.com/jerryzhao173985/ai-ml-sdk-scenario-runner
- https://github.com/jerryzhao173985/ai-ml-sdk-vgf-library
- https://github.com/jerryzhao173985/ai-ml-emulation-layer-for-vulkan

### ARM Repos (upstream)
- https://github.com/ARM-software/ai-ml-sdk-manifest
- https://github.com/ARM-software/ai-ml-sdk-for-vulkan
- https://github.com/ARM-software/ai-ml-sdk-model-converter
- https://github.com/ARM-software/ai-ml-sdk-scenario-runner
- https://github.com/ARM-software/ai-ml-sdk-vgf-library
- https://github.com/ARM-software/ai-ml-emulation-layer-for-vulkan

---

Remember: Your forks are your workspace. The parent Vulkan repo ties everything together. Always push to your forks, and create Pull Requests when ready to contribute upstream!
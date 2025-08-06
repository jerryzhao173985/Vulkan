#!/bin/bash
# Clean up and organize the Vulkan directory

echo "=== Organizing ARM ML SDK Directory ==="
echo ""

# Create organized structure
mkdir -p archive docs scripts

# Move old scripts to archive
echo "Archiving old scripts..."
mv setup_git_workflow.sh archive/ 2>/dev/null
mv setup_git_workflow_fixed.sh archive/ 2>/dev/null
mv push_all_to_github.sh archive/ 2>/dev/null
mv prepare_repos_for_github.sh archive/ 2>/dev/null
mv github_setup_wizard.sh archive/ 2>/dev/null
mv check_everything.sh archive/ 2>/dev/null
mv fix_upstream_remotes.sh archive/ 2>/dev/null
mv test_upstream_sync.sh archive/ 2>/dev/null
mv git_workflow_helpers.sh scripts/ 2>/dev/null

# Move old documentation to archive
echo "Archiving old documentation..."
mv SETUP_INSTRUCTIONS.md archive/ 2>/dev/null
mv GIT_WORKFLOW_GUIDE.md archive/ 2>/dev/null
mv COMPLETE_GITHUB_SETUP.md archive/ 2>/dev/null
mv VERIFICATION_REPORT.md archive/ 2>/dev/null
mv GITHUB_SYNC_VERIFICATION.md archive/ 2>/dev/null
mv UPSTREAM_SYNC_GUIDE.md archive/ 2>/dev/null
mv SUCCESS_REPORT.md archive/ 2>/dev/null

# Keep only essential files in root
echo ""
echo "Current root directory files:"
ls -la | grep -v "^d" | grep -v "^total"

echo ""
echo "=== Cleanup Complete ==="
echo ""
echo "Structure:"
echo "  vulkan-ml-sdk    - Main tool (all commands)"
echo "  README.md        - Project overview"
echo "  .gitignore       - Git ignore rules"
echo "  .gitmodules      - Submodule configuration"
echo "  docs/            - Documentation"
echo "  scripts/         - Helper scripts (if needed)"
echo "  archive/         - Old files (can be deleted)"
echo ""
echo "Use './vulkan-ml-sdk help' for all commands"
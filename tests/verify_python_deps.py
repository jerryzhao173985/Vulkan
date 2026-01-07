#!/usr/bin/env python3
"""Verify Python dependencies are available for the test suite."""

import sys
import os
import warnings

# Suppress deprecation warnings for version checks
warnings.filterwarnings('ignore', category=DeprecationWarning)

def add_user_site_packages():
    """Add user site-packages to path if not enabled."""
    import site
    user_site = site.getusersitepackages()
    if user_site and os.path.exists(user_site) and user_site not in sys.path:
        sys.path.insert(0, user_site)

    # Also try the common user library location for Python 3.9
    user_lib = os.path.expanduser('~/Library/Python/3.9/lib/python/site-packages')
    if os.path.exists(user_lib) and user_lib not in sys.path:
        sys.path.insert(0, user_lib)

def get_version(mod):
    """Get module version safely."""
    # Try __version__ attribute first
    if hasattr(mod, '__version__'):
        return mod.__version__
    # For jsonschema, use importlib.metadata
    try:
        from importlib.metadata import version
        return version(mod.__name__)
    except Exception:
        return 'unknown'

def check_dependencies():
    """Check that all required Python dependencies are importable."""
    # Try to add user site-packages if not already in path
    add_user_site_packages()

    # Map of import names to package names
    deps = {
        'numpy': 'numpy',
        'PIL': 'Pillow',
        'pytest': 'pytest',
        'jsonschema': 'jsonschema'
    }

    missing = []
    available = []

    for import_name, package_name in deps.items():
        try:
            mod = __import__(import_name)
            version = get_version(mod)
            available.append(f"{import_name} ({version})")
        except ImportError:
            missing.append(f"{import_name} ({package_name})")

    if available:
        print(f"Available: {', '.join(available)}")

    if missing:
        print(f"Missing: {', '.join(missing)}")
        print("\nTo install missing dependencies:")
        print("  pip3 install --user numpy pillow pytest jsonschema")
        return 1

    print("OK")
    return 0

if __name__ == "__main__":
    sys.exit(check_dependencies())

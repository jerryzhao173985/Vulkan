#!/usr/bin/env python3
"""Run API import tests using pytest."""

import sys
import os

# Add user site-packages to path
user_site = os.path.expanduser('~/Library/Python/3.9/lib/python/site-packages')
if os.path.exists(user_site) and user_site not in sys.path:
    sys.path.insert(0, user_site)

import pytest

if __name__ == "__main__":
    sys.exit(pytest.main(['-v', '--tb=short', 'unit/test_api_import.py']))

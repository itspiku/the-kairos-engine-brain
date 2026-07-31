"""
The Kairos Engine - Test Runner Script
"""

import sys
import unittest

if __name__ == "__main__":
    print("⚡ Running Kairos Engine Test Suite...\n")
    loader = unittest.TestLoader()
    suite = loader.discover("tests")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)

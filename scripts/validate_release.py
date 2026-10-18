"""Run explicitly scoped static tests only, without scientific module imports."""
from pathlib import Path
import sys
import unittest

def main():
    sys.dont_write_bytecode = True
    root = Path(__file__).resolve().parents[1]
    suite = unittest.TestSuite()
    for pattern in ('test_static_repository.py', 'test_release.py', 'test_final_release.py'):
        suite.addTests(unittest.defaultTestLoader.discover(str(root / 'tests'), pattern=pattern))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1

if __name__ == '__main__':
    raise SystemExit(main())

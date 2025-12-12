"""
Clean test runner for Livnium-O tests.

Provides cleaner, more readable output.
"""

import unittest
import sys
from pathlib import Path

# Add core-o directory to path
core_o_path = Path(__file__).parent.parent
sys.path.insert(0, str(core_o_path))

from test_livnium_o import TestLivniumOSystem


class CleanTestResult(unittest.TextTestResult):
    """Custom test result with cleaner output."""
    
    def __init__(self, stream, descriptions, verbosity):
        super().__init__(stream, descriptions, verbosity)
        self.test_count = 0
        self.passed = []
        self.failed = []
        self.errors = []
        self.test_names = []
    
    def startTest(self, test):
        super().startTest(test)
        self.test_count += 1
        test_name = self._get_test_name(test)
        self.test_names.append(test_name)
        # Print progress dot
        print(".", end="", flush=True)
    
    def addSuccess(self, test):
        super().addSuccess(test)
        test_name = self._get_test_name(test)
        self.passed.append(test_name)
    
    def addError(self, test, err):
        super().addError(test, err)
        test_name = self._get_test_name(test)
        self.errors.append((test_name, err))
        print("E", end="", flush=True)
    
    def addFailure(self, test, err):
        super().addFailure(test, err)
        test_name = self._get_test_name(test)
        self.failed.append((test_name, err))
        print("F", end="", flush=True)
    
    def _get_test_name(self, test):
        """Extract clean test name."""
        name = test._testMethodName
        # Remove 'test_' prefix and convert to readable format
        name = name.replace('test_', '').replace('_', ' ').title()
        return name
    
    def printErrors(self):
        """Print summary instead of full errors."""
        if self.failed or self.errors:
            print("\n\n" + "="*70)
            print("FAILURES & ERRORS")
            print("="*70)
            for test_name, err in self.failed + self.errors:
                print(f"\n❌ {test_name}")
                print(f"   {err[1]}")


def main():
    """Run tests with clean output."""
    print("Livnium-O Test Suite")
    print("="*70)
    print("Running tests", end=" ", flush=True)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestLivniumOSystem)
    
    # Run with custom result (suppress default output)
    import io
    stream = io.StringIO()
    runner = unittest.TextTestRunner(
        stream=stream,
        verbosity=0,
        resultclass=CleanTestResult
    )
    
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    if result.wasSuccessful():
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print("="*70)
    print(f"Total:  {result.test_count}")
    print(f"Passed: {len(result.passed)} ✅")
    if result.failed:
        print(f"Failed: {len(result.failed)} ❌")
    if result.errors:
        print(f"Errors: {len(result.errors)} ⚠️")
    
    # Print test categories
    if result.passed:
        print("\nTest Categories:")
        categories = {
            'S1 - Structure': [t for t in result.passed if 'S1' in t or 'Structure' in t or 'Tangency' in t or 'Position' in t],
            'S3 - Exposure': [t for t in result.passed if 'S3' in t or 'Exposure' in t or 'Solid Angle' in t or 'Symbolic Weight' in t],
            'K1 - Kissing': [t for t in result.passed if 'K1' in t or 'Kissing' in t],
            'D1-D5 - Derived': [t for t in result.passed if any(f'D{i}' in t for i in [1,2,3,4,5]) or 'Equilibrium' in t or 'Concentration' in t or 'Conservation' in t or 'Rotation' in t or 'Encoding' in t],
            'Other': [t for t in result.passed if not any(cat in t for cat in ['S1', 'S3', 'K1', 'D1', 'D2', 'D3', 'D4', 'D5', 'Structure', 'Exposure', 'Kissing', 'Equilibrium', 'Concentration', 'Conservation', 'Rotation', 'Encoding'])]
        }
        
        for category, tests in categories.items():
            if tests:
                print(f"  {category}: {len(tests)} tests")
    
    print()
    
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())


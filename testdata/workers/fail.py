"""
Test fixture: always fails with a structured error.
Used by Go tests to verify error handling in the pyworker contract.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

from common.run import run


def fail(request):
    raise ValueError("intentional test failure")


if __name__ == "__main__":
    run(fail)

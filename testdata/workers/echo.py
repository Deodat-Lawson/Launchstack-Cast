"""
Test fixture: echoes the inputs back as data.
Used by Go tests to verify the pyworker subprocess contract
without requiring any ML models.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

from common.run import run


def echo(request):
    return request.get("inputs", {})


if __name__ == "__main__":
    run(echo)

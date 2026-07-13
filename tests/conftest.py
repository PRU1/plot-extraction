import sys
from pathlib import Path

import pytest

# allow `import synthetic` from the tests dir
sys.path.insert(0, str(Path(__file__).parent))


def run_stub(fn, *args, **kwargs):
    """Call a function that may still be an unimplemented stub.

    Skips the test (rather than failing) while the stub raises
    NotImplementedError; the test activates automatically the moment the
    implementation lands.
    """
    try:
        return fn(*args, **kwargs)
    except NotImplementedError as e:
        pytest.skip(f"stub not yet implemented: {e}")

"""
Thin CLI entry point for the Databento term-structure transform. All the
actual decode logic (raw archive I/O, the anchor-leg symbol-resolution
algorithm, merge-and-write) lives in `src/data/databento_transform.py` as
importable, reusable library code — this script's only job is to expose it
as `python databento/transform_databento.py [asset ...]`.

Requires the raw `Data/databento_raw/*.zip` archives (from `submit_jobs.py`
+ `retry_jobs.py`) to already be present.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from data.databento_transform import run  # noqa: E402

if __name__ == "__main__":
    args = sys.argv[1:]
    run(assets=args if args else None)

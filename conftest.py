"""Makes the `app` package importable for pytest after app/ was moved into
the "Automated Fake News Detection" wrapper folder for organisation. Without
this, `from app.xxx import ...` in tests/ and run_pipeline.py would fail
with ModuleNotFoundError since app/ is no longer a direct child of the repo
root."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "Automated Fake News Detection"))

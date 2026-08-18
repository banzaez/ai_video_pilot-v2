"""Run Job API: python -m app.jobs"""

from __future__ import annotations

import os
import sys


def main() -> None:
    # Fallback if deps installed via --target vendor/
    root = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
    vendor = os.path.join(root, "vendor")
    if os.path.isdir(vendor) and vendor not in sys.path:
        sys.path.insert(0, vendor)
    # Ensure child reloader / import of app.jobs.api sees vendor
    prev = os.environ.get("PYTHONPATH", "")
    if vendor not in prev.split(os.pathsep):
        os.environ["PYTHONPATH"] = vendor + (os.pathsep + prev if prev else "")

    import uvicorn

    host = os.environ.get("JOBS_API_HOST", "127.0.0.1")
    port = int(os.environ.get("JOBS_API_PORT", "8765"))
    uvicorn.run(
        "app.jobs.api:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()

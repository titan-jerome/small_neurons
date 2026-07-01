#!/usr/bin/env python3
"""Download OpenNeuro ds005284 (The 26 By Biosemi Laser Pain Dataset) locally.

This wraps `openneuro-py`, the official OpenNeuro download client, so you get
resumable, checksum-verified downloads straight from the OpenNeuro S3 bucket
without needing AWS credentials (the bucket is public).

Usage
-----
    pip install openneuro-py

    # Full dataset (26 subjects, likely tens of GB -- expect a long download):
    python download_dataset.py --target-dir ./ds005284

    # Just a couple of subjects, e.g. to smoke-test compressibility_hypothesis.py:
    python download_dataset.py --target-dir ./ds005284 --subjects 01 02

The result is a standard BIDS tree you can point straight at the analysis
script:

    python compressibility_hypothesis.py --bids-root ./ds005284
"""

from __future__ import annotations

import argparse
import sys

try:
    import openneuro
except ImportError:
    sys.exit(
        "This script requires openneuro-py. Install it with:\n"
        "    pip install openneuro-py"
    )

DATASET_ID = "ds005284"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target-dir", default=f"./{DATASET_ID}",
                   help=f"Directory to download into (default: ./{DATASET_ID}).")
    p.add_argument("--subjects", nargs="*", default=None,
                   help="Subject labels to download without the 'sub-' prefix, "
                        "e.g. --subjects 01 02. Omit to download every subject.")
    p.add_argument("--tag", default=None,
                   help="Specific dataset snapshot/tag to download (default: latest).")
    return p


def run(target_dir: str, subjects: list[str] | None, tag: str | None) -> None:
    include = None
    if subjects:
        # openneuro-py expects path prefixes relative to the dataset root.
        include = [f"sub-{s}/" for s in subjects]
        print(f"Downloading {DATASET_ID} subjects: {', '.join(subjects)} -> {target_dir}")
    else:
        print(f"Downloading FULL dataset {DATASET_ID} -> {target_dir} "
              "(this can be tens of GB and take a while)")

    openneuro.download(
        dataset=DATASET_ID,
        target_dir=target_dir,
        tag=tag,
        include=include,
    )
    print(f"\nDone. Point the analysis script at it with:\n"
          f"    python compressibility_hypothesis.py --bids-root {target_dir}")


if __name__ == "__main__":
    args = build_parser().parse_args()
    run(args.target_dir, args.subjects, args.tag)

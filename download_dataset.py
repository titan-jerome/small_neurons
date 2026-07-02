#!/usr/bin/env python3
"""Download OpenNeuro ds005284 (The 26 By Biosemi Laser Pain Dataset) locally.

This wraps `openneuro-py`, the official OpenNeuro download client, so you get
resumable, checksum-verified downloads straight from the OpenNeuro S3 bucket
without needing AWS credentials (the bucket is public).

Usage
-----
    pip install openneuro-py requests

    # First, find out how subjects are actually labelled in this dataset --
    # BIDS subject labels are NOT guaranteed to be zero-padded numbers, and
    # guessing wrong (e.g. "01" when the real label is "1" or "pain01") makes
    # openneuro-py fail with "Could not find path in the dataset".
    python download_dataset.py --list-subjects

    # Full dataset (26 subjects, likely tens of GB -- expect a long download).
    # By default this SKIPS derivatives/ (~3 GB of the authors' own
    # preprocessed EEGLAB exports) since compressibility_hypothesis.py only
    # reads the raw sub-*/eeg/*_eeg.bdf files. Pass --include-derivatives to
    # get everything.
    python download_dataset.py --target-dir ./ds005284

    # Just a couple of subjects, using labels from --list-subjects above:
    python download_dataset.py --target-dir ./ds005284 --subjects 01 02

The result is a standard BIDS tree you can point straight at the analysis
script:

    python compressibility_hypothesis.py --bids-root ./ds005284
"""

from __future__ import annotations

import argparse
import re
import sys

try:
    import openneuro
except ImportError:
    sys.exit(
        "This script requires openneuro-py. Install it with:\n"
        "    pip install openneuro-py"
    )

DATASET_ID = "ds005284"
GRAPHQL_URL = "https://openneuro.org/crn/graphql"


def list_subjects(dataset_id: str, tag: str | None) -> list[str]:
    """Query OpenNeuro's GraphQL API for the real subject folder names.

    openneuro-py's --subjects/include filter needs an exact path prefix match
    (e.g. "sub-01/"); it does not guess or zero-pad numbers for you. Rather
    than assume a naming convention, we ask OpenNeuro directly for the actual
    top-level "sub-*" directories in this dataset/snapshot and print them, so
    you can copy the correct labels into --subjects.
    """
    try:
        import requests
    except ImportError:
        sys.exit("This requires the 'requests' package. Install it with:\n"
                 "    pip install requests")

    if tag:
        query = """
        query {
          snapshot(datasetId: "%s", tag: "%s") {
            files(recursive: true) { filename }
          }
        }""" % (dataset_id, tag)
        key_path = ("snapshot", "files")
    else:
        query = """
        query {
          dataset(id: "%s") {
            latestSnapshot {
              files(recursive: true) { filename }
            }
          }
        }""" % (dataset_id,)
        key_path = ("dataset", "latestSnapshot", "files")

    resp = requests.post(GRAPHQL_URL, json={"query": query}, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("errors"):
        sys.exit(f"OpenNeuro API error: {payload['errors']}")

    node = payload["data"]
    for key in key_path:
        node = node[key]
    filenames = [f["filename"] for f in node]

    subjects = sorted({
        m.group(1) for fn in filenames
        if (m := re.match(r"sub-([^/]+)/", fn))
    })
    return subjects


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target-dir", default=f"./{DATASET_ID}",
                   help=f"Directory to download into (default: ./{DATASET_ID}).")
    p.add_argument("--subjects", nargs="*", default=None,
                   help="Subject labels to download without the 'sub-' prefix, "
                        "e.g. --subjects 01 02. Omit to download every subject. "
                        "Run with --list-subjects first to see the real labels.")
    p.add_argument("--tag", default=None,
                   help="Specific dataset snapshot/tag to download (default: latest).")
    p.add_argument("--list-subjects", action="store_true",
                   help="Print the dataset's actual subject labels and exit "
                        "(no download).")
    p.add_argument("--include-derivatives", action="store_true",
                   help="Also download derivatives/ (~3 GB of the authors' own "
                        "preprocessed EEGLAB exports). Skipped by default since "
                        "compressibility_hypothesis.py only needs the raw "
                        "sub-*/eeg/*_eeg.bdf files.")
    return p


def run(target_dir: str, subjects: list[str] | None, tag: str | None,
        include_derivatives: bool) -> None:
    include = None
    exclude = None if include_derivatives else ["derivatives/"]
    if subjects:
        # openneuro-py matches --include against exact path prefixes, so a
        # wrong guess (e.g. "01" when the real label is "1") fails loudly
        # rather than downloading nothing. Verify against the real listing
        # first so we fail fast with a helpful message instead of the raw
        # RuntimeError from openneuro-py.
        available = list_subjects(DATASET_ID, tag)
        missing = [s for s in subjects if s not in available]
        if missing:
            sys.exit(
                f"Requested subject label(s) not found in {DATASET_ID}: {missing}\n"
                f"Available labels: {available}\n"
                "(Run with --list-subjects to see this list without downloading.)"
            )
        include = [f"sub-{s}/" for s in subjects]
        print(f"Downloading {DATASET_ID} subjects: {', '.join(subjects)} -> {target_dir}")
    else:
        print(f"Downloading FULL dataset {DATASET_ID} -> {target_dir} "
              "(this can be tens of GB and take a while)")

    if exclude:
        print(f"Skipping {exclude} (pass --include-derivatives to fetch it too).")

    openneuro.download(
        dataset=DATASET_ID,
        target_dir=target_dir,
        tag=tag,
        include=include,
        exclude=exclude,
    )
    print(f"\nDone. Point the analysis script at it with:\n"
          f"    python compressibility_hypothesis.py --bids-root {target_dir}")


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.list_subjects:
        labels = list_subjects(DATASET_ID, args.tag)
        print(f"{DATASET_ID} subject labels ({len(labels)} total):")
        for label in labels:
            print(f"  {label}  (--subjects {label})")
        sys.exit(0)
    run(args.target_dir, args.subjects, args.tag, args.include_derivatives)

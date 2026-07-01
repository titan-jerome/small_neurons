#!/usr/bin/env python3
"""Test Michael Johnson's algorithmic compressibility hypothesis on EEG.

Background
----------
The Symmetry Theory of Valence (STV), proposed by Michael Johnson and the
Qualia Research Institute, conjectures that the *valence* (pleasantness) of an
experience corresponds to the symmetry of the underlying brain activity.
A related, testable corollary is the "algorithmic compressibility" hypothesis:
more symmetric / more ordered states should be *more compressible* (they contain
more redundancy), whereas noisy, high-valence-negative states such as acute pain
should be *less compressible* (more algorithmically random).

This script operationalises that idea in the crudest possible way that is still
falsifiable: we take short EEG windows during laser-evoked pain and matched
resting windows, serialise the raw sample matrix to bytes, run general-purpose
lossless compressors (gzip / bz2) over the bytes, and compare the resulting
compression ratios. Prediction under the hypothesis:

    compression_ratio(baseline) < compression_ratio(pain)

i.e. the calm/baseline window compresses to a *smaller* fraction of its original
size (more redundancy) than the painful window.

Dataset
-------
OpenNeuro ds005284 -- "The 26 By Biosemi Laser Pain Dataset". BIDS-formatted,
recorded on a BioSemi ActiveTwo system, so the raw files are 24-bit ``.bdf``.

Caveats (read before believing any p-value this prints)
-------------------------------------------------------
* Compression ratio of floating-point EEG is dominated by amplitude/variance and
  by the text/byte encoding, *not* purely by "algorithmic symmetry". A pain
  window with larger evoked potentials can look less compressible simply because
  it has more dynamic range. This is a toy probe, not a validation of STV.
* Event trigger codes differ per dataset; the value used here is a best guess and
  is exposed as a CLI flag. Always inspect ``raw.annotations`` / the events array
  for your data before trusting the epoching.

Usage
-----
    python compressibility_hypothesis.py --bids-root /path/to/ds005284

Dependencies: mne, numpy, scipy (and optionally mne-bids). Install with:
    pip install mne mne-bids numpy scipy
"""

from __future__ import annotations

import argparse
import bz2
import glob
import gzip
import os
import sys
from dataclasses import dataclass, field

import numpy as np

try:
    import mne
except ImportError:  # pragma: no cover - dependency hint
    sys.exit("This script requires MNE-Python. Install it with: pip install mne")

from scipy import stats


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Occipital / parietal channels. Restricting to the back of the head keeps us
# away from frontal EMG (jaw clench), EOG (blinks) and temporalis muscle bursts
# that would otherwise inflate the "randomness" of the pain window for reasons
# that have nothing to do with valence. Names follow the 10-20 system.
POSTERIOR_CHANNELS = ["O1", "O2", "Oz", "P3", "P4", "Pz", "P7", "P8", "PO3", "PO4"]

# BioSemi ActiveTwo systems record with hardware channel labels ("A1".."A32",
# "B1".."B32", ...) rather than 10-20 names -- BIDS conversion does not always
# rename them. These are BioSemi's own published wiring layouts for their
# standard 32- and 64-electrode caps, used to translate hardware labels to
# 10-20 names so POSTERIOR_CHANNELS above can match. If this dataset uses a
# non-standard or larger (128/256-channel) cap, neither table will apply and
# pick_posterior_channels() will raise rather than silently mis-map channels.
BIOSEMI_32_TO_1020 = {
    "A1": "Fp1", "A2": "AF3", "A3": "F7", "A4": "F3", "A5": "FC1", "A6": "FC5",
    "A7": "T7", "A8": "C3", "A9": "CP1", "A10": "CP5", "A11": "P7", "A12": "P3",
    "A13": "Pz", "A14": "PO3", "A15": "O1", "A16": "Oz", "A17": "O2", "A18": "PO4",
    "A19": "P4", "A20": "P8", "A21": "CP6", "A22": "CP2", "A23": "C4", "A24": "T8",
    "A25": "FC6", "A26": "FC2", "A27": "F4", "A28": "F8", "A29": "AF4", "A30": "Fp2",
    "A31": "Fz", "A32": "Cz",
}
BIOSEMI_64_TO_1020 = {
    "A1": "Fp1", "A2": "AF7", "A3": "AF3", "A4": "F1", "A5": "F3", "A6": "F5",
    "A7": "F7", "A8": "FT7", "A9": "FC5", "A10": "FC3", "A11": "FC1", "A12": "C1",
    "A13": "C3", "A14": "C5", "A15": "T7", "A16": "TP7", "A17": "CP5", "A18": "CP3",
    "A19": "CP1", "A20": "P1", "A21": "P3", "A22": "P5", "A23": "P7", "A24": "P9",
    "A25": "PO7", "A26": "PO3", "A27": "O1", "A28": "Iz", "A29": "Oz", "A30": "POz",
    "A31": "Pz", "A32": "CPz",
    "B1": "Fpz", "B2": "Fp2", "B3": "AF8", "B4": "AF4", "B5": "AFz", "B6": "Fz",
    "B7": "F2", "B8": "F4", "B9": "F6", "B10": "F8", "B11": "FT8", "B12": "FC6",
    "B13": "FC4", "B14": "FC2", "B15": "FCz", "B16": "Cz", "B17": "C2", "B18": "C4",
    "B19": "C6", "B20": "T8", "B21": "TP8", "B22": "CP6", "B23": "CP4", "B24": "CP2",
    "B25": "P2", "B26": "P4", "B27": "P6", "B28": "P8", "B29": "P10", "B30": "PO8",
    "B31": "PO4", "B32": "O2",
}

# Pain window: 0.0 -> 3.0 s after the laser trigger.
PAIN_TMIN, PAIN_TMAX = 0.0, 3.0

# Baseline window: a matched 3.0 s of rest *before* the stimulus (-4.0 -> -1.0 s).
# We stop 1 s before the trigger to avoid any anticipatory contamination.
BASE_TMIN, BASE_TMAX = -4.0, -1.0

# Fixed decimal precision used when serialising the float matrix to text. See
# matrix_to_bytes() for why this matters.
DECIMALS = 4


# ---------------------------------------------------------------------------
# Data serialisation: turning a float matrix into compressible bytes
# ---------------------------------------------------------------------------

def matrix_to_bytes(data: np.ndarray, decimals: int = DECIMALS) -> bytes:
    """Flatten a (channels x time) float matrix into a deterministic byte string.

    Why this step exists
    --------------------
    ``gzip`` and ``bz2`` operate on *bytes*, not on numpy arrays. To measure the
    "algorithmic redundancy" of an EEG window we must first commit to a single,
    reproducible byte representation of the numbers. The representation we choose
    changes what "compressible" means, so we make the choice explicit:

    1. Scale to microvolts. MNE stores EEG in volts, so raw values look like
       1.7e-05. Multiplying by 1e6 gives human-scale microvolt numbers (~17.0)
       whose fixed-precision text is compact and comparable across windows.
    2. Round to a fixed number of decimals. This throws away sub-precision noise
       so that the compressor is judging the *structure* of the signal rather
       than the entropy of meaningless trailing float digits. Crucially it makes
       the byte length depend only on the magnitude/pattern of the data, not on
       floating-point representation artefacts.
    3. Render to a flat, fixed-format ASCII string. We join every value with a
       single space using a fixed ``%.<d>f`` format so each sample occupies a
       predictable field. A regular, self-similar signal then produces long runs
       of similar characters that gzip/bz2 can exploit; an irregular signal does
       not. Finally we ``.encode("ascii")`` to get the raw bytes the compressors
       consume.

    Using a *text* encoding (rather than raw float bytes via ``.tobytes()``) is a
    deliberate choice: fixed-precision decimal text exposes redundancy in a way
    that is intuitive and identical across platforms/endianness, at the cost of
    being a coarser probe. Both windows go through the exact same pipeline, so
    the comparison between them stays fair.
    """
    # 1. volts -> microvolts for readable, compact fixed-precision text.
    scaled = np.asarray(data, dtype=np.float64) * 1e6
    # 2 + 3. flatten row-major (channel by channel) and format every value with
    # the same fixed decimal precision, space-separated.
    fmt = "%.{}f".format(decimals)
    flat = scaled.ravel(order="C")
    text = " ".join(fmt % v for v in flat)
    # Raw bytes handed to the compressor.
    return text.encode("ascii")


def compression_ratios(raw_bytes: bytes) -> dict[str, float]:
    """Return gzip and bz2 compression ratios for a byte string.

    Compression Ratio = compressed_size / original_size  (lower = more redundant).
    """
    original = len(raw_bytes)
    if original == 0:
        return {"gzip": float("nan"), "bz2": float("nan")}
    gz = len(gzip.compress(raw_bytes, compresslevel=9))
    bz = len(bz2.compress(raw_bytes, compresslevel=9))
    return {"gzip": gz / original, "bz2": bz / original}


# ---------------------------------------------------------------------------
# File discovery / loading
# ---------------------------------------------------------------------------

def find_subject_files(bids_root: str) -> list[tuple[str, str]]:
    """Discover raw EEG files in a BIDS tree.

    Returns a list of (subject_id, filepath). We look for BioSemi ``.bdf`` first
    (this dataset), then fall back to ``.edf`` / ``.fif`` so the script also works
    on re-exported copies.
    """
    matches: list[tuple[str, str]] = []
    for ext in ("bdf", "edf", "fif"):
        pattern = os.path.join(bids_root, "sub-*", "**", f"*_eeg.{ext}")
        for path in sorted(glob.glob(pattern, recursive=True)):
            # Extract the BIDS subject label ("sub-01") from the path.
            sub = next((p for p in path.split(os.sep) if p.startswith("sub-")), "unknown")
            matches.append((sub, path))
        if matches:
            break  # prefer a single, homogeneous file type
    return matches


def load_raw(path: str):
    """Read a single raw file with the appropriate MNE reader."""
    if path.endswith(".bdf"):
        return mne.io.read_raw_bdf(path, preload=True, verbose="ERROR")
    if path.endswith(".edf"):
        return mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    return mne.io.read_raw_fif(path, preload=True, verbose="ERROR")


def standardize_channel_names(raw) -> None:
    """Rename BioSemi hardware labels (A1, B12, ...) to 10-20 names in-place.

    No-op if the recording already uses 10-20 names. Only renames channels that
    are actually present, so extra externals (EXG1-8, GSR, Erg1, etc.) are left
    alone and simply won't be picked as posterior channels later.
    """
    hw_channels = [ch for ch in raw.ch_names if ch in BIOSEMI_64_TO_1020 or ch in BIOSEMI_32_TO_1020]
    if not hw_channels:
        return  # already using standard names (or an unrecognised layout)

    has_b_channels = any(ch.startswith("B") for ch in raw.ch_names)
    mapping_table = BIOSEMI_64_TO_1020 if has_b_channels else BIOSEMI_32_TO_1020
    cap_size = "64" if has_b_channels else "32"
    mapping = {ch: mapping_table[ch] for ch in raw.ch_names if ch in mapping_table}
    print(f"    [info] renaming {len(mapping)} BioSemi hardware channels "
          f"using the standard {cap_size}-electrode cap layout -- verify this "
          "matches the actual cap used for ds005284 if results look off.")
    raw.rename_channels(mapping)


def pick_posterior_channels(raw) -> None:
    """Restrict the recording in-place to available occipital/parietal channels."""
    standardize_channel_names(raw)
    available = [ch for ch in POSTERIOR_CHANNELS if ch in raw.ch_names]
    if not available:
        raise RuntimeError(
            "None of the expected posterior channels "
            f"{POSTERIOR_CHANNELS} were found. Channels present: {raw.ch_names[:20]}..."
        )
    raw.pick(available)


def get_events(raw, stim_code: int | None):
    """Extract an events array, from a stim channel or from annotations.

    BioSemi files carry triggers on a ``Status`` stim channel; some BIDS exports
    instead store them as annotations. We try both and, if the requested trigger
    code is not present, fall back to the most frequent non-zero code so the
    script still produces epochs (with a loud warning).
    """
    events = None
    # Path A: hardware stim channel (typical for .bdf).
    try:
        events = mne.find_events(raw, stim_channel="Status", verbose="ERROR")
    except (ValueError, RuntimeError):
        events = None

    # Path B: annotations -> events.
    if events is None or len(events) == 0:
        try:
            events, _ = mne.events_from_annotations(raw, verbose="ERROR")
        except (ValueError, RuntimeError):
            events = np.empty((0, 3), dtype=int)

    if len(events) == 0:
        return events, None

    present_codes = np.unique(events[:, 2])
    chosen = stim_code
    if chosen is None or chosen not in present_codes:
        # Fall back to the most common event code.
        codes, counts = np.unique(events[:, 2], return_counts=True)
        chosen = int(codes[np.argmax(counts)])
        print(
            f"    [warn] stim code {stim_code} not found; using most frequent "
            f"code {chosen} (present codes: {present_codes.tolist()})"
        )
    return events, chosen


# ---------------------------------------------------------------------------
# Per-subject processing
# ---------------------------------------------------------------------------

@dataclass
class SubjectResult:
    subject: str
    pain: dict[str, list[float]] = field(default_factory=lambda: {"gzip": [], "bz2": []})
    base: dict[str, list[float]] = field(default_factory=lambda: {"gzip": [], "bz2": []})

    def add(self, condition: str, ratios: dict[str, float]) -> None:
        target = self.pain if condition == "pain" else self.base
        for algo, r in ratios.items():
            if not np.isnan(r):
                target[algo].append(r)

    def mean(self, condition: str, algo: str) -> float:
        vals = (self.pain if condition == "pain" else self.base)[algo]
        return float(np.mean(vals)) if vals else float("nan")


def process_subject(sub: str, path: str, stim_code: int | None) -> SubjectResult | None:
    """Compute mean pain vs baseline compression ratios for one subject."""
    result = SubjectResult(subject=sub)
    print(f"[{sub}] loading {os.path.basename(path)}")
    raw = load_raw(path)
    pick_posterior_channels(raw)

    events, code = get_events(raw, stim_code)
    if code is None:
        print(f"    [warn] no events found for {sub}; skipping.")
        return None

    sfreq = raw.info["sfreq"]

    # Build epochs for the pain window (0 -> 3 s) and the baseline window
    # (-4 -> -1 s) from the *same* trigger. baseline=None so MNE does not
    # subtract any mean; we want the untouched raw signal for compression.
    # We disable rejection so every trial contributes; the posterior-channel
    # restriction is our artefact control instead.
    common = dict(events=events, event_id={"laser": code}, baseline=None,
                  preload=True, reject=None, verbose="ERROR")
    pain_epochs = mne.Epochs(raw, tmin=PAIN_TMIN, tmax=PAIN_TMAX, **common)
    base_epochs = mne.Epochs(raw, tmin=BASE_TMIN, tmax=BASE_TMAX, **common)

    # Guarantee identical sample counts. Rounding differences in tmax can yield an
    # off-by-one length between windows; we crop both to the shared minimum so the
    # byte strings are strictly comparable.
    n_pain = pain_epochs.get_data(copy=False).shape[-1]
    n_base = base_epochs.get_data(copy=False).shape[-1]
    n_samples = min(n_pain, n_base)
    print(f"    fs={sfreq:.0f} Hz, {len(pain_epochs)} trials, "
          f"{n_samples} samples/window ({n_samples / sfreq:.3f} s)")

    pain_data = pain_epochs.get_data(copy=True)[..., :n_samples]  # (trials, ch, time)
    base_data = base_epochs.get_data(copy=True)[..., :n_samples]

    # Some trials may be dropped by one condition (e.g. window runs off the end of
    # the recording). Align on the trials kept by *both* conditions.
    n_trials = min(pain_data.shape[0], base_data.shape[0])
    for i in range(n_trials):
        result.add("pain", compression_ratios(matrix_to_bytes(pain_data[i])))
        result.add("base", compression_ratios(matrix_to_bytes(base_data[i])))

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(bids_root: str, stim_code: int | None, limit: int | None) -> None:
    files = find_subject_files(bids_root)
    if not files:
        sys.exit(f"No raw EEG files found under {bids_root!r}. "
                 "Expected a BIDS tree like sub-01/eeg/sub-01_task-*_eeg.bdf")
    if limit:
        files = files[:limit]
    print(f"Found {len(files)} recording(s).\n")

    results: list[SubjectResult] = []
    for sub, path in files:
        try:
            res = process_subject(sub, path, stim_code)
        except Exception as exc:  # keep going; one bad file shouldn't kill the run
            print(f"    [error] {sub}: {exc}")
            continue
        if res is not None and res.pain["gzip"] and res.base["gzip"]:
            results.append(res)

    if len(results) < 2:
        sys.exit("\nNeed at least 2 usable subjects for a paired t-test.")

    # Aggregate to one mean ratio per subject per condition, then compare.
    print("\n" + "=" * 70)
    print("Per-subject mean compression ratio (lower = more compressible)")
    print("=" * 70)
    print(f"{'subject':<12}{'gzip pain':>11}{'gzip base':>11}"
          f"{'bz2 pain':>11}{'bz2 base':>11}")

    summary = {algo: {"pain": [], "base": []} for algo in ("gzip", "bz2")}
    for res in results:
        row = [res.subject]
        for algo in ("gzip", "bz2"):
            mp, mb = res.mean("pain", algo), res.mean("base", algo)
            summary[algo]["pain"].append(mp)
            summary[algo]["base"].append(mb)
        print(f"{res.subject:<12}"
              f"{res.mean('pain', 'gzip'):>11.4f}{res.mean('base', 'gzip'):>11.4f}"
              f"{res.mean('pain', 'bz2'):>11.4f}{res.mean('base', 'bz2'):>11.4f}")

    print("\n" + "=" * 70)
    print("Paired t-test: are BASELINE windows more compressible than PAIN?")
    print("H1: ratio(baseline) < ratio(pain)   [baseline has more redundancy]")
    print("=" * 70)
    for algo in ("gzip", "bz2"):
        pain = np.array(summary[algo]["pain"])
        base = np.array(summary[algo]["base"])
        # One-sided paired t-test (baseline < pain).
        t_stat, p_two = stats.ttest_rel(base, pain)
        # Convert the two-sided p to a one-sided p for the directional hypothesis.
        p_one = p_two / 2 if t_stat < 0 else 1 - p_two / 2
        print(f"\n[{algo}]  mean baseline={base.mean():.4f}  "
              f"mean pain={pain.mean():.4f}  (n={len(pain)} subjects)")
        print(f"        t = {t_stat:+.3f}   one-sided p = {p_one:.4g}")
        if p_one < 0.05 and base.mean() < pain.mean():
            print("        => Baseline significantly MORE compressible than pain "
                  "(consistent with the hypothesis).")
        else:
            print("        => No significant support for the hypothesis in this data.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bids-root", required=True,
                   help="Path to the ds005284 BIDS root directory.")
    p.add_argument("--stim-code", type=int, default=None,
                   help="Trigger code for the laser stimulus. If omitted or "
                        "absent from the data, the most frequent event code is "
                        "used (with a warning). INSPECT YOUR EVENTS FIRST.")
    p.add_argument("--limit", type=int, default=None,
                   help="Process only the first N recordings (for quick testing).")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    run(args.bids_root, args.stim_code, args.limit)

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
* The laser stimulus in ds005284 is the "condition 54" event (16 trials/subject,
  confirmed against the README and events.tsv sidecars). "condition 64" is a
  burst of spurious triggers at recording onset in some subjects and is excluded.
  This is the ``--trial-type`` default; override it for other datasets.

Output
------
Instead of dumping to the terminal, the script writes a self-contained Markdown
report (default: ``compressibility_report.md``, override with ``--report``)
covering the assumptions, the preprocessing pipeline, per-recording provenance
(channels, events, trials), the per-subject compression ratios, and the paired
t-test with a verdict. The terminal only shows brief progress + the report path.

Usage
-----
    python compressibility_hypothesis.py --bids-root /path/to/ds005284
    python compressibility_hypothesis.py --bids-root ./ds005284 --report run1.md

This dataset stores triggers in BIDS ``*_events.tsv`` sidecars rather than a
raw stim channel, so ``mne-bids`` is required (not optional) to read them.

Dependencies: mne, mne-bids, numpy, scipy. Install with:
    pip install mne mne-bids numpy scipy
"""

from __future__ import annotations

import argparse
import bz2
import glob
import gzip
import os
import re
import sys
from dataclasses import dataclass, field

import numpy as np

try:
    import mne
except ImportError:  # pragma: no cover - dependency hint
    sys.exit("This script requires MNE-Python. Install it with: pip install mne")

try:
    import mne_bids
except ImportError:
    mne_bids = None

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

# Laser-stimulus event label for ds005284. The events.tsv files use a "value"
# column (there is no "trial_type" column) whose entries MNE surfaces verbatim
# as annotation descriptions. Inspecting the sidecars shows two distinct labels:
#   * "condition 54" -- the 16 laser pain trials, spaced ~12-13 s apart, present
#     identically in every subject. This matches the README ("16 trials,
#     approximately every 20 seconds") and the task events.json ("Laser stimli").
#   * "condition 64" -- NOT a stimulus: in sub-001 it is a burst of 14 triggers
#     crammed into a ~200 ms window at t~=7.8 s (a recording-onset/status-channel
#     glitch), ~2 min before the first real laser. It appears in only some
#     subjects and must be excluded.
# We therefore pin "condition 54" rather than trusting a "most frequent code"
# heuristic, which would silently select the artifact burst in any subject where
# it happened to contain more triggers than the 16 real trials.
DEFAULT_TRIAL_TYPE = "condition 54"

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


def _bids_entities(filename: str) -> dict:
    """Parse BIDS entities (sub, ses, task, acq, run) out of a filename."""
    base = os.path.basename(filename)
    entities = {}
    for key in ("sub", "ses", "task", "acq", "run"):
        m = re.search(rf"{key}-([A-Za-z0-9]+)", base)
        if m:
            entities[key] = m.group(1)
    return entities


def load_raw(path: str, bids_root: str):
    """Read a raw file, preferring MNE-BIDS so ``*_events.tsv`` is attached.

    BIDS EEG datasets store trigger onsets/labels in a companion
    ``*_events.tsv`` sidecar, not (necessarily) in the raw file's own stim
    channel or annotations. Reading the ``.bdf``/``.edf`` directly with a plain
    MNE reader silently skips that sidecar, which is why a naive
    ``read_raw_bdf`` call can find zero events even though the dataset has
    them. ``mne_bids.read_raw_bids`` reads ``events.tsv`` and converts each row
    into an ``mne.Annotations`` entry (description = the ``trial_type``/
    ``value`` column), so ``mne.events_from_annotations`` downstream then has
    something to work with.
    """
    if mne_bids is not None and path.endswith((".bdf", ".edf")):
        entities = _bids_entities(path)
        if "sub" in entities and "task" in entities:
            try:
                bids_path = mne_bids.BIDSPath(
                    subject=entities["sub"], session=entities.get("ses"),
                    task=entities["task"], acquisition=entities.get("acq"),
                    run=entities.get("run"), datatype="eeg", suffix="eeg",
                    root=bids_root,
                )
                raw = mne_bids.read_raw_bids(
                    bids_path, extra_params={"preload": True}, verbose="ERROR"
                )
                raw.load_data()
                return raw
            except Exception as exc:
                print(f"    [warn] mne_bids read failed ({exc}); falling back "
                      "to a plain reader (events.tsv annotations will NOT be "
                      "attached).")

    if path.endswith(".bdf"):
        return mne.io.read_raw_bdf(path, preload=True, verbose="ERROR")
    if path.endswith(".edf"):
        return mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    return mne.io.read_raw_fif(path, preload=True, verbose="ERROR")


def standardize_channel_names(raw) -> str:
    """Rename BioSemi hardware labels (A1, B12, ...) to 10-20 names in-place.

    No-op if the recording already uses 10-20 names. Only renames channels that
    are actually present, so extra externals (EXG1-8, GSR, Erg1, etc.) are left
    alone and simply won't be picked as posterior channels later. Returns a
    short human-readable description of what was done, for the report.
    """
    hw_channels = [ch for ch in raw.ch_names if ch in BIOSEMI_64_TO_1020 or ch in BIOSEMI_32_TO_1020]
    if not hw_channels:
        return "none (already 10-20 names)"

    has_b_channels = any(ch.startswith("B") for ch in raw.ch_names)
    mapping_table = BIOSEMI_64_TO_1020 if has_b_channels else BIOSEMI_32_TO_1020
    cap_size = "64" if has_b_channels else "32"
    mapping = {ch: mapping_table[ch] for ch in raw.ch_names if ch in mapping_table}
    raw.rename_channels(mapping)
    return f"{len(mapping)} channels via standard BioSemi {cap_size}-electrode layout"


def pick_posterior_channels(raw) -> tuple[list[str], str]:
    """Restrict the recording in-place to available occipital/parietal channels.

    Returns (channels_kept, cap_mapping_description).
    """
    cap_mapping = standardize_channel_names(raw)
    available = [ch for ch in POSTERIOR_CHANNELS if ch in raw.ch_names]
    if not available:
        raise RuntimeError(
            "None of the expected posterior channels "
            f"{POSTERIOR_CHANNELS} were found. Channels present: {raw.ch_names[:20]}..."
        )
    raw.pick(available)
    return available, cap_mapping


def get_events(raw, stim_code: int | None, trial_type: str | None):
    """Extract an events array, from a stim channel or from annotations.

    BioSemi files carry triggers on a ``Status`` stim channel; BIDS exports
    (this dataset) instead attach ``events.tsv`` as annotations once loaded via
    ``mne_bids.read_raw_bids``. We try the hardware channel first, then fall
    back to annotations. Annotation descriptions come from the ``trial_type``/
    ``value`` column and are arbitrary strings, so ``--stim-code`` (an int)
    cannot select them directly -- use ``--trial-type`` instead. By default
    ``trial_type`` is the dataset's known laser label (see DEFAULT_TRIAL_TYPE).
    If it is empty (auto-detect), we look for a description containing "laser",
    and failing that fall back to the single most frequent code so the script
    still produces *something*. How the code was chosen (and any warning) is
    recorded in the returned ``info`` dict for the report rather than printed.
    Only events matching the chosen label are epoched, so unrelated labels
    (e.g. the "condition 64" artifact burst) are excluded.

    Returns ``(events, chosen_code_or_None, info)``.
    """
    info: dict = {"available": {}, "method": None, "note": None,
                  "chosen_label": None, "chosen_code": None}
    events = None
    # Path A: hardware stim channel (typical for .bdf).
    try:
        events = mne.find_events(raw, stim_channel="Status", verbose="ERROR")
    except (ValueError, RuntimeError):
        events = None

    # Path B: annotations -> events (this is where BIDS events.tsv rows land).
    event_id_map = None
    if events is None or len(events) == 0:
        try:
            events, event_id_map = mne.events_from_annotations(raw, verbose="ERROR")
        except (ValueError, RuntimeError):
            events = np.empty((0, 3), dtype=int)

    if event_id_map is not None:
        # Record available labels (as plain str) with their trigger codes.
        info["available"] = {str(desc): int(c) for desc, c in event_id_map.items()}

    if len(events) == 0:
        info["note"] = "no events found"
        return events, None, info

    present_codes = np.unique(events[:, 2])
    chosen = stim_code if stim_code in present_codes else None
    if chosen is not None:
        info["method"] = "stim-code"

    if chosen is None and event_id_map is not None:
        # Treat an empty --trial-type as "auto-detect" (an empty substring would
        # otherwise match every description).
        if trial_type:
            matches = [c for desc, c in event_id_map.items()
                       if desc == trial_type or trial_type.lower() in desc.lower()]
            if matches:
                chosen = matches[0]
                info["method"] = f"--trial-type {trial_type!r}"
            else:
                info["note"] = (f"--trial-type {trial_type!r} not found in "
                                f"{list(info['available'])}")
        if chosen is None:
            laser_matches = [c for desc, c in event_id_map.items()
                              if "laser" in desc.lower()]
            if laser_matches:
                chosen = laser_matches[0]
                info["method"] = "auto-detect ('laser' in label)"

    if chosen is None:
        # Last resort: the single most frequent code, whatever it is.
        codes, counts = np.unique(events[:, 2], return_counts=True)
        chosen = int(codes[np.argmax(counts)])
        info["method"] = "fallback: most frequent code"
        info["note"] = (info["note"] + "; " if info["note"] else "") + \
            "no --stim-code/--trial-type match -- used most frequent code"

    inv = {c: str(desc) for desc, c in (event_id_map or {}).items()}
    info["chosen_code"] = int(chosen)
    info["chosen_label"] = inv.get(chosen, str(chosen))
    return events, chosen, info


# ---------------------------------------------------------------------------
# Per-subject processing
# ---------------------------------------------------------------------------

@dataclass
class SubjectResult:
    """Everything we learned about one recording, for both stats and the report."""
    subject: str
    filename: str = ""
    status: str = "ok"                 # "ok" | "skipped: ..." | "error: ..."
    sfreq: float = float("nan")
    channels_used: list[str] = field(default_factory=list)
    cap_mapping: str = ""
    event_label: str = ""
    event_code: int | None = None
    event_method: str = ""
    available_events: dict[str, int] = field(default_factory=dict)
    n_stim_events: int = 0             # events matching the chosen label
    n_other_events: int = 0            # events with other labels (excluded)
    n_trials: int = 0                  # trials actually compressed
    n_samples: int = 0                 # samples per window
    window_sec: float = float("nan")
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

    @property
    def usable(self) -> bool:
        return bool(self.pain["gzip"]) and bool(self.base["gzip"])


def process_subject(sub: str, path: str, bids_root: str, stim_code: int | None,
                    trial_type: str | None) -> SubjectResult:
    """Compute mean pain vs baseline compression ratios for one subject.

    Always returns a SubjectResult; on skip/error the ``status`` field explains
    why and the ratio lists stay empty (so it is documented in the report rather
    than silently dropped).
    """
    result = SubjectResult(subject=sub, filename=os.path.basename(path))
    print(f"[{sub}] processing {result.filename} ...")
    raw = load_raw(path, bids_root)
    result.channels_used, result.cap_mapping = pick_posterior_channels(raw)

    events, code, info = get_events(raw, stim_code, trial_type)
    result.available_events = info["available"]
    result.event_method = info["method"] or ""
    if code is None:
        result.status = f"skipped: {info.get('note') or 'no usable events'}"
        return result
    result.event_label = info["chosen_label"]
    result.event_code = code
    result.n_stim_events = int(np.sum(events[:, 2] == code))
    result.n_other_events = int(len(events) - result.n_stim_events)
    result.sfreq = float(raw.info["sfreq"])

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
    result.n_samples = min(n_pain, n_base)
    result.window_sec = result.n_samples / result.sfreq

    pain_data = pain_epochs.get_data(copy=True)[..., :result.n_samples]  # (trials, ch, time)
    base_data = base_epochs.get_data(copy=True)[..., :result.n_samples]

    # Some trials may be dropped by one condition (e.g. window runs off the end of
    # the recording). Align on the trials kept by *both* conditions.
    result.n_trials = min(pain_data.shape[0], base_data.shape[0])
    for i in range(result.n_trials):
        result.add("pain", compression_ratios(matrix_to_bytes(pain_data[i])))
        result.add("base", compression_ratios(matrix_to_bytes(base_data[i])))

    if not result.usable:
        result.status = "skipped: no trials survived epoching"
    print(f"    {result.status}"
          + (f" | fs={result.sfreq:.0f} Hz, {result.n_trials} trials, "
             f"{result.n_samples} samples/window" if result.usable else ""))
    return result


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _fmt(x: float, nd: int = 4) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def paired_test(results: list[SubjectResult], algo: str) -> dict:
    """One-sided paired t-test (baseline < pain) on per-subject mean ratios."""
    pain = np.array([r.mean("pain", algo) for r in results])
    base = np.array([r.mean("base", algo) for r in results])
    out = {"algo": algo, "n": len(results),
           "mean_pain": float(pain.mean()), "mean_base": float(base.mean()),
           "t": float("nan"), "p_one": float("nan"), "supported": False}
    if len(results) >= 2:
        t_stat, p_two = stats.ttest_rel(base, pain)
        p_one = p_two / 2 if t_stat < 0 else 1 - p_two / 2
        out["t"], out["p_one"] = float(t_stat), float(p_one)
        out["supported"] = bool(p_one < 0.05 and base.mean() < pain.mean())
    return out


def build_report(all_results: list[SubjectResult], config: dict) -> str:
    """Render the full run -- assumptions, preprocessing, data, stats -- as Markdown."""
    usable = [r for r in all_results if r.usable]
    skipped = [r for r in all_results if not r.usable]
    L: list[str] = []

    L.append("# Algorithmic Compressibility of Pain vs. Baseline EEG")
    L.append("")
    L.append(f"*Generated {config['timestamp']}*")
    L.append("")
    L.append("Test of the **algorithmic compressibility** corollary of the Symmetry "
             "Theory of Valence: negative-valence states (acute laser pain) are "
             "predicted to be **less compressible** (more algorithmically random) "
             "than matched calm/baseline states. Operationalised by gzip/bz2 "
             "compression of short posterior-EEG windows.")
    L.append("")
    L.append("**Directional hypothesis (H1):** "
             "`compression_ratio(baseline) < compression_ratio(pain)` "
             "(baseline carries more redundancy). Lower ratio = more compressible.")
    L.append("")

    # ---- Assumptions & configuration --------------------------------------
    L.append("## 1. Assumptions & configuration")
    L.append("")
    L.append("| Parameter | Value | Rationale |")
    L.append("|---|---|---|")
    L.append(f"| Dataset (BIDS root) | `{config['bids_root']}` | OpenNeuro ds005284 |")
    L.append(f"| Posterior channels targeted | {', '.join(POSTERIOR_CHANNELS)} | "
             "occipital/parietal only, to keep frontal EMG (jaw clench) & EOG "
             "(blinks) out of the compression score |")
    L.append(f"| Pain window | {PAIN_TMIN:.1f} to {PAIN_TMAX:.1f} s post-stimulus | "
             "laser-evoked response |")
    L.append(f"| Baseline window | {BASE_TMIN:.1f} to {BASE_TMAX:.1f} s | "
             "matched-length rest, ending 1 s pre-stimulus to avoid anticipation |")
    L.append(f"| Stimulus event | `{config['trial_type'] or 'auto-detect'}` | "
             "the laser trials; other labels (e.g. the `condition 64` onset-glitch "
             "burst) are excluded |")
    L.append(f"| Serialisation | µV, {DECIMALS} decimals, space-separated ASCII | "
             "fixed-precision text exposes signal redundancy identically across "
             "platforms (see `matrix_to_bytes`) |")
    L.append("| Compressors | gzip -9, bz2 -9 | general-purpose lossless |")
    L.append("| Baseline correction | none | raw signal fed to compressor |")
    L.append("| Artefact rejection | none | posterior-channel restriction is the "
             "only artefact control |")
    L.append("| Statistic | one-sided paired t-test (`scipy.stats.ttest_rel`) | "
             "within-subject pain vs. baseline |")
    L.append("")
    L.append("**Key caveat:** compression ratio of raw float EEG is driven largely "
             "by amplitude/variance and the byte encoding, *not* purely by "
             "\"algorithmic symmetry\". Larger evoked potentials can look less "
             "compressible for reasons unrelated to valence. Treat this as a "
             "falsifiable toy probe, not a validation of STV.")
    L.append("")
    L.append("**Channel-mapping caveat:** BioSemi hardware labels (`A1`…`B32`) are "
             "renamed to 10-20 names using the *standard* published cap layout; if "
             "ds005284 used a non-standard montage the posterior selection could be "
             "off. Confirm against `*_electrodes.tsv` if in doubt.")
    L.append("")

    # ---- Preprocessing pipeline -------------------------------------------
    L.append("## 2. Preprocessing pipeline (per recording)")
    L.append("")
    L.append("1. **Load** raw `.bdf` via `mne_bids.read_raw_bids`, which attaches "
             "the `*_events.tsv` sidecar as annotations.")
    L.append("2. **Rename** BioSemi hardware channels to 10-20 names.")
    L.append("3. **Pick** the posterior channels present.")
    L.append("4. **Locate events** and select the laser-stimulus trigger.")
    L.append("5. **Epoch** each trigger into the pain and baseline windows "
             "(`baseline=None`, no rejection), cropped to identical sample counts.")
    L.append("6. **Serialise** each (channels × time) window to bytes and record "
             "gzip & bz2 compression ratios; average within subject.")
    L.append("")

    # ---- Per-subject provenance -------------------------------------------
    L.append("## 3. Per-recording processing")
    L.append("")
    L.append(f"Recordings found: **{len(all_results)}** | usable: **{len(usable)}** "
             f"| skipped: **{len(skipped)}**")
    L.append("")
    L.append("| Subject | File | fs (Hz) | Channels used | Cap mapping | "
             "Stim event (code) | Selection | Stim/other events | Trials | "
             "Samples/window |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in all_results:
        chans = ", ".join(r.channels_used) if r.channels_used else "—"
        label = (f"`{r.event_label}` ({r.event_code})"
                 if r.event_code is not None else "—")
        win = f"{r.n_samples} ({_fmt(r.window_sec, 3)} s)" if r.n_samples else "—"
        L.append(f"| {r.subject} | `{r.filename}` | "
                 f"{_fmt(r.sfreq, 0) if not np.isnan(r.sfreq) else '—'} | {chans} | "
                 f"{r.cap_mapping or '—'} | {label} | {r.event_method or '—'} | "
                 f"{r.n_stim_events}/{r.n_other_events} | {r.n_trials or '—'} | {win} |")
    L.append("")
    if skipped:
        L.append("Skipped recordings:")
        L.append("")
        for r in skipped:
            L.append(f"- **{r.subject}** — {r.status}"
                     + (f" (available events: {r.available_events})"
                        if r.available_events else ""))
        L.append("")

    # ---- Compression ratios ------------------------------------------------
    L.append("## 4. Per-subject mean compression ratios")
    L.append("")
    L.append("Lower = more compressible (more redundancy). Each value is the mean "
             "over that subject's trials.")
    L.append("")
    L.append("| Subject | Trials | gzip pain | gzip base | Δ (pain−base) | "
             "bz2 pain | bz2 base | Δ (pain−base) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in usable:
        gp, gb = r.mean("pain", "gzip"), r.mean("base", "gzip")
        bp, bb = r.mean("pain", "bz2"), r.mean("base", "bz2")
        L.append(f"| {r.subject} | {r.n_trials} | {_fmt(gp)} | {_fmt(gb)} | "
                 f"{_fmt(gp - gb)} | {_fmt(bp)} | {_fmt(bb)} | {_fmt(bp - bb)} |")
    L.append("")
    L.append("A **positive Δ** (pain − baseline > 0) means the pain window was "
             "*less* compressible than baseline — the direction H1 predicts.")
    L.append("")

    # ---- Statistics --------------------------------------------------------
    L.append("## 5. Statistical comparison")
    L.append("")
    if len(usable) < 2:
        L.append(f"> **Only {len(usable)} usable subject(s)** — a paired t-test "
                 "needs ≥2, and is meaningless below a handful. No test run.")
        L.append("")
    else:
        L.append("Paired t-test on per-subject mean ratios, "
                 "H1: baseline more compressible than pain.")
        L.append("")
        L.append("| Compressor | n | mean baseline | mean pain | t | one-sided p | "
                 "Supports H1? |")
        L.append("|---|---|---|---|---|---|---|")
        verdicts = []
        for algo in ("gzip", "bz2"):
            s = paired_test(usable, algo)
            verdicts.append(s)
            L.append(f"| {algo} | {s['n']} | {_fmt(s['mean_base'])} | "
                     f"{_fmt(s['mean_pain'])} | {_fmt(s['t'], 3)} | "
                     f"{_fmt(s['p_one'], 4)} | "
                     f"{'**yes** ✓' if s['supported'] else 'no'} |")
        L.append("")
        L.append("### Verdict")
        L.append("")
        if all(v["supported"] for v in verdicts):
            L.append("Baseline windows were significantly **more compressible** than "
                     "pain windows on both compressors — **consistent** with the "
                     "algorithmic compressibility hypothesis in this sample.")
        elif any(v["supported"] for v in verdicts):
            L.append("Mixed: significant in the predicted direction on one compressor "
                     "but not the other. Weak/ambiguous support.")
        else:
            L.append("No significant support for the hypothesis in this sample: "
                     "baseline windows were **not** reliably more compressible than "
                     "pain windows.")
        L.append("")
        n = len(usable)
        if n < 10:
            L.append(f"> ⚠️ With only **n = {n}** subjects this test has very low "
                     "power; treat any p-value (significant or not) as indicative "
                     "only. Run the full cohort before drawing conclusions.")
            L.append("")

    L.append("---")
    L.append(f"*Command: `{config['argv']}`*")
    L.append("")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(bids_root: str, stim_code: int | None, trial_type: str | None,
        limit: int | None, report_path: str) -> None:
    import datetime

    files = find_subject_files(bids_root)
    if not files:
        sys.exit(f"No raw EEG files found under {bids_root!r}. "
                 "Expected a BIDS tree like sub-01/eeg/sub-01_task-*_eeg.bdf")
    if limit:
        files = files[:limit]
    print(f"Found {len(files)} recording(s); processing...\n")

    all_results: list[SubjectResult] = []
    for sub, path in files:
        try:
            res = process_subject(sub, path, bids_root, stim_code, trial_type)
        except Exception as exc:  # keep going; one bad file shouldn't kill the run
            res = SubjectResult(subject=sub, filename=os.path.basename(path),
                                status=f"error: {exc}")
            print(f"    error: {exc}")
        all_results.append(res)

    config = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "bids_root": bids_root,
        "trial_type": trial_type,
        "argv": " ".join(sys.argv),
    }
    report = build_report(all_results, config)
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(report)

    usable = [r for r in all_results if r.usable]
    print(f"\nDone: {len(usable)}/{len(all_results)} recordings usable.")
    print(f"Report written to: {report_path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bids-root", required=True,
                   help="Path to the ds005284 BIDS root directory.")
    p.add_argument("--stim-code", type=int, default=None,
                   help="Trigger code for the laser stimulus, if it comes from "
                        "a hardware stim channel rather than events.tsv.")
    p.add_argument("--trial-type", default=DEFAULT_TRIAL_TYPE,
                   help="events.tsv value/trial_type string identifying the "
                        f"laser stimulus onset (default: {DEFAULT_TRIAL_TYPE!r}, "
                        "the 16 laser trials in ds005284; 'condition 64' is an "
                        "artifact burst and is deliberately excluded). Pass a "
                        "different string for other datasets, or '' to auto-"
                        "detect (looks for 'laser', else the most frequent code).")
    p.add_argument("--report", default="compressibility_report.md",
                   help="Path for the Markdown report (default: "
                        "compressibility_report.md).")
    p.add_argument("--limit", type=int, default=None,
                   help="Process only the first N recordings (for quick testing).")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    run(args.bids_root, args.stim_code, args.trial_type, args.limit, args.report)

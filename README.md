# small_neurons

Testing Michael Johnson's **algorithmic compressibility hypothesis** (a corollary
of the [Symmetry Theory of Valence](https://opentheory.net/2016/11/principia-qualia-part-ii-valence/))
on real EEG.

The hypothesis, stated operationally: negative-valence states such as acute pain
should be *less algorithmically compressible* (more random) than calm/resting
states, which should be *more compressible* (more redundant / symmetric).

`compressibility_hypothesis.py` probes this on OpenNeuro
[`ds005284`](https://openneuro.org/datasets/ds005284) — *The 26 By Biosemi Laser
Pain Dataset* — by:

1. Loading the BIDS-formatted BioSemi `.bdf` raw files via `mne_bids.read_raw_bids`,
   which attaches each `*_events.tsv` sidecar as annotations (a plain `read_raw_bdf`
   call would silently see zero events, since the triggers live in that sidecar,
   not in the raw file's own stim channel).
2. Renaming BioSemi's raw hardware channel labels (`A1`…`A32`, `B1`…`B32`, …) to
   10-20 names, then restricting to occipital/parietal channels (O1, O2, Oz, P3,
   P4, Pz, …) to keep jaw-clench EMG and blink EOG artefacts out of the
   compression score.
3. Epoching each laser trigger into a **pain window** (0–3 s post-stimulus) and a
   matched **baseline window** (−4 to −1 s), cropped to identical sample counts.
4. Serialising each (channels × time) matrix to a fixed-precision ASCII byte
   string (see `matrix_to_bytes` for the why).
5. Running `gzip` and `bz2` over the bytes and computing
   `compression_ratio = compressed_size / original_size`.
6. Aggregating per subject and running a one-sided paired t-test
   (`scipy.stats.ttest_rel`) for `ratio(baseline) < ratio(pain)`.

The run produces a self-contained **Markdown report** (default
`compressibility_report.md`) rather than raw terminal output — it documents the
assumptions, the preprocessing pipeline, per-recording provenance (channels,
selected event, trial/sample counts, anything skipped), the per-subject
compression ratios, and the paired t-test with a verdict and power warning.

## Usage

```bash
pip install -r requirements.txt

# 1. Check how subjects are actually labelled in this dataset (BIDS labels
#    aren't guaranteed to be zero-padded numbers -- don't guess "01").
python download_dataset.py --list-subjects

# 2. Download ds005284 from OpenNeuro (public S3 bucket, no credentials needed).
#    Full dataset (raw EEG only, derivatives/ skipped by default -- see below)
#    is ~1.8 GB. Use --subjects with labels from step 1 to grab just a couple
#    for testing.
python download_dataset.py --target-dir ./ds005284 --subjects 01 02

# 3. Run the analysis against the downloaded BIDS root:
python compressibility_hypothesis.py --bids-root ./ds005284

# Options:
#   --trial-type S  events.tsv value/trial_type string for the laser stimulus
#                   (default: "condition 54", the 16 laser trials in ds005284)
#   --stim-code N   trigger code, only if it comes from a hardware stim channel
#   --report PATH   where to write the Markdown report (default:
#                   compressibility_report.md)
#   --limit N       process only the first N recordings (quick test)
```

### Which event is the laser stimulus?

Confirmed from the dataset's README and `*_events.tsv` sidecars:

* **`condition 54`** — the laser pain stimulus: 16 trials per subject spaced
  ~12–13 s apart ("16 trials, approximately every 20 seconds"). This is the
  default the script epochs on.
* **`condition 64`** — *not* a stimulus. In sub-001 it's 14 triggers packed into
  a ~200 ms burst at t≈7.8 s (a recording-onset glitch), ~2 minutes before the
  first real laser. It appears in only some subjects and is deliberately excluded.

Because the script pins `condition 54` rather than guessing the most frequent
code, the artifact burst can never be mistaken for the stimulus.

### Dataset size: why the site says ~1.6 GB but a full download is bigger

OpenNeuro's displayed dataset size only counts the raw `sub-*/eeg/*.bdf` files
(~1.8 GB). The snapshot also ships a `derivatives/session_merged_data/` folder
with ~3 GB of the original authors' preprocessed EEGLAB (`.set`/`.fdt`) exports,
which isn't reflected in that headline number but *is* included in a full
download. `compressibility_hypothesis.py` never reads `derivatives/`, so
`download_dataset.py` excludes it by default -- pass `--include-derivatives` if
you want it anyway. If you already downloaded before this default existed, just
delete the folder:

```bash
rm -rf ds005284/derivatives
```

## Important caveat

Compression ratio of raw floating-point EEG is driven largely by signal
amplitude/variance and by the byte encoding, **not** purely by "algorithmic
symmetry". A pain window with larger evoked potentials can look less compressible
for reasons unrelated to valence. Treat this as a falsifiable toy probe, not a
validation of STV — and always inspect the event triggers before trusting any
p-value it prints.

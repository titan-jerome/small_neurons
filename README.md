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

## Usage

```bash
pip install -r requirements.txt

# 1. Check how subjects are actually labelled in this dataset (BIDS labels
#    aren't guaranteed to be zero-padded numbers -- don't guess "01").
python download_dataset.py --list-subjects

# 2. Download ds005284 from OpenNeuro (public S3 bucket, no credentials needed).
#    Full dataset is ~tens of GB; use --subjects with labels from step 1 to
#    grab just a couple for testing.
python download_dataset.py --target-dir ./ds005284 --subjects 01 02

# 3. Run the analysis against the downloaded BIDS root:
python compressibility_hypothesis.py --bids-root ./ds005284

# Options:
#   --trial-type S  events.tsv trial_type/value string for the laser stimulus
#                   onset (default: looks for a description containing "laser")
#   --stim-code N   trigger code, only if it comes from a hardware stim channel
#   --limit N       process only the first N recordings (quick test)
```

The first run against real data will print an `[info] available trial types: {...}`
line — check that the auto-picked description is actually the laser stimulus
onset (not a rating prompt, rest marker, etc.) and pass `--trial-type` explicitly
if not.

## Important caveat

Compression ratio of raw floating-point EEG is driven largely by signal
amplitude/variance and by the byte encoding, **not** purely by "algorithmic
symmetry". A pain window with larger evoked potentials can look less compressible
for reasons unrelated to valence. Treat this as a falsifiable toy probe, not a
validation of STV — and always inspect the event triggers before trusting any
p-value it prints.

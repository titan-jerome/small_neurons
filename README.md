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

1. Loading the BIDS-formatted BioSemi `.bdf` raw files with MNE.
2. Restricting to occipital/parietal channels (O1, O2, Oz, P3, P4, Pz, …) to keep
   jaw-clench EMG and blink EOG artefacts out of the compression score.
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

# Download ds005284 from OpenNeuro first (e.g. via `openneuro-py` or datalad),
# then point the script at its BIDS root:
python compressibility_hypothesis.py --bids-root /path/to/ds005284

# Options:
#   --stim-code N   trigger code for the laser stimulus (inspect your events!)
#   --limit N       process only the first N recordings (quick test)
```

## Important caveat

Compression ratio of raw floating-point EEG is driven largely by signal
amplitude/variance and by the byte encoding, **not** purely by "algorithmic
symmetry". A pain window with larger evoked potentials can look less compressible
for reasons unrelated to valence. Treat this as a falsifiable toy probe, not a
validation of STV — and always inspect the event triggers before trusting any
p-value it prints.

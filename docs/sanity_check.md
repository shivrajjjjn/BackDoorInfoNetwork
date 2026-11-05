# Sanity Check: Known-Target Recovery

**What I ran:** Neural Cleanse reverse engineering on all 43 labels of the
provided infected GTSRB model (white 4-pixel trigger at the bottom right,
target label 33), followed by MAD anomaly detection. Run on an RTX 5060 GPU
using a TF2 port of the authors' code.

**What I expected:** Label 33 is flagged, its anomaly index is above 2,
and its trigger's L1 norm is much smaller than the median of other labels.

**What I observed:**
- Flagged label(s): 33
- Anomaly index: 2.679646
- Label 33 L1 norm: 11.92 Median L1 norm: 60.68    MAD: 18.20

**Did it pass?** Yes. The flagged label matches the known infected target, it is the only label flagged, the anomaly index is above the threshold of 2, and the norm of that label is roughly 5 times smaller than the median.

**Differences from the authors' README:** The anomaly index is lower. Possible causes are the environment changes I needed to do to have it work with my GPU. The detection passes regardless. 

**Early partial check (CPU run):** Before the GPU port, a CPU run completed
labels 33 and 0 before I stopped it: label 33 mask norm 17.34, label 0 mask
norm 54.70, consistent with a small trigger on the target label.

**Runtime:** The full 43-label GPU run took 285 s (about 4.75 min). On CPU
(Ryzen 5 5600X), label 33 alone took 86.5 s, which projected to roughly an
hour for all labels.

**Environment:** Python 3.11, TensorFlow 2.21.0, GPU RTX 5060

**Log files:** logs/baseline_gtsrb_infected_gpu_run1.txt,
logs/baseline_gtsrb_infected_gpu_mad1.txt
Results folder: results_gpu

# Project Baseline: Neural Cleanse Reproduction

Reproduction of Neural Cleanse (Wang et al., IEEE S&P 2019) on the authors' provided infected GTSRB model, using their code from https://github.com/bolunwang/backdoor (MIT license, retained in `LICENSE`). The authors' original README is kept as `README.md`.

- Repository: https://github.com/shivrajjjjn/BackDoorInfoNetwork
- Submission branch: `shivraj/gpu-port`

## Fallback trace (use this if a later snapshot is missing or invalid)

| Item | File | Commit |
|---|---|---|
| Reverse-engineering run (all 43 labels, GPU) | `logs/baseline_gtsrb_infected_gpu_run1.txt` | `72e2f4e` |
| MAD anomaly detection output | `logs/baseline_gtsrb_infected_gpu_mad1.txt` | `3abedc0` |
| Sanity check write-up | `docs/sanity_check.md` | latest commit on `shivraj/gpu-port` |

Commit links: https://github.com/shivrajjjjn/BackDoorInfoNetwork/commit/72e2f4e and https://github.com/shivrajjjjn/BackDoorInfoNetwork/commit/3abedc0

**Sanity check result: passed.** Label 33 (the known infected target) is the only label flagged. Anomaly index 2.68 (threshold 2), label 33 L1 norm 11.92 versus a median of 60.68.

## Hardware and environment

- CPU: AMD Ryzen 5 5600X. GPU: NVIDIA RTX 5060 (about 7.8 GB usable, compute capability 12.0). Driver version: see the `nvidia-smi` output saved in `docs/env_notes.txt`.
- OS: Linux
- Python 3.11, TensorFlow 2.21.0, conda environment `nc-gpu`
- Exact package versions: `requirements-gpu.txt`
- First attempt (legacy environment `nc`: Python 3.6, TensorFlow 1.10.1, Keras 2.2.2, CPU only): package list in `requirements.txt`. Used only for the partial CPU run described below.

## Setup

1. Create the environment:

   ```bash
   conda create -n nc-gpu python=3.11 -y
   conda activate nc-gpu
   pip install "tensorflow[and-cuda]" tf-keras h5py pillow scipy
   ```

2. Set environment variables (Keras 2 behavior, and a larger GPU compile cache):

   ```bash
   conda env config vars set KERAS_BACKEND=tensorflow TF_USE_LEGACY_KERAS=1 CUDA_CACHE_MAXSIZE=4294967296
   ```

3. Point TensorFlow at the pip-installed NVIDIA libraries and save it to the environment:

   ```bash
   export LD_LIBRARY_PATH=$(python -c "import os,glob,site; sp=site.getsitepackages()[0]; print(':'.join(glob.glob(os.path.join(sp,'nvidia','*','lib'))))"):$LD_LIBRARY_PATH
   conda env config vars set LD_LIBRARY_PATH="$LD_LIBRARY_PATH"
   conda deactivate
   conda activate nc-gpu
   ```

4. Check that the GPU is visible (it should list `GPU:0`):

   ```bash
   python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
   ```

   On the RTX 50 series, TensorFlow compiles GPU kernels from PTX the first time it runs, so the first run can be slow. Later runs reuse the cache.

## Data and model

Both are included in the authors' repo, so no download is needed:

- `models/gtsrb_bottom_right_white_4_target_33.h5`: infected model (white 4-pixel patch, bottom-right corner, target label 33)
- `data/gtsrb_dataset_int.h5`: test data (12,630 images, 43 labels)

## How to run

1. Reverse-engineer a trigger for each of the 43 labels:

   ```bash
   python gtsrb_visualize_example.py 2>&1 | tee logs/baseline_gtsrb_infected_gpu_run1.txt
   ```

2. Run the MAD anomaly detection on the reversed triggers:

   ```bash
   python mad_outlier_detection.py 2>&1 | tee logs/baseline_gtsrb_infected_gpu_mad1.txt
   ```

Settings used: `DEVICE = '0'`. Reversed triggers are written to `results_gpu/`, and `mad_outlier_detection.py` reads from the same folder. All other parameters are the authors' defaults. Expected final lines: `anomaly index` above 2 and `flagged label list: 33: ...`.

## Results (GPU run, 2026-11-05)

| Item | Authors' README | My run |
|---|---|---|
| Labels analyzed | 43 | 43 |
| Median L1 norm | 64.47 | 60.68 |
| MAD | 13.24 | 18.20 |
| Anomaly index | 3.65 | 2.68 |
| Flagged label | 33 | 33 (only label flagged) |
| Flagged L1 norm | 16.12 | 11.92 |

- Full 43-label run on the RTX 5060: 285 s. The last label (42) took 7.7 s.
- Partial CPU run (Ryzen 5600X, legacy environment): label 33 took 86.5 s (mask norm 17.34) and label 0 gave mask norm 54.70, which projected to about 1 hour for all labels. I stopped it and moved to the GPU. Log: `logs/baseline_gtsrb_infected_cpu_partial.txt`.
- The anomaly index is lower than the authors' because the spread of the other labels' norms (MAD) is larger in my run. I did not test whether random initialization or the TF2 port causes this. Details in `docs/sanity_check.md`.

## Changes made to the authors' code (TensorFlow 1 to TensorFlow 2 port)

- `keras` imports changed to `tensorflow.keras`.
- `tf.Session`, `tf.ConfigProto`, `tf.GPUOptions`, `tf.global_variables_initializer`, and `K.set_session` changed to their `tf.compat.v1` equivalents.
- Added `tf.compat.v1.disable_eager_execution()` at the top of `utils_backdoor.py` and `gtsrb_visualize_example.py`.
- `set_random_seed` imported from `tensorflow.compat.v1`.
- `Adam(lr=...)` changed to `Adam(learning_rate=...)`, and Adam imported from `tensorflow.compat.v1.keras.optimizers` (the old optimizer API is needed for `get_updates`).
- Installed `scipy`, which Keras image preprocessing needs.

## Known issues and fixes

| Problem | Cause | Fix |
|---|---|---|
| Keras error "Unable to import backend: torch" | Keras backend was set to torch by an old setting | Set the backend to tensorflow |
| "Cannot dlopen some GPU libraries", GPU list empty | pip-installed NVIDIA libraries not on the library path | Added them to `LD_LIBRARY_PATH` (Setup step 3) |
| `ImportError: cannot import name 'set_random_seed'` | TF1 API | Import from `tensorflow.compat.v1` |
| `AttributeError: 'Adam' object has no attribute 'get_updates'` | Authors' code uses the old optimizer API | Use the `compat.v1` Adam |
| `NameError: name 'scipy' is not defined` | scipy not installed | `pip install scipy` |

Failure logs kept in `logs/`: `failure_gpu_run0_set_random_seed.txt`, `failure_gpu_run1_adam_get_updates.txt`, `failure_gpu_run2_scipy_missing.txt`.

Harmless warnings that can be ignored: `SyntaxWarning: "is" with a literal` (authors' code), "Operation ... was changed by setting attribute" (graph-mode noise), the AVX2/FMA CPU message, and the PTX compile message.

## Evidence of work (Shivraj, sdn55)

- Baseline reproduction and detection: the logs and results listed above
- Environment records: `requirements-gpu.txt`, `requirements.txt`, `docs/env_notes.txt`
- Dated lab notebook: `docs/lab_notebook.md`
- Sanity check: `docs/sanity_check.md`
- Code port: commits on `shivraj/gpu-port`

## Team and status

- Shivraj (sdn55): Neural Cleanse reproduction, MAD detection, and the MNIST adaptation of the detector (next step, waiting on a trained MNIST model).
- Roger (rtm141): MNIST backdoor training, sweep orchestration, clean-model control, and the experimental protocol. See his logs and commits.
- The third team member has left the group; the instructor said their part does not need to be covered.

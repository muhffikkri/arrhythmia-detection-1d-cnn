# Training Environment Setup (Windows / Linux)

This repository is fully cross-platform: the same code and the same commands run on
**Windows (PowerShell)** and **Linux (bash)**. All paths inside the source are built with
`os.path.join`, so there are no hard-coded `\` or `/` separators — the only thing that
differs is how you create the virtual environment and which compute device TensorFlow uses.

Everything below must be executed from the **repository root**
(`/path/to/arrhythmia-detection-1d-cnn`), because the package layout
(`src.config`, `src.training`, ...) relies on the root being on `PYTHONPATH`.

---

## 1. Requirements

| Component | Windows | Linux |
| :-------- | :------ | :---- |
| Python | 3.10 – 3.12 | 3.10 – 3.12 |
| TensorFlow | pip wheel is **CPU-only** on native Windows | pip wheel **bundles CUDA + cuDNN** (GPU ready) |
| GPU training | via **WSL2 + Ubuntu** (recommended) | native (recommended) |

> The `tensorflow` dependency without a version tag follows the TF 2.x line. On Linux the
> pip wheel is self-contained (GPU). On native Windows, TensorFlow publishes CPU wheels
> (GPU support moved to WSL2 / containers), so use **WSL2** when you want GPU speed.

---

## 2. Setup

### 2.1 Linux (bash) — recommended for GPU

```bash
# Create and activate the virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify the GPU is visible to TensorFlow
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
# Expected: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU'), ...]
```

### 2.2 Windows (PowerShell)

```powershell
# Create and activate the virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

CPU-only is fine for smoke tests; for real training speed on Windows use WSL2:

```bash
# Inside WSL2 (Ubuntu 22.04+) with NVIDIA driver installed on the Windows host:
cd /mnt/c/Project/arrhythmia-detection-1d-cnn   # or wherever the repo is
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

> The working copy can stay inside the Windows filesystem (`/mnt/c/...`); only the venv
> and runtime live under WSL2. If you prefer `os.name` conditions in scripts — you do not
> need any: the code already handles both.

---

## 3. Run Training

Platform differences stop at the activation command:

| Step | Windows (PowerShell) | Linux (bash) |
| :--- | :------------------- | :----------- |
| Activate venv | `.\venv\Scripts\Activate.ps1` | `source venv/bin/activate` |
| Preprocess PTB-XL | `python src/preprocessing/proccess_ptbxl.py` | same |
| Preprocess Chapman | `python src/preprocessing/proccess_chapman.py` | same |
| Train experiment | `python src/experiments/run_experiment.py` | same |

`run_experiment.py` prints a device banner at startup
(`[Env] OS / # GPU device(s) / TF version`) and enables **GPU memory growth** automatically,
so the same command works on CPU or GPU without extra flags.

### Useful environment variables

```bash
# Show only warnings+ from TensorFlow (less noise)
export TF_CPP_MIN_LOG_LEVEL=2

# Pin a specific GPU on multi-GPU machines (Linux / WSL2)
export CUDA_VISIBLE_DEVICES=0

# Force CPU (e.g. to compare timing)
export CUDA_VISIBLE_DEVICES=-1
```

---

## 4. Troubleshooting

- **"Could not load dynamic library 'cudnn...'" on Linux**: pip `tensorflow` bundles the
  CUDA runtime, but the NVIDIA **driver** must still be installed and `nvidia-smi` must
  succeed. Update the driver, then re-create the venv.
- **Native Windows shows CPU only**: expected — use WSL2 as described above.
- **OOM during training**: the runner sets `set_memory_growth`; if you still hit OOM,
  reduce `Config.BATCH_SIZE` in `src/config/experiment_configs.py`.
- **Scripts using `sys.stdout.reconfigure(encoding="utf-8")`** (EDA scripts): safe on both
  platforms; on Windows it avoids cp1252 crashes, on Linux it is a no-op.

---

## 5. Reproducibility

- Global seed `42` is fixed once in `run_experiment.py` (see `set_global_seeds`).
- Splits come from `src/training/dataset_loader.py` (PTB-XL `strat_fold`; Chapman stratified
  70/15/15, seed 42) — identical on both platforms.
- Model artifacts, `experiment_config.json`, and `metrics.csv` are saved per experiment under
  `output/research_experiments/` and are platform-independent.
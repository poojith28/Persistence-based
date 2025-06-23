# Persistence‑based Active Learning

A compact guide for setting up and running experiments that combine Topological Data Analysis (TDA) with classical active‑learning strategies.

---

## 1. Prerequisites

This repository extends the [**TypiClust**](https://github.com/avihu111/TypiClust) project. Make sure you are familiar with its structure and conventions before proceeding.

### Key Python packages

| Package       | Version (pinned) |
| ------------- | ---------------- |
| black         | 19.3b0           |
| flake8        | 3.8.4            |
| isort         | 4.3.21           |
| matplotlib    | 3.3.4            |
| numpy         | *latest*         |
| opencv‑python | 4.2.0.34         |
| torch         | 1.7.1            |
| torchvision   | 0.8.2            |
| parameterized | *latest*         |
| setuptools    | *latest*         |
| simplejson    | *latest*         |
| yacs          | *latest*         |
| **gudhi**     | ≥ 3.4            |
| scikit‑learn  | ≥ 0.22           |
| pandas        | ≥ 1.0            |
| joblib        | *latest*         |

> **Why these pins?** They match the versions used by the original TypiClust authors and have been tested with this codebase.

---

## 2. Environment setup

```bash
# 1. Clone the repo
$ git clone https://github.com/poojith28/Persistence-based.git
$ cd Persistence-based

# 2. Create & activate a conda environment (Python 3.7)
$ conda create --name tdaAL python=3.7 -y
$ conda activate tdaAL

# 3. Install PyTorch with your CUDA version (replace <CUDA_VERSION>)
$ conda install pytorch torchvision torchaudio cudatoolkit=<CUDA_VERSION> -c pytorch -y

# 4. Classic scientific stack
$ conda install matplotlib scipy scikit-learn pandas -y

# 5. FAISS (GPU build)
$ conda install -c conda-forge faiss-gpu -y

# 6. Remaining Python deps
$ pip install pyyaml easydict termcolor tqdm simplejson yacs gudhi
```

Check the installation:

```bash
python - <<'PY'
import torch, gudhi, sklearn, pandas as pd
print('Torch', torch.__version__)
print('GUDHI', gudhi.__version__)
print('sklearn', sklearn.__version__)
print('pandas', pd.__version__)
PY
```

---

## 3. (Optional) Representation learning

Some strategies—**ProbCover**, in particular—require pretrained features. The recommended backbone is **SimCLR** trained on the target dataset.

```bash
# Train SimCLR on CIFAR‑10 (example)
$ cd scan
$ python simclr.py \
    --config_env configs/env.yml \
    --config_exp configs/pretext/simclr_cifar10.yml
$ cd ..
```

When training completes, you should have:

```
results/cifar-10/pretext/features_seed1.npy
```

---

## 4. Running active‑learning experiments

The main entry point is \`\`.

```bash
python train_al.py \
    --cfg configs/<dataset>/al/RESNET18.yaml \
    --al <strategy> \
    --exp-name <run_name> \
    --budget <label_budget> \
    --initial_size <initial_labeled>
```

### 4.1 Choosing a dataset

| Dataset      | Config path                             |
| ------------ | --------------------------------------- |
| CIFAR‑10     | `configs/cifar10/al/RESNET18.yaml`      |
| CIFAR‑100    | `configs/cifar100/al/RESNET18.yaml`     |
| TinyImageNet | `configs/tinyimagenet/al/RESNET18.yaml` |

### 4.2 Available selection strategies (`--al`)

- `random`
- `entropy`
- `margin`
- `uncertainty`
- `dbal` 
- `coreset`
- `typiclust`
- `probcover`
- `most_repeated_effective_death`  ⬅️ **persistence‑based**



### 4.3 Examples

**Entropy on TinyImageNet**

```bash
python train_al.py \
    --cfg configs/tinyimagenet/al/RESNET18.yaml \
    --al entropy \
    --exp-name entropy_2_50b \
    --budget 50 \
    --initial_size 50
```

**Random on CIFAR‑10**

```bash
python train_al.py \
    --cfg configs/cifar10/al/RESNET18.yaml \
    --al random \
    --exp-name random_c10_50b \
    --budget 50 \
    --initial_size 50
```

---

## 5. Persistence‑based pipeline

> The core idea is to select samples whose **effective death** appears most frequently in their \(H_0\) persistence diagrams.

### Step 1 – Compute persistence diagrams

```bash
# For CIFAR‑10
python PD_Calculations/persistence_cifar10.py

# For CIFAR‑100
python PD_Calculations/persistence_cifar100.py

# For TinyImageNet
python PD_Calculations/persistence_tinyimagenet.py
```

Each script produces a CSV file, e.g. `persistence_cifar10.csv`, containing `[birth, death]` pairs per data point.

### Step 2 – Vertex counting

```bash
python PD_Calculations/vertex_counter.py \
    --pd_csv path/to/persistence_cifar10.csv
```

This generates an auxiliary file with the **effective death counts** required by the strategy.

### Step 3 – Active learning with persistence

```bash
python train_al.py \
    --cfg configs/cifar10/al/RESNET18.yaml \
    --al most_repeated_effective_death \
    --exp-name persistence_c10_50b \
    --budget 50 \
    --initial_size 50
```

---

## 6. Tips & troubleshooting

- **GPU memory** – SimCLR and TypiClust can be demanding; adjust batch sizes if you hit OOM.
- **Custom backbones** – Swap `RESNET18.yaml` for other architectures (`RESNET50.yaml`, ViT, …) as long as the config file exists.
- **Logging** – All logs and checkpoints are written to `./results/<dataset>/<exp-name>/`.
- **Adding new datasets** – Replicate an existing config folder and update paths, class counts, and image resolutions accordingly.

---

### Happy experimenting! 🎉

If you find issues or have questions, feel free to open an issue or reach out.


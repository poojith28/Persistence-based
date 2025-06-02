"""
persistence_cifar10_normalised.py
─────────────────────────────────
Build a Vietoris–Rips persistence diagram for ALL CIFAR-10 images after
NORMALISING the data set (no PCA, no sub-sampling).

Normalisation modes
-------------------
--norm pixel      : values ∈ [0,1]   (default; what torchvision.ToTensor() gives)
--norm zscore     : per-pixel standard score  (x - μ) / σ  across the dataset
--norm unit       : per-sample L2-normalise each image vector  ||x||₂ = 1

USAGE EXAMPLES
==============
# 1) plain [0,1] scaling, whole train split
python persistence_cifar10_normalised.py --root ~/data

# 2) z-score normalisation, test split, max H₂ homology
python persistence_cifar10_normalised.py --root ~/data --split test \
        --norm zscore --max_rips_dim 2
"""
import argparse, logging
from pathlib import Path
import numpy as np
import logging
from gudhi import RipsComplex
import matplotlib.pyplot as plt
import csv
import pandas as pd
import numpy as np
import torch
from torchvision import datasets, transforms
from sklearn.preprocessing import StandardScaler, normalize

# ───────────────────────────────────────────────────────────────────────────
# ❶  Import YOUR helper exactly as-is (edit this path if needed)
# ───────────────────────────────────────────────────────────────────────────

def compute_and_display_persistence_diagram(features, max_dim=1, verbose=True, save_path=None):
    """
    Compute and display the persistence diagram for the given features and 
    save the persistence pairs (without 'Involved Simplices') as a CSV file.

    Parameters:
    -----------
    features : np.ndarray
        Input feature data points (n_samples, n_features).
    max_dim : int, optional
        Maximum homology dimension to compute (default: 2).
    verbose : bool, optional
        Whether to print detailed logs (default: True).
    save_path : str, optional
        Path to save the persistence pairs as a CSV file (default: None).

    Returns:
    --------
    pd.DataFrame or None
        Returns a DataFrame with the persistence pairs if save_path is provided.
    """
    if verbose:
        print("Shape of input features:", features.shape)
    logging.info("Shape of input features: %s", features.shape)

    # Step 1: Build Rips Complex
    if verbose:
        print("Building Rips Complex...")
    logging.info("Building Rips Complex with max_edge_length=10000")
    rips_complex = RipsComplex(points=features, max_edge_length=10000)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=1)
    num_simplices = simplex_tree.num_simplices()
    logging.info("Rips Complex built with %d simplices", num_simplices)
    
    # Step 2: Compute Persistence
    if verbose:
        print("Computing persistence...")
    logging.info("Computing persistence...")
    simplex_tree.compute_persistence()
    
    # Step 3: Save Persistence Pairs (without 'Involved Simplices') to CSV if save_path is provided.
    if save_path:
        logging.info("Saving persistence pairs to CSV at %s", save_path)
        with open(save_path, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            # Write header without "Involved Simplices" column
            writer.writerow(["Dimension", "Birth", "Death", "Birth Simplex", "Death Simplex"])
            
            # Loop for each homology dimension up to max_dim
            for dim in range(max_dim + 1):
                intervals = simplex_tree.persistence_intervals_in_dimension(dim)
                pairs = simplex_tree.persistence_pairs()  # Contains all birth-death pairs with simplex info
                
                # Use the same index order as intervals when possible
                for i, (birth, death) in enumerate(intervals):
                    if i < len(pairs):
                        birth_simplex, death_simplex = pairs[i]
                        # Convert infinite death values: use -1 as placeholder for CSV
                        death_val = death if death != float('inf') else -1
                        writer.writerow([
                            dim,
                            birth,
                            death_val,
                            list(birth_simplex) if birth_simplex else [],
                            list(death_simplex) if death_simplex else []
                        ])
                        logging.info("CSV row: dim=%d, birth=%s, death=%s, birth_simplex=%s, death_simplex=%s",
                                     dim, birth, death_val,
                                     list(birth_simplex) if birth_simplex else [],
                                     list(death_simplex) if death_simplex else [])
        if verbose:
            print(f"Persistence pairs saved to {save_path}")
        logging.info("Persistence pairs saved to %s", save_path)
    else:
        logging.info("No save path provided, not saving CSV.")

    # Step 4: Plot Persistence Diagram
    logging.info("Plotting persistence diagram...")
    plt.figure(figsize=(8, 6))
    for dim in range(max_dim + 1):
        intervals = simplex_tree.persistence_intervals_in_dimension(dim)
        # For plotting, map float('inf') to a large number based on the features
        plotted_intervals = [
            (birth, death if death != float('inf') else np.max(features)) 
            for birth, death in intervals
        ]
        births, deaths = zip(*plotted_intervals) if plotted_intervals else ([], [])
        plt.scatter(births, deaths, label=f"H{dim}", s=10)

    plt.axline((0, 0), slope=1, color="red", linestyle="--", label="Diagonal")
    plt.xlabel("Birth")
    plt.ylabel("Death")
    plt.title("Persistence Diagram")
    plt.legend()
    plt.grid(True)
    plt.show()
    logging.info("Persistence diagram plotted.")

    # If save_path was provided, return the CSV data as a pandas DataFrame.
    if save_path:
        logging.info("Returning DataFrame from saved CSV.")
        return pd.read_csv(save_path)
    else:
        return None

# ───────────────────────────────────────────────────────────────────────────
# ❷  Logging (optional)
# ───────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    filename="cifar10_normalised_persistence.log",
    filemode="w"
)


# ───────────────────────────────────────────────────────────────────────────
# ❸  Data loading
# ───────────────────────────────────────────────────────────────────────────
def load_cifar10_flat(root: str, split: str = "train") -> np.ndarray:
    """
    Return a float32 array of shape (N, 3·32·32 = 3 072) in [0,1].
    """
    ds = datasets.CIFAR10(
        root,
        train=(split == "train"),
        download=True,
        transform=transforms.ToTensor(),   # already /255 → [0,1]
    )
    imgs = torch.stack([img for img, _ in ds])        # (N, 3, 32, 32)
    arr  = imgs.view(imgs.size(0), -1).numpy()        # (N, 3072)
    return arr.astype(np.float32)


# ───────────────────────────────────────────────────────────────────────────
# ❹  Normalisation helpers
# ───────────────────────────────────────────────────────────────────────────
def normalise(X: np.ndarray, mode: str) -> np.ndarray:
    """
    Parameters
    ----------
    X    : (N, D) float32
    mode : 'pixel' | 'zscore' | 'unit'
    """
    if mode == "pixel":             # already [0,1]
        return X

    if mode == "zscore":
        # per-feature mean/std across entire dataset
        scaler = StandardScaler(with_mean=True, with_std=True, copy=False)
        return scaler.fit_transform(X)

    if mode == "unit":
        # per-sample L2 normalisation
        return normalize(X, norm="l2", axis=1, copy=False)

    raise ValueError(f"Unknown normalisation mode: {mode}")


# ───────────────────────────────────────────────────────────────────────────
# ❺  Main
# ───────────────────────────────────────────────────────────────────────────
def main(args):
    # 1. load
    logging.info("Loading CIFAR-10 (%s split) …", args.split)
    X = load_cifar10_flat(args.root, split=args.split)
    print(f"Loaded {X.shape[0]} images — raw shape {X.shape}")

    # 2. normalise
    print(f"Applying '{args.norm}' normalisation …")
    X = normalise(X, args.norm)
    logging.info("Normalisation '%s' done.", args.norm)

    # 3. persistence
    csv_path = Path(args.csv).expanduser()
    logging.info("Calling compute_and_display_persistence_diagram …")
    df = compute_and_display_persistence_diagram(
        X,
        max_dim=args.max_rips_dim,
        verbose=True,
        save_path=str(csv_path),
    )

    print("\nFirst few persistence pairs:")
    print(df.head())
    print(f"\nCSV written to: {csv_path.resolve()}")


# ───────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default="./data",
                   help="Folder where torchvision downloads CIFAR-10")
    p.add_argument("--split", choices=["train", "test"], default="train",
                   help="Which split to use")
    p.add_argument("--norm", choices=["pixel", "zscore", "unit"], default="zscore",
                   help="Normalisation mode (see docstring)")
    p.add_argument("--max_rips_dim", type=int, default=1,
                   help="Maximum homology dimension")
    p.add_argument("--csv", type=str, default="cifar10_normalised_pairs_RAWDATA.csv",
                   help="Path for the persistence-pair CSV")
    args = p.parse_args()
    main(args)

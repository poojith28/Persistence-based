"""
persistence_tinyimagenet_from_pickle.py
───────────────────────────────────────
Compute a Vietoris–Rips persistence diagram for Tiny-ImageNet images stored
in a single train.pkl (or val.pkl) file.

Normalisation modes
-------------------
--norm pixel   : raw [0,1] scaling (default)
--norm zscore  : per-pixel standard score
--norm unit    : per-sample L2 normalisation

USAGE EXAMPLE
-------------
python persistence_tinyimagenet_from_pickle.py \
       --pkl ~/data/tiny-imagenet-200/train.pkl \
       --norm zscore --max_rips_dim 1
"""
import argparse, pickle, logging
from pathlib import Path
import argparse, logging
from pathlib import Path
import argparse, logging
from pathlib import Path
import numpy as np
import logging
from gudhi import RipsComplex
import matplotlib.pyplot as plt
import csv
import pandas as pd
import numpy as np
import numpy as np
import torch
from torchvision import datasets, transforms
from sklearn.preprocessing import StandardScaler, normalize

import numpy as np
from sklearn.preprocessing import StandardScaler, normalize
from gudhi import RipsComplex                     # only needed for helper check
import matplotlib.pyplot as plt                   #   "
                                                   #   "
# ───────────────────────────────────────────────────────────────────────────
# Your helper copied verbatim (shortened here ⬇︎ for brevity — leave yours unchanged)
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
# Logging
# ───────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    filename="tinyimagenet_pickle_persistence.log",
    filemode="w"
)

# ───────────────────────────────────────────────────────────────────────────
# Loader for train.pkl / val.pkl
# ───────────────────────────────────────────────────────────────────────────
def load_pickle_flat(pkl_path: str) -> np.ndarray:
    """
    Returns (N, 12288) float32 array in [0,1], regardless of pickle format.
    """
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f, encoding="latin1")

    # 1) CIFAR-style dict  {"data": (N,12288), "labels": …}
    if isinstance(obj, dict) and any(k in obj for k in ("data", b"data")):
        key = "data" if "data" in obj else b"data"
        X = obj[key]                               # uint8
        if X.ndim != 2:
            raise ValueError(f"Expected 2-D array in dict['{key}'], got {X.shape}")
        X = X.reshape(X.shape[0], -1)              # already (N,12288)
        X = X.astype(np.float32) / 255.0
        return X

    # 2) Tuple / list  (images, labels) or just images
    if isinstance(obj, (tuple, list)):
        images = obj[0] if isinstance(obj[0], (np.ndarray, list)) else obj
        images = np.asarray(images)
        if images.ndim == 4:                       # (N, H, W, C) or (N,C,H,W)
            if images.shape[1] == 3:               # (N,3,64,64)  → transpose
                images = images.transpose(0, 2, 3, 1)
            images = images.reshape(images.shape[0], -1)
            images = images.astype(np.float32) / 255.0
            return images

    raise RuntimeError(
        f"Unsupported pickle structure — please inspect {pkl_path} manually.")

# ───────────────────────────────────────────────────────────────────────────
# Normalisation
# ───────────────────────────────────────────────────────────────────────────
def normalise(X: np.ndarray, mode: str) -> np.ndarray:
    if mode == "pixel":
        return X
    if mode == "zscore":
        scaler = StandardScaler(with_mean=True, with_std=True, copy=False)
        return scaler.fit_transform(X)
    if mode == "unit":
        return normalize(X, norm="l2", axis=1, copy=False)
    raise ValueError(f"Unknown --norm mode: {mode}")

# ───────────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────────
def main(args):
    logging.info("Loading Tiny-ImageNet from pickle: %s", args.pkl)
    X = load_pickle_flat(args.pkl)
    print(f"Loaded {X.shape[0]} images  → shape {X.shape}")

    print(f"Applying '{args.norm}' normalisation …")
    X = normalise(X, args.norm)

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
    p.add_argument("--pkl", type=str,
                   default="/vast/s219110279/TypiClust_3/scan/datasets/tiny-imagenet/train.pkl")
    p.add_argument("--norm", choices=["pixel", "zscore", "unit"],
                   default="zscore")
    p.add_argument("--max_rips_dim", type=int, default=1)
    p.add_argument("--csv", type=str,
                   default="tinyimagenet_pickle_pairs_RAWDATA.csv")
    args = p.parse_args()
    main(args)

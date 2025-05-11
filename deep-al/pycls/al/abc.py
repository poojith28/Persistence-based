import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans
from scipy.stats import gaussian_kde
from gudhi import RipsComplex
import matplotlib.pyplot as plt
import numpy as np 
import torch
from statistics import mean
import gc
import numpy as np
import time
import os
import csv
import math
import sys
import time
import pickle
import math
from copy import deepcopy
from tqdm import tqdm
import gudhi
from gudhi import RipsComplex
from gudhi.representations import PersistenceImage
import numpy as np
from collections import defaultdict
from pycls.datasets.persistence_save import extract_persistence_pairs
from joblib import Parallel, delayed
from scipy.spatial import distance_matrix
import torch.nn as nn

def typiclust_like_selection_h0_no_features(
        uSet, lSet,all_features, dataset, budgetSize, cur_episode, randomSampleSize=10000,
        max_dim=0, verbose=True, n_clusters=10, density_weight=0.7
    ):
        """
        TypiClust-inspired persistence-based active sampling for H0 features,
        without dependency on all_features, using only indices in uSet.
        """
        cur_episode += 2  # Increment episode number for naming and tracking
        # Ensure valid inputs
        assert budgetSize > 0 and budgetSize <= len(uSet)
        assert randomSampleSize > 0

        print(f"\n=== Persistence-Based Selection: Episode {cur_episode-1} ===",flush=True)
        print(f"Selecting {budgetSize} samples",flush=True)

        # Step 1: Randomly sample subset from uSet
        seed = int(time.time())  # Seed based on current time
        np.random.seed(seed)
        sample_size = min(randomSampleSize, len(uSet))  # Ensure we don't exceed uSet size
        sampled_indices = np.random.choice(list(uSet), size=sample_size, replace=False)
        #X_sampled = np.vstack([dataset[idx][0].numpy().reshape(-1) for idx in sampled_indices]).astype(np.float32)
        X_sampled = all_features[sampled_indices].astype(np.float32)
        print(f"Sampled {sample_size} data points from unlabeled set.and shape is {X_sampled.shape}",flush=True)
        # Step 2: Build Rips Complex and compute persistence

        print("Computing persistence...",flush=True)
        rips_complex = RipsComplex(points=X_sampled)
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dim)
        simplex_tree.compute_persistence()
        persistence_pairs = simplex_tree.persistence_pairs()
   
        # Step 3: Extract H0 intervals
        h0_intervals = [
            pt for pt in simplex_tree.persistence_intervals_in_dimension(0)
            if pt[1] != float('inf') and pt[0] >= 0 and pt[1] >= 0  # Exclude negatives
        ]
        
        print(f"Total H0 intervals: {len(h0_intervals)}",flush=True)
        
        # Step 4: Compute persistence lifetimes and density
        persistence_lifetimes = np.array([pt[1] - pt[0] for pt in h0_intervals])
        kde = gaussian_kde(persistence_lifetimes)
        densities = kde(persistence_lifetimes)
        persistence_features = np.array([[pt[0], pt[1]] for pt in h0_intervals])
        n_clusters = min(10, max(20, len(h0_intervals) // 50))  # Dynamically adjust clusters
        print("number of clusters", n_clusters)
        kmeans = KMeans(n_clusters=n_clusters, random_state=seed, init='k-means++').fit(persistence_features)
        cluster_labels = kmeans.labels_

        if verbose:
            print(f"Number of clusters: {n_clusters}")

        # Step 6: Adaptive Selection from Clusters
        def process_cluster(cluster):
            cluster_indices = np.where(cluster_labels == cluster)[0]
            cluster_densities = densities[cluster_indices]
            cluster_lifetimes = persistence_lifetimes[cluster_indices]

            # Adaptive density-lifetime weighting
            density_range = cluster_densities.max() - cluster_densities.min()
            lifetime_range = cluster_lifetimes.max() - cluster_lifetimes.min()
            adaptive_weight = density_range / (density_range + lifetime_range) if (density_range + lifetime_range) > 0 else 0.5

            # Compute adaptive scores
            scores = adaptive_weight * cluster_densities + (1 - adaptive_weight) * cluster_lifetimes
            ranked_indices = cluster_indices[np.argsort(-scores)]  # Descending order

            # Minimum selection ensures representation
            min_samples = max(1, len(cluster_indices) // 25)
            top_k = max(min_samples, len(cluster_indices) // 2)
            return ranked_indices[:top_k]


        # Parallel processing for cluster-wise selection
        selected_indices = Parallel(n_jobs=-1)(
            delayed(process_cluster)(cluster) for cluster in range(n_clusters)
        )

        # Flatten and limit selection to budgetSize
        selected_indices = np.concatenate(selected_indices)[:budgetSize]

        # Step 7: Map persistence intervals to original data indices
        sampled_to_original = {i: sampled_indices[i] for i in range(len(sampled_indices))}
        point_to_images = defaultdict(set)
        for i, (birth_simplex, _) in enumerate(persistence_pairs):
            involved_indices = set(birth_simplex)
            for idx in involved_indices:
                if idx in sampled_to_original:
                    point_to_images[i].add(sampled_to_original[idx])

        queried_indices_set = set()
        for idx in selected_indices:
            images_involved = point_to_images.get(idx, set())
            for image in images_involved:
                if image not in lSet:
                    queried_indices_set.add(image)
                    if len(queried_indices_set) >= budgetSize:
                        break
            if len(queried_indices_set) >= budgetSize:
                break

        queried_indices = np.array(list(queried_indices_set))[:budgetSize]
        activeSet = queried_indices
        remainSet = np.setdiff1d(uSet, activeSet)
        #intervals = simplex_tree.persistence_intervals_in_dimension(0)
        #persis = extract_persistence_pairs(persistence_pairs,intervals)
        print(activeSet)
        return activeSet, remainSet

import numpy as np
import time
import torch
from collections import defaultdict, Counter
from gudhi import RipsComplex

def most_repeated_h0_selection(
    uSet,                # Set (or list) of unlabeled indices
    lSet,                # Set (or list) of labeled indices
    all_features,        # Feature matrix (or placeholder, not strictly used here)
    dataset,             # (Optional) for consistency, not used in selection logic
    budgetSize,          # How many samples to pick
    cur_episode,         # Current episode number
    randomSampleSize=10000,
    max_dim=0,           # Typically 0 or 1 for H0
    verbose=True
):
    """
    Simpler persistence-based active sampling for H0 features only.
    This function selects the 'budgetSize' most frequent birth-simplex pairs
    from dimension-0 persistence, *without* clustering or KDE.

    Parameters
    ----------
    uSet : array-like
        Unlabeled set of indices.
    lSet : array-like
        Already labeled set of indices.
    all_features : np.array
        (Optional) A feature matrix. Here, we only use it for building the Rips complex.
        If you do not need these features, you could skip them or pass dummy placeholders.
    dataset : object
        Unused here in the logic, but kept for interface consistency if needed.
    budgetSize : int
        Number of samples to select.
    cur_episode : int
        Current episode index for logging or checkpoint naming.
    randomSampleSize : int, optional
        How many points to sample from uSet for building the Rips complex.
    max_dim : int, optional
        Max dimension for Rips complex. For H0, typically 0 or 1 is enough.
    verbose : bool, optional
        Whether to print progress/logging messages.

    Returns
    -------
    activeSet : np.ndarray
        Indices selected from unlabeled set for labeling.
    remainSet : np.ndarray
        Updated unlabeled set after removal of the newly labeled samples.
    """

    # 1. Basic setup and random sampling
    cur_episode += 1
    if verbose:
        print(f"\n=== Most-Repeated-H0-Selection: Episode {cur_episode} ===")
        print(f"Selecting {budgetSize} samples")

    assert budgetSize > 0 and budgetSize <= len(uSet), (
        f"budgetSize must be between 1 and |uSet|={len(uSet)}."
    )
    assert randomSampleSize > 0, "randomSampleSize must be positive."

    seed = int(time.time())
    np.random.seed(seed)

    # Randomly sample from uSet (without replacement)
    sample_size = min(randomSampleSize, len(uSet))
    sampled_indices = np.random.choice(list(uSet), size=sample_size, replace=False)

    # We only need their coordinates if we are truly constructing a Rips complex
    # from actual features (could be placeholders otherwise)
    X_sampled = all_features[sampled_indices].astype(np.float32)
    if verbose:
        print(f"Sampled {sample_size} data points from unlabeled set. Shape: {X_sampled.shape}")

    # 2. Build Rips Complex and compute persistence
    if verbose:
        print("Computing persistence...")
    rips_complex = RipsComplex(points=X_sampled)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dim)
    simplex_tree.compute_persistence()

    # 3. Filter to H0 intervals (dimension 0)
    #    GUDHI returns a list of ( (birth_simplex, death_simplex), (birth_time, death_time) ) or similar
    #    The dimension is in the birth_simplex/death_simplex or from other calls. For dimension=0, it's typically single vertices.
    h0_pairs = []
    persistence_pairs = simplex_tree.persistence_pairs()
    for i, (birth_simplex, death_simplex) in enumerate(persistence_pairs):
        # If you only want dimension 0, you can check length of the simplex or
        # call 'persistence_intervals_in_dimension(0)' directly:
        #   for pt in simplex_tree.persistence_intervals_in_dimension(0):
        # But let's assume dimension=0 => birth_simplex is typically a single vertex
        if len(birth_simplex) == 1:  # this is a 0-simplex
            h0_pairs.append((i, birth_simplex, death_simplex))

    if verbose:
        print(f"Found {len(h0_pairs)} H0 pairs (dimension 0).")

    # 4. Count how many times each birth simplex appears
    #    birth_simplex is typically a tuple like (3,) or (17,)

    # Assume birth_simplex_counter is already computed
    birth_simplex_counter = Counter()
    for i, birth_simplex, death in h0_pairs:
        for vertex in death:  # Count individual vertices
            birth_simplex_counter[vertex] += 1

    # Sort vertices by descending frequency
    sorted_vertices = [vertex for vertex, count in birth_simplex_counter.most_common()]

    # Map sampled indices to original dataset indices
    sampled_to_original = {i: sampled_indices[i] for i in range(len(sampled_indices))}

    # Select top vertices until budgetSize is reached
    queried_indices_set = set()
    for vertex in sorted_vertices:
        if vertex in sampled_to_original:
            queried_indices_set.add(sampled_to_original[vertex])
            if len(queried_indices_set) >= budgetSize:
                break
            
    # Final selected set
    queried_indices = np.array(list(queried_indices_set))[:budgetSize]
    activeSet = queried_indices
    remainSet = np.setdiff1d(uSet, activeSet)

    if verbose:
        print(f"Selected {len(activeSet)} new indices to label.")
        print(activeSet)

    return activeSet, remainSet


def most_repeated_h0_selection_NF(
    uSet,                # Set (or list) of unlabeled indices
    lSet,                # Set (or list) of labeled indices
    all_features,        # Feature matrix (or placeholder, not strictly used here)
    dataset,             # (Optional) for consistency, not used in selection logic
    budgetSize,          # How many samples to pick
    cur_episode,         # Current episode number
    randomSampleSize=10000,
    max_dim=0,           # Typically 0 or 1 for H0
    verbose=True
):
    """
    Simpler persistence-based active sampling for H0 features only.
    This function selects the 'budgetSize' most frequent birth-simplex pairs
    from dimension-0 persistence, *without* clustering or KDE.

    Parameters
    ----------
    uSet : array-like
        Unlabeled set of indices.
    lSet : array-like
        Already labeled set of indices.
    all_features : np.array
        (Optional) A feature matrix. Here, we only use it for building the Rips complex.
        If you do not need these features, you could skip them or pass dummy placeholders.
    dataset : object
        Unused here in the logic, but kept for interface consistency if needed.
    budgetSize : int
        Number of samples to select.
    cur_episode : int
        Current episode index for logging or checkpoint naming.
    randomSampleSize : int, optional
        How many points to sample from uSet for building the Rips complex.
    max_dim : int, optional
        Max dimension for Rips complex. For H0, typically 0 or 1 is enough.
    verbose : bool, optional
        Whether to print progress/logging messages.

    Returns
    -------
    activeSet : np.ndarray
        Indices selected from unlabeled set for labeling.
    remainSet : np.ndarray
        Updated unlabeled set after removal of the newly labeled samples.
    """

    # 1. Basic setup and random sampling
    cur_episode += 1
    if verbose:
        print(f"\n=== Most-Repeated-H0-Selection: Episode {cur_episode} ===")
        print(f"Selecting {budgetSize} samples")

    assert budgetSize > 0 and budgetSize <= len(uSet), (
        f"budgetSize must be between 1 and |uSet|={len(uSet)}."
    )
    assert randomSampleSize > 0, "randomSampleSize must be positive."

    seed = int(time.time())
    np.random.seed(seed)

    # Randomly sample from uSet (without replacement)
    sample_size = min(randomSampleSize, len(uSet))
    sampled_indices = np.random.choice(list(uSet), size=sample_size, replace=False)

    # We only need their coordinates if we are truly constructing a Rips complex
    # from actual features (could be placeholders otherwise)
    X_sampled = np.vstack([
                dataset[idx][0].numpy().reshape(-1)
                for idx in sampled_indices
            ]).astype(np.float32)

    if verbose:
        print(f"Sampled {sample_size} data points from unlabeled set. Shape: {X_sampled.shape}")

    # 2. Build Rips Complex and compute persistence
    if verbose:
        print("Computing persistence...")
    rips_complex = RipsComplex(points=X_sampled)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dim)
    simplex_tree.compute_persistence()

    # 3. Filter to H0 intervals (dimension 0)
    #    GUDHI returns a list of ( (birth_simplex, death_simplex), (birth_time, death_time) ) or similar
    #    The dimension is in the birth_simplex/death_simplex or from other calls. For dimension=0, it's typically single vertices.
    h0_pairs = []
    persistence_pairs = simplex_tree.persistence_pairs()
    for i, (birth_simplex, death_simplex) in enumerate(persistence_pairs):
        # If you only want dimension 0, you can check length of the simplex or
        # call 'persistence_intervals_in_dimension(0)' directly:
        #   for pt in simplex_tree.persistence_intervals_in_dimension(0):
        # But let's assume dimension=0 => birth_simplex is typically a single vertex
        if len(birth_simplex) == 1:  # this is a 0-simplex
            h0_pairs.append((i, birth_simplex, death_simplex))

    if verbose:
        print(f"Found {len(h0_pairs)} H0 pairs (dimension 0).")

    # 4. Count how many times each birth simplex appears
    #    birth_simplex is typically a tuple like (3,) or (17,)

    # Assume birth_simplex_counter is already computed
    birth_simplex_counter = Counter()
    for i, birth_simplex, death in h0_pairs:
        for vertex in death:  # Count individual vertices
            birth_simplex_counter[vertex] += 1

    # Sort vertices by descending frequency
    sorted_vertices = [vertex for vertex, count in birth_simplex_counter.most_common()]

    # Map sampled indices to original dataset indices
    sampled_to_original = {i: sampled_indices[i] for i in range(len(sampled_indices))}

    # Select top vertices until budgetSize is reached
    queried_indices_set = set()
    for vertex in sorted_vertices:
        if vertex in sampled_to_original:
            queried_indices_set.add(sampled_to_original[vertex])
            if len(queried_indices_set) >= budgetSize:
                break
            
    # Final selected set
    queried_indices = np.array(list(queried_indices_set))[:budgetSize]
    activeSet = queried_indices
    remainSet = np.setdiff1d(uSet, activeSet)

    if verbose:
        print(f"Selected {len(activeSet)} new indices to label.")
        print(activeSet)

    return activeSet, remainSet

import numpy as np
from gudhi import RipsComplex
from collections import Counter
import time

def persistence_frequency_sampling(
    uSet,
    lSet,
    all_features,
    dataset=None,
    budgetSize=10,
    cur_episode=0,
    randomSampleSize=1000,
    max_dim=1,
    alpha=0.5,
    beta=10.0,
    verbose=True
):
    """
    Active learning sample selection combining frequency of involvement
    and persistence values for vertex scoring.
    """
    from collections import Counter
    from gudhi import RipsComplex
    import numpy as np

    print("alpha and beta =", alpha, beta)
    cur_episode += 1
    if verbose:
        print(f"\n=== Persistence-Frequency Sampling: Episode {cur_episode} ===")
        print(f"Selecting {budgetSize} samples.")

    # --- Validation
    assert 1 <= budgetSize <= len(uSet), f"budgetSize must be between 1 and |uSet|={len(uSet)}."
    assert randomSampleSize > 0, "randomSampleSize must be positive."

    # --- Random Sampling
    sample_size = min(randomSampleSize, len(uSet))
    sampled_indices = np.random.choice(list(uSet), size=sample_size, replace=False)
    X_sampled = all_features[sampled_indices].astype(np.float32)

    if verbose:
        print(f"Sampled {sample_size} data points from unlabeled set.")

    # --- Rips Complex and Persistence
    rips_complex = RipsComplex(points=X_sampled)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dim)
    simplex_tree.compute_persistence()

    # --- Count Vertex Involvement
    involvement_counter = Counter()
    for simplex, filtration_value in simplex_tree.get_filtration():
        for vertex in simplex:
            involvement_counter[vertex] += 1

    # --- Persistence Contribution
    persistence_contribution = Counter()
    try:
        # Use persistence generators (if available)
        for dim, (birth, death), generators in simplex_tree.persistence_generators():
            if death == float('inf'):
                continue
            persistence_value = death - birth
            for simplex in generators:
                for vertex in simplex:
                    persistence_contribution[vertex] += persistence_value
    except AttributeError:
        # Fall back to older GUDHI versions
        for dim, (birth, death) in simplex_tree.persistence():
            if death == float('inf'):
                continue
            persistence_value = death - birth
            for simplex, _ in simplex_tree.get_filtration():
                for vertex in simplex:
                    persistence_contribution[vertex] += persistence_value

    # --- Combine Scores
    combined_scores = {
        v: alpha * involvement_counter[v] + beta * persistence_contribution[v]
        for v in involvement_counter.keys()
    }

    # --- Rank and Select
    sorted_vertices = sorted(combined_scores.keys(), key=lambda v: -combined_scores[v])
    sampled_to_original = {i: sampled_indices[i] for i in range(len(sampled_indices))}
    queried_indices_set = set()
    for vertex in sorted_vertices:
        if vertex in sampled_to_original:
            queried_indices_set.add(sampled_to_original[vertex])
            if len(queried_indices_set) >= budgetSize:
                break

    # Final sets
    activeSet = np.array(list(queried_indices_set))[:budgetSize]
    remainSet = np.setdiff1d(uSet, activeSet)

    if verbose:
        print(f"Selected {len(activeSet)} new indices to label.")

    return activeSet, remainSet
def persistence_frequency_sampling_flag(
    uSet,
    lSet,
    all_features,
    dataset=None,
    budgetSize=10,
    cur_episode=0,
    randomSampleSize=1000,
    max_dim=1,
    alpha=1.0,
    beta=5.0,
    verbose=True,
    distance_threshold=None,  # optional cutoff for Rips
    skip_dim0=True,          # skip dimension 0 features
    use_kmeans=True,         # NEW: Toggle K-Means for diversity
    model_uncertainty=None,  # optional for active learning
    gamma=0.3                # weight for uncertainty
):
    """
    Improved sampling that:
      - (Optional) Picks a subset using K-Means for diversity if use_kmeans=True.
      - Builds a Rips (flag) complex on that subset.
      - Uses 'flag_persistence_generators()' to compute topological features + generators.
      - Skips dimension 0 if skip_dim0=True.
      - Normalizes involvement and persistence.
      - Optionally adds a model-based uncertainty term.
    """
    from collections import Counter
    from gudhi import RipsComplex
    import numpy as np
    from sklearn.cluster import KMeans

    cur_episode += 1
    if verbose:
        print(f"\n=== Persistence-Frequency Sampling (Flag-Based): Episode {cur_episode} ===")
        print(f"Selecting {budgetSize} samples.")

    # --- Validation
    assert 1 <= budgetSize <= len(uSet), "budgetSize must be <= size of unlabeled set."
    sample_size = min(randomSampleSize, len(uSet))

    # --- Subset Selection: K-Means or Random
    if use_kmeans and sample_size < len(uSet):
        if verbose:
            print("Using K-Means for diversity-based sampling.")
        # Fit k-means on all unlabeled points
        kmeans = KMeans(n_clusters=sample_size, n_init=10)
        X_unlabeled = all_features[list(uSet)]
        kmeans.fit(X_unlabeled)
        
        # For each cluster, pick the closest point to its center
        closest_points = []
        for i in range(sample_size):
            cluster_indices = np.where(kmeans.labels_ == i)[0]
            if len(cluster_indices) == 0:
                continue
            # Convert cluster_indices -> actual IDs in uSet
            sub_indices = [list(uSet)[j] for j in cluster_indices]
            center = kmeans.cluster_centers_[i]
            distances = np.linalg.norm(all_features[sub_indices] - center, axis=1)
            min_idx = np.argmin(distances)
            closest_points.append(sub_indices[min_idx])

        sampled_indices = np.array(closest_points, dtype=int)
    else:
        if verbose:
            print("Skipping K-Means; using random subset sampling.")
        # If not using K-Means or randomSampleSize >= len(uSet), just use random sampling
        sampled_indices = np.random.choice(list(uSet), size=sample_size, replace=False)

    X_sampled = all_features[sampled_indices].astype(np.float32)
    if verbose:
        print(f"Selected a subset of size {len(X_sampled)} for topological analysis.")

    # --- Construct Rips Complex (a flag complex)
    if distance_threshold is not None:
        rips_complex = RipsComplex(points=X_sampled, max_edge_length=distance_threshold)
    else:
        rips_complex = RipsComplex(points=X_sampled)

    simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dim)

    # --- Count Vertex Involvement in the Filtration
    involvement_counter = Counter()
    for simplex, filtration_value in simplex_tree.get_filtration():
        for vertex in simplex:
            involvement_counter[vertex] += 1

    # --- Compute Persistence First
    simplex_tree.persistence()
    # --- Compute Flag Persistence Generators
    flag_gens = simplex_tree.flag_persistence_generators()
   # --- Compute Flag Persistence Generators
   # --- Accumulate Persistence Contributions
    persistence_contribution = Counter()

    # Process dimension 0 persistence pairs
    dim0_pairs = flag_gens[0]  # numpy array of shape (n, 3)
    for birth_vertex, death_vertex1, death_vertex2 in dim0_pairs:
        # Skip if skipping dim 0
        if skip_dim0:
            continue
        persistence_value = 1  # Set a value; dimension-0 persistence intervals often don't have well-defined persistence
        # Add to vertices
        persistence_contribution[birth_vertex] += persistence_value
        persistence_contribution[death_vertex1] += persistence_value
        persistence_contribution[death_vertex2] += persistence_value

    # Process higher-dimensional persistence pairs
    dim_gt_0_pairs = flag_gens[1]  # list of numpy arrays
    for dim, pairs in enumerate(dim_gt_0_pairs, start=1):  # dimensions start at 1 here
        for birth_vertex1, birth_vertex2, death_vertex1, death_vertex2 in pairs:
            # Skip infinite intervals
            persistence_value = 1  # Compute persistence if needed
            persistence_contribution[birth_vertex1] += persistence_value
            persistence_contribution[birth_vertex2] += persistence_value
            persistence_contribution[death_vertex1] += persistence_value
            persistence_contribution[death_vertex2] += persistence_value

    # Process connected components (dim 0, essential)
    connected_components = flag_gens[2]  # numpy array of shape (l,)
    for vertex in connected_components:
        persistence_contribution[vertex] += 1  # Add some value for essential features

    # Process other essential features (higher dimensions)
    essential_features = flag_gens[3]  # list of numpy arrays
    for dim, pairs in enumerate(essential_features, start=1):  # dimensions start at 1 here
        for birth_vertex, death_vertex in pairs:
            persistence_value = 1
            persistence_contribution[birth_vertex] += persistence_value
            persistence_contribution[death_vertex] += persistence_value
    # --- Normalization
    max_involvement = max(involvement_counter.values()) if involvement_counter else 1
    max_persistence = max(persistence_contribution.values()) if persistence_contribution else 1

    # --- Combine Scores
    combined_scores = {}
    for v in involvement_counter.keys():
        # Scale each term by its max
        inv_score = involvement_counter[v] / max_involvement
        per_score = persistence_contribution[v] / max_persistence
        
        # Combine with alpha and beta
        topo_score = alpha * inv_score + beta * per_score
        
        # Optionally add a model-based uncertainty
        if model_uncertainty is not None:
            original_idx = sampled_indices[v]
            unc_score = model_uncertainty.get(original_idx, 0.0)
            combined_scores[v] = topo_score + gamma * unc_score
        else:
            combined_scores[v] = topo_score

    # --- Rank and Select based on combined scores
    sorted_vertices = sorted(combined_scores.keys(), key=lambda v: -combined_scores[v])
    sampled_to_original = {i: sampled_indices[i] for i in range(len(sampled_indices))}
    queried_indices_set = set()

    for vertex in sorted_vertices:
        if vertex in sampled_to_original:
            queried_indices_set.add(sampled_to_original[vertex])
            if len(queried_indices_set) >= budgetSize:
                break

    activeSet = np.array(list(queried_indices_set))[:budgetSize]
    remainSet = np.setdiff1d(uSet, activeSet)

    if verbose:
        print(f"Selected {len(activeSet)} new indices to label: {activeSet}")

    return activeSet, remainSet

##VERTEX COUNTER THING

import numpy as np
import pandas as pd
from collections import Counter

def load_vertex_counter(counter_input, verbose=True):
    """
    Converts the vertex counter input into a Counter object.
    
    Parameters
    ----------
    counter_input : str or pandas.DataFrame or dict or Counter
        - If str: treated as a file path to an .npy file containing a dictionary.
        - If pandas.DataFrame: expected to have columns 'vertex' and 'count'.
        - If dict or Counter: will be converted to a Counter.
    verbose : bool, optional
        Whether to print progress messages.
    
    Returns
    -------
    vertex_counter : collections.Counter
        A counter mapping each vertex (index) to its occurrence count.
    """
    if isinstance(counter_input, str):
        # Assume a file path to an .npy file
        if counter_input.endswith(".npy"):
            loaded = np.load(counter_input, allow_pickle=True).item()
            if verbose:
                print(f"Loaded vertex counter from npy file: {counter_input}")
            return Counter(loaded)
        else:
            raise ValueError("Unsupported file type. Expected an .npy file.")
    elif isinstance(counter_input, pd.DataFrame):
        # Check that required columns exist
        if 'vertex' not in counter_input.columns or 'count' not in counter_input.columns:
            raise ValueError("DataFrame must contain columns 'vertex' and 'count'")
        d = dict(zip(counter_input['vertex'], counter_input['count']))
        if verbose:
            print("Converted DataFrame to vertex counter.")
        return Counter(d)
    elif isinstance(counter_input, dict):
        return Counter(counter_input)
    elif isinstance(counter_input, Counter):
        return counter_input
    else:
        raise ValueError("Unsupported vertex counter input type.")

def most_repeated_h0_selection_Vetrex(
    uSet,                # Unlabeled indices
    lSet,                # Labeled indices
    vertex_counter_input,  # Precomputed vertex counter as .npy file path, DataFrame, or dict
    budgetSize,          # Number of samples to pick
    cur_episode,         # Current episode number (for logging)
    verbose=True
):
    """
    Selects the top vertices (by frequency) from a precomputed vertex counter.
    The vertex counter can be provided as an .npy file, a pandas DataFrame, or a dictionary.
    Only vertices that are in the unlabeled set (uSet) and not in the labeled set (lSet) are chosen.
    
    Parameters
    ----------
    uSet : array-like or set
        The set or list of unlabeled indices.
    lSet : array-like or set
        The set or list of already labeled indices.
    vertex_counter_input : str or pandas.DataFrame or dict or Counter
        The precomputed vertex counter. If a string, it is assumed to be the path to an .npy file.
        If a DataFrame, it should have columns 'vertex' and 'count'.
    budgetSize : int
        The number of indices to select.
    cur_episode : int
        The current episode number (used for logging).
    verbose : bool, optional
        Whether to print progress/logging messages.
    
    Returns
    -------
    activeSet : np.ndarray
        Array of selected indices from uSet.
    remainSet : np.ndarray
        Array of remaining indices in uSet after selection.
    """
    # Increment episode number for logging
    cur_episode += 1
    if verbose:
        print(f"\n=== Most-Repeated-H0-Selection: Episode {cur_episode} ===")
        print(f"Selecting {budgetSize} samples based on vertex counts.")
    
    # Ensure uSet and lSet are Python sets for efficient membership testing
    if not isinstance(uSet, set):
        uSet = set(uSet)
    if not isinstance(lSet, set):
        lSet = set(lSet)
    
    # Convert the input vertex counter into a Counter object
    vertex_counter = load_vertex_counter(vertex_counter_input, verbose=verbose)
    
    # Sort vertices by descending frequency
    sorted_vertices = [vertex for vertex, count in vertex_counter.most_common()]
    if verbose:
        print(f"Total vertices in counter: {len(sorted_vertices)}")
    
    # Select top vertices that are in uSet and not in lSet until budgetSize is reached
    active_set = set()
    for vertex in sorted_vertices:
        if vertex in uSet and vertex not in lSet:
            active_set.add(vertex)
        if len(active_set) >= budgetSize:
            break
    
    activeSet = np.array(list(active_set))[:budgetSize]
    remainSet = np.array(list(uSet - active_set))
    
    if verbose:
        print(f"Selected {len(activeSet)} active indices:")
        print(activeSet)
        print(f"Remaining unlabeled set size: {len(remainSet)}")
    
    return activeSet, remainSet

def least_repeated_h0_selection_Vetrex(
    uSet,                # Unlabeled indices
    lSet,                # Labeled indices
    vertex_counter_input,  # Precomputed vertex counter as .npy file path, DataFrame, or dict
    budgetSize,          # Number of samples to pick
    cur_episode,         # Current episode number (for logging)
    verbose=True
):
    """
    Selects the least frequent vertices from a precomputed vertex counter.
    The vertex counter can be provided as an .npy file, a pandas DataFrame, or a dictionary.
    Only vertices that are in the unlabeled set (uSet) and not in the labeled set (lSet) are chosen.
    
    Parameters
    ----------
    uSet : array-like or set
        The set or list of unlabeled indices.
    lSet : array-like or set
        The set or list of already labeled indices.
    vertex_counter_input : str or pandas.DataFrame or dict or Counter
        The precomputed vertex counter. If a string, it is assumed to be the path to an .npy file.
        If a DataFrame, it should have columns 'vertex' and 'count'.
    budgetSize : int
        The number of indices to select.
    cur_episode : int
        The current episode number (used for logging).
    verbose : bool, optional
        Whether to print progress/logging messages.
    
    Returns
    -------
    activeSet : np.ndarray
        Array of selected indices from uSet.
    remainSet : np.ndarray
        Array of remaining indices in uSet after selection.
    """
    # Increment episode number for logging
    cur_episode += 1
    if verbose:
        print(f"\n=== Least-Repeated-H0-Selection: Episode {cur_episode} ===")
        print(f"Selecting {budgetSize} samples based on least frequent vertex counts.")
    
    # Ensure uSet and lSet are Python sets for efficient membership testing
    if not isinstance(uSet, set):
        uSet = set(uSet)
    if not isinstance(lSet, set):
        lSet = set(lSet)
    
    # Convert the input vertex counter into a Counter object
    vertex_counter = load_vertex_counter(vertex_counter_input, verbose=verbose)
    
    # Sort vertices by ascending frequency (least frequent first)
    sorted_vertices = [vertex for vertex, count in sorted(vertex_counter.items(), key=lambda x: x[1])]
    if verbose:
        print(f"Total vertices in counter: {len(sorted_vertices)}")
    
    # Select least frequent vertices that are in uSet and not in lSet until budgetSize is reached
    active_set = set()
    for vertex in sorted_vertices:
        if vertex in uSet and vertex not in lSet:
            active_set.add(vertex)
        if len(active_set) >= budgetSize:
            break
    
    activeSet = np.array(list(active_set))[:budgetSize]
    remainSet = np.array(list(uSet - active_set))
    
    if verbose:
        print(f"Selected {len(activeSet)} least frequent active indices:")
        print(activeSet)
        print(f"Remaining unlabeled set size: {len(remainSet)}")
    
    return activeSet, remainSet

import numpy as np
import csv
import ast

def least_persistence_h0_birth_selection_from_csv(
    csv_file,   # Path to the CSV file containing persistence pairs
    uSet,       # Unlabeled indices (array-like or set)
    lSet,       # Labeled indices (array-like or set)
    budgetSize, # Number of vertices to select
    verbose=True
):
    """
    Select vertices from a CSV document (containing persistence pairs) based on the smallest
    H₀ birth times.

    The CSV file is assumed to have the following columns:
        "Dimension", "Birth", "Death", "Birth Simplex", "Death Simplex", "Involved Simplices"
    
    Only rows with Dimension == 0 are considered. For each such row, the first element of the
    "Birth Simplex" (assumed to be a list with one vertex) is taken as the vertex. Rows are then
    sorted by the birth time (lowest first), and vertices are selected if they belong to uSet (and
    are not already in lSet) until the budget is exhausted.

    Parameters
    ----------
    csv_file : str
        Path to the CSV file with persistence pairs.
    uSet : array-like or set
        The set or list of unlabeled indices.
    lSet : array-like or set
        The set or list of already labeled indices.
    budgetSize : int
        The number of vertices to select.
    verbose : bool, optional
        Whether to print progress/logging messages (default is True).

    Returns
    -------
    activeSet : np.ndarray
        Array of selected vertices (from uSet) based on the smallest H₀ birth times.
    remainSet : np.ndarray
        Array of remaining vertices in uSet after selection.
    """
    # Ensure uSet and lSet are Python sets for efficient membership testing.
    if not isinstance(uSet, set):
        uSet = set(uSet)
    if not isinstance(lSet, set):
        lSet = set(lSet)

    # List to hold tuples of (vertex, birth time)
    h0_entries = []
    
    if verbose:
        print(f"Reading CSV file: {csv_file}")

    # Open and read the CSV file.
    with open(csv_file, mode='r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            try:
                dim = int(row["Dimension"])
            except (KeyError, ValueError):
                continue  # Skip rows with missing or invalid Dimension
            if dim == 0:
                # Parse the birth time
                try:
                    birth = float(row["Birth"])
                except (KeyError, ValueError):
                    continue
                # Parse the "Birth Simplex" string to a list using ast.literal_eval.
                try:
                    birth_simplex = ast.literal_eval(row["Birth Simplex"])
                except Exception as e:
                    if verbose:
                        print(f"Error parsing Birth Simplex: {row['Birth Simplex']}. Skipping this row.")
                    continue
                # For H0, we expect the birth simplex to be a list containing a single vertex.
                if isinstance(birth_simplex, list) and len(birth_simplex) > 0:
                    vertex = birth_simplex[0]
                    h0_entries.append((vertex, birth))
                else:
                    if verbose:
                        print(f"Unexpected Birth Simplex format: {row['Birth Simplex']}. Skipping.")
                    continue

    if verbose:
        print(f"Found {len(h0_entries)} H₀ entries in the CSV file.")

    # Sort the H₀ entries by birth time (ascending order)
    h0_entries_sorted = sorted(h0_entries, key=lambda x: x[1])
    
    # Select vertices from the sorted list that are in uSet and not already in lSet.
    active_set = set()
    for vertex, birth in h0_entries_sorted:
        if vertex in uSet and vertex not in lSet:
            active_set.add(vertex)
        if len(active_set) >= budgetSize:
            break

    activeSet = np.array(list(active_set))[:budgetSize]
    remainSet = np.array(list(uSet - active_set))
    
    if verbose:
        print("Selected vertices based on H₀ birth times:", activeSet)
        print("Remaining unlabeled set size:", len(remainSet))
    
    return activeSet, remainSet

import csv
import ast
import numpy as np

def h0_persistence_selection_from_csv(
    csv_file,    # Path to the CSV file containing persistence pairs.
    uSet,        # Unlabeled indices (array-like or set).
    lSet,        # Already labeled indices (array-like or set).
    budgetSize,  # Number of vertices to select.
    strategy,  # Selection strategy.
    verbose=True
):
    """
    Select vertices from a CSV file (containing H₀ persistence pairs) using various strategies.

    Since the birth time is always zero, persistence is simply equal to the death value.

    The CSV file is assumed to have the following columns:
        "Dimension", "Death", "Birth Simplex", "Death Simplex", "Involved Simplices"
    Only rows with Dimension == 0 are considered. For each such row:
      - The "Death" value is parsed.
      - Persistence is set equal to death (since birth = 0).
      - The "Birth Simplex" is expected to be a list with at least one vertex; the first vertex is used.

    Available strategies:
      - "least_persistence": Select vertices with the smallest persistence (lowest death value).
      - "most_persistence":  Select vertices with the largest persistence (highest death value).
      - "uniform_sampling": After filtering for valid vertices, sort entries by persistence (ascending)
                            and then sample uniformly (evenly spread) along the sorted order.

    Parameters
    ----------
    csv_file : str
        Path to the CSV file with persistence pairs.
    uSet : array-like or set
        The set or list of unlabeled indices.
    lSet : array-like or set
        The set or list of already labeled indices.
    budgetSize : int
        The number of vertices to select.
    strategy : str, optional
        The strategy to use for selection. Options are:
          - "least_persistence" (default)
          - "most_persistence"
          - "uniform_sampling"
    verbose : bool, optional
        Whether to print progress/logging messages (default is True).

    Returns
    -------
    activeSet : np.ndarray
        Array of selected vertices (from uSet) based on the chosen strategy.
    remainSet : np.ndarray
        Array of remaining vertices in uSet after selection.
    """
    # Ensure uSet and lSet are Python sets for efficient membership testing.
    if not isinstance(uSet, set):
        uSet = set(uSet)
    if not isinstance(lSet, set):
        lSet = set(lSet)

    h0_entries = []  # List of tuples: (vertex, death, persistence)

    if verbose:
        print(f"Reading CSV file: {csv_file}")

    # Read the CSV file.
    with open(csv_file, mode='r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            try:
                dim = int(row["Dimension"])
            except (KeyError, ValueError):
                continue  # Skip rows with missing/invalid Dimension.
            if dim == 0:
                # Since birth is always 0, parse death.
                try:
                    death = float(row["Death"])
                except (KeyError, ValueError):
                    death = float('inf')  # Use infinity if death is missing/invalid.
                persistence = death  # Persistence equals death (birth = 0)
                # Parse the "Birth Simplex" string into a list.
                try:
                    birth_simplex = ast.literal_eval(row["Birth Simplex"])
                except Exception as e:
                    if verbose:
                        print(f"Error parsing Birth Simplex: {row.get('Birth Simplex', None)}. Skipping row.")
                    continue
                # Expect a list with at least one vertex.
                if isinstance(birth_simplex, list) and len(birth_simplex) > 0:
                    vertex = birth_simplex[0]
                    h0_entries.append((vertex, death, persistence))
                else:
                    if verbose:
                        print(f"Unexpected Birth Simplex format: {row.get('Birth Simplex', None)}. Skipping row.")
                    continue

    if verbose:
        print(f"Found {len(h0_entries)} H₀ entries in the CSV file.")

    # Selection based on the chosen strategy.
    if strategy in ["least_persistence", "most_persistence"]:
        # For these strategies, sort the entries by persistence.
        reverse_sort = (strategy == "most_persistence")
        sorted_entries = sorted(h0_entries, key=lambda entry: entry[2], reverse=reverse_sort)
        active_list = []
        for vertex, death, persistence in sorted_entries:
            if vertex in uSet and vertex not in lSet:
                active_list.append(vertex)
            if len(active_list) >= budgetSize:
                break
        activeSet = np.array(active_list)[:budgetSize]

    elif strategy == "uniform_sampling":
        # Filter entries to include only vertices in uSet and not in lSet.
        filtered_entries = [entry for entry in h0_entries if entry[0] in uSet and entry[0] not in lSet]
        # Sort the filtered entries by persistence (ascending order).
        filtered_entries = sorted(filtered_entries, key=lambda entry: entry[2])
        n_candidates = len(filtered_entries)
        if n_candidates == 0:
            activeSet = np.array([])
        elif n_candidates <= budgetSize:
            activeSet = np.array([entry[0] for entry in filtered_entries])
        else:
            # Select indices that are evenly spaced along the sorted list.
            indices = np.linspace(0, n_candidates - 1, budgetSize).astype(int)
            activeSet = np.array([filtered_entries[i][0] for i in indices])
    else:
        raise ValueError("Invalid strategy provided. Choose from 'least_persistence', 'most_persistence', or 'uniform_sampling'.")

    remainSet = np.array(list(uSet - set(activeSet)))
    
    if verbose:
        print(f"Selected vertices based on strategy '{strategy}':", activeSet)
        print("Remaining unlabeled set size:", len(remainSet))
    
    return activeSet, remainSet

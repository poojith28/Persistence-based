# This file is modified from a code implementation shared with me by Prateek Munjal et al., authors of the paper https://arxiv.org/abs/2002.09564
# GitHub: https://github.com/PrateekMunjal
# ----------------------------------------------------------
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

from scipy.spatial import distance_matrix
import torch.nn as nn

# import pycls.datasets.loader as imagenet_loader
from .vaal_util import train_vae_disc
def compute_persistence_diagram(data, max_dim=2):
    """
    Compute persistence diagrams for the given data.
    Args:
        data (numpy.ndarray): Subsample of the dataset (e.g., 500 samples).
        max_dim (int): Maximum dimension of persistent homology.
    Returns:
        list: Persistence diagrams for each sample.
    """
    persistence_diagrams = []
    for sample in data:
        rips = RipsComplex(points=sample, max_edge_length=21000)
        simplex_tree = rips.create_simplex_tree(max_dimension=max_dim)
        persistence = simplex_tree.persistence()
        persistence_diagrams.append(persistence)
    return persistence_diagrams

def persistence_to_features(persistence_diagrams):
    """
    Convert persistence diagrams to persistence image features.
    Args:
        persistence_diagrams (list): List of persistence diagrams.
    Returns:
        numpy.ndarray: Array of persistence image features.
    """
    pi = PersistenceImage(bandwidth=0.1, weight=lambda x: x[1], resolution=[10, 10])
    features = np.array([pi.fit_transform([diag]) for diag in persistence_diagrams])
    return features

def select_from_persistence_diagrams(persistence_diagrams, budget, strategy="h0_top"):
    """
    Select indices based on persistence diagrams and a specific strategy.
    Args:
        persistence_diagrams (list): List of persistence diagrams for all samples.
        budget (int): Number of samples to select.
        strategy (str): Sampling strategy.
    Returns:
        list: Indices of selected samples.
    """
    scores = []
    for i, diagram in enumerate(persistence_diagrams):
        if strategy in ["h0_top", "h0_bottom", "h0_h1_half"]:
            h0 = [point[1] - point[0] for point in diagram if point[0] != point[1] and point[1] != float('inf')]
        else:
            h0 = []

        if strategy in ["h1_top", "h0_h1_half"]:
            h1 = [point[1] - point[0] for point in diagram if point[1] == float('inf')]
        else:
            h1 = []

        if strategy == "h0_top":
            scores.append((i, max(h0) if h0 else 0))
        elif strategy == "h0_bottom":
            scores.append((i, min(h0) if h0 else 0))
        elif strategy == "h1_top":
            scores.append((i, max(h1) if h1 else 0))
        elif strategy == "h0_h1_half":
            scores.append((i, max(h0 + h1) if h0 or h1 else 0))

    scores.sort(key=lambda x: x[1], reverse=(strategy != "h0_bottom"))
    selected_indices = [score[0] for score in scores[:budget]]
    return selected_indices

class EntropyLoss(nn.Module):
    """
    This class contains the entropy function implemented.
    """
    def __init__(self):
        super(EntropyLoss, self).__init__()

    def forward(self, x, applySoftMax=True):
        #Assuming x : [BatchSize, ]

        if applySoftMax:
            entropy = torch.nn.functional.softmax(x, dim=1)*torch.nn.functional.log_softmax(x, dim=1)
        else:
            entropy = x * torch.log2(x)
        entropy = -1*entropy.sum(dim=1)
        return entropy 


class CoreSetMIPSampling():
    """
    Implements coreset MIP sampling operation
    """
    def __init__(self, cfg, dataObj, isMIP = False):
        self.dataObj = dataObj
        self.cuda_id = torch.cuda.current_device()
        self.cfg = cfg
        self.isMIP = isMIP

    @torch.no_grad()
    def get_representation(self, clf_model, idx_set, dataset):

        clf_model.cuda(self.cuda_id)
        # if self.cfg.TRAIN.DATASET == "IMAGENET":
        #     print("Loading the model in data parallel where num_GPUS: {}".format(self.cfg.NUM_GPUS))
        #     clf_model = torch.nn.DataParallel(clf_model, device_ids = [i for i in range(self.cfg.NUM_GPUS)])
        
        #     tempIdxSetLoader = imagenet_loader.construct_loader_no_aug(cfg=self.cfg, indices=idx_set, isDistributed=False, isShuffle=False, isVaalSampling=False)
        # else:
        tempIdxSetLoader = self.dataObj.getSequentialDataLoader(indexes=idx_set, batch_size=int(self.cfg.TRAIN.BATCH_SIZE/self.cfg.NUM_GPUS), data=dataset)
        features = []
        
        print(f"len(dataLoader): {len(tempIdxSetLoader)}")

        for i, (x, _) in enumerate(tqdm(tempIdxSetLoader, desc="Extracting Representations")):
            with torch.no_grad():
                x = x.cuda(self.cuda_id)
                x = x.type(torch.cuda.FloatTensor)
                temp_z, _ = clf_model(x)
                features.append(temp_z.cpu().numpy())

        features = np.concatenate(features, axis=0)
        return features

    def gpu_compute_dists(self,M1,M2):
        """
        Computes L2 norm square on gpu
        Assume 
        M1: M x D matrix
        M2: N x D matrix

        output: M x N matrix
        """
        #print(f"Function call to gpu_compute dists; M1: {M1.shape} and M2: {M2.shape}")
        M1_norm = (M1**2).sum(1).reshape(-1,1)

        M2_t = torch.transpose(M2, 0, 1)
        M2_norm = (M2**2).sum(1).reshape(1,-1)
        dists = M1_norm + M2_norm - 2.0 * torch.mm(M1, M2_t)
        return dists

    def compute_dists(self, X, X_train):
        dists = -2 * np.dot(X, X_train.T) + np.sum(X_train**2,axis=1) + np.sum(X**2, axis=1).reshape((-1,1))
        return dists

    def optimal_greedy_k_center(self, labeled, unlabeled):
        n_lSet = labeled.shape[0]
        lSetIds = np.arange(n_lSet)
        n_uSet = unlabeled.shape[0]
        uSetIds = n_lSet + np.arange(n_uSet)

        #order is important
        features = np.vstack((labeled,unlabeled))
        print("Started computing distance matrix of {}x{}".format(features.shape[0], features.shape[0]))
        start = time.time()
        distance_mat = self.compute_dists(features, features)
        end = time.time()
        print("Distance matrix computed in {} seconds".format(end-start))
        greedy_indices = []
        for i in range(self.cfg.ACTIVE_LEARNING.BUDGET_SIZE):
            if i!=0 and i%500==0:
                print("Sampled {} samples".format(i))
            lab_temp_indexes = np.array(np.append(lSetIds, greedy_indices),dtype=int)
            min_dist = np.min(distance_mat[lab_temp_indexes, n_lSet:],axis=0)
            active_index = np.argmax(min_dist)
            greedy_indices.append(n_lSet + active_index)
        
        remainSet = set(np.arange(features.shape[0])) - set(greedy_indices) - set(lSetIds)
        remainSet = np.array(list(remainSet))

        return greedy_indices-n_lSet, remainSet

    def greedy_k_center(self, labeled, unlabeled):
        greedy_indices = [None for i in range(self.cfg.ACTIVE_LEARNING.BUDGET_SIZE)]
        greedy_indices_counter = 0
        #move cpu to gpu
        labeled = torch.from_numpy(labeled).cuda(0)
        unlabeled = torch.from_numpy(unlabeled).cuda(0)

        print(f"[GPU] Labeled.shape: {labeled.shape}")
        print(f"[GPU] Unlabeled.shape: {unlabeled.shape}")
        # get the minimum distances between the labeled and unlabeled examples (iteratively, to avoid memory issues):
        st = time.time()
        min_dist,_ = torch.min(self.gpu_compute_dists(labeled[0,:].reshape((1,labeled.shape[1])), unlabeled), dim=0)
        min_dist = torch.reshape(min_dist, (1, min_dist.shape[0]))
        print(f"time taken: {time.time() - st} seconds")

        temp_range = 500
        dist = np.empty((temp_range, unlabeled.shape[0]))
        for j in tqdm(range(1, labeled.shape[0], temp_range), desc="Getting first farthest index"):
            if j + temp_range < labeled.shape[0]:
                dist = self.gpu_compute_dists(labeled[j:j+temp_range, :], unlabeled)
            else:
                dist = self.gpu_compute_dists(labeled[j:, :], unlabeled)
            
            min_dist = torch.cat((min_dist, torch.min(dist,dim=0)[0].reshape((1,min_dist.shape[1]))))

            min_dist = torch.min(min_dist, dim=0)[0]
            min_dist = torch.reshape(min_dist, (1, min_dist.shape[0]))

        # iteratively insert the farthest index and recalculate the minimum distances:
        _, farthest = torch.max(min_dist, dim=1)
        greedy_indices [greedy_indices_counter] = farthest.item()
        greedy_indices_counter += 1

        amount = self.cfg.ACTIVE_LEARNING.BUDGET_SIZE-1
        
        for i in tqdm(range(amount), desc = "Constructing Active set"):
            dist = self.gpu_compute_dists(unlabeled[greedy_indices[greedy_indices_counter-1], :].reshape((1,unlabeled.shape[1])), unlabeled)
            
            min_dist = torch.cat((min_dist, dist.reshape((1, min_dist.shape[1]))))
            
            min_dist, _ = torch.min(min_dist, dim=0)
            min_dist = min_dist.reshape((1, min_dist.shape[0]))
            _, farthest = torch.max(min_dist, dim=1)
            greedy_indices [greedy_indices_counter] = farthest.item()
            greedy_indices_counter += 1

        remainSet = set(np.arange(unlabeled.shape[0])) - set(greedy_indices)
        remainSet = np.array(list(remainSet))
        if self.isMIP:
            return greedy_indices,remainSet,math.sqrt(np.max(min_dist))
        else:
            return greedy_indices, remainSet

    def query(self, lSet, uSet, clf_model, dataset):

        assert clf_model.training == False, "Classification model expected in training mode"
        assert clf_model.penultimate_active == True,"Classification model is expected in penultimate mode"    
        
        print("Extracting Lset Representations")
        lb_repr = self.get_representation(clf_model=clf_model, idx_set=lSet, dataset=dataset)
        print("Extracting Uset Representations")
        ul_repr = self.get_representation(clf_model=clf_model, idx_set=uSet, dataset=dataset)
        
        print("lb_repr.shape: ",lb_repr.shape)
        print("ul_repr.shape: ",ul_repr.shape)
        
        if self.isMIP == True:
            raise NotImplementedError
        else:
            print("Solving K Center Greedy Approach")
            start = time.time()
            greedy_indexes, remainSet = self.greedy_k_center(labeled=lb_repr, unlabeled=ul_repr)
            # greedy_indexes, remainSet = self.optimal_greedy_k_center(labeled=lb_repr, unlabeled=ul_repr)
            end = time.time()
            print("Time taken to solve K center: {} seconds".format(end-start))
            activeSet = uSet[greedy_indexes]
            remainSet = uSet[remainSet]
        return activeSet, remainSet


class Sampling:
    """
    Here we implement different sampling methods which are used to sample
    active learning points from unlabelled set.
    """

    def rips_persistence_sampling_learned_BINS(
        self,
        uSet,
        all_features,
        dataset,
        budgetSize,cur_episode,
        randomSampleSize=750,
        max_dim=2,
        strategy="h1_bins",
        verbose=True,
        num_bins=5
    ):
        """
        Rips Persistence Sampling for active learning using binning strategies.

        Parameters
        ----------
        uSet : np.ndarray
            Indices of the unlabeled set.
        all_features : np.ndarray or None
            Precomputed features array (N, d) or None to fall back to raw data.
        dataset : Dataset
            Data object for fallback if all_features is None.
        budgetSize : int
            Number of samples to select.
        randomSampleSize : int
            Number of random samples to process.
        max_dim : int
            Maximum homology dimension to compute.
        strategy : str
            "h0_bins", "h1_bins", or "h0_h1_bins".
        verbose : bool
            Print logs if True.
        num_bins : int
            Number of bins for persistence scores.

        Returns
        -------
        activeSet : np.ndarray
            Indices of selected samples.
        remainSet : np.ndarray
            Remaining unlabeled indices.
        """

        assert budgetSize > 0
        assert randomSampleSize > 0
        assert budgetSize <= len(uSet)


        print(f"Sample size: {randomSampleSize} | Strategy: {strategy} | num_bins: {num_bins}")

        seed = int(time.time())
        np.random.seed(seed)
        uSet = uSet.astype(int)
        sample_size = min(randomSampleSize, len(uSet))
        sampled_indices = np.random.choice(uSet, size=sample_size, replace=False)

        if all_features is not None:
            X_sampled = all_features[sampled_indices].astype(np.float32)
        else:
            X_sampled = np.vstack([
                dataset[idx][0].numpy().reshape(-1) for idx in sampled_indices
            ]).astype(np.float32)

        print("Data shape:", X_sampled.shape)

        rips_complex = RipsComplex(points=X_sampled)
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dim)
        simplex_tree.compute_persistence()
        persistence_pairs = simplex_tree.persistence_pairs()

        h0_intervals = [pt for pt in simplex_tree.persistence_intervals_in_dimension(0) if pt[1] != float('inf')]
        h1_intervals = [pt for pt in simplex_tree.persistence_intervals_in_dimension(1) if pt[1] != float('inf')]

        h0_persistence = [(i, pt[1] - pt[0]) for i, pt in enumerate(h0_intervals)]
        h1_persistence = [(i, pt[1] - pt[0]) for i, pt in enumerate(h1_intervals)]

        h0_persistence_sorted = sorted(h0_persistence, key=lambda x: x[1], reverse=True)
        h1_persistence_sorted = sorted(h1_persistence, key=lambda x: x[1], reverse=True)

        if verbose:
            print(f"Top H1: {h1_persistence_sorted[:5]}")
            print(f"Top H0: {h0_persistence_sorted[:5]}")


        def binning_strategy(intervals_sorted, budget, bins=num_bins):
            if len(intervals_sorted) == 0 or budget == 0:
                return []
            vals = [x[1] for x in intervals_sorted]
            bin_edges = np.percentile(vals, np.linspace(0, 100, bins+1))
            bin_groups = [[] for _ in range(bins)]
            for (idx_i, p) in intervals_sorted:
                placed = False
                for b_idx in range(bins):
                    if bin_edges[b_idx] <= p <= bin_edges[b_idx+1]:
                        bin_groups[b_idx].append(idx_i)
                        placed = True
                        break
                if not placed:
                    bin_groups[-1].append(idx_i)
            chosen = []
            per_bin = max(1, budget // bins)
            total_taken = 0
            for group in bin_groups:
                np.random.shuffle(group)
                take = min(per_bin, len(group))
                chosen.extend(group[:take])
                total_taken += take
                if total_taken >= budget:
                    break
            return chosen[:budget]

        if strategy == "h0_bins":
            top_feature_indices = binning_strategy(h0_persistence_sorted, budgetSize, num_bins)
        elif strategy == "h1_bins":
            top_feature_indices = binning_strategy(h1_persistence_sorted, budgetSize, num_bins)
        elif strategy == "h0_h1_bins":
            half_budget = budgetSize // 2
            leftover = budgetSize - half_budget
            top_h0 = binning_strategy(h0_persistence_sorted, half_budget, num_bins)
            top_h1 = binning_strategy(h1_persistence_sorted, leftover, num_bins)
            top_feature_indices = top_h0 + top_h1
        else:
            raise ValueError(f"Invalid strategy: {strategy}")


        print(f"Selected Interval Indices: {top_feature_indices}")

        # Step 5: Map top features to data points
        point_to_images = defaultdict(set)
        for i, (birth_simplex, death_simplex) in enumerate(persistence_pairs):
            involved_indices = set(birth_simplex)
            for idx in involved_indices:
                point_to_images[i].add(sampled_indices[idx])

        queried_indices_set = set()
        for idx in top_feature_indices:
            images_involved = point_to_images.get(idx, set())
            queried_indices_set.update(images_involved)
            if len(queried_indices_set) >= budgetSize:
                break

        # Trim to budgetSize
        queried_indices = np.array(list(queried_indices_set))[:budgetSize]
        activeSet = queried_indices

        # Update remaining unlabeled set
        remainSet = np.setdiff1d(uSet, activeSet)
        print(f"Selected {activeSet} samples for the active set.")

        print(f"Selected {len(activeSet)} samples for the active set.")
        print(f"Remaining unlabeled set size: {len(remainSet)}")

        return activeSet, remainSet

    def rips_persistence_sampling_learned_Changed(self,lSet, uSet,all_features, dataset, budgetSize,cur_episode, randomSampleSize=750, max_dim=2, strategy="h1_top", verbose=True,):
        """
        Implements Rips Persistence Sampling for active learning.

        Parameters:
        -----------
        uSet : np.ndarray
            Indices of the unlabeled set.
        dataset : Dataset
            Data object containing the data points.
        budgetSize : int
            Number of samples to select.
        randomSampleSize : int, optional
            Number of random samples to process (default: 750).
        max_dim : int, optional
            Maximum homology dimension to compute (default: 2).
        strategy : str, optional
            Sampling strategy: "h0_top", "h0_bottom", "h1_top", or "h0_h1_half" (default: "h1_top").
        verbose : bool, optional
            Whether to print detailed logs (default: False).

        Returns:
        --------
        activeSet : np.ndarray
            Selected active set.
        remainSet : np.ndarray
            Remaining unlabeled set.
        """

        assert budgetSize > 0, "Budget size must be greater than 0."
        assert randomSampleSize > 0, "Random sample size must be greater than 0."
        assert budgetSize <= len(uSet), "Budget size cannot exceed the size of the unlabeled set."

        print(f"sample size is : {randomSampleSize} and techinque is : {strategy}")

        # Step 1: Randomly sample subset from uSet
        seed = int(time.time())  # or fix a seed for reproducibility
        np.random.seed(seed)
        uSet = uSet.astype(int)     # ensure it's an integer array
        sample_size = min(randomSampleSize, len(uSet))
        sampled_indices = np.random.choice(uSet, size=sample_size, replace=False)

    # Step 2: Retrieve features for the sampled indices
        if all_features is not None:
            # If the entire dataset's features are already loaded, just slice them
            X_sampled = all_features[sampled_indices].astype(np.float32)
        else:
            # Fallback: gather raw data from dataset and flatten it.
            # (You'll only need this if you want to keep that fallback.)
            X_sampled = np.vstack([
            dataset[idx][0].numpy().reshape(-1) for idx in sampled_indices
            ]).astype(np.float32)
        
        print("shape of data is " ,X_sampled.shape)
        print("Computing persistence")
        # Step 2: Build Rips Complex and compute persistence
        rips_complex = RipsComplex(points=all_features)
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dim)
        simplex_tree.compute_persistence()
        persistence_pairs = simplex_tree.persistence_pairs()

        # Step 3: Extract persistence intervals
        h0_intervals = simplex_tree.persistence_intervals_in_dimension(0)
        h1_intervals = simplex_tree.persistence_intervals_in_dimension(1)
        h1_intervals = [pt for pt in h1_intervals if pt[1] != float('inf')]  # Exclude infinite intervals
        h0_intervals = [pt for pt in h0_intervals if pt[1] != float('inf')]  # Exclude infinite intervals
        print("")
        # Compute persistence scores
        h1_persistence = [(i, pt[1] - pt[0]) for i, pt in enumerate(h1_intervals)]
        h0_persistence = [(i, pt[1] - pt[0]) for i, pt in enumerate(h0_intervals)]
        h1_persistence_sorted = sorted(h1_persistence, key=lambda x: x[1], reverse=True)
        h0_persistence_sorted = sorted(h0_persistence, key=lambda x: x[1], reverse=True)
        h1_persistence = sorted(h1_persistence, key=lambda x: x[1], reverse=False)
        h0_persistence = sorted(h0_persistence, key=lambda x: x[1], reverse=False)
        print("Done")

        print(f"Top H1 Persistence Scores: {h1_persistence_sorted[:10]}")
        print(f"Top H0 Persistence Scores: {h0_persistence_sorted[:10]}")

        # Step 4: Identify top features based on strategy
        if strategy == "h1_top":
            top_feature_indices = [i for i, _ in h1_persistence_sorted]
        elif strategy == "h0_top":
            top_feature_indices = [i for i, _ in h0_persistence_sorted]
        elif strategy == "h0_bottom":
            top_feature_indices = [i for i, _ in h0_persistence]
        elif strategy == "h1_bottom":
            top_feature_indices = [i for i, _ in h1_persistence]

        elif strategy == "h0_top_h0_bottom":
            half_budget = budgetSize // 2
            top_feature_indices = (
        [i for i, _ in h0_persistence_sorted[:half_budget]] +
        [i for i, _ in reversed(h0_persistence_sorted[-half_budget:])]
    )
        elif strategy == "h1_top_h1_bottom":
            half_budget = budgetSize // 2
            top_feature_indices = (
        [i for i, _ in h1_persistence_sorted[:half_budget]] +
        [i for i, _ in reversed(h1_persistence_sorted[-half_budget:])]
    )
        elif strategy == "h0_top_h1_top":
            half_budget = budgetSize // 2
            top_feature_indices = (
        [i for i, _ in h0_persistence_sorted[:half_budget]] +
        [i for i, _ in h1_persistence_sorted[:half_budget]]
    )
        elif strategy == "h0_top_h1_bottom":
            half_budget = budgetSize // 2
            top_feature_indices = (
        [i for i, _ in h0_persistence_sorted[:half_budget]] +
        [i for i, _ in reversed(h1_persistence_sorted[-half_budget:])]
    )
        elif strategy == "h0_bottom_h1_top":
            half_budget = budgetSize // 2
            top_feature_indices = (
        [i for i, _ in reversed(h0_persistence_sorted[-half_budget:])] +
        [i for i, _ in h1_persistence_sorted[:half_budget]]
    )
        elif strategy == "h0_bottom_h1_bottom":
            half_budget = budgetSize // 2
            top_feature_indices = (
        [i for i, _ in reversed(h0_persistence_sorted[-half_budget:])] +
        [i for i, _ in reversed(h1_persistence_sorted[-half_budget:])]
    )
        elif strategy == "h0_middle_out":
            n = len(h0_persistence_sorted)
            mid = n // 2   # midpoint index
            top_feature_indices = []
            left = mid
            right = mid + 1

            # Keep picking from the midpoint outward until we have enough
            while len(top_feature_indices) < budgetSize and (left >= 0 or right < n):
                # Pick from `left` (the middle going down) if in range
                if left >= 0:
                    idx = h0_persistence_sorted[left][0]
                    if idx not in lSet:  # Ensure no overlap with lSet
                        top_feature_indices.append(idx)
                    left -= 1
                    if len(top_feature_indices) >= budgetSize:
                        break

                # Pick from `right` (the middle going up) if in range
                if right < n:
                    idx = h0_persistence_sorted[right][0]
                    if idx not in lSet:  # Ensure no overlap with lSet
                        top_feature_indices.append(idx)
                    right += 1
                    if len(top_feature_indices) >= budgetSize:
                        break

       
        else:
            raise ValueError(f"Invalid strategy: {strategy}")

        if verbose:
            print(f"Selected Top Feature Indices: {top_feature_indices}")
        # Step 5: Map top features to data points
        point_to_images = defaultdict(set)
        for i, (birth_simplex, death_simplex) in enumerate(persistence_pairs):
            involved_indices = set(birth_simplex)
            for idx in involved_indices:
                point_to_images[i].add(idx)
        
        queried_indices_set = set()

        for idx in top_feature_indices:
            # Get all images involved with the current feature index
            images_involved = point_to_images.get(idx, set())
            
            # Check each image against lSet and add if not in lSet
            for image in images_involved:
                if image not in lSet:  # Exclude if already in lSet
                    queried_indices_set.add(image)
                    if len(queried_indices_set) >= budgetSize:  # Stop once we reach the budget size
                        break
            if len(queried_indices_set) >= budgetSize:
                break
        print("quired set",queried_indices_set)
        print("active set",lSet)

        # Trim to budgetSize
        queried_indices = np.array(list(queried_indices_set))[:budgetSize]
        activeSet = queried_indices
        
        # Update remaining unlabeled set
        remainSet = np.setdiff1d(uSet, activeSet)

        if verbose:
            print(f"Selected {len(activeSet)} samples for the active set.")
            print(f"Remaining unlabeled set size: {len(remainSet)}")
        print(activeSet)
        return activeSet, remainSet

    def rips_persistence_sampling_learned(self, uSet,all_features, dataset, budgetSize,cur_episode, randomSampleSize=750, max_dim=2, strategy="h1_top", verbose=True,):
        """
        Implements Rips Persistence Sampling for active learning.

        Parameters:
        -----------
        uSet : np.ndarray
            Indices of the unlabeled set.
        dataset : Dataset
            Data object containing the data points.
        budgetSize : int
            Number of samples to select.
        randomSampleSize : int, optional
            Number of random samples to process (default: 750).
        max_dim : int, optional
            Maximum homology dimension to compute (default: 2).
        strategy : str, optional
            Sampling strategy: "h0_top", "h0_bottom", "h1_top", or "h0_h1_half" (default: "h1_top").
        verbose : bool, optional
            Whether to print detailed logs (default: False).

        Returns:
        --------
        activeSet : np.ndarray
            Selected active set.
        remainSet : np.ndarray
            Remaining unlabeled set.
        """

        assert budgetSize > 0, "Budget size must be greater than 0."
        assert randomSampleSize > 0, "Random sample size must be greater than 0."
        assert budgetSize <= len(uSet), "Budget size cannot exceed the size of the unlabeled set."

        print(f"sample size is : {randomSampleSize} and techinque is : {strategy}")

        # Step 1: Randomly sample subset from uSet
        seed = int(time.time())  # or fix a seed for reproducibility
        np.random.seed(seed)
        uSet = uSet.astype(int)     # ensure it's an integer array
        sample_size = min(randomSampleSize, len(uSet))
        sampled_indices = np.random.choice(uSet, size=sample_size, replace=False)

    # Step 2: Retrieve features for the sampled indices
        if all_features is not None:
            # If the entire dataset's features are already loaded, just slice them
            X_sampled = all_features[sampled_indices].astype(np.float32)
        else:
            # Fallback: gather raw data from dataset and flatten it.
            # (You'll only need this if you want to keep that fallback.)
            X_sampled = np.vstack([
            dataset[idx][0].numpy().reshape(-1) for idx in sampled_indices
            ]).astype(np.float32)

        print("shape of data is " ,X_sampled.shape)
        print("Computing persistence")
        # Step 2: Build Rips Complex and compute persistence
        rips_complex = RipsComplex(points=X_sampled)
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dim)
        simplex_tree.compute_persistence()
        persistence_pairs = simplex_tree.persistence_pairs()

        # Step 3: Extract persistence intervals
        h0_intervals = simplex_tree.persistence_intervals_in_dimension(0)
        h1_intervals = simplex_tree.persistence_intervals_in_dimension(1)
        h1_intervals = [pt for pt in h1_intervals if pt[1] != float('inf')]  # Exclude infinite intervals
        h0_intervals = [pt for pt in h0_intervals if pt[1] != float('inf')]  # Exclude infinite intervals

        # Compute persistence scores
        h1_persistence = [(i, pt[1] - pt[0]) for i, pt in enumerate(h1_intervals)]
        h0_persistence = [(i, pt[1] - pt[0]) for i, pt in enumerate(h0_intervals)]
        h1_persistence_sorted = sorted(h1_persistence, key=lambda x: x[1], reverse=True)
        h0_persistence_sorted = sorted(h0_persistence, key=lambda x: x[1], reverse=True)
        print("Done")

        
        print(f"Top H1 Persistence Scores: {h1_persistence_sorted[:10]}")
        print(f"Top H0 Persistence Scores: {h0_persistence_sorted[:10]}")

        # Step 4: Identify top features based on strategy
        if strategy == "h1_top":
            top_feature_indices = [i for i, _ in h1_persistence_sorted[:budgetSize]]
        elif strategy == "h0_top":
            top_feature_indices = [i for i, _ in h0_persistence_sorted[:budgetSize]]
        elif strategy == "h0_bottom":
            top_feature_indices = [i for i, _ in reversed(h0_persistence_sorted[-budgetSize:])]
        elif strategy == "h1_bottom":
            top_feature_indices = [i for i, _ in reversed(h1_persistence_sorted[-budgetSize:])]

        elif strategy == "h0_top_h0_bottom":
            half_budget = budgetSize // 2
            top_feature_indices = (
        [i for i, _ in h0_persistence_sorted[:half_budget]] +
        [i for i, _ in reversed(h0_persistence_sorted[-half_budget:])]
    )
        elif strategy == "h1_top_h1_bottom":
            half_budget = budgetSize // 2
            top_feature_indices = (
        [i for i, _ in h1_persistence_sorted[:half_budget]] +
        [i for i, _ in reversed(h1_persistence_sorted[-half_budget:])]
    )
        elif strategy == "h0_top_h1_top":
            half_budget = budgetSize // 2
            top_feature_indices = (
        [i for i, _ in h0_persistence_sorted[:half_budget]] +
        [i for i, _ in h1_persistence_sorted[:half_budget]]
    )
        elif strategy == "h0_top_h1_bottom":
            half_budget = budgetSize // 2
            top_feature_indices = (
        [i for i, _ in h0_persistence_sorted[:half_budget]] +
        [i for i, _ in reversed(h1_persistence_sorted[-half_budget:])]
    )
        elif strategy == "h0_bottom_h1_top":
            half_budget = budgetSize // 2
            top_feature_indices = (
        [i for i, _ in reversed(h0_persistence_sorted[-half_budget:])] +
        [i for i, _ in h1_persistence_sorted[:half_budget]]
    )
        elif strategy == "h0_bottom_h1_bottom":
            half_budget = budgetSize // 2
            top_feature_indices = (
        [i for i, _ in reversed(h0_persistence_sorted[-half_budget:])] +
        [i for i, _ in reversed(h1_persistence_sorted[-half_budget:])]
    )
        elif strategy == "h0_middle_out":
            """
        New strategy that starts picking from the midpoint of h0_persistence_sorted
        and expands outward to both sides.
            """
            n = len(h0_persistence_sorted)
            mid = n // 2   # midpoint index
            top_feature_indices = []
            left = mid
            right = mid + 1

        # Keep picking from the midpoint outward until we have enough
            while len(top_feature_indices) < budgetSize and (left >= 0 or right < n):
                # Pick from `left` (the middle going down) if in range
                if left >= 0:
                    top_feature_indices.append(h0_persistence_sorted[left][0])
                    left -= 1
                    if len(top_feature_indices) >= budgetSize:
                        break

            # Pick from `right` (the middle going up) if in range
                if right < n:
                    top_feature_indices.append(h0_persistence_sorted[right][0])
                    right += 1
                    if len(top_feature_indices) >= budgetSize:
                        break
        else:
            raise ValueError(f"Invalid strategy: {strategy}")

        if verbose:
            print(f"Selected Top Feature Indices: {top_feature_indices}")

        # Step 5: Map top features to data points
        point_to_images = defaultdict(set)
        for i, (birth_simplex, death_simplex) in enumerate(persistence_pairs):
            involved_indices = set(birth_simplex)
            for idx in involved_indices:
                point_to_images[i].add(sampled_indices[idx])

        queried_indices_set = set()
        for idx in top_feature_indices:
            images_involved = point_to_images.get(idx, set())
            queried_indices_set.update(images_involved)
            if len(queried_indices_set) >= budgetSize:
                break

        # Trim to budgetSize
        queried_indices = np.array(list(queried_indices_set))[:budgetSize]
        activeSet = queried_indices

        # Update remaining unlabeled set
        remainSet = np.setdiff1d(uSet, activeSet)

        if verbose:
            print(f"Selected {len(activeSet)} samples for the active set.")
            print(f"Remaining unlabeled set size: {len(remainSet)}")

        return activeSet, remainSet

    def rips_persistence_sampling_learned_multi(
    self,
    uSet,
    all_features,
    dataset,
    budgetSize,
    cur_episode,
    strategies,  # Pass a list/array of strategies
    randomSampleSize=1000,
    max_dim=2,
    verbose=True,
):    
        """
        Implements Rips Persistence Sampling for active learning.

        Parameters:
        -----------
        uSet : np.ndarray
            Indices of the unlabeled set.
        dataset : Dataset
            Data object containing the data points.
        budgetSize : int
            Number of samples to select.
        randomSampleSize : int, optional
            Number of random samples to process (default: 750).
        max_dim : int, optional
            Maximum homology dimension to compute (default: 2).
        strategy : str, optional
            Sampling strategy: "h0_top", "h0_bottom", "h1_top", or "h0_h1_half" (default: "h1_top").
        verbose : bool, optional
            Whether to print detailed logs (default: False).

        Returns:
        --------
        activeSet : np.ndarray
            Selected active set.
        remainSet : np.ndarray
            Remaining unlabeled set.
        """
        """
    Implements Rips Persistence Sampling for active learning.
    The strategy is determined by an array based on the current episode index.

    Parameters:
    -----------
    uSet : np.ndarray
        Indices of the unlabeled set.
    all_features : np.ndarray or None
        Features array or None for raw data fallback.
    dataset : Dataset
        Data object for fallback if all_features is None.
    budgetSize : int
        Number of samples to select.
    cur_episode : int
        Current episode index (1-based).
    strategies : list of str
        List of strategies for each episode (e.g., ["h0_top", "h0_bottom", "h0_middle_out"]).
    randomSampleSize : int, optional
        Number of random samples to process (default: 750).
    max_dim : int, optional
        Maximum homology dimension to compute (default: 2).
    verbose : bool, optional
        Whether to print logs (default: True).

    Returns:
    --------
    activeSet : np.ndarray
        Selected active set.
    remainSet : np.ndarray
        Remaining unlabeled set.
        """

    # Determine the strategy for the current episode
        if cur_episode < 0 or cur_episode >= len(strategies):
           raise ValueError(f"Invalid episode index: {cur_episode}. Ensure it is within [0, {len(strategies) - 1}].")

        strategy = strategies[cur_episode]  # Direct indexing, 0-based

        if verbose:
            print(f"Episode {cur_episode}: selected strategy = {strategy}")

    # Validate strategy (optional, to catch typos in the strategy array)
        valid_strategies = {
        "h0_top",
        "h0_bottom",
        "h1_top",
        "h0_middle_out",
        "h0_top_h0_bottom",
        "h1_top_h1_bottom",
        "h0_top_h1_top",
        "h0_top_h1_bottom",
        "h0_bottom_h1_top",
        "h0_bottom_h1_bottom",
    }
        if strategy not in valid_strategies:
            raise ValueError(f"Invalid strategy: {strategy}")

        assert budgetSize > 0, "Budget size must be greater than 0."
        assert randomSampleSize > 0, "Random sample size must be greater than 0."
        assert budgetSize <= len(uSet), "Budget size cannot exceed the size of the unlabeled set."

        print(f"sample size is : {randomSampleSize} and techinque is : {strategy}")

        # Step 1: Randomly sample subset from uSet
        seed = int(time.time())  # or fix a seed for reproducibility
        np.random.seed(seed)
        uSet = uSet.astype(int)     # ensure it's an integer array
        sample_size = min(randomSampleSize, len(uSet))
        sampled_indices = np.random.choice(uSet, size=sample_size, replace=False)

    # Step 2: Retrieve features for the sampled indices
        if all_features is not None:
            # If the entire dataset's features are already loaded, just slice them
            X_sampled = all_features[sampled_indices].astype(np.float32)
        else:
            # Fallback: gather raw data from dataset and flatten it.
            # (You'll only need this if you want to keep that fallback.)
            X_sampled = np.vstack([
            dataset[idx][0].numpy().reshape(-1) for idx in sampled_indices
            ]).astype(np.float32)

        print("shape of data is " ,X_sampled.shape)
        print("Computing persistence")
        # Step 2: Build Rips Complex and compute persistence
        rips_complex = RipsComplex(points=X_sampled)
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dim)
        simplex_tree.compute_persistence()
        persistence_pairs = simplex_tree.persistence_pairs()

        # Step 3: Extract persistence intervals
        h0_intervals = simplex_tree.persistence_intervals_in_dimension(0)
        h1_intervals = simplex_tree.persistence_intervals_in_dimension(1)
        h1_intervals = [pt for pt in h1_intervals if pt[1] != float('inf')]  # Exclude infinite intervals
        h0_intervals = [pt for pt in h0_intervals if pt[1] != float('inf')]  # Exclude infinite intervals

        # Compute persistence scores
        h1_persistence = [(i, pt[1] - pt[0]) for i, pt in enumerate(h1_intervals)]
        h0_persistence = [(i, pt[1] - pt[0]) for i, pt in enumerate(h0_intervals)]
        h1_persistence_sorted = sorted(h1_persistence, key=lambda x: x[1], reverse=True)
        h0_persistence_sorted = sorted(h0_persistence, key=lambda x: x[1], reverse=True)
        print("Done")

        
        print(f"Top H1 Persistence Scores: {h1_persistence_sorted[:10]}")
        print(f"Top H0 Persistence Scores: {h0_persistence_sorted[:10]}")

        # Step 4: Identify top features based on strategy
        if strategy == "h1_top":
            top_feature_indices = [i for i, _ in h1_persistence_sorted[:budgetSize]]
        elif strategy == "h0_top":
            top_feature_indices = [i for i, _ in h0_persistence_sorted[:budgetSize]]
        elif strategy == "h0_bottom":
            top_feature_indices = [i for i, _ in reversed(h0_persistence_sorted[-budgetSize:])]
        elif strategy == "h1_bottom":
            top_feature_indices = [i for i, _ in reversed(h1_persistence_sorted[-budgetSize:])]

        elif strategy == "h0_top_h0_bottom":
            half_budget = budgetSize // 2
            top_feature_indices = (
        [i for i, _ in h0_persistence_sorted[:half_budget]] +
        [i for i, _ in reversed(h0_persistence_sorted[-half_budget:])]
    )
        elif strategy == "h1_top_h1_bottom":
            half_budget = budgetSize // 2
            top_feature_indices = (
        [i for i, _ in h1_persistence_sorted[:half_budget]] +
        [i for i, _ in reversed(h1_persistence_sorted[-half_budget:])]
    )
        elif strategy == "h0_top_h1_top":
            half_budget = budgetSize // 2
            top_feature_indices = (
        [i for i, _ in h0_persistence_sorted[:half_budget]] +
        [i for i, _ in h1_persistence_sorted[:half_budget]]
    )
        elif strategy == "h0_top_h1_bottom":
            half_budget = budgetSize // 2
            top_feature_indices = (
        [i for i, _ in h0_persistence_sorted[:half_budget]] +
        [i for i, _ in reversed(h1_persistence_sorted[-half_budget:])]
    )
        elif strategy == "h0_bottom_h1_top":
            half_budget = budgetSize // 2
            top_feature_indices = (
        [i for i, _ in reversed(h0_persistence_sorted[-half_budget:])] +
        [i for i, _ in h1_persistence_sorted[:half_budget]]
    )
        elif strategy == "h0_bottom_h1_bottom":
            half_budget = budgetSize // 2
            top_feature_indices = (
        [i for i, _ in reversed(h0_persistence_sorted[-half_budget:])] +
        [i for i, _ in reversed(h1_persistence_sorted[-half_budget:])]
    )
        elif strategy == "h0_middle_out":
            """
        New strategy that starts picking from the midpoint of h0_persistence_sorted
        and expands outward to both sides.
            """
            n = len(h0_persistence_sorted)
            mid = n // 2   # midpoint index
            top_feature_indices = []
            left = mid
            right = mid + 1

        # Keep picking from the midpoint outward until we have enough
            while len(top_feature_indices) < budgetSize and (left >= 0 or right < n):
                # Pick from `left` (the middle going down) if in range
                if left >= 0:
                    top_feature_indices.append(h0_persistence_sorted[left][0])
                    left -= 1
                    if len(top_feature_indices) >= budgetSize:
                        break

            # Pick from `right` (the middle going up) if in range
                if right < n:
                    top_feature_indices.append(h0_persistence_sorted[right][0])
                    right += 1
                    if len(top_feature_indices) >= budgetSize:
                        break
        else:
            raise ValueError(f"Invalid strategy: {strategy}")

        if verbose:
            print(f"Selected Top Feature Indices: {top_feature_indices}")

        # Step 5: Map top features to data points
        point_to_images = defaultdict(set)
        for i, (birth_simplex, death_simplex) in enumerate(persistence_pairs):
            involved_indices = set(birth_simplex)
            for idx in involved_indices:
                point_to_images[i].add(sampled_indices[idx])

        queried_indices_set = set()
        for idx in top_feature_indices:
            images_involved = point_to_images.get(idx, set())
            queried_indices_set.update(images_involved)
            if len(queried_indices_set) >= budgetSize:
                break

        # Trim to budgetSize
        queried_indices = np.array(list(queried_indices_set))[:budgetSize]
        activeSet = queried_indices

        # Update remaining unlabeled set
        remainSet = np.setdiff1d(uSet, activeSet)

        if verbose:
            print(f"Selected {len(activeSet)} samples for the active set.")
            print(f"Remaining unlabeled set size: {len(remainSet)}")

        return activeSet, remainSet
    
    def rips_persistence_sampling_learned_multi2(
        self,
        uSet,
        all_features,
        dataset,
        budgetSize,
        cur_episode,
        randomSampleSize=750,
        max_dim=0,  # Only H0 needed
        verbose=True,
    ):
        """
        Implements Rips Persistence Sampling for active learning.
        Selection always focuses on H0 and expands symmetrically around the middle based on cur_episode.

        Parameters:
        -----------
        uSet : np.ndarray
            Indices of the unlabeled set.
        all_features : np.ndarray or None
            Features array or None for raw data fallback.
        dataset : Dataset
            Data object for fallback if all_features is None.
        budgetSize : int
            Number of samples to select.
        cur_episode : int
            Current episode index (1-based). Determines how far the range expands.
        randomSampleSize : int, optional
            Number of random samples to process (default: 750).
        max_dim : int, optional
            Maximum homology dimension to compute (default: 0).
        verbose : bool, optional
            Whether to print logs (default: True).

        Returns:
        --------
        activeSet : np.ndarray
            Selected active set.
        remainSet : np.ndarray
            Remaining unlabeled set.
        """

        # Validate inputs
        assert budgetSize > 0, "Budget size must be greater than 0."
        assert randomSampleSize > 0, "Random sample size must be greater than 0."
        assert budgetSize <= len(uSet), "Budget size cannot exceed the size of the unlabeled set."
        cur_episode = cur_episode+1
        if verbose:
            print(f"Episode {cur_episode}: starting selection from middle, expanding outward.")
            print(f"Sample size is: {randomSampleSize}")

        # Step 1: Randomly sample subset from uSet
        seed = int(time.time())
        np.random.seed(seed)
        uSet = uSet.astype(int)
        sample_size = min(randomSampleSize, len(uSet))
        sampled_indices = np.random.choice(uSet, size=sample_size, replace=False)

        # Step 2: Retrieve features
        if all_features is not None:
            X_sampled = all_features[sampled_indices].astype(np.float32)
        else:
            X_sampled = np.vstack([
                dataset[idx][0].numpy().reshape(-1) for idx in sampled_indices
            ]).astype(np.float32)

        if verbose:
            print("Shape of data:", X_sampled.shape)
            print("Computing persistence")

        # Step 3: Build Rips Complex and compute persistence
        rips_complex = RipsComplex(points=X_sampled)
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dim)
        simplex_tree.compute_persistence()

        # Extract H0 persistence intervals (exclude infinite intervals)
        h0_intervals = [
            pt for pt in simplex_tree.persistence_intervals_in_dimension(0)
            if pt[1] != float('inf')
        ]
        h0_persistence = [(i, pt[1] - pt[0]) for i, pt in enumerate(h0_intervals)]
        h0_persistence_sorted = sorted(h0_persistence, key=lambda x: x[1], reverse=True)

        if verbose:
            print(f"Top H0 Persistence Scores: {h0_persistence_sorted[:10]}")

        # Step 4: Determine the range for selection
        n = len(h0_persistence_sorted)
        mid = n // 2  # Middle index
        range_expand = 2 * cur_episode  # Range expansion factor: expands ±5 per episode

        # Calculate start and end indices for this episode
        start_idx = max(0, mid - range_expand)
        end_idx = min(n, mid + range_expand)

        if verbose:
            print(f"Episode {cur_episode}: Selecting indices from {start_idx} to {end_idx}")

        # Get indices from the selected range
        range_indices = list(range(start_idx, end_idx))
        top_feature_indices = [h0_persistence_sorted[i][0] for i in range_indices]

        # Ensure we select only up to budgetSize
        top_feature_indices = top_feature_indices[:budgetSize]

        if verbose:
            print(f"Selected Top Feature Indices: {top_feature_indices}")

        # Step 5: Map top features to data points
        point_to_images = defaultdict(set)
        for i, (birth_simplex, death_simplex) in enumerate(simplex_tree.persistence_pairs()):
            involved_indices = set(birth_simplex)
            for idx in involved_indices:
                point_to_images[i].add(sampled_indices[idx])

        queried_indices_set = set()
        for idx in top_feature_indices:
            images_involved = point_to_images.get(idx, set())
            queried_indices_set.update(images_involved)
            if len(queried_indices_set) >= budgetSize:
                break

        # Trim to budgetSize
        queried_indices = np.array(list(queried_indices_set))[:budgetSize]
        activeSet = queried_indices

        # Update remaining unlabeled set
        remainSet = np.setdiff1d(uSet, activeSet)

        if verbose:
            print(f"Selected {len(activeSet)} samples for the active set.")
            print(f"Remaining unlabeled set size: {len(remainSet)}")

        return activeSet, remainSet
        
    def ring_based_selection_h0(
        self,
        uSet, lSet,
        all_features,
        dataset,
        budgetSize,
        cur_episode,
        randomSampleSize=1000,
        max_dim=0,  # Only H0 needed
        verbose=True,
        ring_width_per_episode=5,
        half_per_episode=5,
    ):
        """
        Ring-based selection with randomness and robust episode handling.
        """
        # Ensure valid inputs
        assert budgetSize > 0 and budgetSize <= len(uSet)
        assert randomSampleSize > 0

        # Ensure cur_episode starts at 1
        if cur_episode <= 0:
            print(f"Warning: cur_episode = {cur_episode}. Adjusting to 1-based indexing.")
            cur_episode = 1

        if verbose:
            print(f"\n=== Episode {cur_episode} ===")
            print(f"Selecting {budgetSize} total => {half_per_episode} from left + {half_per_episode} from right")

        # 1. Random sampling from uSet
        seed = int(time.time())
        np.random.seed(seed)
        uSet = uSet.astype(int)
        sample_size = min(randomSampleSize, len(uSet))
        sampled_indices = np.random.choice(uSet, size=sample_size, replace=False)

        # 2. Retrieve features
        if all_features is not None:
            X_sampled = all_features[sampled_indices].astype(np.float32)
        else:
            X_sampled = np.vstack([
                dataset[idx][0].numpy().reshape(-1)
                for idx in sampled_indices
            ]).astype(np.float32)

        # 3. Compute persistence (H0 only)
        if verbose:
            print("Computing Rips Complex for H0 intervals only...")
        rips_complex = RipsComplex(points=all_features)
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dim)
        simplex_tree.compute_persistence()

        # 4. Extract and sort H0 intervals
        h0_intervals = [
            pt for pt in simplex_tree.persistence_intervals_in_dimension(0)
            if pt[1] != float('inf')
        ]
        h0_persistence = [(i, pt[1] - pt[0]) for i, pt in enumerate(h0_intervals)]
        h0_persistence_sorted = sorted(h0_persistence, key=lambda x: x[1], reverse=True)

        n = len(h0_persistence_sorted)
        mid = n // 2  # midpoint in sorted list

        # Calculate offset based on cur_episode
        offset = ring_width_per_episode * (cur_episode - 1)

        # Define left and right chunks
        left_start = max(mid - offset - half_per_episode, 0)
        left_end = max(mid - offset, 0)
        right_start = min(mid + offset, n)
        right_end = min(mid + offset + half_per_episode, n)

        left_indices = list(range(left_start, left_end))
        right_indices = list(range(right_start, right_end))

        # Randomly select indices from left and right chunks
        np.random.seed(seed)
        random_left = np.random.choice(left_indices, min(len(left_indices), half_per_episode), replace=False)
        random_right = np.random.choice(right_indices, min(len(right_indices), half_per_episode), replace=False)

        # Combine indices to form the ring
        ring_indices = list(random_left) + list(random_right)
        top_feature_indices = [h0_persistence_sorted[i][0] for i in ring_indices if 0 <= i < n]

        if verbose:
            print(f"Episode {cur_episode}: H0-sorted length={n}, mid={mid}")
            print(f"Selected indices: {top_feature_indices}")

        # 5. Map features to data points
        point_to_images = defaultdict(set)
        for i, (birth_simplex, _) in enumerate(simplex_tree.persistence_pairs()):
            involved_indices = set(birth_simplex)
            for idx in involved_indices:
                point_to_images[i].add(idx)

        # Collect data indices from selected intervals
        queried_indices_set = set()
        for idx in top_feature_indices:
            images_involved = point_to_images.get(idx, set())
            for image in images_involved:
                if image not in lSet:
                    queried_indices_set.add(image)
                    if len(queried_indices_set) >= budgetSize:
                        break
            if len(queried_indices_set) >= budgetSize:
                break

        # Trim to budgetSize
        queried_indices = np.array(list(queried_indices_set))[:budgetSize]
        activeSet = queried_indices

        # Compute remaining unlabeled set
        remainSet = np.setdiff1d(uSet, activeSet)

        if verbose:
            print(f"Selected {len(activeSet)} samples for the active set.")
            print(f"Remaining unlabeled set size: {len(remainSet)}")

        return activeSet, remainSet

    def typiclust_like_selection_h0(
        uSet, lSet, all_features, dataset, budgetSize, cur_episode, randomSampleSize=1000,
        max_dim=0, verbose=True, n_clusters=5, density_weight=0.7
    ):
        """
        TypiClust-inspired persistence-based active sampling for H0 features.
        """
              # Compute remaining unlabeled set
        cur_episode = cur_episode + 1 
        # Ensure valid inputs
        assert budgetSize > 0 and budgetSize <= len(uSet)
        assert randomSampleSize > 0


        # 3. Compute persistence (H0 only)
        if verbose:
            print("Computing Rips Complex for H0 intervals only...")
        rips_complex = RipsComplex(points=all_features)
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dim)
        simplex_tree.compute_persistence()
        
        # 4. Extract persistence intervals
        h0_intervals = [
            pt for pt in simplex_tree.persistence_intervals_in_dimension(0)
            if pt[1] != float('inf') and pt[0] >= 0 and pt[1] >= 0  # Exclude negatives
        ]
        if verbose:
            print(f"Total H0 intervals: {len(h0_intervals)}")

        # 5. Compute persistence lifetimes and density
        persistence_lifetimes = np.array([pt[1] - pt[0] for pt in h0_intervals])
        kde = gaussian_kde(persistence_lifetimes)
        densities = kde(persistence_lifetimes)

        # 6. Cluster intervals based on persistence values
        persistence_features = np.array([[pt[0], pt[1]] for pt in h0_intervals])
        kmeans = KMeans(n_clusters=n_clusters, random_state=1).fit(persistence_features)
        cluster_labels = kmeans.labels_

        # 7. Select intervals based on density and cluster representativeness
        selected_indices = []
        for cluster in range(n_clusters):
            cluster_indices = np.where(cluster_labels == cluster)[0]
            cluster_densities = densities[cluster_indices]
            cluster_lifetimes = persistence_lifetimes[cluster_indices]

            # Combine density and lifetime ranking
            scores = density_weight * cluster_densities + (1 - density_weight) * cluster_lifetimes
            ranked_indices = cluster_indices[np.argsort(-scores)]  # Descending order
            selected_indices.extend(ranked_indices[:max(1, len(cluster_indices) // 2)])  # Take top 50% per cluster

        # Limit selection to budgetSize
        selected_indices = selected_indices[:budgetSize]
        selected_h0_intervals = [h0_intervals[i] for i in selected_indices]

        if verbose:
            print(f"Selected {len(selected_h0_intervals)} persistence intervals.")

        # 8. Map intervals to data points
        point_to_images = defaultdict(set)
        for i, (birth_simplex, _) in enumerate(simplex_tree.persistence_pairs()):
            involved_indices = set(birth_simplex)
            for idx in involved_indices:
                point_to_images[i].add(idx)

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

        # Trim to budgetSize
        queried_indices = np.array(list(queried_indices_set))[:budgetSize]
        activeSet = queried_indices
        print(lSet, len(lSet))
        # Compute remaining unlabeled set
        remainSet = np.setdiff1d(uSet, activeSet)

        if verbose:
            print(f"Selected {len(activeSet)} samples for the active set.")
            print(f"Remaining unlabeled set size: {len(remainSet)}")

        return activeSet, remainSet

    def rips_persistence_sampling(self, uSet, dataset, budgetSize,cur_episode, randomSampleSize=750, max_dim=2, strategy="h1_top", verbose=True):
        """
        Implements Rips Persistence Sampling for active learning.

        Parameters:
        -----------
        uSet : np.ndarray
            Indices of the unlabeled set.
        dataset : Dataset
            Data object containing the data points.
        budgetSize : int
            Number of samples to select.
        randomSampleSize : int, optional
            Number of random samples to process (default: 750).
        max_dim : int, optional
            Maximum homology dimension to compute (default: 2).
        strategy : str, optional
            Sampling strategy: "h0_top", "h0_bottom", "h1_top", or "h0_h1_half" (default: "h1_top").
        verbose : bool, optional
            Whether to print detailed logs (default: False).

        Returns:
        --------
        activeSet : np.ndarray
            Selected active set.
        remainSet : np.ndarray
            Remaining unlabeled set.
        """

        assert budgetSize > 0, "Budget size must be greater than 0."
        assert randomSampleSize > 0, "Random sample size must be greater than 0."
        assert budgetSize <= len(uSet), "Budget size cannot exceed the size of the unlabeled set."
     
        # Step 1: Randomly sample subset from uSet
        seed = int(time.time())  # Get the current time as an integer
        np.random.seed(seed)
        sample_size = min(randomSampleSize, len(uSet))
        sampled_indices = np.random.choice(list(uSet), size=sample_size, replace=False)
        X_sampled = np.vstack([dataset[idx][0].numpy().reshape(-1) for idx in sampled_indices]).astype(np.float32)

        print(f"Sampled {sample_size} data points from unlabeled set. stratergy {strategy}")

        print("Computing persistence")
        # Step 2: Build Rips Complex and compute persistence
        rips_complex = RipsComplex(points=X_sampled)
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dim)
        simplex_tree.compute_persistence()
        persistence_pairs = simplex_tree.persistence_pairs()

        # Step 3: Extract persistence intervals
        h0_intervals = simplex_tree.persistence_intervals_in_dimension(0)
        h1_intervals = simplex_tree.persistence_intervals_in_dimension(1)
        h1_intervals = [pt for pt in h1_intervals if pt[1] != float('inf')]  # Exclude infinite intervals
        h0_intervals = [pt for pt in h0_intervals if pt[1] != float('inf')]  # Exclude infinite intervals

        # Compute persistence scores
        h1_persistence = [(i, pt[1] - pt[0]) for i, pt in enumerate(h1_intervals)]
        h0_persistence = [(i, pt[1] - pt[0]) for i, pt in enumerate(h0_intervals)]
        h1_persistence_sorted = sorted(h1_persistence, key=lambda x: x[1], reverse=True)
        h0_persistence_sorted = sorted(h0_persistence, key=lambda x: x[1], reverse=True)
        print("Done")

        
        print(f"Top H1 Persistence Scores: {h1_persistence_sorted[:10]}")
        print(f"Top H0 Persistence Scores: {h0_persistence_sorted[:10]}")

        # Step 4: Identify top features based on strategy
        if strategy == "h1_top":
            top_feature_indices = [i for i, _ in h1_persistence_sorted[:budgetSize]]
        elif strategy == "h0_top":
            top_feature_indices = [i for i, _ in h0_persistence_sorted[:budgetSize]]
        elif strategy == "h0_bottom":
            top_feature_indices = [i for i, _ in reversed(h0_persistence_sorted[-budgetSize:])]
        elif strategy == "h1_bottom":
            top_feature_indices = [i for i, _ in reversed(h1_persistence_sorted[-budgetSize:])]

        elif strategy == "h0_top_h0_bottom":
            half_budget = budgetSize // 2
            top_feature_indices = (
        [i for i, _ in h0_persistence_sorted[:half_budget]] +
        [i for i, _ in reversed(h0_persistence_sorted[-half_budget:])]
    )
        elif strategy == "h1_top_h1_bottom":
            half_budget = budgetSize // 2
            top_feature_indices = (
        [i for i, _ in h1_persistence_sorted[:half_budget]] +
        [i for i, _ in reversed(h1_persistence_sorted[-half_budget:])]
    )
        elif strategy == "h0_top_h1_top":
            half_budget = budgetSize // 2
            top_feature_indices = (
        [i for i, _ in h0_persistence_sorted[:half_budget]] +
        [i for i, _ in h1_persistence_sorted[:half_budget]]
    )
        elif strategy == "h0_top_h1_bottom":
            half_budget = budgetSize // 2
            top_feature_indices = (
        [i for i, _ in h0_persistence_sorted[:half_budget]] +
        [i for i, _ in reversed(h1_persistence_sorted[-half_budget:])]
    )
        elif strategy == "h0_bottom_h1_top":
            half_budget = budgetSize // 2
            top_feature_indices = (
        [i for i, _ in reversed(h0_persistence_sorted[-half_budget:])] +
        [i for i, _ in h1_persistence_sorted[:half_budget]]
    )
        elif strategy == "h0_bottom_h1_bottom":
            half_budget = budgetSize // 2
            top_feature_indices = (
        [i for i, _ in reversed(h0_persistence_sorted[-half_budget:])] +
        [i for i, _ in reversed(h1_persistence_sorted[-half_budget:])]
    )
        elif strategy == "h0_middle_out":
            """
        New strategy that starts picking from the midpoint of h0_persistence_sorted
        and expands outward to both sides.
            """
            n = len(h0_persistence_sorted)
            mid = n // 2   # midpoint index
            top_feature_indices = []
            left = mid
            right = mid + 1

        # Keep picking from the midpoint outward until we have enough
            while len(top_feature_indices) < budgetSize and (left >= 0 or right < n):
                # Pick from `left` (the middle going down) if in range
                if left >= 0:
                    top_feature_indices.append(h0_persistence_sorted[left][0])
                    left -= 1
                    if len(top_feature_indices) >= budgetSize:
                        break

            # Pick from `right` (the middle going up) if in range
                if right < n:
                    top_feature_indices.append(h0_persistence_sorted[right][0])
                    right += 1
                    if len(top_feature_indices) >= budgetSize:
                        break
        else:
            raise ValueError(f"Invalid strategy: {strategy}")

        if verbose:
            print(f"Selected Top Feature Indices: {top_feature_indices}")

        # Step 5: Map top features to data points
        point_to_images = defaultdict(set)
        for i, (birth_simplex, death_simplex) in enumerate(persistence_pairs):
            involved_indices = set(birth_simplex)
            for idx in involved_indices:
                point_to_images[i].add(sampled_indices[idx])

        queried_indices_set = set()
        for idx in top_feature_indices:
            images_involved = point_to_images.get(idx, set())
            queried_indices_set.update(images_involved)
            if len(queried_indices_set) >= budgetSize:
                break

        # Trim to budgetSize
        queried_indices = np.array(list(queried_indices_set))[:budgetSize]
        activeSet = queried_indices

        # Update remaining unlabeled set
        remainSet = np.setdiff1d(uSet, activeSet)

        if verbose:
            print(f"Selected {len(activeSet)} samples for the active set.")
            print(f"Remaining unlabeled set size: {len(remainSet)}")

        return activeSet, remainSet


    def __init__(self, dataObj, cfg):
        self.cfg = cfg
        self.cuda_id = 0 if cfg.ACTIVE_LEARNING.SAMPLING_FN.startswith("ensemble") else torch.cuda.current_device()
        self.dataObj = dataObj

    def gpu_compute_dists(self,M1,M2):
        """
        Computes L2 norm square on gpu
        Assume 
        M1: M x D matrix
        M2: N x D matrix

        output: M x N matrix
        """
        M1_norm = (M1**2).sum(1).reshape(-1,1)

        M2_t = torch.transpose(M2, 0, 1)
        M2_norm = (M2**2).sum(1).reshape(1,-1)
        dists = M1_norm + M2_norm - 2.0 * torch.mm(M1, M2_t)
        return dists

    def get_predictions(self, clf_model, idx_set, dataset):

        clf_model.cuda(self.cuda_id)
        #Used by bald acquisition
        # if self.cfg.TRAIN.DATASET == "IMAGENET":
        #     tempIdxSetLoader = imagenet_loader.construct_loader_no_aug(cfg=self.cfg, indices=idx_set, isDistributed=False, isShuffle=False, isVaalSampling=False)
        # else:
        tempIdxSetLoader = self.dataObj.getSequentialDataLoader(indexes=idx_set, batch_size=int(self.cfg.TRAIN.BATCH_SIZE/self.cfg.NUM_GPUS),data=dataset)
        tempIdxSetLoader.dataset.no_aug = True
        preds = []
        for i, (x, _) in enumerate(tqdm(tempIdxSetLoader, desc="Collecting predictions in get_predictions function")):
            with torch.no_grad():
                x = x.cuda(self.cuda_id)
                x = x.type(torch.cuda.FloatTensor)

                temp_pred = clf_model(x)

                #To get probabilities
                temp_pred = torch.nn.functional.softmax(temp_pred,dim=1)
                preds.append(temp_pred.cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        tempIdxSetLoader.dataset.no_aug = False
        return preds


    def random(self, uSet, budgetSize):
        """
        Chooses <budgetSize> number of data points randomly from uSet.
        
        NOTE: The returned uSet is modified such that it does not contain active datapoints.

        INPUT
        ------

        uSet: np.ndarray, It describes the index set of unlabelled set.

        budgetSize: int, The number of active data points to be chosen for active learning.

        OUTPUT
        -------

        Returns activeSet, uSet   
        """

        np.random.seed(self.cfg.RNG_SEED)

        assert isinstance(uSet, np.ndarray), "Expected uSet of type np.ndarray whereas provided is dtype:{}".format(type(uSet))
        assert isinstance(budgetSize,int), "Expected budgetSize of type int whereas provided is dtype:{}".format(type(budgetSize))
        assert budgetSize > 0, "Expected a positive budgetSize"
        assert budgetSize < len(uSet), "BudgetSet cannot exceed length of unlabelled set. Length of unlabelled set: {} and budgetSize: {}"\
            .format(len(uSet), budgetSize)

        tempIdx = [i for i in range(len(uSet))]
        np.random.shuffle(tempIdx)
        activeSet = uSet[tempIdx[0:budgetSize]]
        uSet = uSet[tempIdx[budgetSize:]]
        return activeSet, uSet


    def bald(self, budgetSize, uSet, clf_model, dataset):
        "Implements BALD acquisition function where we maximize information gain."

        clf_model.cuda(self.cuda_id)

        assert self.cfg.ACTIVE_LEARNING.DROPOUT_ITERATIONS != 0, "Expected dropout iterations > 0."

        #Set Batchnorm in eval mode whereas dropout in train mode
        clf_model.train()
        for m in clf_model.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.eval()

        # if self.cfg.TRAIN.DATASET == "IMAGENET":
        #     uSetLoader = imagenet_loader.construct_loader_no_aug(cfg=self.cfg, indices=uSet, isDistributed=False, isShuffle=False, isVaalSampling=False)
        # else:
        uSetLoader = self.dataObj.getSequentialDataLoader(indexes=uSet, batch_size=int(self.cfg.TRAIN.BATCH_SIZE/self.cfg.NUM_GPUS),data=dataset)
        uSetLoader.dataset.no_aug = True
        n_uPts = len(uSet)
        # Source Code was in tensorflow
        # To provide same readability we use same variable names where ever possible
        # Original TF-Code: https://github.com/Riashat/Deep-Bayesian-Active-Learning/blob/master/MC_Dropout_Keras/Dropout_Bald_Q10_N1000_Paper.py#L223

        # Heuristic: G_X - F_X
        score_All = np.zeros(shape=(n_uPts, self.cfg.MODEL.NUM_CLASSES))
        all_entropy_dropout = np.zeros(shape=(n_uPts))

        for d in tqdm(range(self.cfg.ACTIVE_LEARNING.DROPOUT_ITERATIONS), desc="Dropout Iterations"):
            dropout_score = self.get_predictions(clf_model=clf_model, idx_set=uSet, dataset=dataset)
            
            score_All += dropout_score

            #computing F_x
            dropout_score_log = np.log2(dropout_score+1e-6)#Add 1e-6 to avoid log(0)
            Entropy_Compute = -np.multiply(dropout_score, dropout_score_log)
            Entropy_per_Dropout = np.sum(Entropy_Compute, axis=1)

            all_entropy_dropout += Entropy_per_Dropout

        Avg_Pi = np.divide(score_All, self.cfg.ACTIVE_LEARNING.DROPOUT_ITERATIONS)
        Log_Avg_Pi = np.log2(Avg_Pi+1e-6)
        Entropy_Avg_Pi = -np.multiply(Avg_Pi, Log_Avg_Pi)
        Entropy_Average_Pi = np.sum(Entropy_Avg_Pi, axis=1)
        G_X = Entropy_Average_Pi
        Average_Entropy = np.divide(all_entropy_dropout, self.cfg.ACTIVE_LEARNING.DROPOUT_ITERATIONS)
        F_X = Average_Entropy

        U_X = G_X - F_X
        print("U_X.shape: ",U_X.shape)
        sorted_idx = np.argsort(U_X)[::-1] # argsort helps to return the indices of u_scores such that their corresponding values are sorted.
        activeSet = sorted_idx[:budgetSize]

        activeSet = uSet[activeSet]
        remainSet = uSet[sorted_idx[budgetSize:]]
        # Setting task model in train mode for further learning
        clf_model.train()
        uSetLoader.dataset.no_aug = False
        return activeSet, remainSet


    def dbal(self, budgetSize, uSet, clf_model, dataset):
        """
        Implements deep bayesian active learning where uncertainty is measured by 
        maximizing entropy of predictions. This uncertainty method is choosen following
        the recent state of the art approach, VAAL. [SOURCE: Implementation Details in VAAL paper]
        
        In bayesian view, predictions are computed with the help of dropouts and 
        Monte Carlo approximation 
        """
        clf_model.cuda(self.cuda_id)

        # Set Batchnorm in eval mode whereas dropout in train mode
        clf_model.train()
        for m in clf_model.modules():
            #print("True")
            if isinstance(m, torch.nn.BatchNorm2d):
                m.eval()

        assert self.cfg.ACTIVE_LEARNING.DROPOUT_ITERATIONS != 0, "Expected dropout iterations > 0."

        # if self.cfg.TRAIN.DATASET == "IMAGENET":
        #     uSetLoader = imagenet_loader.construct_loader_no_aug(cfg=self.cfg, indices=uSet, isDistributed=False, isShuffle=False, isVaalSampling=False)
        # else:
        uSetLoader = self.dataObj.getSequentialDataLoader(indexes=uSet, batch_size=int(self.cfg.TRAIN.BATCH_SIZE/self.cfg.NUM_GPUS),data=dataset)
        uSetLoader.dataset.no_aug = True
        u_scores = []
        n_uPts = len(uSet)
        ptsProcessed = 0

        entropy_loss = EntropyLoss()

        print("len usetLoader: {}".format(len(uSetLoader)))
        temp_i=0
        
        for k,(x_u,_) in enumerate(tqdm(uSetLoader, desc="uSet Feed Forward")):
            temp_i += 1
            x_u = x_u.type(torch.cuda.FloatTensor)
            z_op = np.zeros((x_u.shape[0], self.cfg.MODEL.NUM_CLASSES), dtype=float)
            for i in range(self.cfg.ACTIVE_LEARNING.DROPOUT_ITERATIONS):
                with torch.no_grad():
                    x_u = x_u.cuda(self.cuda_id)
                    temp_op = clf_model(x_u)
                    # Till here z_op represents logits of p(y|x).
                    # So to get probabilities
                    temp_op = torch.nn.functional.softmax(temp_op,dim=1)
                    z_op = np.add(z_op, temp_op.cpu().numpy())

            z_op /= self.cfg.ACTIVE_LEARNING.DROPOUT_ITERATIONS
            
            z_op = torch.from_numpy(z_op).cuda(self.cuda_id)
            entropy_z_op = entropy_loss(z_op, applySoftMax=False)
            
            # Now entropy_z_op = Sum over all classes{ -p(y=c|x) log p(y=c|x)}
            u_scores.append(entropy_z_op.cpu().numpy())
            ptsProcessed += x_u.shape[0]
            
        u_scores = np.concatenate(u_scores, axis=0)
        sorted_idx = np.argsort(u_scores)[::-1] # argsort helps to return the indices of u_scores such that their corresponding values are sorted.
        activeSet = sorted_idx[:budgetSize]

        activeSet = uSet[activeSet]
        remainSet = uSet[sorted_idx[budgetSize:]]
        uSetLoader.dataset.no_aug = False
        return activeSet, remainSet


    def ensemble_var_R(self, budgetSize, uSet, clf_models, dataset):
        """
        Implements ensemble variance_ratio measured as the number of disagreement in committee 
        with respect to the predicted class. 
        If f_m is number of members agreeing to predicted class then 
        variance ratio(var_r) is evaludated as follows:
        
            var_r = 1 - (f_m / T); where T is number of commitee members

        For more details refer equation 4 in 
        http://openaccess.thecvf.com/content_cvpr_2018/papers/Beluch_The_Power_of_CVPR_2018_paper.pdf
        """
        from scipy import stats
        T = len(clf_models)

        for cmodel in clf_models:
            cmodel.cuda(self.cuda_id)
            cmodel.eval()

        # if self.cfg.TRAIN.DATASET == "IMAGENET":
        #     uSetLoader = imagenet_loader.construct_loader_no_aug(cfg=self.cfg, indices=uSet, isDistributed=False, isShuffle=False, isVaalSampling=False)
        # else:
        uSetLoader = self.dataObj.getSequentialDataLoader(indexes=uSet, batch_size=int(self.cfg.TRAIN.BATCH_SIZE/self.cfg.NUM_GPUS),data=dataset)
        uSetLoader.dataset.no_aug = True
        print("len usetLoader: {}".format(len(uSetLoader)))

        temp_i=0
        var_r_scores = np.zeros((len(uSet),1), dtype=float)
        
        for k, (x_u, _) in enumerate(tqdm(uSetLoader, desc="uSet Forward Passes through "+str(T)+" models")):
            x_u = x_u.type(torch.cuda.FloatTensor)
            ens_preds = np.zeros((x_u.shape[0], T), dtype=float)
            for i in range(len(clf_models)):
               with torch.no_grad():
                    x_u = x_u.cuda(self.cuda_id)
                    temp_op = clf_models[i](x_u)
                    _, temp_pred = torch.max(temp_op, 1)
                    temp_pred = temp_pred.cpu().numpy()
                    ens_preds[:,i] = temp_pred
            _, mode_cnt = stats.mode(ens_preds, 1)
            temp_varr = 1.0 - (mode_cnt / T*1.0)
            var_r_scores[temp_i : temp_i+x_u.shape[0]] = temp_varr

            temp_i = temp_i + x_u.shape[0]

        var_r_scores = np.squeeze(np.array(var_r_scores))
        print("var_r_scores: ")
        print(var_r_scores.shape)

        sorted_idx = np.argsort(var_r_scores)[::-1] #argsort helps to return the indices of u_scores such that their corresponding values are sorted.
        activeSet = sorted_idx[:budgetSize]

        activeSet = uSet[activeSet]
        remainSet = uSet[sorted_idx[budgetSize:]]
        uSetLoader.dataset.no_aug = False
        return activeSet, remainSet

    def uncertainty(self, budgetSize, lSet, uSet, model, dataset):

        """
        Implements the uncertainty principle as a acquisition function.
        """
        num_classes = self.cfg.MODEL.NUM_CLASSES
        assert model.training == False, "Model expected in eval mode whereas currently it is in {}".format(model.training)
        
        clf = model.cuda()
        
        u_ranks = []
        # if self.cfg.TRAIN.DATASET == "IMAGENET":
        #     print("Loading the model in data parallel where num_GPUS: {}".format(self.cfg.NUM_GPUS))
        #     clf = torch.nn.DataParallel(clf, device_ids = [i for i in range(self.cfg.NUM_GPUS)])
        #     uSetLoader = imagenet_loader.construct_loader_no_aug(cfg=self.cfg, indices=uSet, isDistributed=False, isShuffle=False, isVaalSampling=False)
        # else:
        uSetLoader = self.dataObj.getSequentialDataLoader(indexes=uSet, batch_size=int(self.cfg.TRAIN.BATCH_SIZE),data=dataset)
        uSetLoader.dataset.no_aug = True

        n_uLoader = len(uSetLoader)
        print("len(uSetLoader): {}".format(n_uLoader))
        for i, (x_u, _) in enumerate(tqdm(uSetLoader, desc="uSet Activations")):
            with torch.no_grad():
                x_u = x_u.cuda(0)

                temp_u_rank = torch.nn.functional.softmax(clf(x_u), dim=1)
                temp_u_rank, _ = torch.max(temp_u_rank, dim=1)
                temp_u_rank = 1 - temp_u_rank
                u_ranks.append(temp_u_rank.detach().cpu().numpy())

        u_ranks = np.concatenate(u_ranks, axis=0)
        # Now u_ranks has shape: [U_Size x 1]

        # index of u_ranks serve as key to refer in u_idx
        print(f"u_ranks.shape: {u_ranks.shape}")
        # we add -1 for reversing the sorted array
        sorted_idx = np.argsort(u_ranks)[::-1] # argsort helps to return the indices of u_ranks such that their corresponding values are sorted.
        activeSet = sorted_idx[:budgetSize]

        activeSet = uSet[activeSet]
        remainSet = uSet[sorted_idx[budgetSize:]]
        uSetLoader.dataset.no_aug = False
        return activeSet, remainSet


    def entropy(self, budgetSize, lSet, uSet, model, dataset):

        """
        Implements the uncertainty principle as a acquisition function.
        """
        num_classes = self.cfg.MODEL.NUM_CLASSES
        assert model.training == False, "Model expected in eval mode whereas currently it is in {}".format(model.training)

        clf = model.cuda()

        u_ranks = []
        # if self.cfg.TRAIN.DATASET == "IMAGENET":
        #     print("Loading the model in data parallel where num_GPUS: {}".format(self.cfg.NUM_GPUS))
        #     clf = torch.nn.DataParallel(clf, device_ids = [i for i in range(self.cfg.NUM_GPUS)])
        #     uSetLoader = imagenet_loader.construct_loader_no_aug(cfg=self.cfg, indices=uSet, isDistributed=False, isShuffle=False, isVaalSampling=False)
        # else:
        uSetLoader = self.dataObj.getSequentialDataLoader(indexes=uSet, batch_size=int(self.cfg.TRAIN.BATCH_SIZE), data=dataset)
        uSetLoader.dataset.no_aug = True

        n_uLoader = len(uSetLoader)
        print("len(uSetLoader): {}".format(n_uLoader))
        for i, (x_u, _) in enumerate(tqdm(uSetLoader, desc="uSet Activations")):
            with torch.no_grad():
                x_u = x_u.cuda(0)

                temp_u_rank = torch.nn.functional.softmax(clf(x_u), dim=1)
                temp_u_rank = temp_u_rank * torch.log2(temp_u_rank)
                temp_u_rank = -1*torch.sum(temp_u_rank, dim=1)
                u_ranks.append(temp_u_rank.detach().cpu().numpy())
        u_ranks = np.concatenate(u_ranks, axis=0)
        # Now u_ranks has shape: [U_Size x 1]

        # index of u_ranks serve as key to refer in u_idx
        print(f"u_ranks.shape: {u_ranks.shape}")
        # we add -1 for reversing the sorted array
        sorted_idx = np.argsort(u_ranks)[::-1] # argsort helps to return the indices of u_ranks such that their corresponding values are sorted.
        activeSet = sorted_idx[:budgetSize]

        activeSet = uSet[activeSet]
        remainSet = uSet[sorted_idx[budgetSize:]]
        uSetLoader.dataset.no_aug = False
        return activeSet, remainSet


    def margin(self, budgetSize, lSet, uSet, model, dataset):

        """
        Implements the uncertainty principle as a acquisition function.
        """
        num_classes = self.cfg.MODEL.NUM_CLASSES
        assert model.training == False, "Model expected in eval mode whereas currently it is in {}".format(model.training)

        clf = model.cuda()

        u_ranks = []
        # if self.cfg.TRAIN.DATASET == "IMAGENET":
        #     print("Loading the model in data parallel where num_GPUS: {}".format(self.cfg.NUM_GPUS))
        #     clf = torch.nn.DataParallel(clf, device_ids = [i for i in range(self.cfg.NUM_GPUS)])
        #     uSetLoader = imagenet_loader.construct_loader_no_aug(cfg=self.cfg, indices=uSet, isDistributed=False, isShuffle=False, isVaalSampling=False)
        # else:
        uSetLoader = self.dataObj.getSequentialDataLoader(indexes=uSet, batch_size=int(self.cfg.TRAIN.BATCH_SIZE), data=dataset)
        uSetLoader.dataset.no_aug = True

        n_uLoader = len(uSetLoader)
        print("len(uSetLoader): {}".format(n_uLoader))
        for i, (x_u, _) in enumerate(tqdm(uSetLoader, desc="uSet Activations")):
            with torch.no_grad():
                x_u = x_u.cuda(0)

                temp_u_rank = torch.nn.functional.softmax(clf(x_u), dim=1)
                temp_u_rank, _ = torch.sort(temp_u_rank, descending=True)
                difference = temp_u_rank[:, 0] - temp_u_rank[:, 1]
                # for code consistency across uncertainty, entropy methods i.e., picking datapoints with max value  
                difference = -1*difference 
                u_ranks.append(difference.detach().cpu().numpy())
        u_ranks = np.concatenate(u_ranks, axis=0)
        # Now u_ranks has shape: [U_Size x 1]

        # index of u_ranks serve as key to refer in u_idx
        print(f"u_ranks.shape: {u_ranks.shape}")
        # we add -1 for reversing the sorted array
        sorted_idx = np.argsort(u_ranks)[::-1] #argsort helps to return the indices of u_ranks such that their corresponding values are sorted.
        activeSet = sorted_idx[:budgetSize]

        activeSet = uSet[activeSet]
        remainSet = uSet[sorted_idx[budgetSize:]]
        uSetLoader.dataset.no_aug = False
        return activeSet, remainSet


class AdversarySampler:


    def __init__(self, dataObj, cfg):
        self.cfg = cfg
        self.dataObj = dataObj
        self.budget = cfg.ACTIVE_LEARNING.BUDGET_SIZE
        self.cuda_id = torch.cuda.current_device()
        if cfg.DATASET.NAME == 'TINYIMAGENET':
            cfg.VAAL.Z_DIM = 64
            cfg.VAAL.IM_SIZE = 64
        else:
            cfg.VAAL.Z_DIM = 32
            cfg.VAAL.IM_SIZE = 32


    def compute_dists(self, X, X_train):
        dists = -2 * np.dot(X, X_train.T) + np.sum(X_train**2,axis=1) + np.sum(X**2, axis=1)[:, np.newaxis]
        return dists

    def vaal_perform_training(self, lSet, uSet, dataset, debug=False):
        oldmode = self.dataObj.eval_mode
        self.dataObj.eval_mode = True
        self.dataObj.eval_mode = oldmode

        # First train vae and disc
        vae, disc = train_vae_disc(self.cfg, lSet, uSet, dataset, self.dataObj, debug)
        uSetLoader = self.dataObj.getSequentialDataLoader(indexes=uSet, batch_size=int(self.cfg.TRAIN.BATCH_SIZE / self.cfg.NUM_GPUS) \
            ,data=dataset)

        # Do active sampling
        vae.eval()
        disc.eval()

        return vae, disc, uSetLoader

    def greedy_k_center(self, labeled, unlabeled):
        greedy_indices = []
    
        # get the minimum distances between the labeled and unlabeled examples (iteratively, to avoid memory issues):
        min_dist = np.min(self.compute_dists(labeled[0, :].reshape((1, labeled.shape[1])), unlabeled), axis=0)
        min_dist = min_dist.reshape((1, min_dist.shape[0]))
        temp_range = 1000
        for j in range(1, labeled.shape[0], temp_range):
            if j + temp_range < labeled.shape[0]:
                dist = self.compute_dists(labeled[j:j+temp_range, :], unlabeled)
            else:
                # for last iteration only :)
                dist = self.compute_dists(labeled[j:, :], unlabeled)
            min_dist = np.vstack((min_dist, np.min(dist, axis=0).reshape((1, min_dist.shape[1]))))
            min_dist = np.min(min_dist, axis=0)
            min_dist = min_dist.reshape((1, min_dist.shape[0]))

        # iteratively insert the farthest index and recalculate the minimum distances:
        farthest = np.argmax(min_dist)
        greedy_indices.append(farthest)

        amount = cfg.ACTIVE_LEARNING.BUDGET_SIZE-1
        for i in range(amount):
            if i!=0 and i%500 == 0:
                print("{} Sampled out of {}".format(i, amount+1))
            dist = self.compute_dists(unlabeled[greedy_indices[-1], :].reshape((1,unlabeled.shape[1])), unlabeled)
            min_dist = np.vstack((min_dist, dist.reshape((1, min_dist.shape[1]))))
            min_dist = np.min(min_dist, axis=0)
            min_dist = min_dist.reshape((1, min_dist.shape[0]))
            farthest = np.argmax(min_dist)
            greedy_indices.append(farthest)

        remainSet = set(np.arange(unlabeled.shape[0])) - set(greedy_indices)
        remainSet = np.array(list(remainSet))
        return greedy_indices, remainSet


    def get_vae_activations(self, vae, dataLoader):
        acts = []
        vae.eval()
        
        temp_max_iter = len(dataLoader)
        print("len(dataloader): {}".format(temp_max_iter))
        temp_iter = 0
        for x,y in dataLoader:
            x = x.type(torch.cuda.FloatTensor)
            x = x.cuda(self.cuda_id)
            _, _, mu, _ = vae(x)
            acts.append(mu.cpu().numpy())
            if temp_iter%100 == 0:
                print(f"Iteration [{temp_iter}/{temp_max_iter}] Done!!")

            temp_iter += 1
        
        acts = np.concatenate(acts, axis=0)
        return acts


    def get_predictions(self, vae, discriminator, data, cuda):
        all_preds = []
        all_indices = []

        assert vae.training == False,"Expected vae model to be in eval mode"
        assert discriminator.training == False, "Expected discriminator model to be in eval mode"

        temp_idx = 0
        for images,_ in data:
            if cuda:
                images = images.cuda()

            with torch.no_grad():
                _, _, mu, _ = vae(images)
                preds = discriminator(mu)

            preds = preds.cpu().data
            all_preds.extend(preds)
            temp_idx += images.shape[0]
        
        all_indices = np.arange(temp_idx)
        all_preds = torch.stack(all_preds)
        all_preds = all_preds.view(-1)
        all_preds = all_preds.cpu().numpy()
        return all_preds


    def gpu_compute_dists(self, M1, M2):
        """
        Computes L2 norm square on gpu
        Assume 
        M1: M x D matrix
        M2: N x D matrix

        output: M x N matrix
        """
        M1_norm = (M1**2).sum(1).reshape(-1,1)

        M2_t = torch.transpose(M2, 0, 1)
        M2_norm = (M2**2).sum(1).reshape(1,-1)
        dists = M1_norm + M2_norm - 2.0 * torch.mm(M1, M2_t)
        return dists


    def efficient_compute_dists(self, labeled, unlabeled):
        """
        """
        N_L = labeled.shape[0]
        N_U = unlabeled.shape[0]
        dist_matrix = None

        temp_range = 1000

        unlabeled = torch.from_numpy(unlabeled).cuda(self.cuda_id)
        temp_dist_matrix = np.empty((N_U, temp_range))
        for i in tqdm(range(0, N_L, temp_range), desc="Computing Distance Matrix"):
            end_index = i+temp_range if i+temp_range < N_L else N_L
            temp_labeled = labeled[i:end_index, :]
            temp_labeled = torch.from_numpy(temp_labeled).cuda(self.cuda_id)
            temp_dist_matrix = self.gpu_compute_dists(unlabeled, temp_labeled)
            temp_dist_matrix = torch.min(temp_dist_matrix, dim=1)[0]
            temp_dist_matrix = torch.reshape(temp_dist_matrix,(temp_dist_matrix.shape[0],1))
            if dist_matrix is None:
                dist_matrix = temp_dist_matrix
            else:
                dist_matrix = torch.cat((dist_matrix, temp_dist_matrix), dim=1)
                dist_matrix = torch.min(dist_matrix, dim=1)[0]
                dist_matrix = torch.reshape(dist_matrix,(dist_matrix.shape[0],1))
        
        return dist_matrix.cpu().numpy()


    @torch.no_grad()
    def vae_sample_for_labeling(self, vae, uSet, lSet, unlabeled_dataloader, lSetLoader):
        
        vae.eval()
        print("Computing activattions for uset....")
        u_scores = self.get_vae_activations(vae, unlabeled_dataloader)
        print("Computing activattions for lset....")
        l_scores = self.get_vae_activations(vae, lSetLoader)
        
        print("l_scores.shape: ",l_scores.shape)
        print("u_scores.shape: ",u_scores.shape)
        
        dist_matrix = self.efficient_compute_dists(l_scores, u_scores)
        print("Dist_matrix.shape: ",dist_matrix.shape)

        min_scores = np.min(dist_matrix, axis=1)
        sorted_idx = np.argsort(min_scores)[::-1]

        activeSet = uSet[sorted_idx[0:self.budget]]
        remainSet = uSet[sorted_idx[self.budget:]]

        return activeSet, remainSet


    def sample_vaal_plus(self, vae, disc_task, data, cuda):
        all_preds = []
        all_indices = []

        assert vae.training == False,"Expected vae model to be in eval mode"
        assert disc_task.training == False, "Expected disc_task model to be in eval mode"

        temp_idx = 0
        for images,_ in data:
            if cuda:
                images = images.cuda()

            with torch.no_grad():
                _, _, mu, _ = vae(images)
                preds,_ = disc_task(mu)

            preds = preds.cpu().data
            all_preds.extend(preds)
            temp_idx += images.shape[0]
        
        all_indices = np.arange(temp_idx)
        all_preds = torch.stack(all_preds)
        all_preds = all_preds.view(-1)
        # need to multiply by -1 to be able to use torch.topk 
        all_preds *= -1

        # select the points which the discriminator things are the most likely to be unlabeled
        _, querry_indices = torch.topk(all_preds, int(self.budget))
        querry_indices = querry_indices.numpy()
        remain_indices = np.asarray(list(set(all_indices) - set(querry_indices)))
        assert len(remain_indices) + len(querry_indices) == len(all_indices)," Indices are overlapped between activeSet and uSet"
        activeSet = all_indices[querry_indices]
        uSet = all_indices[remain_indices]
        return activeSet,uSet


    def sample(self, vae, discriminator, data):
        all_preds = []
        all_indices = []

        assert vae.training == False,"Expected vae model to be in eval mode"
        assert discriminator.training == False, "Expected discriminator model to be in eval mode"

        vae.cuda(self.cuda_id)
        discriminator.cuda(self.cuda_id)

        temp_idx = 0
        for images,_ in data:
            images = images.type(torch.cuda.FloatTensor)
            images = images.cuda()

            with torch.no_grad():
                _, _, mu, _ = vae(images)
                preds = discriminator(mu)

            preds = preds.cpu().data
            all_preds.extend(preds)
            temp_idx += images.shape[0]
        
        all_indices = np.arange(temp_idx)
        all_preds = torch.stack(all_preds)
        all_preds = all_preds.view(-1)
        # need to multiply by -1 to be able to use torch.topk 
        all_preds *= -1

        # select the points which the discriminator things are the most likely to be unlabeled
        _, querry_indices = torch.topk(all_preds, int(self.budget))
        querry_indices = querry_indices.numpy()
        remain_indices = np.asarray(list(set(all_indices) - set(querry_indices)))
        assert len(remain_indices) + len(querry_indices) == len(all_indices), " Indices are overlapped between activeSet and uSet"
        activeSet = all_indices[querry_indices]
        uSet = all_indices[remain_indices]
        return activeSet, uSet


    @torch.no_grad()
    def sample_for_labeling(self, vae, discriminator, unlabeled_dataloader, uSet):
        """
        Picks samples from uSet to form activeSet.

        INPUT
        ------
        vae: object of model VAE

        discriminator: object of model discriminator

        unlabeled_dataloader: Sequential dataloader iterating over uSet

        uSet: Collection of unlabelled datapoints

        NOTE: Please pass the unlabelled dataloader as sequential dataloader else the
        results won't be appropriate.

        OUTPUT
        -------

        Returns activeSet, [remaining]uSet
        """
        unlabeled_dataloader.dataset.no_aug = True
        activeSet, remainSet = self.sample(vae, 
                                             discriminator, 
                                             unlabeled_dataloader, 
                                             )

        activeSet = uSet[activeSet]
        remainSet = uSet[remainSet]
        unlabeled_dataloader.dataset.no_aug = False
        return activeSet, remainSet


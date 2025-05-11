# This file is slightly modified from a code implementation by Prateek Munjal et al., authors of the paper https://arxiv.org/abs/2002.09564
# GitHub: https://github.com/PrateekMunjal
# ----------------------------------------------------------

from .Sampling import Sampling, CoreSetMIPSampling, AdversarySampler
import pycls.utils.logging as lu
import pycls.datasets.utils as ds_utils
from pycls.al.abc import typiclust_like_selection_h0_no_features,most_repeated_h0_selection_NF,most_repeated_h0_selection,persistence_frequency_sampling,persistence_frequency_sampling_flag,most_repeated_h0_selection_Vetrex, least_repeated_h0_selection_Vetrex,least_persistence_h0_birth_selection_from_csv,h0_persistence_selection_from_csv
import pandas as pd
logger = lu.get_logger(__name__)

class ActiveLearning:
    """
    Implements standard active learning methods.
    """

    def __init__(self, dataObj, cfg):
        self.dataObj = dataObj
        self.sampler = Sampling(dataObj=dataObj,cfg=cfg)
        self.cfg = cfg
       
    def sample_from_uSet(self, clf_model, lSet, uSet, trainDataset,cur_episode, supportingModels=None):
        """
        Sample from uSet using cfg.ACTIVE_LEARNING.SAMPLING_FN.

        INPUT
        ------
        clf_model: Reference of task classifier model class [Typically VGG]

        supportingModels: List of models which are used for sampling process.

        OUTPUT
        -------
        Returns activeSet, uSet
        """
        assert self.cfg.ACTIVE_LEARNING.BUDGET_SIZE > 0, "Expected a positive budgetSize"
        assert self.cfg.ACTIVE_LEARNING.BUDGET_SIZE < len(uSet), "BudgetSet cannot exceed length of unlabelled set. Length of unlabelled set: {} and budgetSize: {}"\
        .format(len(uSet), self.cfg.ACTIVE_LEARNING.BUDGET_SIZE)

        if self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "random":

            activeSet, uSet = self.sampler.random(uSet=uSet, budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE)
        
        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "uncertainty":
            oldmode = clf_model.training
            clf_model.eval()
            activeSet, uSet = self.sampler.uncertainty(budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,lSet=lSet,uSet=uSet \
                ,model=clf_model,dataset=trainDataset)
            clf_model.train(oldmode)
        
        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "entropy":
            oldmode = clf_model.training
            clf_model.eval()
            activeSet, uSet = self.sampler.entropy(budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,lSet=lSet,uSet=uSet \
                ,model=clf_model,dataset=trainDataset)
            clf_model.train(oldmode)
        
        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "margin":
            oldmode = clf_model.training
            clf_model.eval()
            activeSet, uSet = self.sampler.margin(budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,lSet=lSet,uSet=uSet \
                ,model=clf_model,dataset=trainDataset)
            clf_model.train(oldmode)

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "coreset":
            waslatent = clf_model.penultimate_active
            wastrain = clf_model.training
            clf_model.penultimate_active = True
            # if self.cfg.TRAIN.DATASET == "IMAGENET":
            #     clf_model.cuda(0)
            clf_model.eval()
            coreSetSampler = CoreSetMIPSampling(cfg=self.cfg, dataObj=self.dataObj)
            activeSet, uSet = coreSetSampler.query(lSet=lSet, uSet=uSet, clf_model=clf_model, dataset=trainDataset)
            
            clf_model.penultimate_active = waslatent
            clf_model.train(wastrain)

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN.startswith("typiclust"):
            from .typiclust import TypiClust
            is_scan = self.cfg.ACTIVE_LEARNING.SAMPLING_FN.endswith('dc')
            tpc = TypiClust(self.cfg, lSet, uSet, budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE, is_scan=is_scan)
            activeSet, uSet = tpc.select_samples()

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN.lower() in ["prob_cover", 'probcover']:
            from .prob_cover import ProbCover
            probcov = ProbCover(self.cfg, lSet, uSet, budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,
                            delta=self.cfg.ACTIVE_LEARNING.INITIAL_DELTA)
            activeSet, uSet = probcov.select_samples()

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN.lower() in ["dcom"]:
            from .DCoM import DCoM
            dcom = DCoM(self.cfg, lSet, uSet, budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,
                        max_delta=self.cfg.ACTIVE_LEARNING.MAX_DELTA,
                        lSet_deltas=self.cfg.ACTIVE_LEARNING.DELTA_LST)
            activeSet, uSet = dcom.select_samples(clf_model, trainDataset, self.dataObj)

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "dbal" or self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "DBAL":
            activeSet, uSet = self.sampler.dbal(budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE, \
                uSet=uSet, clf_model=clf_model,dataset=trainDataset)
            
        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "bald" or self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "BALD":
            activeSet, uSet = self.sampler.bald(budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE, uSet=uSet, clf_model=clf_model, dataset=trainDataset)

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "ensemble_var_R":
            activeSet, uSet = self.sampler.ensemble_var_R(budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE, uSet=uSet, clf_models=supportingModels, dataset=trainDataset)

        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "vaal":
            adv_sampler = AdversarySampler(cfg=self.cfg, dataObj=self.dataObj)

            # Train VAE and discriminator first
            vae, disc, uSet_loader = adv_sampler.vaal_perform_training(lSet=lSet, uSet=uSet, dataset=trainDataset)

            # Do active sampling
            activeSet, uSet = adv_sampler.sample_for_labeling(vae=vae, discriminator=disc, \
                                unlabeled_dataloader=uSet_loader, uSet=uSet)
        
        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "rips_persistence":
        # Use Rips Persistence Sampling
            strategy = self.cfg.ACTIVE_LEARNING.PERSISTENCE_STRATEGY  # e.g., "h1_top", "h0_h1_half"
            verbose = self.cfg.VERBOSE if hasattr(self.cfg, "VERBOSE") else False
            activeSet, uSet = self.sampler.rips_persistence_sampling(
            uSet=uSet,
            dataset=trainDataset,
            budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE, cur_episode = cur_episode,
            randomSampleSize=self.cfg.ACTIVE_LEARNING.RANDOM_SAMPLE_SIZE,
            max_dim=self.cfg.ACTIVE_LEARNING.MAX_DIM,
            strategy=strategy,
            verbose=verbose
        )
        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "rips_persistence_embeddings":
        # Use Rips Persistence Sampling
            strategy = self.cfg.ACTIVE_LEARNING.PERSISTENCE_STRATEGY  # e.g., "h1_top", "h0_h1_half"
            verbose = self.cfg.VERBOSE if hasattr(self.cfg, "VERBOSE") else False
            activeSet, uSet = self.sampler.rips_persistence_sampling_learned_Changed(lSet=lSet,
            uSet=uSet,
            all_features=ds_utils.load_features(self.cfg['DATASET']['NAME'], self.cfg['RNG_SEED']),
            dataset=trainDataset,
            budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE, 
            cur_episode = cur_episode,
            randomSampleSize=self.cfg.ACTIVE_LEARNING.RANDOM_SAMPLE_SIZE,
            max_dim=self.cfg.ACTIVE_LEARNING.MAX_DIM,
            strategy=strategy,
            verbose=verbose
        )
        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "rips_persistence_embeddings_bins":
        # Use Rips Persistence Sampling
            strategy = self.cfg.ACTIVE_LEARNING.PERSISTENCE_STRATEGY  # e.g., "h1_top", "h0_h1_half"
            verbose = self.cfg.VERBOSE if hasattr(self.cfg, "VERBOSE") else False
            bins = self.cfg.ACTIVE_LEARNING.bins
            activeSet, uSet = self.sampler.rips_persistence_sampling_learned_BINS(
            uSet=uSet,
            all_features=ds_utils.load_features(self.cfg['DATASET']['NAME'], self.cfg['RNG_SEED']),
            dataset=trainDataset,
            budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE, cur_episode = cur_episode,
            randomSampleSize=self.cfg.ACTIVE_LEARNING.RANDOM_SAMPLE_SIZE,
            max_dim=self.cfg.ACTIVE_LEARNING.MAX_DIM,
            strategy=strategy,
            verbose=verbose,
            num_bins = bins
        )
        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "rips_persistence_embeddings_Multi":
        # Use Rips Persistence Sampling
        
            strategy = self.cfg.ACTIVE_LEARNING.PERSISTENCE_STRATEGY  # e.g., "h1_top", "h0_h1_half"
            verbose = self.cfg.VERBOSE if hasattr(self.cfg, "VERBOSE") else False
            bins = self.cfg.ACTIVE_LEARNING.bins
            activeSet, uSet = self.sampler.rips_persistence_sampling_learned_multi(
            uSet=uSet,
            all_features=ds_utils.load_features(self.cfg['DATASET']['NAME'], self.cfg['RNG_SEED']),
            dataset=trainDataset,
            budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE, cur_episode = cur_episode,
            strategies=["h0_top","h0_middle_out","h0_bottom","h0_middle_out","h0_bottom","h0_bottom"],
            randomSampleSize=self.cfg.ACTIVE_LEARNING.RANDOM_SAMPLE_SIZE,
            max_dim=self.cfg.ACTIVE_LEARNING.MAX_DIM,
            verbose=verbose
        )
        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "rips_persistence_embeddings_Multi2":
        # Use Rips Persistence Sampling
            strategy = self.cfg.ACTIVE_LEARNING.PERSISTENCE_STRATEGY  # e.g., "h1_top", "h0_h1_half"
            verbose = self.cfg.VERBOSE if hasattr(self.cfg, "VERBOSE") else False
            bins = self.cfg.ACTIVE_LEARNING.bins
            activeSet, uSet = self.sampler.rips_persistence_sampling_learned_multi2(
            uSet=uSet,
            all_features=ds_utils.load_features(self.cfg['DATASET']['NAME'], self.cfg['RNG_SEED']),
            dataset=trainDataset,
            budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE, cur_episode = cur_episode,
            randomSampleSize=self.cfg.ACTIVE_LEARNING.RANDOM_SAMPLE_SIZE,
            max_dim=self.cfg.ACTIVE_LEARNING.MAX_DIM,
            verbose=verbose
        )
        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "ring_based":
        # Use Rips Persistence Sampling
            strategy = self.cfg.ACTIVE_LEARNING.PERSISTENCE_STRATEGY  # e.g., "h1_top", "h0_h1_half"
            verbose = self.cfg.VERBOSE if hasattr(self.cfg, "VERBOSE") else False
            bins = self.cfg.ACTIVE_LEARNING.bins
            activeSet, uSet = self.sampler.ring_based_selection_h0(
            uSet=uSet, lSet = lSet,
            all_features=ds_utils.load_features(self.cfg['DATASET']['NAME'], self.cfg['RNG_SEED']),
            dataset=trainDataset,
            budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE, cur_episode = cur_episode,
            randomSampleSize=self.cfg.ACTIVE_LEARNING.RANDOM_SAMPLE_SIZE,
            max_dim=self.cfg.ACTIVE_LEARNING.MAX_DIM,
            verbose=verbose 
        )
        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "typiclust_like_selection_h0":
        # Use Rips Persistence Sampling
            strategy = self.cfg.ACTIVE_LEARNING.PERSISTENCE_STRATEGY  # e.g., "h1_top", "h0_h1_half"
            verbose = self.cfg.VERBOSE if hasattr(self.cfg, "VERBOSE") else False
            bins = self.cfg.ACTIVE_LEARNING.bins
            activeSet, uSet= self.sampler.typiclust_like_selection_h0(
            uSet=uSet, lSet = lSet,
            all_features=ds_utils.load_features(self.cfg['DATASET']['NAME'], self.cfg['RNG_SEED']),
            dataset=trainDataset,
            budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE, cur_episode = cur_episode,
            randomSampleSize=self.cfg.ACTIVE_LEARNING.RANDOM_SAMPLE_SIZE,
            max_dim=self.cfg.ACTIVE_LEARNING.MAX_DIM,
            verbose=True,n_clusters=5, density_weight=0.7
        )
        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "_like_selection_h0_NF":
            verbose = self.cfg.VERBOSE if hasattr(self.cfg, "VERBOSE") else False
            activeSet, uSet = typiclust_like_selection_h0_no_features(
            uSet=uSet, lSet = lSet,
            all_features=ds_utils.load_features(self.cfg['DATASET']['NAME'], self.cfg['RNG_SEED']),
            dataset=trainDataset, cur_episode = cur_episode,
            budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,
            randomSampleSize=self.cfg.ACTIVE_LEARNING.RANDOM_SAMPLE_SIZE,
            max_dim=self.cfg.ACTIVE_LEARNING.MAX_DIM, 
            verbose=True
        )
        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "most_repeated":
            verbose = self.cfg.VERBOSE if hasattr(self.cfg, "VERBOSE") else False
            activeSet, uSet = most_repeated_h0_selection(
            uSet=uSet, lSet = lSet,
            all_features=ds_utils.load_features(self.cfg['DATASET']['NAME'], self.cfg['RNG_SEED']),
            dataset=trainDataset, cur_episode = cur_episode,
            budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,
            randomSampleSize=self.cfg.ACTIVE_LEARNING.RANDOM_SAMPLE_SIZE,
            max_dim=self.cfg.ACTIVE_LEARNING.MAX_DIM, 
            verbose=True
        )
        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "most_repeated_NF":
            verbose = self.cfg.VERBOSE if hasattr(self.cfg, "VERBOSE") else False
            activeSet, uSet = most_repeated_h0_selection_NF(
            uSet=uSet, lSet = lSet,
            all_features=ds_utils.load_features(self.cfg['DATASET']['NAME'], self.cfg['RNG_SEED']),
            dataset=trainDataset, cur_episode = cur_episode,
            budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,
            randomSampleSize=self.cfg.ACTIVE_LEARNING.RANDOM_SAMPLE_SIZE,
            max_dim=self.cfg.ACTIVE_LEARNING.MAX_DIM, 
            verbose=True
        )
        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "involved":
            verbose = self.cfg.VERBOSE if hasattr(self.cfg, "VERBOSE") else False
            activeSet, uSet = persistence_frequency_sampling_flag(
            uSet=uSet, lSet = lSet,
            all_features=ds_utils.load_features(self.cfg['DATASET']['NAME'], self.cfg['RNG_SEED']),
            dataset=trainDataset, cur_episode = cur_episode,
            budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,
            randomSampleSize=self.cfg.ACTIVE_LEARNING.RANDOM_SAMPLE_SIZE,
            max_dim=self.cfg.ACTIVE_LEARNING.MAX_DIM, 
            verbose=True
        )
        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "most_repeated_effective_death":
            verbose = self.cfg.VERBOSE if hasattr(self.cfg, "VERBOSE") else False
            activeSet, uSet = most_repeated_h0_selection_Vetrex(
            uSet=uSet, lSet = lSet,vertex_counter_input = "/vast/s219110279/TypiClust/output/vertex_counts.npy",
            budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,cur_episode=cur_episode,
            verbose=True
        )
        
        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "least_repeated_effective_death":
            verbose = self.cfg.VERBOSE if hasattr(self.cfg, "VERBOSE") else False
            activeSet, uSet = least_repeated_h0_selection_Vetrex(
            uSet=uSet, lSet = lSet,vertex_counter_input = "/vast/s219110279/TypiClust/output/vertex_counts.npy",
            budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,cur_episode=cur_episode,
            verbose=True
        )
        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "least_persistence_h0_birth_selection_from_csv":
            verbose = self.cfg.VERBOSE if hasattr(self.cfg, "VERBOSE") else False
            
            activeSet, uSet = least_persistence_h0_birth_selection_from_csv(
            uSet=uSet, lSet = lSet,csv_file = "/vast/s219110279/persistence_pairs_Features_CIFAR10.csv",
            budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,
            verbose=True
        )
        elif self.cfg.ACTIVE_LEARNING.SAMPLING_FN == "h0_persistence_selection_from_csv":
            strategy = self.cfg.ACTIVE_LEARNING.PERSISTENCE_STRATEGY
            verbose = self.cfg.VERBOSE if hasattr(self.cfg, "VERBOSE") else False
            activeSet, uSet = h0_persistence_selection_from_csv(
            uSet=uSet, lSet = lSet,csv_file = "/vast/s219110279/persistence_pairs_Features_CIFAR10.csv",
            budgetSize=self.cfg.ACTIVE_LEARNING.BUDGET_SIZE,strategy=strategy, 
            verbose=True
        )

        else:
            print(f"{self.cfg.ACTIVE_LEARNING.SAMPLING_FN} is either not implemented or there is some spelling mistake.")
            raise NotImplementedError

        return activeSet, uSet
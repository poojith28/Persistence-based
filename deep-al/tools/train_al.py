import os
import sys
from datetime import datetime
import argparse
import numpy as np

import torch
from copy import deepcopy

# local

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

add_path(os.path.abspath('..'))

from pycls.al.ActiveLearning import ActiveLearning
import pycls.core.builders as model_builder
from pycls.core.config import cfg, dump_cfg
import pycls.core.losses as losses
import pycls.core.optimizer as optim
from pycls.datasets.data import Data
import pycls.utils.checkpoint as cu
import pycls.utils.logging as lu
import pycls.utils.metrics as mu
import pycls.utils.net as nu
from pycls.utils.meters import TestMeter
from pycls.utils.meters import TrainMeter
from pycls.utils.meters import ValMeter

logger = lu.get_logger(__name__)

plot_episode_xvalues = []
plot_episode_yvalues = []

plot_epoch_xvalues = []
plot_epoch_yvalues = []

plot_it_x_values = []
plot_it_y_values = []

delta_avg_lst = []
delta_std_lst = []


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def argparser():
    parser = argparse.ArgumentParser(description='Active Learning - Image Classification')
    parser.add_argument('--cfg', dest='cfg_file', help='Config file', required=True, type=str)
    parser.add_argument('--exp-name', help='Experiment Name', required=True, type=str)
    parser.add_argument('--al', help='AL Method', required=True, type=str)
    parser.add_argument('--budget', help='Budget Per Round', required=True, type=int)
    parser.add_argument('--initial_size', help='Size of the initial random labeled set', default=0, type=int)
    parser.add_argument('--seed', help='Random seed', default=1, type=int)
    parser.add_argument('--finetune', help='Whether to continue with existing model between rounds', type=str2bool, default=False)
    parser.add_argument('--linear_from_features', help='Whether to use a linear layer from self-supervised features', action='store_true')
    parser.add_argument('--initial_delta', help='Relevant only for ProbCover and DCoM', default=0.6, type=float)
    parser.add_argument('--k_logistic', default=50, type=int)
    parser.add_argument('--a_logistic', default=0.8, type=float)
    parser.add_argument('--persistence_strategy', help='Persistence strategy for sampling (e.g., h1_top, h0_top, h0_h1_half)', 
                        default='h1_top', type=str)
    parser.add_argument('--random_sample_size', help='Number of random samples to process for Rips Persistence (default: 750)', 
                        default=10000, type=int)
    parser.add_argument('--max_dim', help='Maximum homology dimension for Rips Persistence (default: 2)', 
                        default=2, type=int)
    parser.add_argument('--verbose', help='Enable verbose logging for debugging', 
                        type=str2bool, default=False)
    parser.add_argument('--bins', help='add for persistence diagram', 
                        type=int, default=5)
    return parser


def is_eval_epoch(cur_epoch):
    """Determines if the model should be evaluated at the current epoch."""
    return (
        (cur_epoch + 1) % cfg.TRAIN.EVAL_PERIOD == 0 or
        (cur_epoch + 1) == cfg.OPTIM.MAX_EPOCH
    )


def main(cfg):
    # Setting up GPU args
    use_cuda = (cfg.NUM_GPUS > 0) and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': cfg.DATA_LOADER.NUM_WORKERS, 'pin_memory': cfg.DATA_LOADER.PIN_MEMORY} if use_cuda else {}

    # Auto assign a RNG_SEED when not supplied a value
    if cfg.RNG_SEED is None:
        cfg.RNG_SEED = np.random.randint(100)

    # Using specific GPU
    # os.environ['NVIDIA_VISIBLE_DEVICES'] = str(cfg.GPU_ID)
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # print("Using GPU : {}.\n".format(cfg.GPU_ID))

    # Getting the output directory ready (default is "/output")
    cfg.OUT_DIR = os.path.join(os.path.abspath('../..'), cfg.OUT_DIR)
    if not os.path.exists(cfg.OUT_DIR):
        os.mkdir(cfg.OUT_DIR)
    # Create "DATASET/MODEL TYPE" specific directory
    dataset_out_dir = os.path.join(cfg.OUT_DIR, cfg.DATASET.NAME, cfg.MODEL.TYPE)
    if not os.path.exists(dataset_out_dir):
        os.makedirs(dataset_out_dir)
    # Creating the experiment directory inside the dataset specific directory 
    # all logs, labeled, unlabeled, validation sets are stroed here 
    # E.g., output/CIFAR10/resnet18/{timestamp or cfg.EXP_NAME based on arguments passed}
    if cfg.EXP_NAME == 'auto':
        now = datetime.now()
        exp_dir = f'{now.year}_{now.month}_{now.day}_{now.hour:02}{now.minute:02}{now.second:02}_{now.microsecond}'
    else:
        exp_dir = cfg.EXP_NAME

    exp_dir = os.path.join(dataset_out_dir, exp_dir)
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)
        print("Experiment Directory is {}.\n".format(exp_dir))
    else:
        print("Experiment Directory Already Exists: {}. Reusing it may lead to loss of old logs in the directory.\n".format(exp_dir))
    cfg.EXP_DIR = exp_dir

    # Save the config file in EXP_DIR
    dump_cfg(cfg)

    # Setup Logger
    lu.setup_logging(cfg)

    # Dataset preparing steps
    print("\n======== PREPARING DATA AND MODEL ========\n")
    cfg.DATASET.ROOT_DIR = os.path.join(os.path.abspath('../..'), cfg.DATASET.ROOT_DIR)
    data_obj = Data(cfg)
    train_data, train_size = data_obj.getDataset(save_dir=cfg.DATASET.ROOT_DIR, isTrain=True, isDownload=True)
    test_data, test_size = data_obj.getDataset(save_dir=cfg.DATASET.ROOT_DIR, isTrain=False, isDownload=True)
    cfg.ACTIVE_LEARNING.INIT_L_RATIO = args.initial_size / train_size
    print("\nDataset {} Loaded Sucessfully.\nTotal Train Size: {} and Total Test Size: {}\n".format(cfg.DATASET.NAME, train_size, test_size))
    logger.info("Dataset {} Loaded Sucessfully. Total Train Size: {} and Total Test Size: {}\n".format(cfg.DATASET.NAME, train_size, test_size))

    lSet_path, uSet_path, valSet_path = data_obj.makeLUVSets(train_split_ratio=cfg.ACTIVE_LEARNING.INIT_L_RATIO, \
        val_split_ratio=cfg.DATASET.VAL_RATIO, data=train_data, seed_id=cfg.RNG_SEED, save_dir=cfg.EXP_DIR)

    cfg.ACTIVE_LEARNING.LSET_PATH = lSet_path
    cfg.ACTIVE_LEARNING.USET_PATH = uSet_path
    cfg.ACTIVE_LEARNING.VALSET_PATH = valSet_path

    lSet, uSet, valSet = data_obj.loadPartitions(lSetPath=cfg.ACTIVE_LEARNING.LSET_PATH, \
            uSetPath=cfg.ACTIVE_LEARNING.USET_PATH, valSetPath = cfg.ACTIVE_LEARNING.VALSET_PATH)
    model = model_builder.build_model(cfg).cuda()
    
    if len(lSet) == 0:
        if cfg.ACTIVE_LEARNING.SAMPLING_FN.lower() in ['dcom']:
            print('Labeled Set is Empty - Create and save the first delta values list')
            lSet_deltas = [str(cfg.ACTIVE_LEARNING.INITIAL_DELTA)] * cfg.ACTIVE_LEARNING.BUDGET_SIZE
            cfg.ACTIVE_LEARNING.DELTA_LST = lSet_deltas
            delta_avg_lst.append(cfg.ACTIVE_LEARNING.INITIAL_DELTA)

        print('Labeled Set is Empty - Sampling an Initial Pool')
        print(cfg.EXP_NAME)
        al_obj = ActiveLearning(data_obj, cfg)
        activeSet, new_uSet = al_obj.sample_from_uSet(model, lSet, uSet, train_data,cur_episode = -1)
        print(f'Initial Pool is {activeSet}')
        # Save current lSet, new_uSet and activeSet in the episode directory
        # data_obj.saveSets(lSet, uSet, activeSet, cfg.EPISODE_DIR)
        # Add activeSet to lSet, save new_uSet as uSet and update dataloader for the next episode
        lSet = np.append(lSet, activeSet)
        print("Lset is ", lSet)
        print("lSet length is" , len(lSet))
        uSet = new_uSet

    print("Data Partitioning Complete. \nLabeled Set: {}, Unlabeled Set: {}, Validation Set: {}\n".format(len(lSet), len(uSet), len(valSet)))
    logger.info("Labeled Set: {}, Unlabeled Set: {}, Validation Set: {}\n".format(len(lSet), len(uSet), len(valSet)))

    # Preparing dataloaders for initial training
    lSet_loader = data_obj.getIndexesDataLoader(indexes=lSet, batch_size=cfg.TRAIN.BATCH_SIZE, data=train_data)
    valSet_loader = data_obj.getIndexesDataLoader(indexes=valSet, batch_size=cfg.TRAIN.BATCH_SIZE, data=train_data)
    test_loader = data_obj.getTestLoader(data=test_data, test_batch_size=cfg.TRAIN.BATCH_SIZE, seed_id=cfg.RNG_SEED)
    uSet_loader2 = data_obj.getSequentialDataLoader(indexes=uSet, batch_size=cfg.TRAIN.BATCH_SIZE, data=train_data)
    # Initialize the model.  
    model = model_builder.build_model(cfg)
    print("model: {}\n".format(cfg.MODEL.TYPE))
    logger.info("model: {}\n".format(cfg.MODEL.TYPE))

    # Construct the optimizer
    optimizer = optim.construct_optimizer(cfg, model)
    opt_init_state = deepcopy(optimizer.state_dict())
    model_init_state = deepcopy(model.state_dict().copy())

    print("optimizer: {}\n".format(optimizer))
    logger.info("optimizer: {}\n".format(optimizer))

    print("AL Query Method: {}\nMax AL Episodes: {}\n".format(cfg.ACTIVE_LEARNING.SAMPLING_FN, cfg.ACTIVE_LEARNING.MAX_ITER))
    logger.info("AL Query Method: {}\nMax AL Episodes: {}\n".format(cfg.ACTIVE_LEARNING.SAMPLING_FN, cfg.ACTIVE_LEARNING.MAX_ITER))

    for cur_episode in range(0, cfg.ACTIVE_LEARNING.MAX_ITER+1):

        print("======== EPISODE {} BEGINS ========\n".format(cur_episode))
        logger.info("======== EPISODE {} BEGINS ========\n".format(cur_episode))

        # Creating output directory for the episode
        episode_dir = os.path.join(cfg.EXP_DIR, f'episode_{cur_episode}')
        if not os.path.exists(episode_dir):
            os.mkdir(episode_dir)
        cfg.EPISODE_DIR = episode_dir

        # Train model
        print("======== TRAINING ========")
        logger.info("======== TRAINING ========")

        best_val_acc, best_val_epoch, checkpoint_file = train_model(lSet_loader, valSet_loader, model, optimizer, cfg)

        print("Best Validation Accuracy: {}\nBest Epoch: {}\n".format(round(best_val_acc, 4), best_val_epoch))
        logger.info("EPISODE {} Best Validation Accuracy: {}\tBest Epoch: {}\n".format(cur_episode, round(best_val_acc, 4), best_val_epoch))
        
        # Test best model checkpoint
        print("======== TESTING ========\n")
        logger.info("======== TESTING ========\n")
        test_acc, correctness_list = test_model(test_loader, checkpoint_file, cfg, cur_episode) #changes this 
        #test_acc = test_model(uSet_loader2, checkpoint_file, cfg, cur_episode)
        print("Test Accuracy: {}.\n".format(round(test_acc, 4)))
        logger.info("EPISODE {} Test Accuracy {}.\n".format(cur_episode, test_acc))

        # No need to perform active sampling in the last episode iteration
        if cur_episode == cfg.ACTIVE_LEARNING.MAX_ITER:
            # Save current lSet, uSet in the final episode directory
            data_obj.saveSet(lSet, 'lSet', cfg.EPISODE_DIR)
            data_obj.saveSet(uSet, 'uSet', cfg.EPISODE_DIR)
            break

        # DCoM's delta-s updating
        if cfg.ACTIVE_LEARNING.SAMPLING_FN.lower() in ["dcom"]:
            print("======== Update the deltas dynamically ========\n")
            from pycls.al.DCoM import DCoM
            al_algo = DCoM(cfg, lSet, uSet, budgetSize=cfg.ACTIVE_LEARNING.BUDGET_SIZE,
                                    max_delta=cfg.ACTIVE_LEARNING.MAX_DELTA,
                                    lSet_deltas=cfg.ACTIVE_LEARNING.DELTA_LST)

            lSet_labels = np.take(train_data.targets, np.asarray(lSet, dtype=np.int64))
            all_images_idx = np.array(list(lSet) + list(uSet))
            images_loader = data_obj.getSequentialDataLoader(indexes=all_images_idx,
                                                    batch_size=cfg.TRAIN.BATCH_SIZE, data=train_data)
            all_labels = np.take(train_data.targets, np.asarray(all_images_idx, dtype=np.int64))

            images_pseudo_labels = get_label_from_model(images_loader, checkpoint_file, cfg)
            cfg.ACTIVE_LEARNING.DELTA_LST[
            -1 * cfg.ACTIVE_LEARNING.BUDGET_SIZE:] = al_algo.new_centroids_deltas(lSet_labels,
                                                                          all_labels=all_labels,
                                                                          pseudo_labels=images_pseudo_labels,
                                                                          budget=cfg.ACTIVE_LEARNING.BUDGET_SIZE)

            delta_lst_float = [np.float(delta) for delta in cfg.ACTIVE_LEARNING.DELTA_LST]
            delta_avg_lst.append(np.average(delta_lst_float))
            delta_std_lst.append(np.std(delta_lst_float))

        # Active Sample 
        print("======== ACTIVE SAMPLING ========\n")
        logger.info("======== ACTIVE SAMPLING ========\n")
        al_obj = ActiveLearning(data_obj, cfg)
        clf_model = model_builder.build_model(cfg)
        clf_model = cu.load_checkpoint(checkpoint_file, clf_model)
        activeSet, new_uSet = al_obj.sample_from_uSet(clf_model, lSet, uSet, train_data,cur_episode)

        # Save current lSet, new_uSet and activeSet in the episode directory
        data_obj.saveSets(lSet, uSet, activeSet, cfg.EPISODE_DIR)
        save_correctness_list(correctness_list,cfg.EPISODE_DIR)

        # Directory where the file should be saved


        # Ensure the directory exists
        os.makedirs(cfg.EPISODE_DIR, exist_ok=True)

        # Dynamic suffix for the filename (e.g., using current date and time)
        dynamic_suffix = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Filepath with dynamic suffix
        filename = f"Persis_{cur_episode}.csv"
        file_path = os.path.join(cfg.EPISODE_DIR, filename)
        #save_persistence_diagram(persis,lSet=lSet, output_image_dir=cfg.EPISODE_DIR, episode=cur_episode)
        # Add activeSet to lSet, save new_uSet as uSet and update dataloader for the next episode
        lSet = np.append(lSet, activeSet)
        print("Lset is ", lSet)
        print("lSet length is" , len(lSet))
        uSet = new_uSet


        lSet_loader = data_obj.getIndexesDataLoader(indexes=lSet, batch_size=cfg.TRAIN.BATCH_SIZE, data=train_data)
        valSet_loader = data_obj.getIndexesDataLoader(indexes=valSet, batch_size=cfg.TRAIN.BATCH_SIZE, data=train_data)
        uSet_loader = data_obj.getSequentialDataLoader(indexes=uSet, batch_size=cfg.TRAIN.BATCH_SIZE, data=train_data)

        print("Active Sampling Complete. After Episode {}:\nNew Labeled Set: {}, New Unlabeled Set: {}, Active Set: {}\n".format(cur_episode, len(lSet), len(uSet), len(activeSet)))
        logger.info("Active Sampling Complete. After Episode {}:\nNew Labeled Set: {}, New Unlabeled Set: {}, Active Set: {}\n".format(cur_episode, len(lSet), len(uSet), len(activeSet)))
        print("================================\n\n")
        logger.info("================================\n\n")

        # add avg delta to cfg.ACTIVE_LEARNING.DELTA_LST towards the next active sampling
        if cfg.ACTIVE_LEARNING.SAMPLING_FN.lower() in ['dcom']:
            delta_lst_float = [np.float(delta) for delta in cfg.ACTIVE_LEARNING.DELTA_LST]
            next_initial_deltas = [str(round(np.average(delta_lst_float), 2))] * cfg.ACTIVE_LEARNING.BUDGET_SIZE
            cfg.ACTIVE_LEARNING.DELTA_LST.extend(next_initial_deltas)
            print("Current delta list: ", cfg.ACTIVE_LEARNING.DELTA_LST)
            print("Current delta avg list: ", delta_avg_lst)
            print("Current delta std list: ", delta_std_lst)
        print('Current accuracy values: ', plot_episode_yvalues)

        if not cfg.ACTIVE_LEARNING.FINE_TUNE:
            # start model from scratch
            print('Starting model from scratch - ignoring existing weights.')
            model = model_builder.build_model(cfg)
            # Construct the optimizer
            optimizer = optim.construct_optimizer(cfg, model)
            print(model.load_state_dict(model_init_state))
            print(optimizer.load_state_dict(opt_init_state))

        os.remove(checkpoint_file)



def train_model(train_loader, val_loader, model, optimizer, cfg):
    global plot_episode_xvalues
    global plot_episode_yvalues

    global plot_epoch_xvalues
    global plot_epoch_yvalues

    global plot_it_x_values
    global plot_it_y_values

    start_epoch = 0
    loss_fun = losses.get_loss_fun()

    # Create meters
    train_meter = TrainMeter(len(train_loader))
    val_meter = ValMeter(len(val_loader))

    # Perform the training loop
    # print("Len(train_loader):{}".format(len(train_loader)))
    logger.info('Start epoch: {}'.format(start_epoch + 1))
    val_set_acc = 0.

    temp_best_val_acc = 0.
    temp_best_val_epoch = 0

    # Best checkpoint model and optimizer states
    best_model_state = None
    best_opt_state = None

    val_acc_epochs_x = []
    val_acc_epochs_y = []

    clf_train_iterations = cfg.OPTIM.MAX_EPOCH * int(len(train_loader)/cfg.TRAIN.BATCH_SIZE)
    clf_change_lr_iter = clf_train_iterations // 25
    clf_iter_count = 0

    for cur_epoch in range(start_epoch, cfg.OPTIM.MAX_EPOCH):

        # Train for one epoch
        train_loss, clf_iter_count = train_epoch(train_loader, model, loss_fun, optimizer, train_meter, \
                                        cur_epoch, cfg, clf_iter_count, clf_change_lr_iter, clf_train_iterations)

        # Compute precise BN stats
        if cfg.BN.USE_PRECISE_STATS:
            nu.compute_precise_bn_stats(model, train_loader)


        # Model evaluation
        if is_eval_epoch(cur_epoch):
            # Original code[PYCLS] passes on testLoader but we want to compute on val Set
            val_loader.dataset.no_aug = True
            val_set_err, _ = test_epoch(val_loader, model, val_meter, cur_epoch)
            val_set_acc = 100. - val_set_err
            val_loader.dataset.no_aug = False
            if temp_best_val_acc < val_set_acc:
                temp_best_val_acc = val_set_acc
                temp_best_val_epoch = cur_epoch + 1

                # Save best model and optimizer state for checkpointing
                model.eval()

                best_model_state = model.module.state_dict() if cfg.NUM_GPUS > 1 else model.state_dict()
                best_opt_state = optimizer.state_dict()

                model.train()

            # Since we start from 0 epoch
            val_acc_epochs_x.append(cur_epoch+1)
            val_acc_epochs_y.append(val_set_acc)

        plot_epoch_xvalues.append(cur_epoch+1)
        plot_epoch_yvalues.append(train_loss)

        # save_plot_values([plot_epoch_xvalues, plot_epoch_yvalues, plot_it_x_values, plot_it_y_values, val_acc_epochs_x, val_acc_epochs_y],\
        #     ["plot_epoch_xvalues", "plot_epoch_yvalues", "plot_it_x_values", "plot_it_y_values","val_acc_epochs_x","val_acc_epochs_y"], out_dir=cfg.EPISODE_DIR, isDebug=False)
        logger.info("Successfully logged numpy arrays!!")

        # Plot arrays
        # plot_arrays(x_vals=plot_epoch_xvalues, y_vals=plot_epoch_yvalues, \
        # x_name="Epochs", y_name="Loss", dataset_name=cfg.DATASET.NAME, out_dir=cfg.EPISODE_DIR)
        #
        # plot_arrays(x_vals=val_acc_epochs_x, y_vals=val_acc_epochs_y, \
        # x_name="Epochs", y_name="Validation Accuracy", dataset_name=cfg.DATASET.NAME, out_dir=cfg.EPISODE_DIR)

        # save_plot_values([plot_epoch_xvalues, plot_epoch_yvalues, plot_it_x_values, plot_it_y_values, val_acc_epochs_x, val_acc_epochs_y], \
        #         ["plot_epoch_xvalues", "plot_epoch_yvalues", "plot_it_x_values", "plot_it_y_values","val_acc_epochs_x","val_acc_epochs_y"], out_dir=cfg.EPISODE_DIR)

        print('Training Epoch: {}/{}\tTrain Loss: {}\tVal Accuracy: {}'.format(cur_epoch+1, cfg.OPTIM.MAX_EPOCH, round(train_loss, 4), round(val_set_acc, 4)))

    # Save the best model checkpoint (Episode level)
    checkpoint_file = cu.save_checkpoint(info="vlBest_acc_"+str(int(temp_best_val_acc)), \
        model_state=best_model_state, optimizer_state=best_opt_state, epoch=temp_best_val_epoch, cfg=cfg)

    print('\nWrote Best Model Checkpoint to: {}\n'.format(checkpoint_file.split('/')[-1]))
    logger.info('Wrote Best Model Checkpoint to: {}\n'.format(checkpoint_file))

    # plot_arrays(x_vals=plot_epoch_xvalues, y_vals=plot_epoch_yvalues, \
    #     x_name="Epochs", y_name="Loss", dataset_name=cfg.DATASET.NAME, out_dir=cfg.EPISODE_DIR)
    #
    # plot_arrays(x_vals=plot_it_x_values, y_vals=plot_it_y_values, \
    #     x_name="Iterations", y_name="Loss", dataset_name=cfg.DATASET.NAME, out_dir=cfg.EPISODE_DIR)
    #
    # plot_arrays(x_vals=val_acc_epochs_x, y_vals=val_acc_epochs_y, \
    #     x_name="Epochs", y_name="Validation Accuracy", dataset_name=cfg.DATASET.NAME, out_dir=cfg.EPISODE_DIR)

    plot_epoch_xvalues = []
    plot_epoch_yvalues = []
    plot_it_x_values = []
    plot_it_y_values = []

    best_val_acc = temp_best_val_acc
    best_val_epoch = temp_best_val_epoch

    return best_val_acc, best_val_epoch, checkpoint_file


def test_model(test_loader, checkpoint_file, cfg, cur_episode):

    global plot_episode_xvalues
    global plot_episode_yvalues

    global plot_epoch_xvalues
    global plot_epoch_yvalues

    global plot_it_x_values
    global plot_it_y_values

    test_meter = TestMeter(len(test_loader))

    model = model_builder.build_model(cfg)
    model = cu.load_checkpoint(checkpoint_file, model)

    test_err,correctness_list = test_epoch(test_loader, model, test_meter, cur_episode)
    
    test_acc = 100. - test_err

    plot_episode_xvalues.append(cur_episode)
    plot_episode_yvalues.append(test_acc)

    # plot_arrays(x_vals=plot_episode_xvalues, y_vals=plot_episode_yvalues, \
    #     x_name="Episodes", y_name="Test Accuracy", dataset_name=cfg.DATASET.NAME, out_dir=cfg.EXP_DIR)
    #
    # save_plot_values([plot_episode_xvalues, plot_episode_yvalues], \
    #     ["plot_episode_xvalues", "plot_episode_yvalues"], out_dir=cfg.EXP_DIR)

    return test_acc,correctness_list


def train_epoch(train_loader, model, loss_fun, optimizer, train_meter, cur_epoch, cfg, clf_iter_count, clf_change_lr_iter, clf_max_iter):
    """Performs one epoch of training."""
    global plot_episode_xvalues
    global plot_episode_yvalues

    global plot_epoch_xvalues
    global plot_epoch_yvalues

    global plot_it_x_values
    global plot_it_y_values

    # Shuffle the data
    #loader.shuffle(train_loader, cur_epoch)
    if cfg.NUM_GPUS>1:  train_loader.sampler.set_epoch(cur_epoch)

    # Update the learning rate
    # Currently we only support LR schedules for only 'SGD' optimizer
    lr = optim.get_epoch_lr(cfg, cur_epoch)
    if cfg.OPTIM.TYPE == "sgd":
        optim.set_lr(optimizer, lr)

    if torch.cuda.is_available():
        model.cuda()

    # Enable training mode
    model.train()
    train_meter.iter_tic() #This basically notes the start time in timer class defined in utils/timer.py

    len_train_loader = len(train_loader)
    for cur_iter, (inputs, labels) in enumerate(train_loader):
        #ensuring that inputs are floatTensor as model weights are
        inputs = inputs.type(torch.cuda.FloatTensor)
        inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
        # Perform the forward pass
        preds = model(inputs)
        # Compute the loss
        loss = loss_fun(preds, labels)
        # Perform the backward pass
        optimizer.zero_grad()
        loss.backward()
        # Update the parametersSWA
        optimizer.step()
        # Compute the errors
        top1_err, top5_err = mu.topk_errors(preds, labels, [1, 5])
        # Combine the stats across the GPUs
        # if cfg.NUM_GPUS > 1:
        #     #Average error and losses across GPUs
        #     #Also this this calls wait method on reductions so we are ensured
        #     #to obtain synchronized results
        #     loss, top1_err = du.scaled_all_reduce(
        #         [loss, top1_err]
        #     )
        # Copy the stats from GPU to CPU (sync point)
        loss, top1_err = loss.item(), top1_err.item()
        # #Only master process writes the logs which are used for plotting
        # if du.is_master_proc():
        if cur_iter != 0 and cur_iter%19 == 0:
            #because cur_epoch starts with 0
            plot_it_x_values.append((cur_epoch)*len_train_loader + cur_iter)
            plot_it_y_values.append(loss)
            # save_plot_values([plot_it_x_values, plot_it_y_values],["plot_it_x_values", "plot_it_y_values"], out_dir=cfg.EPISODE_DIR, isDebug=False)
            # print(plot_it_x_values)
            # print(plot_it_y_values)
            #Plot loss graphs
            # plot_arrays(x_vals=plot_it_x_values, y_vals=plot_it_y_values, x_name="Iterations", y_name="Loss", dataset_name=cfg.DATASET.NAME, out_dir=cfg.EPISODE_DIR,)
            print('Training Epoch: {}/{}\tIter: {}/{}'.format(cur_epoch+1, cfg.OPTIM.MAX_EPOCH, cur_iter, len(train_loader)))

        #Compute the difference in time now from start time initialized just before this for loop.
        train_meter.iter_toc()
        train_meter.update_stats(top1_err=top1_err, loss=loss, \
            lr=lr, mb_size=inputs.size(0) * cfg.NUM_GPUS)
        train_meter.log_iter_stats(cur_epoch, cur_iter)
        train_meter.iter_tic()
    # Log epoch stats
    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()
    return loss, clf_iter_count


def get_label_from_model(images_loader, checkpoint_file, cfg, model=None):
    """
    returns the labels of the images according to the checkpoint file model
    """
    get_label_meter = TestMeter(len(images_loader))
    if model is None:
        model = model_builder.build_model(cfg)
        model = cu.load_checkpoint(checkpoint_file, model)

    pred = get_label_epoch(images_loader, model, get_label_meter)
    return pred

@torch.no_grad()
def test_epoch(test_loader, model, test_meter, cur_epoch):
    """Evaluates the model on the test set and returns both the misclassification rate
    and a list of tuples (sample_index, correctness) for each test sample.
    
    Each tuple is:
        (sample_index, True) if the sample was classified correctly,
        (sample_index, False) otherwise.
    """

    global plot_episode_xvalues
    global plot_episode_yvalues

    global plot_epoch_xvalues
    global plot_epoch_yvalues

    global plot_it_x_values
    global plot_it_y_values

    if torch.cuda.is_available():
        model.cuda()

    # Enable eval mode
    model.eval()
    test_meter.iter_tic()

    misclassifications = 0.
    totalSamples = 0.

    # List to store per-sample (index, correctness) info.
    correctness_list = []
    global_sample_index = 0

    for cur_iter, (inputs, labels) in enumerate(test_loader):
        with torch.no_grad():
            # Transfer the data to the current GPU device
            inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
            inputs = inputs.type(torch.cuda.FloatTensor)
            
            # Compute the predictions
            preds = model(inputs)

            # ----------------------------
            # New code to record per-sample correctness
            # ----------------------------
            # Get the top-1 predicted labels.
            # (Assumes that higher scores are better.)
            _, predicted = preds.topk(1, dim=1, largest=True, sorted=True)
            predicted = predicted.squeeze(1)  # shape: (batch_size,)
            # Compare predictions to the true labels.
            correct_batch = predicted.eq(labels)
            
            batch_size = inputs.size(0)
            for i in range(batch_size):
                # Append tuple: (sample index, correctness)
                correctness_list.append((global_sample_index, bool(correct_batch[i].item())))
                global_sample_index += 1
            # ----------------------------

            # Compute the errors using your existing method.
            top1_err, top5_err = mu.topk_errors(preds, labels, [1, 5])
            # Copy the top1 error from GPU to CPU.
            top1_err = top1_err.item()
            # Adjust for number of GPUs.
            misclassifications += top1_err * batch_size * cfg.NUM_GPUS
            totalSamples += batch_size * cfg.NUM_GPUS

            test_meter.iter_toc()
            # Update and log stats.
            test_meter.update_stats(
                top1_err=top1_err, mb_size=batch_size * cfg.NUM_GPUS
            )
            test_meter.log_iter_stats(cur_epoch, cur_iter)
            test_meter.iter_tic()

    # Log epoch stats.
    test_meter.log_epoch_stats(cur_epoch)
    test_meter.reset()

    # Return both the misclassification rate and the list of (sample_index, correctness).
    return misclassifications / totalSamples, correctness_list
import os
import numpy as np

def save_correctness_list(correctness_list, save_dir):
    # Ensure the directory exists.
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, 'correctness_list.npy')
    
    # Save the list as a .npy file.
    np.save(file_path, correctness_list)

# Example usage:
# correctness_list = [(0, True), (1, False), (2, True)]  # example list
# save_dir = "/your/desired/directory"
# save_correctness_list(correctness_list, save_dir)


@torch.no_grad()
def get_label_epoch(images_loader, model, get_label_meter):
    """get labels according to the model."""
    if torch.cuda.is_available():
        model.cuda()

    # Enable eval mode
    model.eval()
    get_label_meter.iter_tic()

    all_preds = []
    for cur_iter, (inputs, _) in enumerate(images_loader):
        with torch.no_grad():
            # Transfer the data to the current GPU device
            inputs = inputs.cuda().type(torch.cuda.FloatTensor)
            # Compute the predictions
            preds = model(inputs)
            all_preds += preds

    final_preds = [torch.argmax(p).item() for p in all_preds]
    model.train()

    return final_preds


if __name__ == "__main__":
    args = argparser().parse_args()
    cfg.merge_from_file(args.cfg_file)
    cfg.EXP_NAME = args.exp_name
    cfg.ACTIVE_LEARNING.SAMPLING_FN = args.al
    cfg.ACTIVE_LEARNING.BUDGET_SIZE = args.budget
    cfg.ACTIVE_LEARNING.INITIAL_DELTA = args.initial_delta
    cfg.RNG_SEED = args.seed
    cfg.MODEL.LINEAR_FROM_FEATURES = args.linear_from_features
    cfg.ACTIVE_LEARNING.A_LOGISTIC = args.a_logistic
    cfg.ACTIVE_LEARNING.K_LOGISTIC = args.k_logistic
    cfg.ACTIVE_LEARNING.PERSISTENCE_STRATEGY = args.persistence_strategy
    cfg.ACTIVE_LEARNING.RANDOM_SAMPLE_SIZE = args.random_sample_size
    cfg.ACTIVE_LEARNING.MAX_DIM = args.max_dim
    cfg.ACTIVE_LEARNING.bins = args.bins

    main(cfg)



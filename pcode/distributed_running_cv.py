# -*- coding: utf-8 -*-
import gc
import copy
import numpy as np
import torch
from torch.autograd import grad
import torch.distributed as dist
import os.path as osp

from pcode.create_dataset import define_dataset, load_data_batch

from pcode.utils.checkpoint import load_local_model, save_local_model
from pcode.utils.logging import (
    display_training_stat,
    display_test_stat,
    dispaly_best_test_stat,
    display_consensus_distance,
    display_custom
)
from pcode.utils.stat_tracker import RuntimeTracker
import pcode.utils.error_handler as error_handler
import pcode.utils.auxiliary as auxiliary
from pcode.utils.gradient import gradient_norm, model_norm, log_gt_diff
from pcode.utils.communication import global_tensor_average, global_tensor_sum
from pcode.optim.sam import SAM

import wandb
from wandb_utils import aggregate_stats

def train_and_validate(
    conf, model, criterion, scheduler, optimizer, metrics, data_loader
):
    if conf.load_model:
        model = load_local_model(
            model,
            osp.join(conf.model_dir, "local_model_{}.pth".format(conf.graph.rank))
        )
    print("=>>>> start training and validation.\n")

    # define runtime stat tracker and start the training.
    tracker_tr = RuntimeTracker(
        metrics_to_track=metrics.metric_names, on_cuda=conf.graph.on_cuda
    )

    # get the timer.
    timer = conf.timer
    # break until finish expected full epoch training.
    conf.logger.log_metric(
        name="hyperparameters",
        values={
            "rank": conf.graph.rank,
            "num_batches_train_per_device_per_epoch": conf.num_batches_train_per_device_per_epoch,
            "batch_size": conf.batch_size,
            "total_epochs": conf.num_epochs
        },
        tags={"type": "hyperparameters"},
        display=True
    )
    conf.logger.save_json()

    # create model holders
    if conf.eval_worst:
        global_models = [copy.deepcopy(
                model.module if "DataParallel" == model.__class__.__name__ else model
            ) for _ in range(conf.graph.n_nodes)]
    
    if conf.sam:
        sam_optimizer = SAM(model.parameters(), optimizer)
    else:
        sam_optimizer = None

    if optimizer.__class__.__name__ == "GtHsgd" or "DoCoM" in optimizer.__class__.__name__ or "BEER" in optimizer.__class__.__name__:
        # First step of GT-HSGD / DoCoM
        with timer("opt_init", epoch=0.0):
            if "BEER" in optimizer.__class__.__name__:
                optimizer.init_gradient_tracker(conf, criterion, data_loader["train_loader"], timer=timer)
            else:
                optimizer.init_momentum_tracker(conf, criterion, data_loader["train_loader"], timer=timer, sam_opt=sam_optimizer)
    
        # reshuffle the data.
        if conf.reshuffle_per_epoch and conf.data != "femnist": # femnist reshuffled by dataloader
            print("\nReshuffle the dataset.")
            del data_loader
            gc.collect()
            data_loader = define_dataset(conf)
    
    
    print("=>>>> enter the training.\n")
    total_bits_transmitted = 0
    dist.barrier()
    while True:

        # configure local step.
        for _input, _target in data_loader["train_loader"]:
            model.train()
            scheduler.step(optimizer, const_stepsize=conf.const_lr)

            # load data
            with timer("load_data", epoch=scheduler.epoch_):
                _input, _target = load_data_batch(conf, _input, _target)

            # inference and get current performance.
            with timer("forward_pass", epoch=scheduler.epoch_):
                optimizer.zero_grad()
                loss = inference(model, criterion, metrics, _input, _target, tracker_tr)

            with timer("backward_pass", epoch=scheduler.epoch_):
                loss.backward()
            
            if conf.sam:
                with timer("sam_loss", epoch=scheduler.epoch_):
                    # sam_optimizer.first_step(rho=0.05 * scheduler.get_lr() / conf.lr ,zero_grad=True) # climb to local maxima
                    sam_optimizer.first_step(zero_grad=True) # climb to local maxima
                    inference(model, criterion, metrics, _input, _target, tracker_tr).backward()
                    sam_optimizer.second_step(zero_grad=False) # revert model prm to the usual state, keep the SAM gradient for decentralized optimizer

            with timer("sync_complete", epoch=scheduler.epoch_):
                n_bits_to_transmit = optimizer.step(timer=timer, scheduler=scheduler, input=_input, target=_target, criterion=criterion,
                                                    model=model, epoch=scheduler.epoch_, conf=conf, dataloader=data_loader["train_loader"])
                total_bits_transmitted += n_bits_to_transmit

            # display the logging info.
            msg = {}
            if "FSP" in  optimizer.__class__.__name__:
                msg = {"eta": optimizer.eta, "gamma": optimizer.gamma / 2, "beta": optimizer.beta}
            display_training_stat(conf, scheduler, tracker_tr, n_bits_to_transmit, display=conf.graph.rank==0, extra_stats=msg)

            # display tracking time.
            if (
                conf.graph.rank == 0
                and conf.display_tracked_time
                and scheduler.local_index % conf.summary_freq == 0
            ):
                print(timer.summary())
            

            if tracker_tr.stat["loss"].avg > 1e5 or np.isnan(tracker_tr.stat["loss"].avg):
                print("\nThe process diverges!!!!!Early stop it. loss = {}".format(tracker_tr.stat["loss"].avg))
                exit()
                error_handler.abort()

            if scheduler.epoch_ % 1 == 0:
                tracker_tr.reset()
            
            if scheduler.is_eval():
                dist.barrier()
                # evaluate gradient tracker consensus
                if optimizer.__class__.__name__ in ["GNSD", "GtHsgd", "DeTAG", "ParallelDoCoM_V", "ParallelBEER_V"]:
                    local_gt = optimizer.get_gt()
                    avged_gt = global_tensor_average(local_gt, conf.graph.n_nodes, conf.on_cuda)
                    log_gt_diff(conf, scheduler, local_gt, avged_gt)
                
                # refresh the logging cache at the begining of each epoch.
                tracker_tr.reset()

                # evaluate (and only inference) on the whole training loader.
                if not conf.train_fast and not conf.skip_eval:

                    # evaluate on the local model.
                    if not conf.eval_consensus_only or (conf.eval_consensus_only and scheduler.is_stop()):
                        if conf.eval_worst:
                            _stats = all_gather_models_and_local_eval_and_cal_consensus(
                                conf,
                                model,
                                optimizer,
                                criterion,
                                scheduler,
                                metrics,
                                data_loader=data_loader,
                                global_models=global_models
                            )
                            _stats["bits_transmitted"] = total_bits_transmitted
                            _stats["iteration"] = torch.tensor(scheduler.local_index, dtype=torch.int64)
                            _stats = aggregate_stats(_stats)
                        else:
                            _stats = all_reduce_models_and_global_eval_and_cal_consensus(
                                conf,
                                model,
                                optimizer,
                                criterion,
                                scheduler,
                                metrics,
                                data_loader
                            )
                            _stats["bits_transmitted"] = total_bits_transmitted
                            _stats["iteration"] = torch.tensor(scheduler.local_index, dtype=torch.int64)
                        if "FSP" in  optimizer.__class__.__name__:
                            msg = {"eta": optimizer.eta, "gamma": optimizer.gamma / 2, "beta": optimizer.beta}
                            _stats = {**_stats, **msg}
                        _stats = {**_stats, "lr": optimizer.lr}
                        if conf.graph.rank == 0:
                            wandb.log(_stats)
                    else:
                        consensus_distance(conf, model, optimizer, scheduler)

                # determine if the training is finished.
                if scheduler.is_stop():
                    # save json.
                    conf.logger.save_json()
                    # save the model.
                    if conf.save_model:
                        save_local_model(
                            {"state_dict": model.state_dict()},
                            conf.model_dir, 
                            "local_model_{}.pth".format(conf.graph.rank)
                        )
                    # temporarily hack the exit parallelchoco
                    if optimizer.__class__.__name__ == "ParallelCHOCO" or optimizer.__class__.__name__ == "ParallelDoCoM":
                        error_handler.abort()
                    return

            

        # reshuffle the data.
        if conf.reshuffle_per_epoch and conf.data != "femnist" and conf.data != "tomshardware": # custom dataset reshuffled by dataloader
            print("\nReshuffle the dataset.")
            del data_loader
            gc.collect()
            data_loader = define_dataset(conf)


def inference(model, criterion, metrics, _input, _target, tracker=None, weight_decay=1e-4, backward=False):
    """Inference on the given model and get loss and accuracy."""
    output = model(_input)
    loss = criterion(output, _target)
    if backward:
        loss.backward()
    performance = metrics.evaluate(loss, output, _target)
    weight_decay_loss = weight_decay * model_norm(model)**2
    if tracker is not None:
        tracker.update_metrics([loss.item() + weight_decay_loss] + performance, n_samples=_input.size(0))
    return loss


def consensus_distance(conf, model, optimizer, scheduler):
    if conf.on_cuda:
        dev = next(model.parameters()).device
        model = model.to("cpu")
    
    copied_model = copy.deepcopy(
        model.module if "DataParallel" == model.__class__.__name__ else model
    )
    if conf.on_cuda:
        copied_model = copied_model.to(dev)
    
    optimizer.world_aggregator.agg_model(copied_model, op="avg", communication_scheme="all_reduce")

    if conf.on_cuda:
        copied_model = copied_model.to("cpu")

    # get the l2 distance of the local model to the averaged model
    consensus_err = auxiliary.get_model_difference(model, copied_model)
    conf.logger.log_metric(
        name="stat",
        values={
            "rank": conf.graph.rank,
            "epoch": scheduler.epoch_,
            "distance": consensus_err,
        },
        tags={"split": "test", "type": "averaged_model"},
        display=True
    )
    conf.logger.save_json()

    if conf.on_cuda:
        model = model.to(dev)

    return consensus_err, copied_model


def all_reduce_models_and_global_eval_and_cal_consensus(
    conf,
    model,
    optimizer,
    criterion,
    scheduler,
    metrics,
    data_loader
):
    def _evaluate(_model, is_val):
        # define stat.
        tracker_te = RuntimeTracker(
            metrics_to_track=metrics.metric_names, on_cuda=conf.graph.on_cuda
        )

        # switch to evaluation mode for logging grad
        if conf.eval_grad:
            _model._eval_layers()
            _model.zero_grad()
        else:
            _model.eval()

        dloader = data_loader["val_loader"] if is_val else data_loader["train_loader"]
        n_samples = 0
        for _input, _target in dloader:
            # load data and check performance.
            _input, _target = load_data_batch(conf, _input, _target)
            n_samples += _input.size(0)
            
            if conf.eval_grad:
                inference(_model, criterion, metrics, _input, _target, tracker_te, backward=conf.eval_grad) 
            else:
                with torch.no_grad():
                    inference(_model, criterion, metrics, _input, _target, tracker_te, backward=conf.eval_grad) 

        # aggregate gradient to get gradient of global function
        if conf.eval_grad:
            optimizer.world_aggregator.agg_grad(_model, op="sum", communication_scheme="reduce", dst_rank=0)

        tracker_dict = tracker_te.get_sum()
        return tracker_dict["top1"][0], tracker_dict["loss"][0], n_samples
    
    def _eval_wrapper(conf, scheduler, avg_model, label, is_val):
        local_top1_sum, local_loss_sum, n_samples = _evaluate(avg_model, is_val)
        
        dev = next(model.parameters()).device
        local_top1_sum = torch.tensor([local_top1_sum], device=dev)
        local_loss_sum = torch.tensor([local_loss_sum], device=dev)
        n_samples = torch.tensor([n_samples], dtype=torch.int64, device=dev)

        # sum among all local datasets
        dist.all_reduce(local_top1_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(local_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(n_samples, op=dist.ReduceOp.SUM)
        
        global_n_samples = n_samples.item()
        perf_log = {"top1": local_top1_sum.item() / global_n_samples,
                    "loss": local_loss_sum.item() / global_n_samples}

        if conf.eval_grad:
            perf_log["grad_norm"] = gradient_norm(avg_model, weight_decay=conf.weight_decay) / global_n_samples
        
        display_custom(conf, scheduler, perf_log, label)
        return perf_log

    consensus_dist, avg_model = consensus_distance(conf, model, optimizer, scheduler)

    if conf.on_cuda:
        dev = next(model.parameters()).device
        model = model.cpu() # offload model to save memory
        avg_model = avg_model.to(dev)
    label_val = "eval_local_model_on_full_testing_data"
    label_train = "eval_local_model_on_full_training_data"
    if not conf.clean_output or conf.graph.rank == 0:
        conf.logger.log("epoch {}: eval averaged model on local training data".format(conf.epoch_))
    train_stat = _eval_wrapper(conf, scheduler, avg_model, label_train, False)
    if not conf.clean_output or conf.graph.rank == 0:
        conf.logger.log("epoch {}: eval averaged model on local testing data".format(conf.epoch_))
    test_stat = _eval_wrapper(conf, scheduler, avg_model, label_val, True)

    result_dict = {
        "train_top1": train_stat["top1"],
        "train_loss": train_stat["loss"],
        "test_top1": test_stat["top1"],
        "test_loss": test_stat["loss"],
        "consensus_error": consensus_dist
    }
    if conf.eval_grad:
        result_dict["train_grad_norm"] = train_stat["grad_norm"]
        result_dict["test_grad_norm"] = test_stat["grad_norm"]

    del avg_model
    
    if conf.on_cuda:
        model = model.to(dev)
    return result_dict

# def do_validate(conf, model, optimizer, criterion, scheduler, metrics, data_loader, is_val=True):
#     """Evaluate the model on the test dataset"""
#     # wait until the whole group enters this function, and then evaluate.
#     print("Enter validation phase.")
#     performance = validate(
#         conf, model, optimizer, criterion, scheduler, metrics, data_loader, is_val=is_val
#     )

#     # remember best performance and display the val info.
#     scheduler.best_tracker.update(performance[0], scheduler.epoch_)
#     dispaly_best_test_stat(conf, scheduler)

#     print("Finished validation.")


# def validate(
#     conf,
#     model,
#     criterion,
#     scheduler,
#     metrics,
#     data_loader,
#     label="local_model",
#     is_val=True,
# ):
#     """A function for model evaluation."""

#     def _evaluate(_model, label):
#         # define stat.
#         tracker_te = RuntimeTracker(
#             metrics_to_track=metrics.metric_names, on_cuda=conf.graph.on_cuda
#         )

#         # switch to evaluation mode
#         _model.eval()
#         dloader = data_loader["val_loader"] if is_val else data_loader["train_loader"]
#         for _input, _target in dloader:
#             # load data and check performance.
#             _input, _target = load_data_batch(conf, _input, _target)

#             with torch.no_grad():
#                 inference(_model, criterion, metrics, _input, _target, tracker_te)

#         # display the test stat.
#         display_test_stat(conf, scheduler, tracker_te, label)

#         # get global (mean) performance
#         global_performance = tracker_te.evaluate_global_metrics()
#         return global_performance

#     # evaluate each local model on the validation dataset.
#     global_performance = _evaluate(model, label=label)
#     return global_performance

# def cal_consensus(conf, model, global_models, optimizer, scheduler):
#     # all gather models
#     my_copied_model = copy.deepcopy(
#         model.module if "DataParallel" == model.__class__.__name__ else model
#     )

#     if not conf.clean_output or conf.graph.rank == 0:
#         conf.logger.log("epoch {}: all gather models.".format(conf.epoch_))
#     avg_model = optimizer.world_aggregator.all_gather_model(global_models, my_copied_model)

#     # get the l2 distance of the local model to the averaged model
#     consensus_dist = auxiliary.get_model_difference(model, avg_model)
#     display_consensus_distance(conf, scheduler, consensus_dist)





def all_gather_models_and_local_eval_and_cal_consensus(  
    conf,
    model,
    optimizer,
    criterion,
    scheduler,
    metrics,
    data_loader,
    global_models
):
    """"Use centralized aggregator to get all models, eval all models on local dataset, 
    and aggregate the performance metrics (memory constrained approach)"""

    def _evaluate(_model, rank, is_val=False):
        # define stat.
        tracker_te = RuntimeTracker(
            metrics_to_track=metrics.metric_names, on_cuda=conf.graph.on_cuda
        )

        # switch to evaluation mode for logging grad
        if conf.eval_grad:
            _model._eval_layers()
            _model.zero_grad()
        else:
            _model.eval()

        dloader = data_loader["val_loader"] if is_val else data_loader["train_loader"]
        n_samples = 0
        for _input, _target in dloader:
            # load data and check performance.
            _input, _target = load_data_batch(conf, _input, _target)
            n_samples += _input.size(0)
            
            if conf.eval_grad:
                inference(_model, criterion, metrics, _input, _target, tracker_te, backward=conf.eval_grad) 
            else:
                with torch.no_grad():
                    inference(_model, criterion, metrics, _input, _target, tracker_te, backward=conf.eval_grad) 

        # aggregate gradient to get gradient of global function
        if conf.eval_grad:
            optimizer.world_aggregator.agg_grad(_model, op="sum", communication_scheme="reduce", dst_rank=rank)

        tracker_dict = tracker_te.get_sum()
        return tracker_dict["top1"][0], tracker_dict["loss"][0], n_samples
    
    def _eval_wrapper(conf, scheduler, global_models, label, is_val):
        performance_list = []
        for rank, agent_model in enumerate(global_models):
            agent_local_top1_sum, agent_local_loss_sum, n_samples = _evaluate(agent_model, rank, is_val)
            performance_list.append([agent_local_top1_sum, agent_local_loss_sum, n_samples])
        
        all_performances = global_tensor_sum(performance_list, conf.on_cuda) # averaged among all local datasets, allreduced all models

        global_n_samples = all_performances[conf.graph.rank][-1].item()
        perf_log = {"top1": all_performances[conf.graph.rank][0].item() / global_n_samples,
                    "loss": all_performances[conf.graph.rank][1].item() / global_n_samples}

        if conf.eval_grad:
            perf_log["grad_norm"] = gradient_norm(global_models[conf.graph.rank], weight_decay=conf.weight_decay) / global_n_samples
        
        display_custom(conf, scheduler, perf_log, label)
        return perf_log
        

    # all gather models
    my_model = model.module if "DataParallel" == model.__class__.__name__ else model

    if not conf.clean_output or conf.graph.rank == 0:
        conf.logger.log("epoch {}: all gather models.".format(conf.epoch_))
    avg_model = optimizer.world_aggregator.all_gather_model(global_models, my_model)

    # get the l2 distance of the local model to the averaged model
    consensus_dist = auxiliary.get_model_difference(model, avg_model)
    display_consensus_distance(conf, scheduler, consensus_dist)

    label_val = "eval_local_model_on_full_testing_data"
    label_train = "eval_local_model_on_full_training_data"

    if not conf.clean_output or conf.graph.rank == 0:
        conf.logger.log("epoch {}: eval all models on local training data".format(conf.epoch_))
    train_stat = _eval_wrapper(conf, scheduler, global_models, label_train, False)
    if not conf.clean_output or conf.graph.rank == 0:
        conf.logger.log("epoch {}: eval all models on local testing data".format(conf.epoch_))
    test_stat = _eval_wrapper(conf, scheduler, global_models, label_val, True)

    result_dict = {
        "train_top1": train_stat["top1"],
        "train_loss": train_stat["loss"],
        "test_top1": test_stat["top1"],
        "test_loss": test_stat["loss"],
        "consensus_error": consensus_dist
    }
    if conf.eval_grad:
        result_dict["train_grad_norm"] = train_stat["grad_norm"]
        result_dict["test_grad_norm"] = test_stat["grad_norm"]
    return result_dict
# -*- coding: utf-8 -*-
from copy import deepcopy

import torch
from torch.optim.optimizer import Optimizer, required

import pcode.optim.utils as utils
import pcode.utils.communication as comm

from pcode.utils.tensor_buffer import TensorBuffer
from pcode.create_dataset import load_data_batch
from .parallel_choco_v import CHOCOCompressor, RandomGraphCompressor


class ParallelBEER_V(Optimizer):
    def __init__(
        self,
        params,
        lr=required,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        conf=None,
        model=None,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(ParallelBEER_V, self).__init__(params, defaults)

        # initialize gradient tracker
        for groups in self.param_groups:
            groups["grad_tracker"] = [torch.zeros_like(prm) for prm in groups["params"]]
            groups["grad_p"] = [torch.zeros_like(prm) for prm in groups["params"]]

        self.model = model

        # store the whole training arguments.
        self.conf = conf
        self.use_cuda = conf.on_cuda
        
        # define the aggregator.
        self.rank = conf.graph.rank
        self.neighbors_info = conf.graph.get_neighborhood()
        self.aggregator = comm.get_aggregators(
            cur_rank=self.rank,
            world=conf.graph.ranks,
            neighbors_info=self.neighbors_info,
            aggregator_type="decentralized",
        )
        self.world_aggregator = comm.get_aggregators(
            cur_rank=self.rank,
            world=conf.graph.ranks,
            neighbors_info=dict(
                (rank, 1.0 / conf.graph.n_nodes) for rank in conf.graph.ranks
            ),
            aggregator_type="centralized",
        )

        # define param names and init model_hat.
        self.param_names = list(
            enumerate([group["name"] for group in self.param_groups])
        )
        self.init_neighbor_hat_params()
        self.init_neighbor_hat_gt()
        self.consensus_stepsize = conf.consensus_stepsize

        # related to sparsification/quantization.
        if self.conf.node_fraction < 1:
            self.compressor = RandomGraphCompressor(
                aggregator=self.aggregator,
                comm_op=conf.comm_op,
                comm_device=self.conf.comm_device,
                compress_ratio=conf.compress_ratio,
                quantize_level=conf.quantize_level,
                is_biased=conf.is_biased,
                backend=conf.backend,
                use_ipc=conf.use_ipc,
                node_fraction=conf.node_fraction,
                rank=self.rank,
                use_cuda=self.use_cuda
            )
        else:
            self.compressor = CHOCOCompressor(
                aggregator=self.aggregator,
                comm_op=conf.comm_op,
                comm_device=self.conf.comm_device,
                compress_ratio=conf.compress_ratio,
                quantize_level=conf.quantize_level,
                is_biased=conf.is_biased,
                backend=conf.backend,
                use_ipc=conf.use_ipc,
                use_cuda=conf.on_cuda,
            )

        # define auxilary functions.
        self.helper_thread = None
        self.sync_buffer = {}
        self.sync_buffer_gt = {}
        self.n_bits = 0
        self.it = 0
    
    def L2_regularize_grad(self):
        weight_decay = self.conf.weight_decay
        for key, prm in self.model.named_parameters():
            if not "bn" in key and weight_decay != 0:
                prm.grad.data = prm.grad.data.detach().clone() + weight_decay * prm.data.detach().clone()

    def init_neighbor_hat_params(self):
        params, self.shapes = comm.get_data(
            self.param_groups, self.param_names, is_get_grad=False
        )
        flatten_params = TensorBuffer(params)
        # flatten_params.buffer = torch.zeros_like(flatten_params.buffer)

        # init the neighbor_params.
        self.neighbor_hat_params = {
            self.rank: deepcopy(flatten_params),
            "memory": deepcopy(flatten_params),
        }

    def init_neighbor_hat_gt(self):
        params, self.shapes = comm.get_data(
            self.param_groups, self.param_names, is_get_grad=False
        )
        flatten_params = TensorBuffer(params)
        flatten_params.buffer = torch.zeros_like(flatten_params.buffer)

        # init the neighbor_params.
        self.neighbor_hat_gt = {
            self.rank: deepcopy(flatten_params),
            "memory": deepcopy(flatten_params),
        }
    
    def __setstate__(self, state):
        super(ParallelBEER_V, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    def inference(self, model, criterion, _input, _target):
        output = model(_input)
        loss = criterion(output, _target)
        return loss

    def init_grad_prev(self):
        for group in self.param_groups:
            for prm, g_p in zip(group["params"], group["grad_p"]):
                # put current gradient into grad_p
                g_p.data = prm.grad.data.detach().clone()
            
    
    def init_gradient_tracker(self, conf, criterion, dataloader, **kargs):
        # update gradient tracker with X_0
        self.model.train()
        num_of_batches = len(dataloader)
        for i, (_input, _target) in enumerate(dataloader):
            _input, _target = load_data_batch(conf, _input, _target)
            loss = self.inference(self.model, criterion, _input, _target) / num_of_batches
            loss.backward()
        self.L2_regularize_grad()
        
        for group in self.param_groups:
            for gt, prm, g_p in zip(group["grad_tracker"], group["params"], group["grad_p"]):
                gt.data =  prm.grad.data.detach().clone()
                # put current gradient into grad_p
                g_p.data = prm.grad.data.detach().clone()
        

    def update_gradient_tracker(self):
        self.L2_regularize_grad()
        for group in self.param_groups:
            for gt, prm, g_p in zip(group["grad_tracker"], group["params"], group["grad_p"]):
                    gt.data = gt.data.detach().clone() + prm.grad.data.detach().clone() - g_p.data.detach().clone()
                    # put current gradient into grad_p
                    g_p.data = prm.grad.data.detach().clone()


    def get_gt(self):
        gts = []
        for groups in self.param_groups:
            for gt in groups["grad_tracker"]:
                gts += gt.data.detach().clone().flatten().tolist()
        return gts


    def step(self, closure=None, **kargs):
        lr = kargs["scheduler"].get_lr()
        self.lr = lr if not lr is None else self.lr
        n_bits = 0
        if self.it == 0:
            self.init_grad_prev()
            do_gt = False
        else:
            do_gt = True
        # =========== gt_round ================

        if do_gt:
            self.update_gradient_tracker()
            
            # recover current params and hat_params
            gts, flatten_gt, flatten_hat_gt = utils.recover_params(
                param_groups=self.param_groups,
                param_names=self.param_names,
                rank=self.rank,
                neighbor_hat_params=self.neighbor_hat_gt,
                get_hat_params=True,
                is_recover_grad_tracker=True
            )
            flatten_gt.buffer += self.consensus_stepsize * (
                self.neighbor_hat_gt["memory"].buffer - self.neighbor_hat_gt[self.rank].buffer
            )
            flatten_gt.unpack(gts)
            
            # eq 6 done

            # start compress/sync.
            self.sync_buffer_gt = {
                "original_shapes": self.shapes,
                "flatten_params": flatten_gt,
                "flatten_hat_params": flatten_hat_gt,
                "edge_result": {},
                "n_bits": 0
            }

            if isinstance(self.compressor, RandomGraphCompressor):
                self.compressor.prepare_round(self.it)

            self.helper_thread = utils.HelperThread(
                name=f"_thread_at_epoch_{self.conf.epoch_}.compress",
                func=self.compressor.pipeline,
                # the arguments below will be feeded into the `func`.
                sync_buffer=self.sync_buffer_gt,
                neighbors_info=self.neighbors_info,
            )
            self.helper_thread.start()
            utils.join_thread(self.helper_thread)
            n_bits += self.sync_buffer_gt.get("n_bits", 0) 

            # update neighbor_hat_gt[self.rank]
            if isinstance(self.compressor, RandomGraphCompressor):
                neighborhood = self.compressor.active_neighbors
                active = self.compressor.active
            else:
                neighborhood = self.neighbors_info
                active = True

            if active:
                if "top_k" in self.conf.comm_op or "random_k" in self.conf.comm_op:
                    idx, vals = self.sync_buffer_gt["edge_result"][self.rank]
                    self.neighbor_hat_gt[self.rank].buffer[idx] += vals
                elif "quantize" in self.conf.comm_op:
                    vals = self.sync_buffer_gt["edge_result"][self.rank]
                    self.neighbor_hat_gt[self.rank].buffer += vals

            
            # update neighbor_hat_gt[nei]
            for nei in neighborhood:
                weight = self.neighbors_info[nei]
                if "top_k" in self.conf.comm_op or "random_k" in self.conf.comm_op:
                    idx, vals = self.sync_buffer_gt["edge_result"][nei]
                    self.neighbor_hat_gt["memory"].buffer[idx] += weight * vals
                elif "quantize" in self.conf.comm_op:
                    vals = self.sync_buffer_gt["edge_result"][nei]
                    self.neighbor_hat_gt["memory"].buffer += weight * vals
                
            # eq 7,8 done

        # =========== theta_round ================

        # Apply the gradients with the weight decay and momentum.
        utils.apply_gradient_from_gradient_tracker(
            self.param_groups, self.state, apply_grad_to_model=True
        )

        

        # recover current params and hat_params
        params, flatten_params, flatten_hat_params = utils.recover_params(
            param_groups=self.param_groups,
            param_names=self.param_names,
            rank=self.rank,
            neighbor_hat_params=self.neighbor_hat_params,
            get_hat_params=True,
        )
        # get updated flatten params.
        flatten_params.buffer += self.consensus_stepsize * (
            self.neighbor_hat_params["memory"].buffer - self.neighbor_hat_params[self.rank].buffer
        )
        # update the local model.
        flatten_params.unpack(params)

        # eq 3 done

    
        # start compress/sync.
        self.sync_buffer = {
            "original_shapes": self.shapes,
            "flatten_params": flatten_params,
            "flatten_hat_params": flatten_hat_params,
            "edge_result": {},
            "n_bits": 0
        }


        if isinstance(self.compressor, RandomGraphCompressor):
            self.compressor.prepare_round(self.it)

        
        
        self.helper_thread = utils.HelperThread(
            name=f"_thread_at_epoch_{self.conf.epoch_}.compress",
            func=self.compressor.pipeline,
            # the arguments below will be feeded into the `func`.
            sync_buffer=self.sync_buffer,
            neighbors_info=self.neighbors_info,
        )
        self.helper_thread.start()
        utils.join_thread(self.helper_thread)
        n_bits += self.sync_buffer.get("n_bits", 0)
        
        
        # update neighbor_hat_params[self.rank]
        if isinstance(self.compressor, RandomGraphCompressor):
            neighborhood = self.compressor.active_neighbors
            active = self.compressor.active
        else:
            neighborhood = self.neighbors_info
            active = True

        if active:
            if "top_k" in self.conf.comm_op or "random_k" in self.conf.comm_op:
                idx, vals = self.sync_buffer["edge_result"][self.rank]
                self.neighbor_hat_params[self.rank].buffer[idx] += vals
            elif "quantize" in self.conf.comm_op:
                vals = self.sync_buffer["edge_result"][self.rank]
                self.neighbor_hat_params[self.rank].buffer += vals

        
        # update neighbor_hat_params[nei]
        for nei in neighborhood:
            weight = self.neighbors_info[nei]
            if "top_k" in self.conf.comm_op or "random_k" in self.conf.comm_op:
                idx, vals = self.sync_buffer["edge_result"][nei]
                self.neighbor_hat_params["memory"].buffer[idx] += weight * vals
            elif "quantize" in self.conf.comm_op:
                vals = self.sync_buffer["edge_result"][nei]
                self.neighbor_hat_params["memory"].buffer += weight * vals
        
        # eq 4,5 done

        self.it += 1
        return n_bits

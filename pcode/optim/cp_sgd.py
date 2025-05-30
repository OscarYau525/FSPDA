# -*- coding: utf-8 -*-
from copy import deepcopy

import torch
from torch.optim.optimizer import Optimizer, required

import pcode.optim.utils as utils
import pcode.utils.communication as comm
# from pcode.utils.sparsification import (
#     get_n_bits,
#     SignCompressor,
#     SparsificationCompressor,
#     QuantizationCompressor,
# )
from pcode.utils.tensor_buffer import TensorBuffer
from .parallel_choco_v import RandomGraphCompressor, CHOCOCompressor

# https://arxiv.org/pdf/2403.01322
class CP_SGD(Optimizer):
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
        super(CP_SGD, self).__init__(params, defaults)

        # store the whole training arguments.
        self.conf = conf
        self.use_cuda = conf.on_cuda
        self.gamma = conf.gamma
        self.omega = conf.omega
        self.alpha = conf.alpha

        # define the aggregator.
        self.rank = conf.graph.rank
        self.neighbors_info = conf.graph.get_neighborhood()
        # convert weights into Laplacian matrix values
        n_neighbors = len(self.neighbors_info) - 1
        self.neighbors_info[self.rank] = n_neighbors
        for i in self.neighbors_info:
            if i != self.rank:
                self.neighbors_info[i] = -1
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
        self.consensus_stepsize = conf.consensus_stepsize

        # initialize dual variable lambda
        for groups in self.param_groups:
            groups["lambdas"] = [torch.zeros_like(prm) for prm in groups["params"]]

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
                use_cuda=self.use_cuda,
                compression_noise=conf.compression_noise
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
                use_cuda=self.use_cuda,
                compression_noise=conf.compression_noise
            )

        # define auxilary functions.
        self.helper_thread = None
        self.sync_buffer = {}
        self.n_bits = 0
        self.it = 0

    def init_neighbor_hat_params(self):
        params, self.shapes = comm.get_data(
            self.param_groups, self.param_names, is_get_grad=False
        )
        flatten_params = TensorBuffer(params, self.use_cuda)
        flatten_params.buffer = torch.zeros_like(flatten_params.buffer)

        # init the neighbor_params.
        self.neighbor_hat_params = {
            self.rank: deepcopy(flatten_params),
            "memory": deepcopy(flatten_params),
        }

    def __setstate__(self, state):
        super(CP_SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)
    

    def get_lambda(self, param_groups, param_names):
        data = []
        for idx, _ in param_names:
            _data = param_groups[idx]["lambdas"][0]
            if _data is not None:
                data.append(_data)
        flatten_lambda = TensorBuffer(data, self.use_cuda)
        return data, flatten_lambda
    
    def get_prm(self, param_groups, param_names):
        data = []
        for idx, _ in param_names:
            _data = param_groups[idx]["params"][0]
            if _data is not None:
                data.append(_data)
        flatten_params = TensorBuffer(data, self.use_cuda)
        return data, flatten_params
    
    def get_zeros_prm_buffer(self, param_groups, param_names):
        _, flatten_params = self.get_prm(param_groups, param_names)
        flatten_params.buffer = torch.zeros_like(flatten_params.buffer)
        return flatten_params


    def step(self, closure=None, **kargs):
        lr = kargs["scheduler"].get_lr()
        self.lr = lr if not lr is None else self.lr
        # start compress/sync.
        params, flatten_params, flatten_hat_params = utils.recover_params(
            param_groups=self.param_groups,
            param_names=self.param_names,
            rank=self.rank,
            neighbor_hat_params=self.neighbor_hat_params,
            get_hat_params=True,
        )
        
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

        if isinstance(self.compressor, RandomGraphCompressor):
            neighborhood = self.compressor.active_neighbors
            active = self.compressor.active
        else:
            neighborhood = self.neighbors_info
            active = True

        # ====== primal update ======
        primal_update_vec = self.get_zeros_prm_buffer(self.param_groups, self.param_names)
        avg_comp_agg = self.get_zeros_prm_buffer(self.param_groups, self.param_names)

        for rank in neighborhood:
            if "top_k" in self.conf.comm_op or "random_k" in self.conf.comm_op:
                idx, vals = self.sync_buffer["edge_result"][rank]
                avg_comp_agg.buffer[idx] += self.neighbors_info[rank] * vals
            elif "quantize" in self.conf.comm_op:
                vals = self.sync_buffer["edge_result"][rank]
                avg_comp_agg.buffer += self.neighbors_info[rank] * vals
            
        primal_update_vec.buffer += self.lr * self.gamma * (self.neighbor_hat_params["memory"].buffer + avg_comp_agg.buffer)
        _lambda, flatten_lambda = self.get_lambda(self.param_groups, self.param_names)
        primal_update_vec.buffer += self.lr * self.omega * flatten_lambda.buffer


        # Apply the gradients with the weight decay and momentum.
        utils.apply_gradient(
            self.param_groups, self.state, apply_grad_to_model=True
        )

        params, flatten_params = self.get_prm(self.param_groups, self.param_names)
        flatten_params.buffer -= primal_update_vec.buffer
        flatten_params.unpack(params)
        # ====== end primal update ======


        # ====== dual update ======
        flatten_lambda.buffer += self.lr * self.omega * (self.neighbor_hat_params["memory"].buffer + avg_comp_agg.buffer)
        flatten_lambda.unpack(_lambda)
        # ====== end dual update ======


        # ====== hat update ======
        if active:
            if "top_k" in self.conf.comm_op or "random_k" in self.conf.comm_op:
                idx, vals = self.sync_buffer["edge_result"][self.rank]
                self.neighbor_hat_params[self.rank].buffer[idx] += self.alpha * vals
            elif "quantize" in self.conf.comm_op:
                vals = self.sync_buffer["edge_result"][self.rank]
                self.neighbor_hat_params[self.rank].buffer += self.alpha * vals
        
        self.neighbor_hat_params["memory"].buffer += self.alpha * avg_comp_agg.buffer
        # ====== end hat update ======

        self.it += 1
        self.n_bits = self.sync_buffer.get("n_bits", 0)

        return self.n_bits


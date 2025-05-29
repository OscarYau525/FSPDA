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

# https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10636509
class DEF_ATC(Optimizer):
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
        super(DEF_ATC, self).__init__(params, defaults)

        # store the whole training arguments.
        self.conf = conf
        self.use_cuda = conf.on_cuda
        self.gamma = conf.gamma
        self.zeta = conf.zeta

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
        self.init_aux_variable()
        self.consensus_stepsize = conf.consensus_stepsize

        # initialize dual variable lambda
        for groups in self.param_groups:
            groups["compress_tracker"] = [torch.zeros_like(prm) for prm in groups["params"]]
            groups["compress_error"] = [torch.zeros_like(prm) for prm in groups["params"]]

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
        self.n_bits = 0
        self.it = 0

    def init_aux_variable(self):
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
        self.compress_tracker = deepcopy(flatten_params)
        self.compress_error = deepcopy(flatten_params)

    def __setstate__(self, state):
        super(DEF_ATC, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)
    
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

        # Apply the gradients with the weight decay and momentum.
        utils.apply_gradient(
            self.param_groups, self.state, apply_grad_to_model=True, lr=self.lr / self.zeta
        )

        # start compress/sync.
        params, flatten_params, flatten_compress_tracker = utils.recover_params(
            param_groups=self.param_groups,
            param_names=self.param_names,
            rank=self.rank,
            neighbor_hat_params=self.neighbor_hat_params,
            get_hat_params=True,
        )

        tmp = self.get_zeros_prm_buffer(self.param_groups, self.param_names)
        tmp.buffer = flatten_compress_tracker.buffer - self.compress_error.buffer

        self.sync_buffer = {
            "original_shapes": self.shapes,
            "flatten_params": flatten_params,
            "flatten_hat_params": tmp,
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

        # ====== aux update ======
        real_msg = flatten_params.buffer - flatten_compress_tracker.buffer + self.compress_error.buffer
        self.compress_error.buffer = real_msg - self.sync_buffer["edge_result"][self.rank]

        if active:
            if "top_k" in self.conf.comm_op or "random_k" in self.conf.comm_op:
                idx, vals = self.sync_buffer["edge_result"][self.rank]
                self.neighbor_hat_params[self.rank].buffer[idx] += self.zeta * vals
            elif "quantize" in self.conf.comm_op:
                vals = self.sync_buffer["edge_result"][self.rank]
                self.neighbor_hat_params[self.rank].buffer += self.zeta * vals
        
        for rank in neighborhood: # including self.rank
            if "top_k" in self.conf.comm_op or "random_k" in self.conf.comm_op:
                idx, vals = self.sync_buffer["edge_result"][rank]
                self.neighbor_hat_params["memory"].buffer[idx] += self.neighbors_info[rank] * self.zeta * vals
            elif "quantize" in self.conf.comm_op:
                vals = self.sync_buffer["edge_result"][rank]
                self.neighbor_hat_params["memory"].buffer += self.neighbors_info[rank] * self.zeta * vals

        # ====== end aux update ======

        params, flatten_params = self.get_prm(self.param_groups, self.param_names)
        flatten_params.buffer = (1 - self.gamma) * self.neighbor_hat_params[self.rank].buffer + self.gamma * self.neighbor_hat_params["memory"].buffer 
        flatten_params.unpack(params)

        self.it += 1
        self.n_bits = self.sync_buffer.get("n_bits", 0)

        return self.n_bits


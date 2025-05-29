# -*- coding: utf-8 -*-

import torch
from torch.optim.optimizer import Optimizer, required

import pcode.optim.utils as utils
import pcode.utils.communication as comm
from pcode.utils.sparsification import (
    get_n_bits,
)
from pcode.utils.tensor_buffer import TensorBuffer
import numpy as np
import torch.distributed as dist


class LED(Optimizer):
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
        super(LED, self).__init__(params, defaults)

        # define the aggregator.
        self.rank = conf.graph.rank
        self.world_size = conf.n_mpi_process
        torch.manual_seed(self.rank)
        np.random.seed(self.rank)
        self.neighbors_info = conf.graph.get_neighborhood()
        self.decentralized_aggregator = comm.get_aggregators(
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

        self.it = 0

        # initialize dual variable lambda and primal memory
        for groups in self.param_groups:
            groups["dual"] = [torch.zeros_like(prm) for prm in groups["params"]]


        self.model = model

        # store the whole training arguments.
        self.conf = conf
        self.use_cuda = conf.on_cuda
        self.local_steps = conf.local_steps
        self.beta = 1 / self.local_steps

        if self.conf.lr_change_epochs is not None:
            self.lr_schedule = [ int(ep) for ep in self.conf.lr_change_epochs.split(",") ]
        
        # define param names and init model_hat.
        self.param_names = list(
            enumerate([group["name"] for group in self.param_groups])
        )

        # define auxilary functions.
        self.helper_thread = None
        self.sync_buffer = {}
        self.sync_buffer_gt = {}
        self.n_bits = 0
        self.first_step = True

        _, self.shapes = comm.get_data(
            self.param_groups, self.param_names, is_get_grad=False
        )
    
    
    def __setstate__(self, state):
        super(LED, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    def inference(self, model, criterion, _input, _target):
        output = model(_input)
        loss = criterion(output, _target)
        return loss

    def get_prm(self, param_groups, param_names, tag="params"):
        data = []
        for idx, _ in param_names:
            _data = param_groups[idx][tag][0]
            if _data is not None:
                data.append(_data)
        flatten_params = TensorBuffer(data, self.use_cuda)
        return data, flatten_params

    
    def get_dual(self, param_groups, param_names):
        return self.get_prm(param_groups, param_names, tag="dual")


    def get_zeros_prm_buffer(self, param_groups, param_names):
        _, flatten_params = self.get_prm(param_groups, param_names)
        flatten_params.buffer = torch.zeros_like(flatten_params.buffer)
        return flatten_params

    def step(self, **kwargs):
        self.n_bits = 0
        lr = kwargs["scheduler"].get_lr()
        self.lr = lr if not lr is None else self.lr

        # Apply the gradients with the weight decay and momentum.
        utils.apply_gradient(
            self.param_groups, self.state, apply_grad_to_model=True
        )

        # Apply control variable to the model.
        for groups in self.param_groups:
            for prm, d in zip(groups["params"], groups["dual"]):
                prm.data -= self.beta * d.data.detach().clone()
        

        if (self.it + 1) % self.local_steps == 0 and self.it > 0:
            # Gossip communication.
            params, flatten_params = self.get_prm(self.param_groups, self.param_names)

            agg_prm = self.decentralized_aggregator._agg(
                flatten_params.buffer, op="weighted"
            )

            # Update dual.
            dual, flatten_dual = self.get_dual(self.param_groups, self.param_names)
            flatten_dual.buffer += flatten_params.buffer - agg_prm
            flatten_dual.unpack(dual)

            # Update prm.
            flatten_params.buffer = agg_prm
            flatten_params.unpack(params)

            self.n_bits += get_n_bits(flatten_params.buffer) * (len(self.neighbors_info) - 1)


        self.it += 1
        return self.n_bits

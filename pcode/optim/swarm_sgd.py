# -*- coding: utf-8 -*-

import torch
from torch.optim.optimizer import Optimizer, required

import pcode.optim.utils as utils
import pcode.utils.communication as comm
from pcode.utils.sparsification import (
    get_n_bits,
    SparsificationCompressor,
)
from pcode.utils.tensor_buffer import TensorBuffer
import numpy as np
import torch.distributed as dist


class SimpleLatticeEncoder:
    def __init__(self, q, s):
        # assume for all inputs, ||x - x'||_infty <= y. then we set side-length = 2y/(q-1), for y = 1
        assert q != 0 and ( q & (q-1) == 0 ) # q should be power of 2
        self.q = q  # quantization level
        self.s = s  # hypercube side length
        self.n_bit = int(np.log2(q))
    
    def encode(self, input_vec):
        scaled_input = input_vec / self.s # divide by s
        scaled_input = torch.round(scaled_input) # make it integer
        encoded_vector = torch.remainder(scaled_input, self.q) # mod q
        
        return encoded_vector
    
class SimpleLatticeDecoder:
    def __init__(self, q, s):
        assert q != 0 and ( q & (q-1) == 0 ) # q should be power of 2
        self.q = q  # quantization level
        self.s = s  # hypercube side length
        self.n_bit = int(np.log2(q))

    def decode(self, quantized_vector, b):
        # Decoding phase:
        part1 = self.q * self.s * torch.round( (b / (self.q * self.s)) - (quantized_vector / self.q) ) # find integer combination of lattices basis
        part2 = self.s * quantized_vector # shift to the same coloring
        decoded_vec = part1 + part2
        
        return decoded_vec
    
class SwarmSGD(Optimizer): # current implementation uses local updates
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
        super(SwarmSGD, self).__init__(params, defaults)

        # define the aggregator.
        self.rank = conf.graph.rank
        torch.manual_seed(self.rank)
        np.random.seed(self.rank)
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

        self.it = 0

        # initialize buffers for gossips
        for groups in self.param_groups:
            groups["outdated_model"] = [prm.detach().clone() for prm in groups["params"]]
            groups["current_model"] = [prm.detach().clone() for prm in groups["params"]]

        self.model = model

        # store the whole training arguments.
        self.conf = conf
        self.use_cuda = conf.on_cuda
        self.gamma = conf.gamma
        self.eta = conf.eta
        self.beta = conf.beta
        
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

        # lattice quantizer
        self.encoder = SimpleLatticeEncoder(conf.quantize_level, conf.side_length)
        self.decoder = SimpleLatticeDecoder(conf.quantize_level, conf.side_length)
    
    
    def __setstate__(self, state):
        super(SwarmSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    def inference(self, model, criterion, _input, _target):
        output = model(_input)
        loss = criterion(output, _target)
        return loss


    def get_outdated_model(self, param_groups, param_names):
        data = []
        for idx, _ in param_names:
            _data = param_groups[idx]["outdated_model"][0]
            if _data is not None:
                data.append(_data)
        flatten_lambda = TensorBuffer(data, self.use_cuda)
        return data, flatten_lambda

    def get_current_model(self, param_groups, param_names):
        data = []
        for idx, _ in param_names:
            _data = param_groups[idx]["current_model"][0]
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
    
    
    def step(self, **kwargs):
        dist.barrier()
        self.n_bits = 0
        lr = kwargs["scheduler"].get_lr()
        self.lr = lr if not lr is None else self.lr

        # implemented 1 edge random graph
        initiator = int(self.it % self.aggregator.world_size)
        if self.aggregator.rank == initiator:
            # I am initiator this round
            rand_neigh_idx = int(np.floor(np.random.rand() * len(self.aggregator.neighbor_ranks)))
            rand_neigh = torch.tensor(self.aggregator.neighbor_ranks[rand_neigh_idx], dtype=torch.int64)
        else:
            rand_neigh = torch.tensor(0)
        
        dist.broadcast(rand_neigh, src=initiator)

        # instead of shared memory, we use communication to exchange encoded outdated models and encoded current models
        if self.aggregator.rank == rand_neigh.item() or self.aggregator.rank == initiator:
            current_params, flatten_current_params = self.get_current_model(self.param_groups, self.param_names)
            my_o_params, flatten_my_o_params = self.get_outdated_model(self.param_groups, self.param_names)

            neighbors_o_params = torch.zeros_like(flatten_my_o_params.buffer)
            neighbors_params = torch.zeros_like(flatten_my_o_params.buffer)

            # procedure to compute the neighbor's quantized current model Q(X_^i_t):
            # - receives encoded outdated model from neighbor, and use the local full-precision outdated model to decode neighbor's outdated model to higher-precision
            # - use neighbor's full-precision outdated model to decode neighbor's current model to higher-precision
            
            if self.aggregator.rank == rand_neigh.item():
                comm_target = initiator
            elif self.aggregator.rank == initiator:
                comm_target = rand_neigh.item()

            # exchange encoded outdated models
            reqs = []
            reqs.append( dist.isend(self.encoder.encode(flatten_my_o_params.buffer), dst=comm_target, tag=0) )
            reqs.append( dist.irecv(neighbors_o_params, src=comm_target, tag=0) )
            # for req in reqs:
            #     req.wait()

            # exchange encoded current models
            # reqs = []
            reqs.append( dist.isend(self.encoder.encode(flatten_current_params.buffer), dst=comm_target, tag=1) )
            reqs.append( dist.irecv(neighbors_params, src=comm_target, tag=1) )
            for req in reqs:
                req.wait()
            
            
            # decode the received encoded outdated model
            neighbors_o_params = self.decoder.decode(
                neighbors_o_params,
                flatten_my_o_params.buffer
            )

            # decode the received encoded current model
            neighbors_params = self.decoder.decode(
                neighbors_params,
                neighbors_o_params
            )
            
            
            # compute the local quantized current model Q(X_^i_t)
            Q_local = self.decoder.decode(
                self.encoder.encode(flatten_current_params.buffer),
                self.decoder.decode(
                    self.encoder.encode(flatten_my_o_params.buffer),
                    flatten_my_o_params.buffer
                )
            )

            self.n_bits = 2 * flatten_current_params.buffer.nelement() * self.encoder.n_bit
        else:
            self.n_bits = 0
        
        # everyone do a local update
        utils.apply_gradient(
            self.param_groups, self.state, apply_grad_to_model=True, set_model_prm_as_grad=False
        )

        if self.aggregator.rank == initiator:
            # use the accumulated gradient steps to update the current model buffer (the one used for gossiping)
            params, flatten_params = self.get_prm(self.param_groups, self.param_names) # get flatten_params after gradient update
            accum_grads = flatten_my_o_params.buffer - flatten_params.buffer # X_hat - (X_hat - eta * h) = eta * h

            current_params, flatten_current_params = self.get_current_model(self.param_groups, self.param_names)
            flatten_current_params.buffer = Q_local / 2 + neighbors_params / 2 - accum_grads
            flatten_current_params.unpack(current_params)

            # update local outdated model
            flatten_current_params.unpack(my_o_params)
            # update local model (the point of next gradient)
            flatten_current_params.unpack(params)
        elif self.aggregator.rank == rand_neigh.item():
            # do not apply gradient, keep the accumulated gradient and local outdated model untouched.
            current_params, flatten_current_params = self.get_current_model(self.param_groups, self.param_names)
            flatten_current_params.buffer = Q_local / 2 + neighbors_params / 2
            flatten_current_params.unpack(current_params)
               
        self.it += 1
        return self.n_bits

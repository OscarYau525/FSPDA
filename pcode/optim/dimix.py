# -*- coding: utf-8 -*-
from copy import deepcopy

import torch
from torch.optim.optimizer import Optimizer, required

import pcode.optim.utils as utils
import pcode.utils.communication as comm
from pcode.utils.sparsification import (
    get_n_bits,
    SignCompressor,
    SparsificationCompressor,
    QuantizationCompressor,
)
from pcode.utils.tensor_buffer import TensorBuffer
from math import sqrt
import torch.distributed as dist
import numpy as np

class DIMIX(Optimizer):
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
        super(DIMIX, self).__init__(params, defaults)

        # store the whole training arguments.
        self.conf = conf
        self.use_cuda = conf.on_cuda
        self.init_lr = conf.lr
        # self.tau = conf.tau
        # self.nu = conf.nu
        # self.mu = conf.mu
        self.alpha = conf.alpha
        self.beta = conf.beta

        # define the aggregator.
        self.rank = conf.graph.rank
        self.world_size = conf.n_mpi_process
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
        self.consensus_stepsize = conf.consensus_stepsize

        # initialize dual variable lambda
        for groups in self.param_groups:
            groups["lambdas"] = [torch.zeros_like(prm) for prm in groups["params"]]
        
        _, self.shapes = comm.get_data(
            self.param_groups, self.param_names, is_get_grad=False
        )

        # related to sparsification/quantization.
        self.compressor = DIMIXCompressor(
            aggregator=self.aggregator,
            comm_op=conf.comm_op,
            comm_device=self.conf.comm_device,
            compress_ratio=conf.compress_ratio,
            quantize_level=conf.quantize_level,
            is_biased=conf.is_biased,
            backend=conf.backend,
            use_ipc=conf.use_ipc,
            use_cuda=conf.on_cuda,
            gamma=conf.gamma,
            compression_noise=conf.compression_noise
        )

        # define auxilary functions.
        self.helper_thread = None
        self.sync_buffer = {}
        self.n_bits = 0
        self.it = 0

 
    def __setstate__(self, state):
        super(DIMIX, self).__setstate__(state)
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


    def sample_random_graph(self):
        if self.conf.one_edge:
            # implemented 1 edge random graph
            initiator = int(self.it % self.world_size)
            if self.aggregator.rank == initiator:
                # I am initiator this round
                rand_neigh_idx = int(np.floor(np.random.rand() * len(self.aggregator.neighbor_ranks)))
                rand_neigh = torch.tensor(self.aggregator.neighbor_ranks[rand_neigh_idx], dtype=torch.int64)
            else:
                rand_neigh = torch.tensor(0)
            
            dist.broadcast(rand_neigh, src=initiator)
            if self.aggregator.rank == initiator:
                active_neighbors = [rand_neigh.item()]
            elif self.aggregator.rank == rand_neigh.item():
                active_neighbors = [initiator]
            else:
                active_neighbors = []
            
        else:
            # edge_activation decides whether an edge is active or not
            edge_activation = {nei: torch.rand(1) for nei in self.aggregator.neighbor_ranks}
            edge_activation = self.aggregator.one_way_consensus(edge_activation, force_wait=True)

            active_neighbors = [nei for nei in edge_activation if edge_activation[nei] <= self.conf.edge_prob]
        return active_neighbors


    # def get_stepsizes(self):
    #     current_alpha = self.alpha / (self.it + self.tau)**self.nu
    #     current_beta = self.beta / (self.it + self.tau)**self.mu
    #     return current_alpha, current_beta
    

    def step(self, closure=None, **kargs):
        lr = kargs["scheduler"].get_lr()
        self.lr = lr if not lr is None else self.lr
        lr_ratio = self.lr / self.init_lr
        # start compress/sync.
        active_neighbors = self.sample_random_graph()
        params, flatten_params = self.get_prm(self.param_groups, self.param_names)

        self.sync_buffer = {
            "original_shapes": self.shapes,
            "flatten_params": flatten_params,
            "active_neighbors": active_neighbors,
            "edge_result": {},
            "n_bits": 0
        }

        self.helper_thread = utils.HelperThread(
            name=f"_thread_at_epoch_{self.conf.epoch_}.compress",
            func=self.compressor.pipeline,
            # the arguments below will be feeded into the `func`.
            sync_buffer=self.sync_buffer,
        )

        self.helper_thread.start()
        utils.join_thread(self.helper_thread)

        self.n_bits = self.sync_buffer.get("n_bits", 0)
        

        # ====== primal update ======
        alpha, beta = self.alpha * lr_ratio, self.beta * lr_ratio

        # Apply the gradients with the weight decay and momentum.
        utils.apply_gradient(
            self.param_groups, self.state, apply_grad_to_model=True, lr=alpha * beta
        )

        if len(active_neighbors) > 0:
            params, flatten_params = self.get_prm(self.param_groups, self.param_names)
            gossip_vec = torch.mean(torch.stack([prm for _, prm in self.sync_buffer["edge_result"].items()]), axis=0)
            flatten_params.buffer -= beta * gossip_vec
            flatten_params.unpack(params)
        
        # ====== end primal update ======

        self.it += 1
        return self.n_bits


"""the entry for DIMIXCompressor."""


class DIMIXCompressor(object):
    def __init__(self, **kargs):
        # assign compressor class.
        if "top_k" in kargs["comm_op"] or "random_k" in kargs["comm_op"]:
            self.compressor_fn = DIMIXSparsificationCompressor(**kargs)
        elif "quantize" in kargs["comm_op"]:
            self.compressor_fn = DIMIXQuantizationCompressor(**kargs)
        else:
            raise NotImplementedError

    def pipeline(self, *args, **kargs):
        return self.compressor_fn.pipeline(*args, **kargs)

    def compress(self, *args, **kargs):
        return self.compressor_fn.compress(*args, **kargs)

    def sync(self, *args, **kargs):
        return self.compressor_fn.sync(*args, **kargs)

    def uncompress(self, *args, **kargs):
        return self.compressor_fn.uncompress(*args, **kargs)


"""Detailed DIMIXCompressors, e.g., top-k/random-k, quantization, sign-based quantization."""


class DIMIXSparsificationCompressor(object):
    def __init__(
        self,
        aggregator,
        comm_op,
        comm_device,
        compress_ratio,
        quantize_level,
        is_biased,
        backend,
        use_ipc,
        use_cuda,
        gamma,
        **kargs,
    ):
        # assign the common hyper-parameters
        self.aggregator_fn = aggregator
        self.comm_op = comm_op
        self.comm_device = comm_device
        self.compress_ratio = compress_ratio
        self.quantize_level = quantize_level
        self.is_biased = is_biased
        self.backend = backend
        self.use_ipc = use_ipc
        self.use_cuda = use_cuda
        self.gamma = gamma
        self.kargs = kargs
        self.compressor_fn = SparsificationCompressor()

        # define gossip_stream
        if torch.cuda.is_available():
            self.gossip_stream = torch.cuda.current_stream()
            self.use_cuda = True

    def pipeline(self, sync_buffer):
        if len(sync_buffer["active_neighbors"]) > 0:
            if torch.cuda.is_available():
                with torch.cuda.stream(self.gossip_stream):
                    try:
                        self.compress(sync_buffer)
                        self.sync(sync_buffer)
                        self.uncompress(sync_buffer)
                    except RuntimeError as e:
                        print("Error: {}".format(e))
            else:
                self.compress(sync_buffer)
                self.sync(sync_buffer)
                self.uncompress(sync_buffer)


    def compress(self, sync_buffer):
        sync_buffer["send_dict"] = {}
        sync_buffer["selected_shapes"] = {}

        for nei in sync_buffer["active_neighbors"]:
            selected_values = []
            selected_indices = []
            for half_param, hat_param in zip(
                sync_buffer["flatten_params"], sync_buffer["local_hat_params"][nei]
            ):
                _selected_values, _selected_indices = self.compressor_fn.compress(
                    half_param - hat_param,
                    self.comm_op,
                    self.compress_ratio,
                    self.is_biased,
                )
                selected_values.append(_selected_values)
                selected_indices.append(_selected_indices)

            # get selected shapes.
            selected_shapes = [len(_value) for _value in selected_values]

            # flatten selected values/indices.
            flatten_selected_values = TensorBuffer(selected_values, self.use_cuda)
            flatten_selected_indices = TensorBuffer(selected_indices, self.use_cuda)

            noise = torch.zeros_like(flatten_selected_values.buffer).uniform_(-self.compression_noise, self.compression_noise)
            flatten_selected_values.buffer += noise

            # update the local hat variable
            q_values, q_indices = self.compressor_fn.uncompress(
                flatten_selected_values.buffer, flatten_selected_indices.buffer, selected_shapes, sync_buffer["original_shapes"]
            )
            sync_buffer["local_hat_params"][nei].buffer[q_indices] += self.gamma * q_values

            sync_buffer["send_dict"][nei] = [flatten_selected_values.buffer, flatten_selected_indices.buffer]

            sync_buffer["selected_shapes"][nei] = selected_shapes
            if self.comm_device == "cpu":
                sync_buffer["send_dict"][nei][0] = sync_buffer["send_dict"][nei][0].cpu().pin_memory()
                sync_buffer["send_dict"][nei][1] = sync_buffer["send_dict"][nei][1].cpu().pin_memory()
            
            # get n_bits to transmit.
            if self.compress_ratio > 0:
                n_bits = get_n_bits(flatten_selected_values.buffer) + get_n_bits(
                    flatten_selected_indices.buffer
                )
            else:
                # no sparsification is applied
                n_bits = get_n_bits(flatten_selected_values.buffer)
            sync_buffer["n_bits"] += n_bits


    def sync(self, sync_buffer):
        # sync.
        sync_buffer["recv_dict"] = {}
        for rank in sync_buffer["send_dict"]:
            sync_buffer["recv_dict"][rank] = [torch.empty_like(sync_buffer["send_dict"][rank][0]), torch.empty_like(sync_buffer["send_dict"][rank][1])]

        sync_message_reqs, synced_message = self.aggregator_fn.two_way_sendrecv_with_tags(sync_buffer["send_dict"], sync_buffer["recv_dict"], 
                        force_wait=False, active_neighbors=sync_buffer["active_neighbors"])
       
        # update sync_buffer.
        sync_buffer["sync_reqs"] = sync_message_reqs
        sync_buffer["synced_message"] = synced_message


    def uncompress(self, sync_buffer):
        # wait the sync.
        self.aggregator_fn.complete_wait(sync_buffer["sync_reqs"])

        # uncompress and update.
        for rank in sync_buffer["active_neighbors"]:
            # recover values/indices to the correct device.
            q_values, q_indices = self._uncompress_helper(
                sync_buffer["flatten_params"].buffer.device,
                rank,
                sync_buffer["synced_message"],
                sync_buffer["selected_shapes"],
                sync_buffer["original_shapes"],
            )

            # have (rank)-neighbour sparse param here
            sync_buffer["neighbor_hat_params"][rank].buffer[q_indices] += self.gamma * q_values
            
    
    def _uncompress_helper(
        self,
        _device,
        _rank,
        synced_message,
        selected_shapes,
        original_shapes,
    ):
        # recover the message and the corresponding device.
        values = comm.recover_device(
            synced_message[_rank][0], device=_device
        )
        indices = comm.recover_device(
            synced_message[_rank][1], device=_device
        )

        # deal with unbalanced values/indieces
        q_values, q_indices = self.compressor_fn.uncompress(
            values, indices, selected_shapes[_rank], original_shapes
        )
        return q_values, q_indices



class BiasedQuantizer(object):
    def quantize(self, x, b):
        norm = torch.norm(x)
        delta = sqrt(x.shape[0]) / (2 **(b - 1))
        xi = 1 + delta if delta > 1 else 1 + delta ** 2
        tmp = (2 ** (b - 1)) / norm * torch.abs(x) + torch.randn(x.shape, device=x.device)
        return torch.sign(x) * torch.floor(tmp) * (norm / (2 ** (b - 1)) / xi)
    
    def compress(self, arr, op, quantize_level, is_biased):
        if quantize_level != 32:
            values = self.quantize(arr, quantize_level)
        else:
            values = arr
        return values

    def uncompress(self, arr):
        return arr

class DIMIXQuantizationCompressor(object):
    def __init__(
        self,
        aggregator,
        comm_op,
        comm_device,
        compress_ratio,
        quantize_level,
        is_biased,
        backend,
        use_ipc,
        use_cuda,
        gamma,
        compression_noise,
        **kargs,
    ):
        # assign the common hyper-parameters
        self.aggregator_fn = aggregator
        self.comm_op = comm_op
        self.comm_device = comm_device
        self.compress_ratio = compress_ratio
        self.quantize_level = quantize_level
        self.is_biased = is_biased
        self.backend = backend
        self.use_ipc = use_ipc
        self.use_cuda = use_cuda
        self.gamma = gamma
        self.compression_noise = compression_noise
        self.kargs = kargs
        self.compressor_fn = QuantizationCompressor()

        # define gossip_stream
        if torch.cuda.is_available():
            self.gossip_stream = torch.cuda.current_stream()
            self.use_cuda = True

    def pipeline(self, sync_buffer):
        if len(sync_buffer["active_neighbors"]) > 0:
            if torch.cuda.is_available():
                with torch.cuda.stream(self.gossip_stream):
                    try:
                        self.compress(sync_buffer)
                        self.sync(sync_buffer)
                        self.uncompress(sync_buffer)
                    except RuntimeError as e:
                        print("Error: {}".format(e))
            else:
                self.compress(sync_buffer)
                self.sync(sync_buffer)
                self.uncompress(sync_buffer)

    
    def compress(self, sync_buffer):
        sync_buffer["send_dict"] = {}
        sync_buffer["selected_shapes"] = {}

        for nei in sync_buffer["active_neighbors"]:
            quantized_values = []
            for param in sync_buffer["flatten_params"]:
                _quantized_values = self.compressor_fn.compress(
                    param,
                    self.comm_op,
                    self.quantize_level,
                    self.is_biased,
                )
                quantized_values.append(_quantized_values)

            # flatten selected values/indices.
            flatten_updates = TensorBuffer(quantized_values, self.use_cuda)
            noise = torch.zeros_like(flatten_updates.buffer).uniform_(-self.compression_noise, self.compression_noise)
            flatten_updates.buffer += noise

            # update the local hat variable
            sync_buffer["edge_result"][self.aggregator_fn.rank] = flatten_updates.buffer # local parameter without noise
            sync_buffer["send_dict"][nei] = flatten_updates.buffer

            if self.comm_device == "cpu":
                sync_buffer["send_dict"][nei] = sync_buffer["send_dict"][nei].cpu().pin_memory()
            
            # get n_bits to transmit.
            n_bits = get_n_bits(flatten_updates.buffer) * self.quantize_level / 32
            sync_buffer["n_bits"] += n_bits


    def sync(self, sync_buffer):
        # sync.
        sync_buffer["recv_dict"] = {}
        for rank in sync_buffer["send_dict"]:
            sync_buffer["recv_dict"][rank] = torch.empty_like(sync_buffer["send_dict"][rank])

        sync_message_reqs, synced_message = self.aggregator_fn.two_way_sendrecv(sync_buffer["send_dict"], sync_buffer["recv_dict"], 
                        force_wait=False, active_neighbors=sync_buffer["active_neighbors"])
       
        # update sync_buffer.
        sync_buffer["sync_reqs"] = sync_message_reqs
        sync_buffer["synced_message"] = synced_message


    def uncompress(self, sync_buffer):
        # wait the sync.
        self.aggregator_fn.complete_wait(sync_buffer["sync_reqs"])

        # uncompress and update.
        for rank in sync_buffer["active_neighbors"]:
            # recover correct values/indices.
            q_values = comm.recover_device(
                sync_buffer["synced_message"][rank], device=sync_buffer["flatten_params"].buffer.device
            )

            # have (rank)-neighbour sparse param here
            sync_buffer["edge_result"][rank] = q_values # neighbors' parameter with noise
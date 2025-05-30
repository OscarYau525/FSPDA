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

class TiCoPD(Optimizer):
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
        super(TiCoPD, self).__init__(params, defaults)

        # store the whole training arguments.
        self.conf = conf
        self.use_cuda = conf.on_cuda
        self.theta = conf.theta
        self.eta = conf.eta
        self.random_lap = conf.random_lap
        self.use_compressor_buffer = conf.use_compressor_buffer
        self.theta_ratio = self.theta * conf.lr

        if conf.shared_mask:
           assert conf.one_edge, "shared mask only support one edge random graphs"

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
        self.init_neighbor_hat_params()
        self.consensus_stepsize = conf.consensus_stepsize

        # initialize dual variable lambda
        for groups in self.param_groups:
            groups["lambdas"] = [torch.zeros_like(prm) for prm in groups["params"]]

        # related to sparsification/quantization.
        self.compressor = TiCoPDCompressor(
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
            compression_noise=conf.compression_noise,
            shared_mask=conf.shared_mask
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
        self.neighbor_hat_params = {i: deepcopy(flatten_params) for i in self.neighbors_info} # neighbor's hat in my perspective
        self.local_hat_params = {i: deepcopy(flatten_params) for i in self.neighbors_info} # my hat in neighbor's perspective
        self.neighbor_hat_buffer = {i: deepcopy(flatten_params) for i in self.neighbors_info}

    def __setstate__(self, state):
        super(TiCoPD, self).__setstate__(state)
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


    def step(self, closure=None, **kargs):
        lr = kargs["scheduler"].get_lr()
        self.lr = lr if not lr is None else self.lr
        # self.theta = self.theta_ratio / self.lr
        # start compress/sync.
        active_neighbors = self.sample_random_graph()
        

        # ====== primal update ======

        # Apply the gradients with the weight decay and momentum.
        utils.apply_gradient(
            self.param_groups, self.state, apply_grad_to_model=True
        )
        params, flatten_params = self.get_prm(self.param_groups, self.param_names)

        primal_update_vec = self.get_zeros_prm_buffer(self.param_groups, self.param_names)

        _lambda, flatten_lambda = self.get_lambda(self.param_groups, self.param_names)
        primal_update_vec.buffer += flatten_lambda.buffer

        flatten_params.buffer -= self.lr * primal_update_vec.buffer

        if self.random_lap:
            lap_neighbors = active_neighbors
        else:
            lap_neighbors = self.neighbors_info

        for nei in lap_neighbors:
            flatten_params.buffer -= self.theta_ratio * (self.local_hat_params[nei].buffer - self.neighbor_hat_params[nei].buffer)

        flatten_params.unpack(params)
        # ====== end primal update ======


        # ====== dual update ======
        for nei in lap_neighbors:
            flatten_lambda.buffer += self.eta * (self.local_hat_params[nei].buffer - self.neighbor_hat_params[nei].buffer)
        flatten_lambda.unpack(_lambda)
        # ====== end dual update ======

        params, flatten_params = self.get_prm(self.param_groups, self.param_names)
        
        if self.use_compressor_buffer:
            active_neighbors = self.neighbors_info
        
        self.sync_buffer = {
            "original_shapes": self.shapes,
            "flatten_params": flatten_params,
            "neighbor_hat_params": self.neighbor_hat_params,
            "local_hat_params": self.local_hat_params,
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


        # ====== hat update ======
        self.local_hat_params = self.sync_buffer["local_hat_params"]
        self.neighbor_hat_params = self.sync_buffer["neighbor_hat_params"]
        
        # ====== end hat update ======

        self.n_bits = self.sync_buffer.get("n_bits", 0)
        self.it += 1
        return self.n_bits


"""the entry for TiCoPDCompressor."""


class TiCoPDCompressor(object):
    def __init__(self, **kargs):
        # assign compressor class.
        if "top_k" in kargs["comm_op"] or "random_k" in kargs["comm_op"]:
            self.compressor_fn = TiCoPDSparsificationCompressor(**kargs)
        elif "quantize" in kargs["comm_op"]:
            self.compressor_fn = TiCoPDQuantizationCompressor(**kargs)
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


"""Detailed TiCoPDCompressors, e.g., top-k/random-k, quantization, sign-based quantization."""


class TiCoPDSparsificationCompressor(object):
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
        shared_mask,
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
        self.compression_noise = compression_noise
        self.gamma = gamma
        self.shared_mask = shared_mask
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
                        if self.shared_mask:
                            self.compress_sync_uncompress_sm(sync_buffer)
                        else:
                            self.compress(sync_buffer)
                            self.sync(sync_buffer)
                            self.uncompress(sync_buffer)
                    except RuntimeError as e:
                        print("Error: {}".format(e))
            else:
                if self.shared_mask:
                    self.compress_sync_uncompress_sm(sync_buffer)
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
                n_bits = get_n_bits(flatten_selected_values.buffer) + get_n_bits(flatten_selected_indices.buffer)
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
    

    def compress_sync_uncompress_sm(self, sync_buffer):
        def get_vi(sync_buffer, nei, compressor_fn, comm_op, compress_ratio, is_biased, use_cuda):
            selected_values = []
            selected_indices = []
            for half_param, hat_param in zip(
                sync_buffer["flatten_params"], sync_buffer["local_hat_params"][nei]
            ):
                _selected_values, _selected_indices = compressor_fn.compress(
                    half_param - hat_param,
                    comm_op,
                    compress_ratio,
                    is_biased,
                )
                selected_values.append(_selected_values)
                selected_indices.append(_selected_indices)

            # get selected shapes.
            selected_shapes = [len(_value) for _value in selected_values]

            # flatten selected values/indices.
            flatten_selected_values = TensorBuffer(selected_values, use_cuda)
            flatten_selected_indices = TensorBuffer(selected_indices, use_cuda)
            return flatten_selected_values, flatten_selected_indices, selected_shapes
        
        def get_v_uncompressed(sync_buffer, nei, use_cuda):
            selected_values = []
            for half_param, hat_param in zip(
                sync_buffer["flatten_params"], sync_buffer["local_hat_params"][nei]
            ):
                selected_values.append(half_param - hat_param)

            # get selected shapes.
            selected_shapes = [len(_value) for _value in selected_values]

            # flatten selected values/indices.
            flatten_values = TensorBuffer(selected_values, use_cuda)
            return flatten_values, selected_shapes
        
        sync_buffer["send_dict"] = {}
        sync_buffer["recv_dict"] = {}
        sync_buffer["selected_shapes"] = {}

        for nei in sync_buffer["active_neighbors"]:
            if self.aggregator_fn.rank < nei:
                # smaller rank determines the top-k/rank-k mask
                flatten_selected_values, flatten_selected_indices, selected_shapes = get_vi(sync_buffer, nei, self.compressor_fn, self.comm_op, self.compress_ratio, self.is_biased, self.use_cuda)

                noise = torch.zeros_like(flatten_selected_values.buffer).uniform_(-self.compression_noise, self.compression_noise)
                flatten_selected_values.buffer += noise

                # update the local hat variable
                q_values, q_indices = self.compressor_fn.uncompress(flatten_selected_values.buffer, flatten_selected_indices.buffer, selected_shapes, sync_buffer["original_shapes"])
                sync_buffer["local_hat_params"][nei].buffer[q_indices] += self.gamma * q_values

                sync_buffer["send_dict"][nei] = [flatten_selected_values.buffer, flatten_selected_indices.buffer]

                sync_buffer["selected_shapes"][nei] = selected_shapes
                if self.comm_device == "cpu":
                    sync_buffer["send_dict"][nei][0] = sync_buffer["send_dict"][nei][0].cpu().pin_memory()
                    sync_buffer["send_dict"][nei][1] = sync_buffer["send_dict"][nei][1].cpu().pin_memory()
            

                # sync 1
                reqs = []
                reqs.append( dist.isend(tensor=sync_buffer["send_dict"][nei][0], dst=nei, tag=0) )
                reqs.append( dist.isend(tensor=sync_buffer["send_dict"][nei][1], dst=nei, tag=1) )
                for req in reqs:
                    req.wait()
                
                sync_buffer["recv_dict"][nei] = [torch.empty_like(flatten_selected_values.buffer), flatten_selected_indices.buffer]
                
                # sync 2
                dist.recv(tensor=sync_buffer["recv_dict"][nei][0], src=nei)

                q_values, q_indices = self._uncompress_helper(
                    sync_buffer["flatten_params"].buffer.device,
                    nei,
                    sync_buffer["recv_dict"],
                    sync_buffer["selected_shapes"],
                    sync_buffer["original_shapes"],
                )

                sync_buffer["neighbor_hat_params"][nei].buffer[q_indices] += self.gamma * q_values


            else:
                flatten_selected_values, flatten_selected_indices, selected_shapes = get_vi(sync_buffer, nei, self.compressor_fn, self.comm_op, self.compress_ratio, self.is_biased, self.use_cuda)
                sync_buffer["selected_shapes"][nei] = selected_shapes
                sync_buffer["recv_dict"][nei] = [torch.empty_like(flatten_selected_values.buffer), torch.empty_like(flatten_selected_indices.buffer)]

                # sync 1
                reqs = []
                reqs.append( dist.irecv(tensor=sync_buffer["recv_dict"][nei][0], src=nei, tag=0) )
                reqs.append( dist.irecv(tensor=sync_buffer["recv_dict"][nei][1], src=nei, tag=1) )
                for req in reqs:
                    req.wait()
                
                q_values, q_indices = self._uncompress_helper(
                    sync_buffer["flatten_params"].buffer.device,
                    nei,
                    sync_buffer["recv_dict"],
                    sync_buffer["selected_shapes"],
                    sync_buffer["original_shapes"],
                )

                # have (rank)-neighbour sparse param here
                sync_buffer["neighbor_hat_params"][nei].buffer[q_indices] += self.gamma * q_values


                # update the local hat variable
                flatten_values, selected_shapes = get_v_uncompressed(sync_buffer, nei, self.use_cuda)

                # sync 2
                dist.send(tensor=flatten_values.buffer[q_indices], dst=nei)

                sync_buffer["local_hat_params"][nei].buffer[q_indices] += self.gamma * flatten_values.buffer[q_indices]
                


            # get n_bits to transmit.
            if self.compress_ratio > 0:
                if self.aggregator_fn.rank < nei:
                    n_bits = get_n_bits(flatten_selected_values.buffer) + get_n_bits(flatten_selected_indices.buffer)
                else:
                    n_bits = get_n_bits(flatten_selected_values.buffer)
            else:
                # no sparsification is applied
                n_bits = get_n_bits(flatten_selected_values.buffer)
            sync_buffer["n_bits"] += n_bits



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

class TiCoPDQuantizationCompressor(object):
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
            for half_param, hat_param in zip(
                sync_buffer["flatten_params"], sync_buffer["local_hat_params"][nei]
            ):
                _quantized_values = self.compressor_fn.compress(
                    half_param - hat_param,
                    self.comm_op,
                    self.quantize_level,
                    self.is_biased,
                )
                quantized_values.append(_quantized_values)

            # flatten selected values/indices.
            flatten_updates = TensorBuffer(quantized_values, self.use_cuda)

            # update the local hat variable
            noise = torch.zeros_like(flatten_updates.buffer).uniform_(-self.compression_noise, self.compression_noise)
            flatten_updates.buffer += noise
            sync_buffer["local_hat_params"][nei].buffer += self.gamma * ( flatten_updates.buffer )

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
            sync_buffer["neighbor_hat_params"][rank].buffer += self.gamma * q_values
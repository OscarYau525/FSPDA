# -*- coding: utf-8 -*-
from copy import deepcopy

import torch
from torch.optim.optimizer import Optimizer, required

import pcode.optim.utils as utils
import pcode.utils.communication as comm
from pcode.utils.sparsification import (
    get_n_bits,
    SparsificationCompressor,
)
from pcode.utils.tensor_buffer import TensorBuffer
from pcode.create_dataset import load_data_batch
import numpy as np

import torch.distributed as dist


class Di_CS_GT(Optimizer):
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
        super(Di_CS_GT, self).__init__(params, defaults)

        # define the aggregator.
        # define the aggregator.
        self.world_size = conf.n_mpi_process
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

        self.use_cuda = conf.on_cuda
        self.gamma = conf.gamma
        self.B = conf.B_connected
        self.edge_fraction = conf.edge_fraction
        self.B_round_active_neighbours = set()
        self.it = 0

        # initialize gradient tracker
        for groups in self.param_groups:
            groups["params_prev"] = [torch.zeros_like(prm) for prm in groups["params"]]
            groups["surplus_y"] = [torch.zeros_like(prm) for prm in groups["params"]]
            groups["surplus_y_mem"] = [torch.zeros_like(prm) for prm in groups["params"]]
            groups["grad_tracker"] = [torch.zeros_like(prm) for prm in groups["params"]]
            groups["grad_prev"] = [torch.zeros_like(prm) for prm in groups["params"]]

        self.model = model
        self.model_w = deepcopy(
            model.module if "DataParallel" == model.__class__.__name__ else model
        )
        self.model_w.zero_grad()
        
        # store the whole training arguments.
        self.conf = conf
        
        
        

        # define param names and init model_hat.
        self.param_names = list(
            enumerate([group["name"] for group in self.param_groups])
        )
        # self.init_neighbor_hat_params()
        # self.init_neighbor_hat_gt()
        self.consensus_stepsize = conf.consensus_stepsize

        # related to sparsification/quantization.
        self.compressor = Di_CS_DirectedGraph_Sparsifier(
            aggregator=self.aggregator,
            comm_device=self.conf.comm_device,
            compress_ratio=conf.compress_ratio,
            is_biased=conf.is_biased,
            backend=conf.backend,
            one_edge=conf.one_edge,
            use_ipc=conf.use_ipc,
            use_cuda=self.use_cuda,
            edge_fraction=self.edge_fraction,
            world_size=self.world_size,
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
        super(Di_CS_GT, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    def inference(self, model, criterion, _input, _target):
        output = model(_input)
        loss = criterion(output, _target)
        return loss

    
    def get_buffer(self, param_groups, param_names, param_tag):
        # param_tag example
        # prm x: "params"
        # surplus y: "surplus_y"
        data = []
        for idx, _ in param_names:
            _data = param_groups[idx][param_tag][0]
            if _data is not None:
                data.append(_data)
        flatten_params = TensorBuffer(data, self.use_cuda)
        return data, flatten_params
    
    def get_zeros_prm_buffer(self, param_groups, param_names):
        _, flatten_params = self.get_buffer(param_groups, param_names, "params")
        flatten_params.buffer = torch.zeros_like(flatten_params.buffer)
        return flatten_params
        

    def di_cs_gossip(self, active_neighbors, edge_result, flatten_local_prms):
        renormalization_sum = self.get_zeros_prm_buffer(self.param_groups, self.param_names)
        for nei in active_neighbors:
            if nei in edge_result:
                _, indices = edge_result[nei]
                renormalization_sum.buffer[indices] += self.neighbors_info[nei]
        renormalization_sum.buffer += self.neighbors_info[self.rank]
        
        weighted_avg = self.get_zeros_prm_buffer(self.param_groups, self.param_names)
        for nei in active_neighbors:
            if nei in edge_result:
                sparse_values, indices = edge_result[nei]
                weighted_avg.buffer[indices] += self.neighbors_info[nei] * sparse_values
        weighted_avg.buffer += self.neighbors_info[self.rank] * flatten_local_prms.buffer

        weighted_avg.buffer /= renormalization_sum.buffer
        return weighted_avg
    
    
    def step(self, **kwargs):
        lr = kwargs["scheduler"].get_lr()
        self.lr = lr if not lr is None else self.lr
        self.n_bits = 0

        # ======================= CONSIDER THIS AS START OF NEW ITERATION =======================

        params, flatten_params = self.get_buffer(self.param_groups, self.param_names, "params")

        # draw new \xi
        self.compressor.prepare_round(flatten_params, self.it)

        self.n_bits += self.sync_buffer.get("n_bits", 0)
        # start compress/sync.
        self.sync_buffer = {
            "original_shapes": self.shapes,
            "flatten_params": flatten_params,
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

        for groups in self.param_groups:
            for prm, prm_prev in zip(groups["params"], groups["params_prev"]):
                prm_prev.data = prm.data.detach().clone()

        #  ==== update model parameters from aggregates ==== 
        
        flatten_params.buffer = self.di_cs_gossip(self.compressor.active_neighbors, 
                                                    self.sync_buffer["edge_result"], 
                                                    flatten_params).buffer
        flatten_params.unpack(params)

        if self.it % self.B == self.B - 1:
            # apply + gamma * y - lr * g
            for groups in self.param_groups:
                for y, g, prm in zip(groups["surplus_y_mem"], groups["grad_tracker"], groups["params"]):
                    prm.data += self.gamma * y.data.detach().clone() - self.lr * g.data.detach().clone()
            
        # swap edge mask 
        self.compressor.set_edge_mask_from_edge_result( self.sync_buffer["edge_result"],
                                                        self.sync_buffer["recv_selected_shapes"],
                                                        self.sync_buffer["original_shapes"] )

        #  ==== do aggregate for surplus variable ==== 
        surplus_y, flatten_surplus_y = self.get_buffer(self.param_groups, self.param_names, "surplus_y")

        self.n_bits += self.sync_buffer.get("n_bits", 0)
        # start compress/sync.
        self.sync_buffer = {
            "original_shapes": self.shapes,
            "flatten_params": flatten_surplus_y,
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

        #  ==== update dual variable y from aggregates ==== 
        utils.join_thread(self.helper_thread)
        _, flatten_params = self.get_buffer(self.param_groups, self.param_names, "params")
        _, flatten_param_prev = self.get_buffer(self.param_groups, self.param_names, "params_prev")
        surplus_y, flatten_surplus_y = self.get_buffer(self.param_groups, self.param_names, "surplus_y")
        flatten_surplus_y.buffer = self.di_cs_gossip(self.compressor.active_neighbors, 
                                                    self.sync_buffer["edge_result"], 
                                                    flatten_surplus_y).buffer
        flatten_surplus_y.buffer -= flatten_params.buffer - flatten_param_prev.buffer
        flatten_surplus_y.unpack(surplus_y)

        if self.it % self.B == 0:
            for groups in self.param_groups:
                for y_mem, y in zip(groups["surplus_y_mem"], groups["surplus_y"]):
                    y_mem.data = y.data.detach().clone()
    
        # ==== Gossip update for grad_tracker ====
        for nei in self.compressor.active_neighbors:
            self.B_round_active_neighbours.add(nei)
        
        if self.it % self.B == self.B - 1:
            grad_tracker, flatten_grad_tracker = self.get_buffer(self.param_groups, self.param_names, "grad_tracker")
            # aggregate grad_tracker on B_round_active_neighbours (uncompressed)
            self.n_bits += get_n_bits(flatten_grad_tracker.buffer) * len(self.B_round_active_neighbours)
            flatten_grad_tracker.buffer = self.aggregator._agg_custom_neighbor(self.B_round_active_neighbours, flatten_grad_tracker.buffer, "weighted")
            flatten_grad_tracker.unpack(grad_tracker)
            self.B_round_active_neighbours = set()

            # update gradient tracker
            for groups in self.param_groups:
                for g, prm, gp in zip(groups["grad_tracker"], groups["params"], groups["grad_prev"]):
                    g.data += prm.grad.data.detach().clone() - gp.data.detach().clone()
                    gp.data = prm.grad.data.detach().clone()
                
        self.it += 1

        return self.n_bits

class Di_CS_DirectedGraph_Sparsifier(object):
    def __init__(
        self,
        aggregator,
        comm_device,
        compress_ratio,
        is_biased,
        backend,
        use_ipc,
        use_cuda,
        world_size,
        **kwargs,
    ):
        # assign the common hyper-parameters
        self.aggregator_fn = aggregator
        self.comm_device = comm_device
        self.compress_ratio = compress_ratio
        self.is_biased = is_biased
        self.backend = backend
        self.use_ipc = use_ipc
        self.use_cuda = use_cuda
        self.world_size = world_size
        self.kwargs = kwargs
        self.compressor_fn = SparsificationCompressor()

        # define gossip_stream
        if torch.cuda.is_available():
            self.gossip_stream = torch.cuda.current_stream()

    def pipeline(self, sync_buffer):
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
    

    def sample_random_graph(self, flatten_params, it):
        if self.kwargs["one_edge"]:
            # implemented 1 edge random graph
            initiator = int(it % self.world_size)
            if self.aggregator_fn.rank == initiator:
                # I am initiator this round
                rand_neigh_idx = int(np.floor(np.random.rand() * len(self.aggregator_fn.neighbor_ranks)))
                rand_neigh = torch.tensor(self.aggregator_fn.neighbor_ranks[rand_neigh_idx], dtype=torch.int64)
            else:
                rand_neigh = torch.tensor(0)
            
            dist.broadcast(rand_neigh, src=initiator)
            if self.aggregator_fn.rank == initiator:
                active_neighbors = [rand_neigh.item()]
                edge_masks = {rand_neigh.item(): self.prepare_compress(flatten_params)}
            elif self.aggregator_fn.rank == rand_neigh.item():
                active_neighbors = [initiator]
                edge_masks = {initiator: self.prepare_compress(flatten_params)}
            else:
                active_neighbors = []
                edge_masks = {}
            
            # print(self.aggregator_fn.rank, initiator, rand_neigh)
        else:
            # edge_activation decides whether an edge is active or not
            edge_activation = {nei: torch.rand(1) for nei in self.aggregator_fn.neighbor_ranks}
            edge_activation = self.aggregator_fn.one_way_consensus(edge_activation, force_wait=True)

            active_neighbors = [nei for nei in edge_activation if edge_activation[nei] <= self.kargs["edge_fraction"]]
            edge_masks = {nei: self.prepare_compress(flatten_params) for nei in active_neighbors}
        return active_neighbors, edge_masks
    
    
    def prepare_round(self, flatten_params, it):
        self.active_neighbors, self.edge_masks = self.sample_random_graph(flatten_params, it)
        
    def set_edge_mask_from_edge_result(self, edge_result, recv_selected_shapes, original_shapes):
        for nei in self.active_neighbors:
            if nei in edge_result:
                _, indices_buffer = edge_result[nei]
                self.edge_masks[nei] = self.buffer_idx_to_shaped_idx(indices_buffer, recv_selected_shapes[nei], original_shapes)
                

    def prepare_compress(self, flatten_params):
        selected_values, selected_indices = [], []

        for param in flatten_params:
            _selected_values, _selected_indices = self.compressor_fn.get_random_k(
                param,
                self.compress_ratio,
                self.is_biased,
            )
            selected_values.append(_selected_values)
            selected_indices.append(_selected_indices)

        return selected_indices
    
    def compress(self, sync_buffer):
        sync_buffer["send_dict"] = {}
        sync_buffer["selected_shapes"] = {}
        sync_buffer["recv_selected_shapes"] = {}
        for nei in self.active_neighbors:
            selected_indices = self.edge_masks[nei]
            selected_values = []

            for param, _selected_indices in zip(sync_buffer["flatten_params"], selected_indices):
                _selected_values = param.view(-1)[_selected_indices]
                selected_values.append(_selected_values)
            
            selected_shapes = torch.tensor([len(_value) for _value in selected_values])

            flatten_selected_values = TensorBuffer(selected_values, self.use_cuda)
            flatten_selected_indices = TensorBuffer(selected_indices, self.use_cuda)
            

            sync_buffer["send_dict"][nei] = [
                flatten_selected_values.buffer, 
                flatten_selected_indices.buffer, 
                selected_shapes
            ]


            sync_buffer["selected_shapes"][nei] = selected_shapes
            if self.comm_device == "cpu":
                sync_buffer["send_dict"][nei] = sync_buffer["send_dict"][nei].cpu().pin_memory()
            
            # get n_bits to transmit.
            n_bits = get_n_bits(flatten_selected_values.buffer) + get_n_bits(
                flatten_selected_indices.buffer
            )
            sync_buffer["n_bits"] += n_bits

    def sync(self, sync_buffer):
        # sync.
        sync_buffer["recv_dict"] = {}
        for rank in sync_buffer["send_dict"]:
            sync_buffer["recv_dict"][rank] = [torch.empty_like(sync_buffer["send_dict"][rank][i]) for i in range(len(sync_buffer["send_dict"][rank]))]

        sync_message_reqs, synced_message = self.aggregator_fn.two_way_sendrecv_with_tags(sync_buffer["send_dict"], sync_buffer["recv_dict"], 
                                        force_wait=False, active_neighbors=self.active_neighbors)
       
        # update sync_buffer.
        sync_buffer["sync_reqs"] = sync_message_reqs
        sync_buffer["synced_message"] = synced_message
        sync_buffer["sycned_message_size"] = {nei: len(sync_buffer["send_dict"][nei]) for nei in sync_buffer["send_dict"]}

    def uncompress(self, sync_buffer):
        # wait the sync.
        self.aggregator_fn.complete_wait(sync_buffer["sync_reqs"])

        # uncompress and update.
        for rank in self.active_neighbors:
            # recover values/indices to the correct device.
            q_values, q_indices, nei_selected_shape = self._uncompress_helper(
                sync_buffer["flatten_params"].buffer.device,
                rank,
                sync_buffer["synced_message"],
                sync_buffer["selected_shapes"],
                sync_buffer["original_shapes"],
            )

            # have (rank)-neighbour sparse param here
            sync_buffer["edge_result"][rank] = (q_values, q_indices) # can be used directly on buffer
            sync_buffer["recv_selected_shapes"][rank] = nei_selected_shape

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
        nei_selected_shape = comm.recover_device(
            synced_message[_rank][2], device=_device
        )

        # deal with unbalanced values/indieces
        q_values, q_indices = self.compressor_fn.uncompress(
            values, indices, selected_shapes[_rank], original_shapes
        )
        return q_values, q_indices, nei_selected_shape
    
    def buffer_idx_to_shaped_idx(self, q_indices, selected_shapes, original_shapes):
        # apply each param.
        sync_pointer = 0
        shape_sum = 0

        shaped_indices = []
        for idx, n_sparse_value in enumerate(selected_shapes):
            indices = q_indices[sync_pointer : sync_pointer + int(n_sparse_value)] - shape_sum
            shaped_indices += [indices.long()]

            sync_pointer += int(n_sparse_value)
            shape_sum += original_shapes[idx][1]

        return shaped_indices

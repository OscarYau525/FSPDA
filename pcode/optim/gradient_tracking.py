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
import numpy as np
import torch.distributed as dist

class GT(Optimizer): # current implementation uses local updates
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
        super(GT, self).__init__(params, defaults)

        # define the aggregator.
        self.rank = conf.graph.rank
        self.world_size = conf.n_mpi_process
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

        self.edge_prob = conf.edge_prob
        self.it = 0

        # initialize dual variable lambda and primal memory
        for groups in self.param_groups:
            groups["gt"] = [torch.zeros_like(prm) for prm in groups["params"]]
            groups["prev_grad"] = [torch.zeros_like(prm) for prm in groups["params"]]


        self.model = model
        self.model_prev = deepcopy(
            model.module if "DataParallel" == model.__class__.__name__ else model
        )
        self.model_prev.zero_grad()


        # store the whole training arguments.
        self.conf = conf
        self.consensus_stepsize = conf.gamma
        self.use_cuda = conf.on_cuda

        if self.conf.lr_change_epochs is not None:
            self.lr_schedule = [ int(ep) for ep in self.conf.lr_change_epochs.split(",") ]
        
        # define param names and init model_hat.
        self.param_names = list(
            enumerate([group["name"] for group in self.param_groups])
        )

        # related to sparsification/quantization.
        self.compressor = RandomGraph_Sparsifier(
            aggregator=self.aggregator,
            comm_device=self.conf.comm_device,
            compress_ratio=conf.compress_ratio,
            is_biased=conf.is_biased,
            backend=conf.backend,
            use_ipc=conf.use_ipc,
            edge_prob=self.edge_prob,
            one_edge=conf.one_edge,
            use_cuda=self.use_cuda,
            world_size=self.world_size,
            compression_noise=conf.compression_noise
        )

        # define auxilary functions.
        self.helper_thread = None
        self.sync_buffer = {}
        self.n_bits = 0
        self.first_step = True

        _, self.shapes = comm.get_data(
            self.param_groups, self.param_names, is_get_grad=False
        )
    
    
    def __setstate__(self, state):
        super(GT, self).__setstate__(state)
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

    def get_gt(self, param_groups, param_names):
        return self.get_prm(param_groups, param_names, tag="gt")

    def get_prev_grad(self, param_groups, param_names):
        return self.get_prm(param_groups, param_names, tag="prev_grad")
    
    def get_zeros_prm_buffer(self, param_groups, param_names):
        _, flatten_params = self.get_prm(param_groups, param_names)
        flatten_params.buffer = torch.zeros_like(flatten_params.buffer)
        return flatten_params

    
    def L2_regularize_grad(self, model):
        weight_decay = self.conf.weight_decay
        if weight_decay > 0:
            for key, prm in model.named_parameters():
                if not "bn" in key and weight_decay != 0:
                    prm.grad.data = prm.grad.data.detach().clone() + weight_decay * prm.data.detach().clone()
      
    
    def step(self, **kwargs):
        self.n_bits = 0
        lr = kwargs["scheduler"].get_lr()
        self.lr = lr if not lr is None else self.lr

        self.L2_regularize_grad(self.model)
        params, flatten_params = self.get_prm(self.param_groups, self.param_names)
            
        # draw new random graph \xi^t
        self.compressor.prepare_round(flatten_params, self.it)

         #  ==== do aggregates ==== 
        params, flatten_params = self.get_prm(self.param_groups, self.param_names)
        _, flatten_gt = self.get_gt(self.param_groups, self.param_names)

        # start compress/sync.
        self.sync_buffer = {
            "original_shapes": self.shapes,
            "flatten_params": flatten_params,
            "flatten_gt": flatten_gt,
            "edge_result": {},
            "n_bits": 0
        }

        self.compressor.pipeline(self.sync_buffer)
        utils.join_thread(self.helper_thread)
        self.n_bits += self.sync_buffer.get("n_bits", 0)
        #  ==== end of aggregates, results stored in self.sync_buffer["edge_result"][nei] ==== 

        params, flatten_params = self.get_prm(self.param_groups, self.param_names)
        for nei in self.sync_buffer["edge_result"]:
            _, prm_values, indices = self.sync_buffer["edge_result"][nei]
            _, my_prm_values, _ = self.sync_buffer["send_dict"][nei]
            flatten_params.buffer[ indices ] -= self.consensus_stepsize * (my_prm_values - prm_values)

        flatten_params.unpack(params)

        for groups in self.param_groups:
            for gt, prm in zip(groups["gt"], groups["params"]):
                prm.data -= self.lr * gt.data.detach().clone()

        # GT update

        grad_tracker, flatten_gt = self.get_gt(self.param_groups, self.param_names)
        for nei in self.sync_buffer["edge_result"]:
            gt_values, _, indices = self.sync_buffer["edge_result"][nei]
            my_gt_values, _, _ = self.sync_buffer["send_dict"][nei]
            flatten_gt.buffer[ indices ] -= self.consensus_stepsize * (my_gt_values - gt_values)
        flatten_gt.unpack(grad_tracker)

        for groups in self.param_groups:
            for gt, prm, pgrd in zip(groups["gt"], groups["params"], groups["prev_grad"]):
                gt.data += prm.grad.data.detach().clone() - pgrd.data.detach().clone()
                pgrd.data = prm.grad.data.detach().clone()


        self.it += 1
        return self.n_bits

class RandomGraph_Sparsifier(object):
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
        compression_noise,
        **kargs,
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
        self.compression_noise = compression_noise
        self.kargs = kargs
        self.compressor_fn = SparsificationCompressor()

        # define gossip_stream
        if torch.cuda.is_available():
            self.gossip_stream = torch.cuda.current_stream()
            self.use_cuda = True

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
        if self.kargs["one_edge"]:
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

            active_neighbors = [nei for nei in edge_activation if edge_activation[nei] <= self.kargs["edge_prob"]]
            edge_masks = {nei: self.prepare_compress(flatten_params) for nei in active_neighbors}
        return active_neighbors, edge_masks


    
    def prepare_round(self, flatten_params, it):
        self.active_neighbors, self.edge_masks = self.sample_random_graph(flatten_params, it)

        n_layers = len(flatten_params)
        # for communication without reindexing we send indices layer by layer
        self.comm_edge_masks = [{nei: self.edge_masks[nei][i] for nei in self.active_neighbors}
                                                                for i in range(n_layers)]

        # two node agrees on the same mask
        for layer_j in range(n_layers):
            self.comm_edge_masks[layer_j] = self.aggregator_fn.one_way_consensus(self.comm_edge_masks[layer_j], 
                                            force_wait=True, active_neighbors=self.active_neighbors)
       
        self.edge_masks = {nei: [self.comm_edge_masks[j][nei] for j in range(n_layers)] for nei in self.active_neighbors}

        

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
        for nei in self.active_neighbors:
            selected_indices = self.edge_masks[nei]
            selected_values = []
            selected_gt_values = []

            for param, prev_param, _selected_indices in zip(sync_buffer["flatten_params"], sync_buffer["flatten_gt"], selected_indices):
                _selected_values = param.view(-1)[_selected_indices]
                selected_values.append(_selected_values)

                _selected_gt_values = prev_param.view(-1)[_selected_indices]
                selected_gt_values.append(_selected_gt_values)
            
            selected_shapes = [len(_value) for _value in selected_values]

            flatten_selected_values = TensorBuffer(selected_values, self.use_cuda)
            flatten_selected_gt_values = TensorBuffer(selected_gt_values, self.use_cuda)
            flatten_selected_indices = TensorBuffer(selected_indices, self.use_cuda)

            noise = torch.zeros_like(flatten_selected_values.buffer).uniform_(-self.compression_noise, self.compression_noise)
            flatten_selected_values.buffer += noise

            sync_buffer["send_dict"][nei] = [flatten_selected_gt_values.buffer, flatten_selected_values.buffer, flatten_selected_indices.buffer]

            sync_buffer["selected_shapes"][nei] = selected_shapes
            if self.comm_device == "cpu":
                sync_buffer["send_dict"][nei][0] = sync_buffer["send_dict"][nei][0].cpu().pin_memory()
                sync_buffer["send_dict"][nei][1] = sync_buffer["send_dict"][nei][1].cpu().pin_memory()
                sync_buffer["send_dict"][nei][2] = sync_buffer["send_dict"][nei][2].cpu().pin_memory()
            
            # get n_bits to transmit.
            if self.compress_ratio > 0:
                if self.aggregator_fn.rank > nei:
                    n_bits = get_n_bits(flatten_selected_values.buffer) * 2 + get_n_bits(
                        flatten_selected_indices.buffer
                    )
                else:
                    n_bits = get_n_bits(flatten_selected_values.buffer) * 2
            else:
                # no sparsification is applied
                n_bits = get_n_bits(flatten_selected_values.buffer) * 2
            sync_buffer["n_bits"] += n_bits

    def sync(self, sync_buffer):
        # sync.
        sync_buffer["recv_dict"] = {}
        for rank in sync_buffer["send_dict"]:
            sync_buffer["recv_dict"][rank] = [torch.empty_like(sync_buffer["send_dict"][rank][0]), 
                                              torch.empty_like(sync_buffer["send_dict"][rank][1]),
                                              torch.empty_like(sync_buffer["send_dict"][rank][2])]

        sync_message_reqs, synced_message = self.aggregator_fn.two_way_sendrecv_with_tags(sync_buffer["send_dict"], sync_buffer["recv_dict"], 
                        force_wait=False, active_neighbors=self.active_neighbors)
       
        # update sync_buffer.
        sync_buffer["sync_reqs"] = sync_message_reqs
        sync_buffer["synced_message"] = synced_message

    def uncompress(self, sync_buffer):
        # wait the sync.
        self.aggregator_fn.complete_wait(sync_buffer["sync_reqs"])

        # uncompress and update.
        for rank in self.active_neighbors:
            # recover values/indices to the correct device.
            prev_q_values, q_values, q_indices = self._uncompress_helper(
                sync_buffer["flatten_params"].buffer.device,
                rank,
                sync_buffer["synced_message"],
                sync_buffer["selected_shapes"],
                sync_buffer["original_shapes"],
            )

            # have (rank)-neighbour sparse param here
            sync_buffer["edge_result"][rank] = (prev_q_values, q_values, q_indices) # can be used directly on buffer
        
    def _uncompress_helper(
        self,
        _device,
        _rank,
        synced_message,
        selected_shapes,
        original_shapes,
    ):
        # recover the message and the corresponding device.
        gt_values = comm.recover_device(
            synced_message[_rank][0], device=_device
        )
        values = comm.recover_device(
            synced_message[_rank][1], device=_device
        )
        indices = comm.recover_device(
            synced_message[_rank][2], device=_device
        )

        # deal with unbalanced values/indieces
        q_values, q_indices = self.compressor_fn.uncompress(
            values, indices, selected_shapes[_rank], original_shapes
        )
        gt_q_values, q_indices = self.compressor_fn.uncompress(
            gt_values, indices, selected_shapes[_rank], original_shapes
        )
        return gt_q_values, q_values, q_indices
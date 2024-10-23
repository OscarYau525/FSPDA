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
from random import random
import torch.distributed as dist

class SPARQ_SGD(Optimizer):
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
        super(SPARQ_SGD, self).__init__(params, defaults)

        # store the whole training arguments.
        self.conf = conf
        self.c_init = conf.c_init
        self.c_incre = conf.c_incre
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
        self.consensus_stepsize = conf.consensus_stepsize

        # related to sparsification/quantization.
        self.compressor = RandomNodeActivation_Sparsifier(
            aggregator=self.aggregator,
            comm_op=conf.comm_op,
            comm_device=self.conf.comm_device,
            compress_ratio=conf.compress_ratio,
            quantize_level=conf.quantize_level,
            is_biased=conf.is_biased,
            backend=conf.backend,
            use_ipc=conf.use_ipc,
            rank=self.rank,
            use_cuda=self.use_cuda
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
        super(SPARQ_SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    
    def step(self, closure=None, **kwargs):
        lr = kwargs["scheduler"].get_lr()
        self.lr = lr if not lr is None else self.lr
        self.c = self.c_init + self.conf.epoch_ // 10 * self.c_incre # as mentioned in the paper: initialized to 2.0 and increases by 1.0 after every 10 epochs
        # Apply the gradients with the weight decay and momentum.
        utils.apply_gradient(
            self.param_groups, self.state, apply_grad_to_model=True
        )

        # recover current params and hat_params
        params, flatten_params, flatten_hat_params = utils.recover_params(
            param_groups=self.param_groups,
            param_names=self.param_names,
            rank=self.rank,
            neighbor_hat_params=self.neighbor_hat_params,
            get_hat_params=True,
            use_cuda=self.use_cuda
        )

        # check event 
        hat_error = torch.norm(flatten_params.buffer - self.neighbor_hat_params[self.rank].buffer, p="fro")**2
        active = hat_error > self.c * self.lr**2

        # start compress/sync.
        self.sync_buffer = {
            "original_shapes": self.shapes,
            "flatten_params": flatten_params,
            "flatten_hat_params": flatten_hat_params,
            "edge_result": {},
            "n_bits": 0,
            "active": active,
        }

        # communicate the compressed hat variables.
        self.helper_thread = utils.HelperThread(
            name=f"_thread_at_epoch_{self.conf.epoch_}.compress",
            func=self.compressor.pipeline,
            # the arguments below will be feeded into the `func`.
            sync_buffer=self.sync_buffer,
            neighbors_info=self.neighbors_info,
        )
        self.helper_thread.start()
        utils.join_thread(self.helper_thread)


        # update neighbor_hat_params[self.rank]
        if active:
            q_values, q_indices = self.sync_buffer["edge_result"][self.rank]
            self.neighbor_hat_params[self.rank].buffer[q_indices] += q_values
        
        # update neighbor_hat_params[nei]
        for nei in self.compressor.active_neighbors:
            q_values, q_indices = self.sync_buffer["edge_result"][nei]
            weight = self.neighbors_info[nei]
            self.neighbor_hat_params["memory"].buffer[q_indices] += weight * q_values

        # get updated flatten params.
        flatten_params.buffer += self.consensus_stepsize * (
            self.neighbor_hat_params["memory"].buffer - self.neighbor_hat_params[self.rank].buffer
        )
       
        # update the local model.
        flatten_params.unpack(params)
        
        self.it += 1
        self.n_bits = self.sync_buffer["n_bits"]
        return self.n_bits


class RandomNodeActivation_Sparsifier(object):
    def __init__(
        self,
        aggregator,
        comm_device,
        compress_ratio,
        is_biased,
        backend,
        use_ipc,
        use_cuda,
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
        self.kargs = kargs
        self.compressor_fn = SparsificationCompressor()

        # define gossip_stream
        if torch.cuda.is_available():
            self.gossip_stream = torch.cuda.current_stream()
            self.use_cuda = True

    def pipeline(self, sync_buffer, neighbors_info):
        if torch.cuda.is_available():
            with torch.cuda.stream(self.gossip_stream):
                try:
                    self.compress(sync_buffer, neighbors_info)
                    self.sync(sync_buffer)
                    self.uncompress(sync_buffer)
                except RuntimeError as e:
                    print("Error: {}".format(e))
        else:
            self.compress(sync_buffer, neighbors_info)
            self.sync(sync_buffer)
            self.uncompress(sync_buffer)
    

    def prepare_compress(self, sync_buffer):
        selected_values, selected_indices = [], []

        for half_param, hat_param in zip(sync_buffer["flatten_params"], sync_buffer["flatten_hat_params"]):
            _selected_values, _selected_indices = self.compressor_fn.get_random_k(
                half_param - hat_param,
                self.compress_ratio,
                self.is_biased,
            )
            selected_values.append(_selected_values)
            selected_indices.append(_selected_indices)

        return selected_values, selected_indices



    def compress(self, sync_buffer, neighbors_info):
        self.active = sync_buffer["active"]
        if self.active:
            self.masked_values, self.masks = self.prepare_compress(sync_buffer)

        if self.active:
            selected_shapes = [len(_value) for _value in self.masked_values]

            flatten_selected_values = TensorBuffer(self.masked_values, self.use_cuda)
            flatten_selected_indices = TensorBuffer(self.masks, self.use_cuda)
            if self.comm_device == "cpu":
                send_message = [flatten_selected_values.buffer.cpu().pin_memory(), flatten_selected_indices.buffer.cpu().pin_memory()]
            else:
                send_message = [flatten_selected_values.buffer, flatten_selected_indices.buffer]

        sync_buffer["send_dict"] = {}
        if self.active:
            for nei in neighbors_info:

                sync_buffer["send_dict"][nei] = send_message

                # get n_bits to transmit.
                n_bits = get_n_bits(flatten_selected_values.buffer) + get_n_bits(
                    flatten_selected_indices.buffer
                )
                sync_buffer["n_bits"] += n_bits


        # check if neighbors are active
        if self.active:
            send_dict = {nei: torch.tensor([int(self.active), send_message[0].shape[0], *selected_shapes], dtype=torch.int64) for nei in neighbors_info}
        else:
            send_dict = {nei: torch.zeros(2 + len(sync_buffer["flatten_params"]), dtype=torch.int64) for nei in neighbors_info}
        recv_dict = {nei: torch.zeros(2 + len(sync_buffer["flatten_params"]), dtype=torch.int64) for nei in neighbors_info}
        _, recv_dict = self.aggregator_fn.one_way_sendrecv(send_dict, recv_dict, force_wait=True)
        self.active_neighbors = {nei: msg[1] for nei, msg in recv_dict.items() if msg[0].item() == 1}
        sync_buffer["selected_shapes"] = {nei: msg[2:] for nei, msg in recv_dict.items() if msg[0].item() == 1}


    def sync(self, sync_buffer):
        # sync.
        sync_buffer["recv_dict"] = {}
        for rank in self.active_neighbors:
            recv_len = self.active_neighbors[rank]
            sync_buffer["recv_dict"][rank] = [torch.empty(recv_len, dtype=torch.float32), torch.empty(recv_len, dtype=torch.int64)]

        sync_message_reqs, synced_message = self.aggregator_fn.one_way_sendrecv_with_tags(sync_buffer["send_dict"], sync_buffer["recv_dict"], 
                        force_wait=False)
       
        # update sync_buffer.
        sync_buffer["sync_reqs"] = sync_message_reqs
        sync_buffer["synced_message"] = synced_message

    def uncompress(self, sync_buffer):
        # wait the sync.
        self.aggregator_fn.complete_wait(sync_buffer["sync_reqs"])

        # uncompress and update.
        for rank in self.active_neighbors:
            # recover values/indices to the correct device.
            q_values, q_indices = self._uncompress_helper(
                sync_buffer["flatten_params"].buffer.device,
                rank,
                sync_buffer["synced_message"],
                sync_buffer["selected_shapes"],
                sync_buffer["original_shapes"],
            )

            # have (rank)-neighbour sparse param here
            sync_buffer["edge_result"][rank] = (q_values, q_indices) # can be used directly on buffer
            
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

"""the entry for CHOCOCompressor."""


class CHOCOCompressor(object):
    def __init__(self, **kargs):
        # assign compressor class.
        if "top_k" in kargs["comm_op"] or "random_k" in kargs["comm_op"]:
            self.compressor_fn = CHOCOSparsificationCompressor(**kargs)
        elif "quantize" in kargs["comm_op"]:
            self.compressor_fn = CHOCOQuantizationCompressor(**kargs)
        elif "sign" in kargs["comm_op"]:
            self.compressor_fn = CHOCOSignCompressor(**kargs)
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


"""Detailed CHOCOCompressors, e.g., top-k/random-k, quantization, sign-based quantization."""


class CHOCOSparsificationCompressor(object):
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
        self.kargs = kargs
        self.compressor_fn = SparsificationCompressor()

        # define gossip_stream
        if torch.cuda.is_available():
            if self.comm_device == "cpu":
                self.gossip_stream = torch.cuda.current_stream()
            else:
                self.gossip_stream = torch.cuda.current_stream()

    def pipeline(self, sync_buffer, neighbor_hat_params, neighbors_info):
        if torch.cuda.is_available():
            with torch.cuda.stream(self.gossip_stream):
                try:
                    self.compress(sync_buffer)
                    self.sync(sync_buffer)
                    self.uncompress(sync_buffer, neighbor_hat_params, neighbors_info)
                except RuntimeError as e:
                    print("Error: {}".format(e))
        else:
            self.compress(sync_buffer)
            self.sync(sync_buffer)
            self.uncompress(sync_buffer, neighbor_hat_params, neighbors_info)

    def compress(self, sync_buffer):
        selected_values, selected_indices = [], []

        for half_param, hat_param in zip(
            sync_buffer["flatten_params"], sync_buffer["flatten_hat_params"]
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
        flatten_selected_values = TensorBuffer(selected_values)
        flatten_selected_indices = TensorBuffer(selected_indices)

        # get n_bits to transmit.
        n_bits = get_n_bits(flatten_selected_values.buffer) + get_n_bits(
            flatten_selected_indices.buffer
        )

        # update shared dict.
        sync_buffer["selected_shapes"] = selected_shapes
        sync_buffer["flatten_selected_values"] = flatten_selected_values
        sync_buffer["flatten_selected_indices"] = flatten_selected_indices
        sync_buffer["n_bits"] = n_bits

    def sync(self, sync_buffer):
        # get the flatten values and prepare the sync.
        message_to_send = torch.cat(
            [
                sync_buffer["flatten_selected_values"].buffer,
                sync_buffer["flatten_selected_indices"].buffer,
            ]
        )

        if self.comm_device == "cpu":
            message_to_send = message_to_send.cpu().pin_memory()

        # sync.
        sync_message_reqs, synced_message = self.aggregator_fn._agg(
            message_to_send, op="get_raw_sync_data", force_wait=False
        )

        # update sync_buffer.
        sync_buffer["sync_reqs"] = sync_message_reqs
        sync_buffer["synced_message"] = synced_message
        sync_buffer["sycned_message_size"] = len(message_to_send)

    def uncompress(self, sync_buffer, neighbor_hat_params, neighbors_info):
        # wait the sync.
        self.aggregator_fn.complete_wait(sync_buffer["sync_reqs"])

        # uncompress and update.
        message_size = int(sync_buffer["sycned_message_size"] / 2)

        for rank, weight in neighbors_info.items():
            hat_params = neighbor_hat_params[
                rank if rank in neighbor_hat_params else "memory"
            ]
            hat_params_memory = neighbor_hat_params["memory"]

            # recover values/indices to the correct device.
            q_values, q_indices = self._uncompress_helper(
                hat_params,
                rank,
                sync_buffer["synced_message"],
                message_size,
                sync_buffer["selected_shapes"],
                sync_buffer["original_shapes"],
            )

            # update neighbor_hat_params
            if rank in neighbor_hat_params:
                hat_params.buffer[q_indices] += q_values
            hat_params_memory.buffer[q_indices] += weight * q_values

    def _uncompress_helper(
        self,
        _hat_params,
        _rank,
        synced_message,
        sycned_message_size,
        selected_shapes,
        original_shapes,
    ):
        # recover the message and the corresponding device.
        _message = comm.recover_device(
            synced_message[_rank], device=_hat_params.buffer.device
        )
        values = _message[:sycned_message_size]
        indices = _message[sycned_message_size:]

        # deal with unbalanced values/indieces
        q_values, q_indices = self.compressor_fn.uncompress(
            values, indices, selected_shapes, original_shapes
        )
        return q_values, q_indices


class CHOCOQuantizationCompressor(object):
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
        self.kargs = kargs
        self.compressor_fn = QuantizationCompressor()

        # define gossip_stream
        if torch.cuda.is_available():
            if self.comm_device == "cpu":
                self.gossip_stream = torch.cuda.current_stream()
            else:
                self.gossip_stream = torch.cuda.current_stream()

    def pipeline(self, sync_buffer, neighbor_hat_params, neighbors_info):
        if torch.cuda.is_available():
            with torch.cuda.stream(self.gossip_stream):
                try:
                    self.compress(sync_buffer)
                    self.sync(sync_buffer)
                    self.uncompress(sync_buffer, neighbor_hat_params, neighbors_info)
                except RuntimeError as e:
                    print("Error: {}".format(e))
        else:
            self.compress(sync_buffer)
            self.sync(sync_buffer)
            self.uncompress(sync_buffer, neighbor_hat_params, neighbors_info)

    def compress(self, sync_buffer):
        quantized_values = []

        for half_param, hat_param in zip(
            sync_buffer["flatten_params"], sync_buffer["flatten_hat_params"]
        ):
            _quantized_values = self.compressor_fn.compress(
                half_param - hat_param,
                self.comm_op,
                self.quantize_level,
                self.is_biased,
            )
            quantized_values.append(_quantized_values)

        # flatten selected values/indices.
        flatten_updates = TensorBuffer(quantized_values)

        # get n_bits to transmit.
        n_bits = get_n_bits(flatten_updates.buffer) * self.quantize_level / 64

        # update shared dict.
        sync_buffer["flatten_updates"] = flatten_updates
        sync_buffer["n_bits"] = n_bits

    def sync(self, sync_buffer):
        # prepare the sync.
        to_sync_message = sync_buffer["flatten_updates"].buffer

        if self.comm_device == "cpu":
            to_sync_message = to_sync_message.cpu().pin_memory()

        # sync.
        sync_message_reqs, synced_message = self.aggregator_fn._agg(
            to_sync_message, op="get_raw_sync_data", force_wait=False
        )

        # update sync_buffer.
        sync_buffer["sync_reqs"] = sync_message_reqs
        sync_buffer["synced_message"] = synced_message

    def uncompress(self, sync_buffer, neighbor_hat_params, neighbors_info):
        # wait the sync.
        self.aggregator_fn.complete_wait(sync_buffer["sync_reqs"])

        for rank, weight in neighbors_info.items():
            hat_params = neighbor_hat_params[
                rank if rank in neighbor_hat_params else "memory"
            ]
            hat_params_memory = neighbor_hat_params["memory"]

            # recover correct values/indices.
            q_values = comm.recover_device(
                sync_buffer["synced_message"][rank], device=hat_params.buffer.device
            )

            # update neighbor_hat_params
            if rank in neighbor_hat_params:
                hat_params.buffer += q_values
            hat_params_memory.buffer += weight * q_values


class CHOCOSignCompressor(object):
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
        self.kargs = kargs
        self.compressor_fn = SignCompressor()

        # define gossip_stream
        if self.comm_device == "cpu":
            self.gossip_stream = torch.cuda.current_stream()
        else:
            self.gossip_stream = torch.cuda.current_stream()

    def pipeline(self, sync_buffer, neighbor_hat_params, neighbors_info):
        with torch.cuda.stream(self.gossip_stream):
            try:
                self.compress(sync_buffer)
                self.sync(sync_buffer)
                self.uncompress(sync_buffer, neighbor_hat_params, neighbors_info)
            except RuntimeError as e:
                print("Error: {}".format(e))

    def compress(self, sync_buffer):
        # get the sign/magnitude for the tensor (to be transmitted).
        norms, updates = [], []
        for half_param, hat_param in zip(
            sync_buffer["flatten_params"], sync_buffer["flatten_hat_params"]
        ):
            _update = half_param - hat_param
            updates += [_update]
            norms += [_update.norm(p=1)]

        # flatten selected values/indices.
        flatten_norms = TensorBuffer(norms)
        flatten_directions = TensorBuffer(updates)
        signs, sign_size = self.compressor_fn.compress(flatten_directions.buffer)

        # get n_bits to transmit.
        n_bits = get_n_bits(flatten_norms.buffer) + get_n_bits(signs)

        # update shared dict.
        sync_buffer["flatten_norms"] = flatten_norms
        sync_buffer["flatten_directions"] = flatten_directions
        sync_buffer["signs"] = signs
        sync_buffer["sign_size"] = sign_size
        sync_buffer["n_bits"] = n_bits

    def sync(self, sync_buffer):
        # prepare sync.
        to_sync_flatten_norms = sync_buffer["flatten_norms"].buffer
        to_sync_signs = sync_buffer["signs"]

        if self.comm_device == "cpu":
            to_sync_flatten_norms = to_sync_flatten_norms.cpu().pin_memory()
            to_sync_signs = to_sync_signs.cpu().pin_memory()

        # sync.
        sync_reqs_1, synced_flatten_norms = self.aggregator_fn._agg(
            to_sync_flatten_norms, op="get_raw_sync_data", force_wait=False
        )
        sync_reqs_2, synced_signs = self.aggregator_fn._agg(
            to_sync_signs, op="get_raw_sync_data", force_wait=False
        )

        # update sync_buffer.
        sync_buffer["sync_reqs_1"] = sync_reqs_1
        sync_buffer["sync_reqs_2"] = sync_reqs_2
        sync_buffer["synced_flatten_norms"] = synced_flatten_norms
        sync_buffer["synced_signs"] = synced_signs

    def uncompress(self, sync_buffer, neighbor_hat_params, neighbors_info):
        # wait the sync.
        self.aggregator_fn.complete_wait(sync_buffer["sync_reqs_1"])
        self.aggregator_fn.complete_wait(sync_buffer["sync_reqs_2"])

        # uncompress and update.
        for rank, weight in neighbors_info.items():
            # get hat_params of the current rank.
            hat_params = neighbor_hat_params[
                rank if rank in neighbor_hat_params else "memory"
            ]

            # recover the message and the corresponding device.
            sync_buffer["flatten_norms"].buffer = comm.recover_device(
                sync_buffer["synced_flatten_norms"][rank],
                device=hat_params.buffer.device,
            )
            sync_buffer["flatten_directions"].buffer = self.compressor_fn.uncompress(
                comm.recover_device(
                    sync_buffer["synced_signs"][rank], device=hat_params.buffer.device
                ),
                sync_buffer["sign_size"],
            )

            # update neighbor_hat_params
            for hat_param, hat_param_memory, norm, sign in zip(
                hat_params,
                neighbor_hat_params["memory"],
                sync_buffer["flatten_norms"],
                sync_buffer["flatten_directions"],
            ):
                _update = norm / sign.nelement() * sign
                if rank in neighbor_hat_params:
                    hat_param.add_(_update)
                hat_param_memory.add_(_update, alpha=weight)

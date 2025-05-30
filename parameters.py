# -*- coding: utf-8 -*-
"""define all global parameters here."""
from os.path import join
import argparse

import pcode.models as models
from pcode.utils.checkpoint import get_checkpoint_folder_name


def get_args():
    ROOT_DIRECTORY = "./"
    RAW_DATA_DIRECTORY = join(ROOT_DIRECTORY, "data/")
    TRAINING_DIRECTORY = join(RAW_DATA_DIRECTORY, "checkpoint")

    model_names = sorted(
        name for name in models.__dict__ if name.islower() and not name.startswith("__")
    )

    # feed them to the parser.
    parser = argparse.ArgumentParser(description="PyTorch Training for ConvNet")
    
    parser.add_argument("--train_size", default=None, type=int)

    # arguments for evaluation only
    parser.add_argument("--model_dir", default=None, type=str) # where saved models are
    parser.add_argument("--checkpoint_dir", default=None, type=str) # where to save the trained model
    parser.add_argument("--inference_dir", default=None, type=str) # where to save the inference jsons
    parser.add_argument("--epochs_ran", default=None, type=int)

    # arguments for DeTAG
    parser.add_argument("--gossip_eta", default=None, type=float)
    parser.add_argument("--gossip_rounds", default=None, type=int)
    parser.add_argument("--oracle_budget", default=None, type=int)
    
    # add arguments.
    parser.add_argument("--clean_output", default=True, type=str2bool)
    parser.add_argument("--work_dir", default=None, type=str)
    parser.add_argument("--remote_exec", default=False, type=str2bool)

    # dataset.
    parser.add_argument("--data", default="cifar10", help="a specific dataset name")
    parser.add_argument(
        "--data_dir", default=RAW_DATA_DIRECTORY, help="path to dataset"
    )
    parser.add_argument(
        "--use_lmdb_data",
        default=False,
        type=str2bool,
        help="use sequential lmdb dataset for better loading.",
    )
    parser.add_argument(
        "--partition_data",
        default=None,
        type=str,
        help="decide if each worker will access to all data.",
    )
    parser.add_argument("--pin_memory", default=True, type=str2bool)

    # model
    parser.add_argument(
        "--arch",
        "-a",
        default="resnet20",
        help="model architecture: " + " | ".join(model_names) + " (default: resnet20)",
    )

    # training and learning scheme
    parser.add_argument("--train_fast", type=str2bool, default=False)
    parser.add_argument("--skip_eval", type=str2bool, default=False)
    parser.add_argument("--stop_criteria", type=str, default="epoch")
    parser.add_argument("--num_epochs", type=int, default=90)
    parser.add_argument("--num_iterations", type=int, default=9800)
    parser.add_argument("--eval_n_points", type=int, default=50)

    # parser.add_argument("--avg_model", type=str2bool, default=False)
    parser.add_argument("--reshuffle_per_epoch", default=False, type=str2bool)
    parser.add_argument(
        "--batch_size",
        "-b",
        default=256,
        type=int,
        help="mini-batch size (default: 256)",
    )
    parser.add_argument("--base_batch_size", default=None, type=int)
    parser.add_argument("--initial_batch_num", default=None, type=int) # for GT-HSGD
    parser.add_argument("--true_gradient", default=False, type=str2bool)
    parser.add_argument("--loss", default="cross_entropy", type=str)

    # for time varying setup
    # PProx-SPDA / FSPPD
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--omega", type=float, default=1.0)
    parser.add_argument("--edge_prob", type=float, default=0.0)
    parser.add_argument("--one_edge", type=str2bool, default=False)

    # FSPDA-STORM
    parser.add_argument("--storm_momentum", type=float, default=0.01)
    parser.add_argument("--storm_dual_momentum", type=float, default=0.01)

    # FSPDA-ADAM
    parser.add_argument("--adam_primal_beta1", type=float, default=0.9)
    parser.add_argument("--adam_primal_beta2", type=float, default=0.999)
    parser.add_argument("--adam_dual_beta1", type=float, default=0.9)
    parser.add_argument("--adam_dual_beta2", type=float, default=0.999)

    # Prox-Skip
    parser.add_argument("--gossip_prob", type=float, default=1.0)

    # CHOCO-SGD
    parser.add_argument("--node_fraction", type=float, default=1.0)

    # CP-SGD
    parser.add_argument("--alpha", type=float, default=0.0)

    # TiCoPD
    parser.add_argument("--theta", type=float, default=0.0)
    parser.add_argument("--random_lap", type=str2bool, default=True)
    parser.add_argument("--use_compressor_buffer", type=str2bool, default=False)
    parser.add_argument("--compression_noise", type=float, default=0.0)
    parser.add_argument("--shared_mask", type=str2bool, default=False)

    # DIMIX
    parser.add_argument("--tau", type=float, default=0.0)
    parser.add_argument("--nu", type=float, default=0.0)
    parser.add_argument("--mu", type=float, default=0.0)

    # DEF-ATC
    parser.add_argument("--zeta", type=float, default=1.0)

    # K-GT
    parser.add_argument("--local_steps", type=int, default=1)
    parser.add_argument("--eta_s", type=float, default=1.0)

    # Di-CS-SVRG
    parser.add_argument("--SVRG", type=str2bool, default=False)
    parser.add_argument("--outer_loop_T", type=int, default=100)
    parser.add_argument("--B_connected", type=int, default=10)

    # SPARQ-SGD
    parser.add_argument("--c_init", type=float, default=2.0)
    parser.add_argument("--c_incre", type=float, default=1.0)

    parser.add_argument("--sam", type=str2bool, default=False)

    
    # learning rate scheme
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--dual_lr", type=float, default=0.01)
    parser.add_argument("--const_lr", default=False, type=str2bool)
    parser.add_argument("--lr_schedule_scheme", type=str, default=None)

    parser.add_argument("--cosine_warmup_epoch", type=int, default=0)
    parser.add_argument("--ramp_up_epoch", type=int, default=0)
    parser.add_argument("--ramp_down_epoch", type=int, default=0)
    parser.add_argument("--min_lambda", type=float, default=0)
    parser.add_argument("--lr_change_epochs", type=str, default=None)
    parser.add_argument("--lr_fields", type=str, default=None)
    parser.add_argument("--lr_scale_indicators", type=str, default=None)

    parser.add_argument("--lr_scaleup", type=str2bool, default=False)
    parser.add_argument("--lr_scaleup_type", type=str, default="linear")
    parser.add_argument(
        "--lr_scaleup_factor",
        type=str,
        default="graph",
        help="scale by the graph connection, or the world size",
    )
    parser.add_argument("--lr_warmup", type=str2bool, default=False)
    parser.add_argument("--lr_warmup_epochs", type=int, default=5)
    parser.add_argument("--lr_decay", type=float, default=10)

    parser.add_argument("--lr_onecycle_low", type=float, default=0.15)
    parser.add_argument("--lr_onecycle_high", type=float, default=3)
    parser.add_argument("--lr_onecycle_extra_low", type=float, default=0.0015)
    parser.add_argument("--lr_onecycle_num_epoch", type=int, default=46)

    parser.add_argument("--lr_gamma", type=float, default=None)
    parser.add_argument("--lr_mu", type=float, default=None)
    parser.add_argument("--lr_alpha", type=float, default=None)

    # optimizer
    parser.add_argument("--optimizer", type=str, default="sgd")

    parser.add_argument("--adam_beta_1", default=0.9, type=float)
    parser.add_argument("--adam_beta_2", default=0.999, type=float)
    parser.add_argument("--adam_eps", default=1e-8, type=float)

    # the topology of the decentralized network.
    parser.add_argument("--graph_topology", default="complete", type=str)
    parser.add_argument("--er_p", default=0.5, type=float)

    # compression scheme.
    parser.add_argument("--comm_algo", default=None, type=str)
    parser.add_argument(
        "--comm_op",
        default=None,
        type=str,
        choices=["compress_top_k", "compress_random_k", "quantize_qsgd", "sign", "quantize"],
    )
    parser.add_argument("--compress_ratio", default=None, type=float)
    parser.add_argument(
        "--compress_warmup_values", default="0.75,0.9375,0.984375,0.996,0.999", type=str
    )
    parser.add_argument("--compress_warmup_epochs", default=0, type=int)
    parser.add_argument("--quantize_level", default=None, type=int)
    parser.add_argument("--quantize_bits", default=None, type=int)
    parser.add_argument("--side_length", default=None, type=float)
    parser.add_argument("--is_biased", default=False, type=str2bool)
    parser.add_argument("--majority_vote", default=False, type=str2bool)

    parser.add_argument("--consensus_stepsize", default=0.9, type=float)
    parser.add_argument("--momentum_beta", default=None, type=float)
    parser.add_argument("--dual_momentum_beta", default=None, type=float)
    # parser.add_argument("--evaluate_consensus", default=False, type=str2bool)
    parser.add_argument("--eval_consensus_only", default=False, type=str2bool)

    parser.add_argument("--mask_momentum", default=False, type=str2bool)
    parser.add_argument("--clip_grad", default=False, type=str2bool)
    parser.add_argument("--clip_grad_val", default=None, type=float)

    parser.add_argument("--local_step", default=1, type=int)
    parser.add_argument("--turn_on_local_step_from", default=0, type=int)
    parser.add_argument("--local_adam_memory_treatment", default=None, type=str)

    # momentum scheme
    parser.add_argument("--momentum_factor", default=0.9, type=float)
    parser.add_argument("--use_nesterov", default=False, type=str2bool)

    # regularization
    parser.add_argument(
        "--weight_decay", default=5e-4, type=float, help="weight decay (default: 1e-4)"
    )
    parser.add_argument("--drop_rate", default=0.0, type=float)

    # configuration for different models.
    parser.add_argument("--densenet_growth_rate", default=12, type=int)
    parser.add_argument("--densenet_bc_mode", default=False, type=str2bool)
    parser.add_argument("--densenet_compression", default=0.5, type=float)

    parser.add_argument("--wideresnet_widen_factor", default=4, type=int)

    parser.add_argument("--rnn_n_hidden", default=200, type=int)
    parser.add_argument("--rnn_n_layers", default=2, type=int)
    parser.add_argument("--rnn_bptt_len", default=35, type=int)
    parser.add_argument("--rnn_clip", type=float, default=0.25)
    parser.add_argument("--rnn_use_pretrained_emb", type=str2bool, default=True)
    parser.add_argument("--rnn_tie_weights", type=str2bool, default=True)
    parser.add_argument("--rnn_weight_norm", type=str2bool, default=False)

    # miscs
    parser.add_argument("--manual_seed", type=int, default=6, help="manual seed")
    parser.add_argument(
        "--evaluate",
        "-e",
        dest="evaluate",
        type=str2bool,
        default=False,
        help="evaluate model on validation set",
    )
    parser.add_argument("--summary_freq", default=100, type=int)
    parser.add_argument("--timestamp", default=None, type=str)
    parser.add_argument("--track_time", default=False, type=str2bool)
    parser.add_argument("--track_detailed_time", default=False, type=str2bool)
    parser.add_argument("--display_tracked_time", default=False, type=str2bool)
    parser.add_argument("--eval_grad", type=str2bool, default=True)
    parser.add_argument("--eval_worst", type=str2bool, default=True, help="evaluate every local model and log the worst performance.")
    parser.add_argument("--eval_avg", type=str2bool, default=True, help="evaluate the performance at the average iterate.")
    # parser.add_argument("--evaluate_avg", default=False, type=str2bool)

    # checkpoint
    parser.add_argument("--resume", default=None, type=str)
    parser.add_argument(
        "--checkpoint",
        "-c",
        default=TRAINING_DIRECTORY,
        type=str,
        help="path to save checkpoint (default: checkpoint)",
    )
    parser.add_argument("--checkpoint_index", type=str, default=None)
    parser.add_argument("--save_epoch_models", type=str2bool, default=False)
    parser.add_argument("--save_all_models", type=str2bool, default=False)
    parser.add_argument("--save_model", type=str2bool, default=False)
    parser.add_argument("--load_model", type=str2bool, default=False)
    parser.add_argument("--save_some_models", type=str, default=None)

    """meta info."""
    parser.add_argument("--user", type=str, default="lin")
    parser.add_argument(
        "--project", type=str, default="distributed_adam_type_algorithm"
    )
    parser.add_argument("--experiment", type=str, default=None)

    # device
    parser.add_argument("--backend", type=str, default="mpi")
    parser.add_argument("--use_ipc", type=str2bool, default=False)
    parser.add_argument("--hostfile", type=str, default="iccluster/hostfile")
    parser.add_argument("--mpi_path", type=str, default="$HOME/.openmpi")
    parser.add_argument("--mpirun_path", type=str, default="$HOME/.openmpi/bin/mpirun")
    parser.add_argument("--mpi_env", type=str, default=None)
    parser.add_argument(
        "--python_path", type=str, default="$HOME/conda/envs/pytorch-py3.6/bin/python"
    )
    parser.add_argument(
        "-j",
        "--num_workers",
        default=4,
        type=int,
        help="number of data loading workers (default: 4)",
    )

    parser.add_argument(
        "--n_mpi_process", default=1, type=int, help="# of the main process."
    )
    parser.add_argument(
        "--n_sub_process",
        default=1,
        type=int,
        help="# of subprocess for each mpi process.",
    )
    parser.add_argument("--world", default=None, type=str)
    parser.add_argument("--on_cuda", type=str2bool, default=True)
    parser.add_argument("--comm_device", type=str, default="cuda")
    parser.add_argument("--local_rank", default=None, type=str)
    parser.add_argument("--clean_python", default=False, type=str2bool)

    parser.add_argument("--log_eval", type=str2bool, default=False)

    # for synthetic data
    parser.add_argument("--data_dim", type=int, default=1000)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--num_train_samples", type=int, default=1000)



    # parse conf.
    conf = parser.parse_args()
    if conf.timestamp is None:
        conf.timestamp = get_checkpoint_folder_name(conf)

    return conf


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


if __name__ == "__main__":
    args = get_args()

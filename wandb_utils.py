import wandb
import torch
import torch.distributed as dist

def wandb_init(conf, name=None):
    # start a new wandb run to track this script
    project_name = "random_graph_decen_opt"
    wandb.init(
        # set the wandb project where this run will be logged
        project=project_name,
        name=name,
        # track hyperparameters and run metadata
        config={
            **conf,
        }
    )

def aggregate_stats(stat):
    for k in stat:
        stat[k] = torch.tensor(stat[k])
    dist.reduce(stat["worst_train_top1"], dst=0, op=dist.ReduceOp.MIN)
    dist.reduce(stat["worst_train_loss"], dst=0, op=dist.ReduceOp.MAX)
    dist.reduce(stat["worst_train_grad_norm"], dst=0, op=dist.ReduceOp.MAX)
    dist.reduce(stat["worst_test_top1"], dst=0, op=dist.ReduceOp.MIN)
    dist.reduce(stat["worst_test_loss"], dst=0, op=dist.ReduceOp.MAX)
    dist.reduce(stat["worst_test_grad_norm"], dst=0, op=dist.ReduceOp.MAX)

    worst_consensus_error = stat["consensus_error"].clone()
    dist.reduce(worst_consensus_error, dst=0, op=dist.ReduceOp.MAX)
    stat["worst_consensus_error"] = worst_consensus_error
    dist.reduce(stat["consensus_error"] / dist.get_world_size(), dst=0, op=dist.ReduceOp.SUM)

    for k in stat:
        stat[k] = stat[k].item()
    return stat
    

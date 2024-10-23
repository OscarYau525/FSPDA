# FSPDA in PyTorch
This repo contains the implementation of Fully Stochastic Primal-Dual Algorithm (FSPDA) for decentralized optimization.
Our implementation extends from https://github.com/epfml/ChocoSGD.

## Installation
1. Setup the local environment using [conda_setup.sh](conda_setup.sh).
2. (For Imagenet exp. only) Follow the instructions in [system_setup.md](system_setup.md) to prepare the Imagenet dataset.
3. (For Imagenet exp. only) Follow the instructions in [torch_build_instruction.md](torch_build_instruction.md) to build PyTorch from source with MPI support for CUDA.

Our experiments are reproducible from a single CPU server (Fig. 2,3,4,5) and a 8-GPU server (Fig. 6).

## Experiments
Use the bash scripts in `FSPDA/exps`. Each file corresponds to one figure of experiment results in the paper. Our script uses [wandb](https://wandb.ai/) for logging by default.

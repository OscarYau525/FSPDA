# sorted mnist
# parallel_choco with compress_random_k
python run.py \
    --arch feedforward --optimizer parallel_choco_v \
    --experiment test \
    --data mnist --pin_memory False \
    --batch_size 256 --num_workers 0 \
    --num_iterations 10000000 --partition_data sorted --reshuffle_per_epoch False --stop_criteria iteration --eval_n_points 500 \
    --n_mpi_process 10 --n_sub_process 1 --world 0,0,0,0,0,0,0,0,0,0 --on_cuda False --use_ipc False --comm_device cuda \
    --lr 1e-4 --node_fraction 0.1 --const_lr True --lr_schedule_scheme custom_multistep --lr_change_epochs 1000 --lr_decay 1 \
    --weight_decay 1e-4 \
    --comm_op compress_random_k --compress_ratio 0.99 --is_biased True --consensus_stepsize 0.001 \
    --hostfile hostfile --graph_topology complete --track_time True --display_tracked_time True --mpirun_path mpirun \
    --python_path $HOME/.conda/envs/torch_mpi/bin/python --mpi_path $HOME/openmpi-4.1.3/bin/mpirun --mpirun_path mpirun --manual_seed 1 --train_fast False


# swarm sgd with 8-bits quantization
python run.py \
    --arch feedforward --optimizer swarm_sgd \
    --experiment test \
    --data mnist --pin_memory False \
    --batch_size 256 --num_workers 0 \
    --num_iterations 1000000 --partition_data sorted --reshuffle_per_epoch False --stop_criteria iteration --eval_n_points 500 \
    --n_mpi_process 10 --n_sub_process 1 --world 0,0,0,0,0,0,0,0,0,0 --on_cuda False --use_ipc False --comm_device cuda \
    --lr 5e-5 --quantize_level 256 --side_length 0.0001 --const_lr True --lr_schedule_scheme custom_multistep --lr_change_epochs 1000 --lr_decay 1 \
    --weight_decay 1e-4 \
    --hostfile hostfile --graph_topology complete --track_time True --display_tracked_time True \
    --python_path $HOME/.conda/envs/torch_mpi/bin/python --mpi_path $HOME/openmpi-4.1.3/bin/mpirun --mpirun_path mpirun --manual_seed 1 --train_fast False

# fspda with compress_random_k
python run.py \
    --arch feedforward --optimizer fspda \
    --experiment test \
    --data mnist --pin_memory False \
    --batch_size 256 --num_workers 0 \
    --num_iterations 10000000 --partition_data sorted --reshuffle_per_epoch False --stop_criteria iteration --eval_n_points 500 \
    --n_mpi_process 10 --n_sub_process 1 --world 0,0,0,0,0,0,0,0,0,0 --on_cuda False --use_ipc False --comm_device cuda \
    --lr 1e-4 --beta 1 --gamma 1 --eta 1e-4 --edge_fraction 0.0 --one_edge True --const_lr True --lr_schedule_scheme custom_multistep --lr_change_epochs 1000 --lr_decay 1 \
    --weight_decay 1e-4 \
    --comm_op compress_random_k --compress_ratio 0.99 --is_biased True \
    --hostfile hostfile --graph_topology complete --track_time True --display_tracked_time True \
    --python_path $HOME/.conda/envs/torch_mpi/bin/python --mpi_path $HOME/openmpi-4.1.3/bin/mpirun --mpirun_path mpirun --manual_seed 1 --train_fast False


# shuffled mnist
# parallel_choco with compress_random_k
python run.py \
    --arch feedforward --optimizer parallel_choco_v \
    --experiment test \
    --data mnist --pin_memory False \
    --batch_size 256 --num_workers 0 \
    --num_iterations 10000000 --partition_data random --reshuffle_per_epoch False --stop_criteria iteration --eval_n_points 500 \
    --n_mpi_process 10 --n_sub_process 1 --world 0,0,0,0,0,0,0,0,0,0 --on_cuda False --use_ipc False --comm_device cuda \
    --lr 1e-4 --node_fraction 0.1 --const_lr True --lr_schedule_scheme custom_multistep --lr_change_epochs 1000 --lr_decay 1 \
    --weight_decay 1e-4 \
    --comm_op compress_random_k --compress_ratio 0.99 --is_biased True --consensus_stepsize 0.001 \
    --hostfile hostfile --graph_topology complete --track_time True --display_tracked_time True --mpirun_path mpirun \
    --python_path $HOME/.conda/envs/torch_mpi/bin/python --mpi_path $HOME/openmpi-4.1.3/bin/mpirun --mpirun_path mpirun --manual_seed 1 --train_fast False

# swarm sgd with 8-bits quantization
python run.py \
    --arch feedforward --optimizer swarm_sgd \
    --experiment test \
    --data mnist --pin_memory False \
    --batch_size 256 --num_workers 0 \
    --num_iterations 1000000 --partition_data random --reshuffle_per_epoch False --stop_criteria iteration --eval_n_points 500 \
    --n_mpi_process 10 --n_sub_process 1 --world 0,0,0,0,0,0,0,0,0,0 --on_cuda False --use_ipc False --comm_device cuda \
    --lr 5e-5 --quantize_level 256 --side_length 0.0001 --const_lr True --lr_schedule_scheme custom_multistep --lr_change_epochs 1000 --lr_decay 1 \
    --weight_decay 1e-4 \
    --hostfile hostfile --graph_topology complete --track_time True --display_tracked_time True \
    --python_path $HOME/.conda/envs/torch_mpi/bin/python --mpi_path $HOME/openmpi-4.1.3/bin/mpirun --mpirun_path mpirun --manual_seed 1 --train_fast False


# fspda with compress_random_k
python run.py \
    --arch feedforward --optimizer fspda \
    --experiment test \
    --data mnist --pin_memory False \
    --batch_size 256 --num_workers 0 \
    --num_iterations 10000000 --partition_data random --reshuffle_per_epoch False --stop_criteria iteration --eval_n_points 500 \
    --n_mpi_process 10 --n_sub_process 1 --world 0,0,0,0,0,0,0,0,0,0 --on_cuda False --use_ipc False --comm_device cuda \
    --lr 1e-4 --beta 1 --gamma 1 --eta 1e-5 --edge_fraction 0.0 --one_edge True --const_lr True --lr_schedule_scheme custom_multistep --lr_change_epochs 1000 --lr_decay 1 \
    --weight_decay 1e-4 \
    --comm_op compress_random_k --compress_ratio 0.99 --is_biased True \
    --hostfile hostfile --graph_topology complete --track_time True --display_tracked_time True \
    --python_path $HOME/.conda/envs/torch_mpi/bin/python --mpi_path $HOME/openmpi-4.1.3/bin/mpirun --mpirun_path mpirun --manual_seed 1 --train_fast False

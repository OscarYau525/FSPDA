# complete graph
python run.py \
    --arch feedforward --optimizer fspda \
    --experiment test \
    --data mnist --pin_memory False \
    --batch_size 256 --num_workers 0 \
    --num_iterations 10000000 --partition_data random --reshuffle_per_epoch False --stop_criteria iteration --eval_n_points 500 \
    --n_mpi_process 10 --n_sub_process 1 --world 0,0,0,0,0,0,0,0,0,0 --on_cuda False --use_ipc False --comm_device cuda \
    --lr 1e-4 --beta 1 --gamma 0.5 --eta 1e-6 --edge_fraction 0.0 --one_edge True --const_lr True --lr_schedule_scheme custom_multistep --lr_change_epochs 1000 --lr_decay 1 \
    --weight_decay 1e-4 \
    --comm_op compress_random_k --compress_ratio 0.99 --is_biased True \
    --hostfile hostfile --graph_topology complete --track_time True --display_tracked_time True \
    --python_path $HOME/.conda/envs/torch_mpi/bin/python --mpi_path $HOME/openmpi-4.1.3/bin/mpirun --mpirun_path mpirun --manual_seed 1 --train_fast False

# ER graph
python run.py \
    --arch feedforward --optimizer fspda \
    --experiment test \
    --data mnist --pin_memory False \
    --batch_size 256 --num_workers 0 \
    --num_iterations 10000000 --partition_data random --reshuffle_per_epoch False --stop_criteria iteration --eval_n_points 500 \
    --n_mpi_process 10 --n_sub_process 1 --world 0,0,0,0,0,0,0,0,0,0 --on_cuda False --use_ipc False --comm_device cuda \
    --lr 1e-4 --beta 1 --gamma 0.5 --eta 1e-6 --edge_fraction 0.0 --one_edge True --const_lr True --lr_schedule_scheme custom_multistep --lr_change_epochs 1000 --lr_decay 1 \
    --weight_decay 1e-4 \
    --comm_op compress_random_k --compress_ratio 0.99 --is_biased True \
    --hostfile hostfile --graph_topology random --er_p 0.5 --track_time True --display_tracked_time True \
    --python_path $HOME/.conda/envs/torch_mpi/bin/python --mpi_path $HOME/openmpi-4.1.3/bin/mpirun --mpirun_path mpirun --manual_seed 1 --train_fast False

# ring graph
python run.py \
    --arch feedforward --optimizer fspda \
    --experiment test \
    --data mnist --pin_memory False \
    --batch_size 256 --num_workers 0 \
    --num_iterations 10000000 --partition_data random --reshuffle_per_epoch False --stop_criteria iteration --eval_n_points 500 \
    --n_mpi_process 10 --n_sub_process 1 --world 0,0,0,0,0,0,0,0,0,0 --on_cuda False --use_ipc False --comm_device cuda \
    --lr 1e-4 --beta 1 --gamma 0.5 --eta 1e-6 --edge_fraction 0.0 --one_edge True --const_lr True --lr_schedule_scheme custom_multistep --lr_change_epochs 1000 --lr_decay 1 \
    --weight_decay 1e-4 \
    --comm_op compress_random_k --compress_ratio 0.99 --is_biased True \
    --hostfile hostfile --graph_topology ring --track_time True --display_tracked_time True \
    --python_path $HOME/.conda/envs/torch_mpi/bin/python --mpi_path $HOME/openmpi-4.1.3/bin/mpirun --mpirun_path mpirun --manual_seed 1 --train_fast False
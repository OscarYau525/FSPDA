# FSPDA-SA 10% sparse
python run.py \
    --arch resnet50 --optimizer fspda \
    --experiment test \
    --data imagenet --use_lmdb_data True --pin_memory True \
    --batch_size 128 --base_batch_size 128 --num_workers 16 \
    --num_epochs 100 --partition_data random --reshuffle_per_epoch False --stop_criteria epoch --eval_n_points 100 \
    --n_mpi_process 8 --n_sub_process 1 --world 0,1,2,3,4,5,6,7 \
    --on_cuda True --use_ipc False --comm_device cuda \
    --lr 1e-1 --lr_schedule_scheme cosine --cosine_warmup_epoch 5 \
    --beta 1 --gamma 0.5 --eta 5e-9 --edge_fraction 0 --one_edge True \
    --weight_decay 1e-4 --eval_consensus_only False \
    --comm_op compress_random_k --compress_ratio 0.9 --is_biased True \
    --hostfile hostfile --graph_topology complete --track_time True --display_tracked_time True --eval_grad False --eval_worst False \
    --backend mpi --data_dir data/imagenet --work_dir /home/ubuntu/FSPDA \
    --python_path /opt/conda/bin/python --mpi_path /home/ubuntu/opt/openmpi \
    --mpirun_path /home/ubuntu/opt/openmpi/bin/mpirun --manual_seed 1

# FSPDA-SA 1% sparse
python run.py \
    --arch resnet50 --optimizer fspda \
    --experiment test \
    --data imagenet --use_lmdb_data True --pin_memory True \
    --batch_size 128 --base_batch_size 128 --num_workers 16 \
    --num_epochs 100 --partition_data random --reshuffle_per_epoch False --stop_criteria epoch --eval_n_points 100 \
    --n_mpi_process 8 --n_sub_process 1 --world 0,1,2,3,4,5,6,7 \
    --on_cuda True --use_ipc False --comm_device cuda \
    --lr 1e-1 --lr_schedule_scheme cosine --cosine_warmup_epoch 5 \
    --beta 1 --gamma 0.5 --eta 1e-9 --edge_fraction 0 --one_edge True \
    --weight_decay 1e-4 --eval_consensus_only False \
    --comm_op compress_random_k --compress_ratio 0.99 --is_biased True \
    --hostfile hostfile --graph_topology complete --track_time True --display_tracked_time True --eval_grad False --eval_worst False \
    --backend mpi --data_dir data/imagenet --work_dir /home/ubuntu/FSPDA \
    --python_path /opt/conda/bin/python --mpi_path /home/ubuntu/opt/openmpi \
    --mpirun_path /home/ubuntu/opt/openmpi/bin/mpirun --manual_seed 1

# FSPDA-SA 0.1% sparse
python run.py \
    --arch resnet50 --optimizer fspda \
    --experiment test \
    --data imagenet --use_lmdb_data True --pin_memory True \
    --batch_size 128 --base_batch_size 128 --num_workers 16 \
    --num_epochs 200 --partition_data random --reshuffle_per_epoch False --stop_criteria epoch --eval_n_points 200 \
    --n_mpi_process 8 --n_sub_process 1 --world 0,1,2,3,4,5,6,7 \
    --on_cuda True --use_ipc False --comm_device cuda \
    --lr 0.05  --lr_schedule_scheme cosine --cosine_warmup_epoch 5 \
    --beta 1 --gamma 0.5 --eta 5e-10 --edge_fraction 0 --one_edge True \
    --weight_decay 1e-4 --eval_consensus_only False \
    --comm_op compress_random_k --compress_ratio 0.999 --is_biased True \
    --hostfile hostfile --graph_topology complete --track_time True --display_tracked_time True --eval_grad False --eval_worst False \
    --backend mpi --data_dir data/imagenet --work_dir /home/ubuntu/FSPDA \
    --python_path /opt/conda/bin/python --mpi_path /home/ubuntu/opt/openmpi \
    --mpirun_path /home/ubuntu/opt/openmpi/bin/mpirun --manual_seed 1


# CHOCO-SGD 10% sparse
python run.py \
    --arch resnet50 --optimizer parallel_choco_v \
    --experiment test \
    --data imagenet --use_lmdb_data True --pin_memory True \
    --batch_size 128 --base_batch_size 128 --num_workers 16 \
    --num_epochs 100 --partition_data random --reshuffle_per_epoch False --stop_criteria epoch --eval_n_points 100 \
    --n_mpi_process 8 --n_sub_process 1 --world 0,1,2,3,4,5,6,7 \
    --on_cuda True --use_ipc False --comm_device cuda \
    --lr 1e-1 --lr_schedule_scheme cosine --cosine_warmup_epoch 5 \
    --weight_decay 1e-4 --use_nesterov False \
    --node_fraction 0.5 \
    --comm_op compress_random_k --compress_ratio 0.9 --is_biased True --consensus_stepsize 0.05 \
    --hostfile hostfile --graph_topology complete --track_time True --display_tracked_time True --eval_grad False --eval_worst False \
    --backend mpi --data_dir data/imagenet --work_dir /home/ubuntu/FSPDA \
    --python_path /opt/conda/bin/python --mpi_path /home/ubuntu/opt/openmpi \
    --mpirun_path /home/ubuntu/opt/openmpi/bin/mpirun --manual_seed 1

# CHOCO-SGD 1% sparse
python run.py \
    --arch resnet50 --optimizer parallel_choco_v \
    --experiment test \
    --data imagenet --use_lmdb_data True --pin_memory True \
    --batch_size 128 --base_batch_size 128 --num_workers 16 \
    --num_epochs 100 --partition_data random --reshuffle_per_epoch False --stop_criteria epoch --eval_n_points 100 \
    --n_mpi_process 8 --n_sub_process 1 --world 0,1,2,3,4,5,6,7 \
    --on_cuda True --use_ipc False --comm_device cuda \
    --lr 1e-1 --lr_schedule_scheme cosine --cosine_warmup_epoch 5 \
    --weight_decay 1e-4 --use_nesterov False \
    --node_fraction 0.5 \
    --comm_op compress_random_k --compress_ratio 0.99 --is_biased True --consensus_stepsize 0.005 \
    --hostfile hostfile --graph_topology complete --track_time True --display_tracked_time True --eval_grad False --eval_worst False \
    --backend mpi --data_dir data/imagenet --work_dir /home/ubuntu/FSPDA \
    --python_path /opt/conda/bin/python --mpi_path /home/ubuntu/opt/openmpi \
    --mpirun_path /home/ubuntu/opt/openmpi/bin/mpirun --manual_seed 1

# Swarm-SGD 8-bits quantization
python run.py \
    --arch resnet50 --optimizer swarm_sgd \
    --experiment test \
    --data imagenet --use_lmdb_data True --pin_memory True \
    --batch_size 128 --base_batch_size 128 --num_workers 16 \
    --num_epochs 200 --partition_data random --reshuffle_per_epoch False --stop_criteria epoch --eval_n_points 200 \
    --n_mpi_process 8 --n_sub_process 1 --world 0,1,2,3,4,5,6,7 \
    --on_cuda True --use_ipc False --comm_device cuda \
    --lr 0.005 --lr_schedule_scheme cosine --cosine_warmup_epoch 5 \
    --weight_decay 1e-4 --use_nesterov False \
    --quantize_level 256 --side_length 0.00003 \
    --hostfile hostfile --graph_topology complete --track_time True --display_tracked_time True --eval_grad False --eval_worst False \
    --backend mpi --data_dir data/imagenet --work_dir /home/ubuntu/FSPDA \
    --python_path /opt/conda/bin/python --mpi_path /home/ubuntu/opt/openmpi \
    --mpirun_path /home/ubuntu/opt/openmpi/bin/mpirun --manual_seed 1
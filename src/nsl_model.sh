#!/bin/bash

# get the plot data for exp config then iterate over all to build the final plots
# resolve fedadagrad fault_tolerant fedavg
# i will change the variables in terms of what is constant in each iteration (rounds); I could technically have the plots in terms of the net rounds I use (not just rounds isolated to each specific experiment)
# compute comms cost (system-level)

# Method(s): FedAvg + NSL FedAvg + GaussianNoise FedAvg + [CORRUPTION_NAME]; FedAdagrad + NSL
# strategy + adv_reg + adv_step_size + adv_grad_norm

# FedAvg + NSL
python3 server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 client.py --model="nsl_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.05 --epochs=1 --steps_per_epoch=1 --nsl_reg=True --adv_grad_norm="infinity"
python3 client.py --model="nsl_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.05 --epochs=1 --steps_per_epoch=1 --nsl_reg=True --adv_grad_norm="infinity" 
python3 client.py --model="nsl_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.05 --epochs=1 --steps_per_epoch=1 --nsl_reg=True --adv_grad_norm="infinity"
python3 client.py --model="nsl_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.05 --epochs=1 --steps_per_epoch=1 --nsl_reg=True --adv_grad_norm="infinity"
python3 client.py --model="nsl_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.05 --epochs=1 --steps_per_epoch=1 --nsl_reg=True --adv_grad_norm="infinity"
python3 client.py --model="nsl_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.05 --epochs=1 --steps_per_epoch=1 --nsl_reg=True --adv_grad_norm="infinity"
python3 client.py --model="nsl_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.05 --epochs=1 --steps_per_epoch=1 --nsl_reg=True --adv_grad_norm="infinity"
python3 client.py --model="nsl_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.05 --epochs=1 --steps_per_epoch=1 --nsl_reg=True --adv_grad_norm="infinity"
python3 client.py --model="nsl_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.05 --epochs=1 --steps_per_epoch=1 --nsl_reg=True --adv_grad_norm="infinity"
python3 client.py --model="nsl_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.05 --epochs=1 --steps_per_epoch=1 --nsl_reg=True --adv_grad_norm="infinity"

python3 server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 client.py --model="nsl_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.1 --epochs=1 --steps_per_epoch=1 --nsl_reg=True --adv_grad_norm="infinity"
python3 client.py --model="nsl_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.1 --epochs=1 --steps_per_epoch=1 --nsl_reg=True --adv_grad_norm="infinity"
python3 client.py --model="nsl_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.1 --epochs=1 --steps_per_epoch=1 --nsl_reg=True --adv_grad_norm="infinity"
python3 client.py --model="nsl_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.1 --epochs=1 --steps_per_epoch=1 --nsl_reg=True --adv_grad_norm="infinity"
python3 client.py --model="nsl_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.1 --epochs=1 --steps_per_epoch=1 --nsl_reg=True --adv_grad_norm="infinity"
python3 client.py --model="nsl_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.1 --epochs=1 --steps_per_epoch=1 --nsl_reg=True --adv_grad_norm="infinity"
python3 client.py --model="nsl_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.1 --epochs=1 --steps_per_epoch=1 --nsl_reg=True --adv_grad_norm="infinity"
python3 client.py --model="nsl_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.1 --epochs=1 --steps_per_epoch=1 --nsl_reg=True --adv_grad_norm="infinity"
python3 client.py --model="nsl_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.1 --epochs=1 --steps_per_epoch=1 --nsl_reg=True --adv_grad_norm="infinity"
python3 client.py --model="nsl_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.1 --epochs=1 --steps_per_epoch=1 --nsl_reg=True --adv_grad_norm="infinity"


trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT
sleep 86400











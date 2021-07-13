#!/bin/bash
# the "complexity" is mapping the combinations to relevant plots/graphs for the paper
# iterate over adv_step_size, strategy, adv_reg_technique
# define each experiment even if redundant
# so fed avg + nsl, fedadagrad + nsl, fedavg + gaussian corruption, fedavg + data corruption set, fedavg + blur corruption set

python3 server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 client.py --model="nsl_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.05
python3 client.py --model="nsl_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.05
python3 client.py --model="nsl_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.05
python3 client.py --model="nsl_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.05
python3 client.py --model="nsl_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.05
python3 client.py --model="nsl_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.05
python3 client.py --model="nsl_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.05
python3 client.py --model="nsl_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.05
python3 client.py --model="nsl_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.05
python3 client.py --model="nsl_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.05

python3 server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.05 
python3 client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.05
python3 client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.05
python3 client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.05
python3 client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.05
python3 client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.05
python3 client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.05
python3 client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.05
python3 client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.05
python3 client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.05


trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT
sleep 86400











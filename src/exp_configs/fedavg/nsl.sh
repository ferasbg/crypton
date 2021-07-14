#!/bin/bash

# change the epochs parameter to 1 and set steps_per_epoch=0. Do this once you've written working code that gets the plot data for each client, as well as the server-side data. This may require flwr-side custom configuration, so keep in mind.
# metadata: ε=range(0.05, 1), norm: l-inf, l2; 10 * exp_config_count = 200 net comm rounds?
# if I need to scale up in terms of my range, that is a simple fix. Either add redundant code or iterate at the exp config level.
# flwr, write to log file or log object (and make checks based on ArgumentParser).
# Specify where each data point is accessed. Write data to logfiles relevant to target plots. 
# server-side model over comm rounds --> write to server_model_data.txt file etc

# FedAvg + NSL
python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.05 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.05 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.05 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.05 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.05 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.05 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.05 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.05 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.05 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.05 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client" 
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.15 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.15 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.15 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.15 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.15 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.15 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.15 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.15 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.15 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.15 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.2 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.2 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client" 
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.2 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.2 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.2 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.2 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.2 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.2 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.2 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.2 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.25 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.25 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.25 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.25 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.25 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.25 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.25 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.25 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.25 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.25 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.3 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.3 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client" 
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.3 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.3 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.3 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.3 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.3 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.3 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.3 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.3 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.35 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.35 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.35 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.35 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.35 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.35 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.35 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.35 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.35 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.35 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.4 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.4 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client" 
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.4 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.4 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.4 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.4 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.4 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.4 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.4 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.4 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.45 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.45 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.45 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.45 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.45 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.45 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.45 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.45 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.45 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.45 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.5 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.5 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client" 
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.5 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.5 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.5 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.5 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.5 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.5 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.5 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.5 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.55 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.55 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.55 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.55 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.55 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.55 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.55 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.55 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.55 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.55 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.6 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.6 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client" 
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.6 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.6 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.6 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.6 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.6 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.6 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.6 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.6 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.65 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.65 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.65 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.65 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.65 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.65 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.65 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.65 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.65 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.65 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.7 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.7 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client" 
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.7 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.7 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.7 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.7 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.7 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.7 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.7 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.7 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.75 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.75 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.75 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.75 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.75 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.75 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.75 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.75 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.75 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.75 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.8 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.8 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client" 
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.8 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.8 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.8 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.8 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.8 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.8 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.8 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.8 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.85 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.85 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.85 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.85 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.85 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.85 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.85 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.85 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.85 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.85 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.9 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.9 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client" 
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.9 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.9 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.9 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.9 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.9 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.9 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.9 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.9 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.95 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.95 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.95 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.95 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.95 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.95 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.95 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.95 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.95 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.95 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"


# FedAvg + NSL; norm: l2
python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.05 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.05 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.05 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.05 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.05 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.05 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.05 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.05 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.05 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.05 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client" 
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.15 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.15 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.15 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.15 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.15 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.15 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.15 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.15 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.15 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.15 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.2 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.2 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client" 
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.2 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.2 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.2 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.2 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.2 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.2 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.2 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.2 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.25 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.25 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.25 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.25 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.25 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.25 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.25 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.25 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.25 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.25 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.3 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.3 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client" 
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.3 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.3 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.3 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.3 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.3 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.3 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.3 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.3 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.35 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.35 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.35 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.35 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.35 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.35 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.35 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.35 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.35 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.35 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.4 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.4 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client" 
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.4 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.4 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.4 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.4 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.4 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.4 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.4 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.4 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.45 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.45 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.45 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.45 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.45 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.45 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.45 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.45 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.45 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.45 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.5 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.5 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client" 
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.5 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.5 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.5 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.5 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.5 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.5 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.5 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.5 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.55 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.55 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.55 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.55 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.55 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.55 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.55 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.55 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.55 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.55 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.6 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.6 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client" 
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.6 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.6 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.6 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.6 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.6 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.6 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.6 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.6 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.65 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.65 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.65 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.65 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.65 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.65 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.65 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.65 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.65 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.65 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.7 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.7 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client" 
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.7 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.7 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.7 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.7 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.7 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.7 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.7 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.7 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.75 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.75 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.75 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.75 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.75 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.75 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.75 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.75 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.75 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.75 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.8 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.8 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client" 
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.8 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.8 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.8 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.8 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.8 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.8 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.8 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.8 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.85 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.85 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.85 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.85 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.85 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.85 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.85 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.85 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.85 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.85 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.9 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.9 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client" 
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.9 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.9 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.9 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.9 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.9 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.9 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.9 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.9 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.95 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.95 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.95 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.95 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.95 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.95 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.95 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.95 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.95 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.95 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"
python3 ../../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client"

trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT
sleep 86400

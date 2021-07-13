# !/bin/bash

# gaussian_noise --> invokes different model that processes GaussianNoise(stddev=0.2)
noise_corruption_set = ["shot_noise" "impulse_noise" "speckle_noise"]

# FedAvg + GaussianNoise; norm = infinity
python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.05 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.05 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.05 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.05 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.05 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.05 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.05 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.05 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.05 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.05 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client" 
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.15 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.15 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.15 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.15 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.15 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.15 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.15 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.15 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.15 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.15 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.2 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.2 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client" 
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.2 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.2 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.2 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.2 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.2 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.2 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.2 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.2 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.25 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.25 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.25 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.25 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.25 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.25 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.25 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.25 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.25 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.25 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.3 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.3 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client" 
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.3 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.3 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.3 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.3 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.3 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.3 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.3 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.3 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.35 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.35 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.35 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.35 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.35 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.35 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.35 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.35 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.35 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.35 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.4 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.4 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client" 
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.4 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.4 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.4 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.4 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.4 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.4 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.4 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.4 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.45 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.45 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.45 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.45 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.45 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.45 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.45 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.45 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.45 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.45 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.5 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.5 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client" 
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.5 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.5 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.5 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.5 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.5 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.5 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.5 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.5 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.55 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.55 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.55 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.55 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.55 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.55 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.55 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.55 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.55 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.55 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.6 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.6 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client" 
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.6 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.6 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.6 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.6 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.6 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.6 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.6 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.6 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.65 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.65 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.65 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.65 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.65 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.65 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.65 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.65 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.65 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.65 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.7 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.7 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client" 
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.7 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.7 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.7 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.7 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.7 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.7 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.7 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.7 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.75 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.75 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.75 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.75 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.75 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.75 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.75 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.75 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.75 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.75 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.8 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.8 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client" 
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.8 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.8 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.8 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.8 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.8 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.8 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.8 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.8 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.85 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.85 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.85 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.85 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.85 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.85 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.85 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.85 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.85 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.85 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.9 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.9 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client" 
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.9 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.9 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.9 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.9 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.9 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.9 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.9 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.9 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.95 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.95 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.95 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.95 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.95 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.95 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.95 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.95 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.95 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.95 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"


# FedAvg + GaussianNoise; norm: l2
python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.05 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.05 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.05 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.05 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.05 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.05 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.05 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.05 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.05 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.05 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client" 
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.15 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.15 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.15 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.15 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.15 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.15 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.15 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.15 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.15 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.15 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.2 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.2 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client" 
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.2 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.2 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.2 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.2 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.2 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.2 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.2 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.2 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.25 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.25 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.25 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.25 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.25 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.25 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.25 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.25 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.25 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.25 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.3 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.3 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client" 
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.3 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.3 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.3 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.3 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.3 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.3 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.3 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.3 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.35 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.35 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.35 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.35 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.35 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.35 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.35 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.35 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.35 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.35 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.4 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.4 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client" 
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.4 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.4 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.4 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.4 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.4 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.4 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.4 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.4 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.45 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.45 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.45 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.45 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.45 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.45 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.45 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.45 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.45 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.45 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.5 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.5 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client" 
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.5 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.5 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.5 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.5 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.5 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.5 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.5 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.5 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.55 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.55 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.55 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.55 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.55 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.55 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.55 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.55 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.55 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.55 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.6 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.6 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client" 
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.6 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.6 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.6 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.6 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.6 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.6 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.6 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.6 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.65 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.65 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.65 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.65 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.65 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.65 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.65 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.65 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.65 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.65 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.7 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.7 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client" 
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.7 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.7 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.7 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.7 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.7 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.7 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.7 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.7 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.75 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.75 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.75 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.75 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.75 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.75 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.75 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.75 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.75 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.75 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.8 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.8 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client" 
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.8 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.8 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.8 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.8 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.8 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.8 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.8 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.8 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.85 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.85 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.85 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.85 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.85 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.85 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.85 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.85 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.85 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.85 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.9 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.9 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client" 
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.9 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.9 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.9 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.9 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.9 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.9 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.9 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.9 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.95 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.95 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.95 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.95 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.95 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.95 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.95 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.95 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.95 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.95 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"
python3 ../../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client"

trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT
sleep 86400

# FedAvg + Shot Noise

# FedAvg + Impulse Noise

# FedAvg + Speckle Noise


trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT
sleep 86400

# !/bin/bash

norms = ["infinity" "l2"]
corruption_names = []

### FedAvg + GaussianNoise
python3 ../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60

for i in `seq 0 9`; do
    echo "Starting client $i"
    python3 ../client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=${i} --adv_step_size=0.05 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=True --client="base_client" &
done

### FedAvg + Shot Noise
python3 ../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60

for i in `seq 0 9`; do
    echo "Starting client $i"
    python3 ../client.py --model="base_model" --num_clients=10 --client_partition_idx=${i} --adv_step_size=0.05 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="shot_noise" &
done

# FedAvg + Impulse Noise
python3 ../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60

for i in `seq 0 9`; do
    echo "Starting client $i"
    python3 ../client.py --model="base_model" --num_clients=10 --client_partition_idx=${i} --adv_step_size=0.05 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="impulse_noise" &
done

# FedAvg + Speckle Noise
python3 ../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
for i in `seq 0 9`; do
    echo "Starting client $i"
    python3 ../client.py --model="base_model" --num_clients=10 --client_partition_idx=${i} --adv_step_size=0.05 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="speckle_noise" &
done

trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT
sleep 86400
wait

# !/bin/bash

# FedAvg + Gaussian Blur
python3 ../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60

for i in `seq 0 9`; do
    echo "Starting client $i"
    python3 ../client.py --model="base_model" --num_clients=10 --client_partition_idx=${i} --adv_step_size=0.05 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="gaussian_blur" 
done

# FedAvg + Glass Blur
python3 ../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
for i in `seq 0 9`; do
    echo "Starting client $i"
    python3 ../client.py --model="base_model" --num_clients=10 --client_partition_idx=${i} --adv_step_size=0.05 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="glass_blur" 
done

# FedAvg + Zoom BLur 
python3 ../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
for i in `seq 0 9`; do
    echo "Starting client $i"
    python3 ../client.py --model="base_model" --num_clients=10 --client_partition_idx=${i} --adv_step_size=0.05 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="zoom_blur" 
done

# FedAvg + Defocus Blur
python3 ../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60

for i in `seq 0 9`; do
    echo "Starting client $i"
    python3 ../client.py --model="base_model" --num_clients=10 --client_partition_idx=${i} --adv_step_size=0.05 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="defocus_blur" 
done

trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT
sleep 86400

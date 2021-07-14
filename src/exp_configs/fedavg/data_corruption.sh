# !/bin/bash

## FedAvg + JPEG Compression (Corruption Type: Data/Noise/Blur, Corruption Name: JPEG Compression)
python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.05 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression" 
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.05 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.05 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.05 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.05 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.05 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.05 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.05 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.05 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.05 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression" 
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.15 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.15 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.15 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.15 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.15 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.15 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.15 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.15 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.15 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.15 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.2 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.2 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression" 
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.2 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.2 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.2 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.2 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.2 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.2 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.2 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.2 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.25 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.25 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.25 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.25 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.25 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.25 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.25 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.25 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.25 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.25 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.3 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.3 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression" 
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.3 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.3 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.3 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.3 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.3 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.3 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.3 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.3 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.35 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.35 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.35 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.35 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.35 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.35 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.35 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.35 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.35 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.35 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.4 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.4 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression" 
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.4 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.4 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.4 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.4 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.4 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.4 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.4 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.4 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.45 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.45 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.45 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.45 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.45 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.45 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.45 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.45 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.45 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.45 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.5 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.5 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression" 
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.5 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.5 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.5 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.5 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.5 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.5 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.5 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.5 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.55 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.55 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.55 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.55 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.55 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.55 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.55 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.55 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.55 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.55 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.6 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.6 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression" 
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.6 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.6 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.6 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.6 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.6 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.6 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.6 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.6 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.65 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.65 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.65 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.65 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.65 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.65 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.65 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.65 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.65 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.65 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.7 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.7 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression" 
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.7 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.7 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.7 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.7 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.7 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.7 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.7 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.7 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.75 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.75 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.75 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.75 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.75 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.75 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.75 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.75 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.75 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.75 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.8 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.8 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression" 
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.8 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.8 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.8 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.8 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.8 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.8 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.8 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.8 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.85 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.85 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.85 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.85 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.85 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.85 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.85 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.85 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.85 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.85 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.9 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.9 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression" 
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.9 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.9 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.9 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.9 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.9 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.9 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.9 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.9 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.95 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.95 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.95 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.95 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.95 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.95 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.95 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.95 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.95 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.95 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"


# FedAvg + GaussianNoise; norm: l2
python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.05 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.05 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.05 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.05 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.05 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.05 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.05 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.05 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.05 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.05 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression" 
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.15 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.15 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.15 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.15 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.15 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.15 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.15 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.15 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.15 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.15 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.2 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.2 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression" 
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.2 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.2 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.2 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.2 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.2 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.2 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.2 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.2 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.25 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.25 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.25 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.25 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.25 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.25 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.25 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.25 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.25 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.25 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.3 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.3 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression" 
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.3 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.3 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.3 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.3 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.3 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.3 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.3 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.3 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.35 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.35 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.35 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.35 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.35 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.35 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.35 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.35 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.35 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.35 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.4 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.4 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression" 
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.4 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.4 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.4 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.4 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.4 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.4 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.4 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.4 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.45 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.45 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.45 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.45 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.45 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.45 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.45 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.45 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.45 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.45 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.5 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.5 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression" 
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.5 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.5 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.5 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.5 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.5 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.5 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.5 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.5 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.55 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.55 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.55 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.55 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.55 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.55 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.55 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.55 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.55 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.55 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.6 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.6 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression" 
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.6 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.6 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.6 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.6 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.6 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.6 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.6 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.6 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.65 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.65 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.65 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.65 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.65 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.65 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.65 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.65 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.65 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.65 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.7 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.7 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression" 
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.7 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.7 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.7 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.7 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.7 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.7 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.7 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.7 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.75 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.75 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.75 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.75 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.75 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.75 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.75 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.75 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.75 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.75 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.8 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.8 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression" 
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.8 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.8 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.8 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.8 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.8 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.8 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.8 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.8 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.85 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.85 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.85 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.85 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.85 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.85 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.85 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.85 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.85 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.85 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.9 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.9 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression" 
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.9 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.9 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.9 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.9 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.9 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.9 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.9 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.9 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.95 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.95 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.95 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.95 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.95 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.95 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.95 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.95 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.95 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.95 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --nsl_reg=False --gaussian_reg=False --client="base_client" --corruption_name="jpeg_compression"

### FedAvg + Elastic Transformation
###
###
###
python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.05 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform" 
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.05 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.05 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.05 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.05 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.05 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.05 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.05 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.05 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.05 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform" 
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.15 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.15 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.15 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.15 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.15 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.15 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.15 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.15 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.15 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.15 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.2 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.2 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform" 
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.2 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.2 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.2 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.2 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.2 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.2 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.2 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.2 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.25 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.25 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.25 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.25 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.25 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.25 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.25 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.25 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.25 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.25 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.3 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.3 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform" 
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.3 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.3 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.3 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.3 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.3 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.3 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.3 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.3 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.35 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.35 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.35 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.35 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.35 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.35 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.35 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.35 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.35 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.35 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.4 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.4 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform" 
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.4 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.4 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.4 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.4 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.4 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.4 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.4 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.4 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.45 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.45 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.45 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.45 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.45 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.45 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.45 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.45 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.45 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.45 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.5 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.5 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform" 
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.5 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.5 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.5 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.5 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.5 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.5 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.5 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.5 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.55 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.55 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.55 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.55 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.55 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.55 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.55 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.55 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.55 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.55 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.6 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.6 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform" 
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.6 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.6 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.6 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.6 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.6 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.6 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.6 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.6 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.65 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.65 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.65 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.65 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.65 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.65 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.65 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.65 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.65 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.65 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.7 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.7 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform" 
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.7 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.7 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.7 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.7 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.7 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.7 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.7 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.7 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.75 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.75 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.75 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.75 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.75 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.75 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.75 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.75 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.75 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.75 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.8 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.8 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform" 
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.8 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.8 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.8 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.8 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.8 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.8 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.8 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.8 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.85 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.85 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.85 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.85 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.85 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.85 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.85 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.85 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.85 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.85 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.9 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.9 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform" 
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.9 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.9 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.9 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.9 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.9 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.9 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.9 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.9 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.95 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.95 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.95 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.95 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.95 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.95 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.95 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.95 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.95 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.95 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"


python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.05 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.05 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.05 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.05 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.05 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.05 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.05 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.05 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.05 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.05 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform" 
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.15 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.15 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.15 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.15 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.15 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.15 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.15 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.15 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.15 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.15 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.2 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.2 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform" 
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.2 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.2 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.2 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.2 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.2 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.2 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.2 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.2 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.25 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.25 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.25 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.25 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.25 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.25 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.25 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.25 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.25 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.25 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.3 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.3 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform" 
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.3 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.3 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.3 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.3 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.3 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.3 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.3 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.3 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.35 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.35 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.35 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.35 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.35 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.35 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.35 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.35 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.35 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.35 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.4 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.4 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform" 
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.4 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.4 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.4 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.4 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.4 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.4 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.4 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.4 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.45 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.45 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.45 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.45 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.45 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.45 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.45 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.45 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.45 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.45 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.5 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.5 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform" 
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.5 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.5 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.5 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.5 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.5 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.5 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.5 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.5 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.55 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.55 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.55 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.55 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.55 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.55 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.55 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.55 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.55 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.55 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.6 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.6 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform" 
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.6 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.6 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.6 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.6 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.6 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.6 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.6 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.6 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.65 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.65 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.65 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.65 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.65 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.65 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.65 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.65 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.65 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.65 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.7 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.7 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform" 
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.7 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.7 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.7 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.7 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.7 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.7 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.7 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.7 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.75 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.75 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.75 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.75 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.75 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.75 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.75 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.75 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.75 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.75 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.8 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.8 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform" 
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.8 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.8 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.8 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.8 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.8 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.8 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.8 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.8 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.85 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.85 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.85 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.85 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.85 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.85 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.85 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.85 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.85 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.85 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.9 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.9 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform" 
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.9 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.9 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.9 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.9 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.9 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.9 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.9 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.9 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.95 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.95 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.95 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.95 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.95 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.95 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.95 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.95 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.95 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.95 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"


python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.05 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform" 
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.05 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.05 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.05 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.05 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.05 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.05 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.05 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.05 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.05 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform" 
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.15 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.15 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.15 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.15 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.15 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.15 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.15 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.15 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.15 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.15 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.2 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.2 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform" 
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.2 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.2 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.2 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.2 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.2 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.2 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.2 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.2 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.25 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.25 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.25 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.25 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.25 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.25 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.25 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.25 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.25 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.25 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.3 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.3 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform" 
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.3 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.3 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.3 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.3 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.3 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.3 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.3 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.3 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.35 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.35 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.35 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.35 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.35 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.35 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.35 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.35 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.35 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.35 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.4 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.4 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform" 
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.4 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.4 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.4 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.4 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.4 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.4 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.4 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.4 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.45 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.45 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.45 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.45 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.45 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.45 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.45 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.45 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.45 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.45 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.5 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.5 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform" 
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.5 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.5 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.5 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.5 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.5 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.5 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.5 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.5 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.55 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.55 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.55 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.55 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.55 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.55 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.55 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.55 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.55 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.55 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.6 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.6 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform" 
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.6 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.6 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.6 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.6 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.6 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.6 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.6 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.6 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.65 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.65 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.65 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.65 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.65 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.65 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.65 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.65 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.65 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.65 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.7 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.7 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform" 
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.7 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.7 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.7 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.7 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.7 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.7 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.7 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.7 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.75 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.75 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.75 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.75 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.75 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.75 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.75 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.75 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.75 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.75 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.8 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.8 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform" 
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.8 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.8 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.8 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.8 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.8 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.8 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.8 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.8 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.85 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.85 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.85 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.85 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.85 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.85 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.85 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.85 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.85 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.85 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.9 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.9 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform" 
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.9 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.9 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.9 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.9 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.9 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.9 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.9 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.9 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.95 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.95 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.95 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.95 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.95 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.95 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.95 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.95 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.95 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.95 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"


python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.05 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.05 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.05 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.05 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.05 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.05 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.05 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.05 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.05 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.05 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform" 
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.15 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.15 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.15 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.15 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.15 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.15 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.15 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.15 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.15 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.15 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.2 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.2 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform" 
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.2 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.2 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.2 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.2 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.2 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.2 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.2 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.2 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.25 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.25 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.25 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.25 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.25 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.25 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.25 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.25 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.25 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.25 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.3 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.3 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform" 
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.3 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.3 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.3 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.3 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.3 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.3 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.3 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.3 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.35 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.35 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.35 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.35 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.35 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.35 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.35 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.35 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.35 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.35 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.4 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.4 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform" 
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.4 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.4 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.4 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.4 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.4 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.4 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.4 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.4 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.45 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.45 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.45 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.45 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.45 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.45 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.45 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.45 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.45 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.45 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.5 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.5 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform" 
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.5 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.5 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.5 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.5 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.5 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.5 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.5 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.5 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.55 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.55 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.55 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.55 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.55 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.55 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.55 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.55 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.55 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.55 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.6 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.6 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform" 
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.6 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.6 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.6 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.6 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.6 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.6 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.6 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.6 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.65 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.65 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.65 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.65 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.65 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.65 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.65 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.65 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.65 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.65 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.7 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.7 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform" 
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.7 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.7 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.7 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.7 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.7 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.7 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.7 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.7 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.75 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.75 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.75 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.75 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.75 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.75 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.75 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.75 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.75 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.75 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.8 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.8 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform" 
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.8 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.8 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.8 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.8 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.8 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.8 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.8 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.8 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.85 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.85 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.85 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.85 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.85 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.85 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.85 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.85 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.85 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.85 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.9 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.9 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform" 
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.9 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.9 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.9 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.9 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.9 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.9 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.9 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.9 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.95 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.95 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.95 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.95 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.95 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.95 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.95 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.95 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.95 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.95 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="elastic_transform"

# FedAvg + Pixelation 
###
###
###

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.05 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate" 
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.05 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.05 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.05 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.05 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.05 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.05 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.05 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.05 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.05 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate" 
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.15 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.15 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.15 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.15 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.15 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.15 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.15 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.15 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.15 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.15 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.2 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.2 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate" 
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.2 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.2 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.2 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.2 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.2 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.2 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.2 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.2 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.25 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.25 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.25 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.25 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.25 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.25 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.25 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.25 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.25 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.25 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.3 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.3 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate" 
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.3 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.3 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.3 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.3 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.3 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.3 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.3 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.3 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.35 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.35 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.35 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.35 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.35 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.35 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.35 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.35 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.35 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.35 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.4 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.4 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate" 
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.4 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.4 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.4 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.4 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.4 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.4 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.4 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.4 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.45 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.45 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.45 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.45 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.45 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.45 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.45 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.45 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.45 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.45 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.5 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.5 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate" 
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.5 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.5 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.5 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.5 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.5 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.5 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.5 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.5 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.55 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.55 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.55 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.55 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.55 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.55 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.55 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.55 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.55 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.55 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.6 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.6 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate" 
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.6 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.6 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.6 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.6 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.6 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.6 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.6 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.6 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.65 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.65 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.65 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.65 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.65 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.65 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.65 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.65 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.65 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.65 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.7 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.7 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate" 
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.7 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.7 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.7 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.7 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.7 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.7 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.7 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.7 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.75 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.75 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.75 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.75 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.75 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.75 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.75 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.75 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.75 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.75 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.8 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.8 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate" 
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.8 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.8 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.8 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.8 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.8 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.8 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.8 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.8 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.85 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.85 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.85 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.85 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.85 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.85 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.85 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.85 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.85 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.85 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.9 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.9 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate" 
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.9 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.9 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.9 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.9 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.9 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.9 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.9 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.9 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.95 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.95 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.95 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.95 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.95 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.95 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.95 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.95 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.95 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.95 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.05 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.05 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.05 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.05 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.05 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.05 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.05 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.05 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.05 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.05 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate" 
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.15 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.15 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.15 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.15 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.15 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.15 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.15 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.15 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.15 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.15 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.2 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.2 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate" 
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.2 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.2 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.2 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.2 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.2 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.2 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.2 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.2 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.25 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.25 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.25 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.25 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.25 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.25 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.25 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.25 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.25 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.25 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.3 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.3 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate" 
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.3 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.3 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.3 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.3 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.3 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.3 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.3 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.3 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.35 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.35 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.35 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.35 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.35 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.35 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.35 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.35 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.35 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.35 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.4 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.4 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate" 
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.4 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.4 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.4 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.4 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.4 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.4 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.4 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.4 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.45 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.45 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.45 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.45 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.45 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.45 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.45 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.45 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.45 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.45 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.5 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.5 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate" 
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.5 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.5 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.5 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.5 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.5 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.5 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.5 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.5 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.55 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.55 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.55 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.55 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.55 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.55 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.55 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.55 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.55 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.55 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.6 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.6 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate" 
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.6 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.6 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.6 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.6 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.6 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.6 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.6 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.6 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.65 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.65 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.65 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.65 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.65 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.65 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.65 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.65 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.65 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.65 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.7 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.7 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate" 
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.7 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.7 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.7 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.7 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.7 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.7 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.7 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.7 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.75 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.75 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.75 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.75 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.75 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.75 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.75 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.75 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.75 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.75 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.8 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.8 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate" 
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.8 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.8 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.8 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.8 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.8 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.8 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.8 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.8 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.85 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.85 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.85 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.85 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.85 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.85 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.85 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.85 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.85 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.85 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.9 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.9 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate" 
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.9 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.9 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.9 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.9 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.9 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.9 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.9 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.9 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.95 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.95 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.95 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.95 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.95 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.95 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.95 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.95 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.95 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.95 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"


python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.05 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate" 
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.05 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.05 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.05 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.05 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.05 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.05 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.05 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.05 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.05 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate" 
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.15 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.15 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.15 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.15 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.15 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.15 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.15 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.15 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.15 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.15 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.2 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.2 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate" 
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.2 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.2 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.2 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.2 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.2 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.2 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.2 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.2 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.25 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.25 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.25 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.25 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.25 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.25 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.25 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.25 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.25 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.25 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.3 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.3 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate" 
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.3 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.3 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.3 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.3 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.3 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.3 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.3 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.3 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.35 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.35 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.35 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.35 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.35 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.35 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.35 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.35 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.35 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.35 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.4 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.4 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate" 
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.4 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.4 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.4 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.4 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.4 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.4 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.4 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.4 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.45 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.45 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.45 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.45 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.45 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.45 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.45 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.45 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.45 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.45 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.5 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.5 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate" 
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.5 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.5 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.5 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.5 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.5 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.5 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.5 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.5 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.55 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.55 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.55 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.55 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.55 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.55 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.55 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.55 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.55 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.55 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.6 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.6 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate" 
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.6 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.6 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.6 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.6 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.6 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.6 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.6 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.6 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.65 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.65 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.65 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.65 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.65 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.65 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.65 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.65 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.65 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.65 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.7 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.7 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate" 
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.7 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.7 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.7 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.7 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.7 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.7 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.7 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.7 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.75 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.75 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.75 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.75 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.75 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.75 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.75 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.75 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.75 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.75 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.8 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.8 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate" 
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.8 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.8 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.8 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.8 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.8 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.8 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.8 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.8 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.85 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.85 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.85 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.85 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.85 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.85 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.85 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.85 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.85 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.85 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.9 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.9 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate" 
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.9 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.9 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.9 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.9 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.9 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.9 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.9 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.9 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.95 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.95 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.95 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.95 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.95 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.95 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.95 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.95 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.95 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.95 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=1 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"


python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.05 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.05 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.05 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.05 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.05 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.05 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.05 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.05 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.05 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.05 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate" 
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.15 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.15 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.15 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.15 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.15 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.15 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.15 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.15 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.15 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.15 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.2 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.2 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate" 
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.2 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.2 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.2 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.2 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.2 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.2 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.2 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.2 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.25 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.25 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.25 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.25 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.25 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.25 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.25 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.25 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.25 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.25 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.3 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.3 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate" 
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.3 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.3 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.3 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.3 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.3 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.3 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.3 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.3 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.35 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.35 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.35 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.35 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.35 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.35 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.35 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.35 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.35 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.35 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.4 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.4 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate" 
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.4 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.4 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.4 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.4 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.4 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.4 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.4 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.4 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.45 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.45 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.45 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.45 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.45 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.45 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.45 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.45 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.45 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.45 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.5 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.5 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate" 
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.5 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.5 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.5 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.5 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.5 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.5 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.5 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.5 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.55 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.55 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.55 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.55 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.55 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.55 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.55 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.55 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.55 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.55 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.6 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.6 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate" 
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.6 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.6 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.6 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.6 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.6 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.6 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.6 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.6 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.65 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.65 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.65 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.65 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.65 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.65 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.65 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.65 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.65 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.65 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.7 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.7 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate" 
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.7 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.7 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.7 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.7 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.7 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.7 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.7 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.7 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.75 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.75 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.75 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.75 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.75 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.75 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.75 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.75 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.75 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.75 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.8 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.8 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate" 
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.8 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.8 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.8 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.8 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.8 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.8 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.8 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.8 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.85 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.85 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.85 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.85 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.85 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.85 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.85 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.85 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.85 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.85 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.9 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.9 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate" 
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.9 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.9 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.9 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.9 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.9 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.9 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.9 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.9 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.95 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.95 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.95 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.95 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.95 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.95 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.95 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.95 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.95 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.95 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"

python3 ../../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"
python3 ../../client.py --model="base_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=1 --adv_grad_norm="l2" --epochs=1 --steps_per_epoch=1 --client="base_client" --corruption_name="pixelate"

trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT
sleep 86400

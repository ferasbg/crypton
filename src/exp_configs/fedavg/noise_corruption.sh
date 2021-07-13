noise_corruption_set = ["shot_noise" "impulse_noise" "speckle_noise"]

python3 server.py --strategy="fedavg" --num_rounds=10 &
sleep 60

# FedAvg + GaussianNoise
python3 client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=0 --adv_step_size=0.05 --epochs=1 --steps_per_epoch=1 --client="base_client" 
python3 client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=1 --adv_step_size=0.05 --epochs=1 --steps_per_epoch=1 --client="base_client"  
python3 client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=2 --adv_step_size=0.05 --epochs=1 --steps_per_epoch=1 --client="base_client" 
python3 client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=3 --adv_step_size=0.05 --epochs=1 --steps_per_epoch=1 --client="base_client" 
python3 client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=4 --adv_step_size=0.05 --epochs=1 --steps_per_epoch=1 --client="base_client" 
python3 client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=5 --adv_step_size=0.05 --epochs=1 --steps_per_epoch=1 --client="base_client" 
python3 client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=6 --adv_step_size=0.05 --epochs=1 --steps_per_epoch=1 --client="base_client" 
python3 client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=7 --adv_step_size=0.05 --epochs=1 --steps_per_epoch=1 --client="base_client" 
python3 client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=8 --adv_step_size=0.05 --epochs=1 --steps_per_epoch=1 --client="base_client" 
python3 client.py --model="gaussian_model" --num_clients=10 --client_partition_idx=9 --adv_step_size=0.05 --epochs=1 --steps_per_epoch=1 --client="base_client" 

# FedAvg + Shot Noise

# FedAvg + Impulse Noise

# FedAvg + Speckle Noise


trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT
sleep 86400

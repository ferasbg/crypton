#!/bin/bash

python3 server.py --strategy="fedavg" --num_rounds=10 &
sleep 60
python3 client.py --model="nsl_model" --num_clients=10 --client_partition_idx=0 
python3 client.py --model="nsl_model" --num_clients=10 --client_partition_idx=1 
python3 client.py --model="nsl_model" --num_clients=10 --client_partition_idx=2 
python3 client.py --model="nsl_model" --num_clients=10 --client_partition_idx=3 
python3 client.py --model="nsl_model" --num_clients=10 --client_partition_idx=4 
python3 client.py --model="nsl_model" --num_clients=10 --client_partition_idx=5 
python3 client.py --model="nsl_model" --num_clients=10 --client_partition_idx=6 
python3 client.py --model="nsl_model" --num_clients=10 --client_partition_idx=7 
python3 client.py --model="nsl_model" --num_clients=10 --client_partition_idx=8 
python3 client.py --model="nsl_model" --num_clients=10 --client_partition_idx=9

trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT
sleep 86400











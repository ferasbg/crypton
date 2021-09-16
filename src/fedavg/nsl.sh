#!/bin/bash

python3 ../server.py --strategy="fedavg" --num_rounds=10 &
sleep 60

for i in `seq 0 9`; do
    echo "Starting client $i"
    # the partition size will affect batch_size, steps_per_epoch, and epochs
    python3 ../client.py --model="nsl_model" --num_clients=10 --client_partition_idx=${i} --adv_step_size=0.05 --adv_grad_norm="infinity" --epochs=1 --steps_per_epoch=1 --nsl_reg=True --gaussian_reg=False --client="nsl_client" &
done

trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT
sleep 86400
wait

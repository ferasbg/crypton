#!/bin/bash

# FedAvg + NSL
python3 server.py &
sleep 60

for i in `seq 0 9`; do
    echo "Starting client $i"
    # the partition size will affect batch_size, steps_per_epoch, and epochs
    python3 client.py --model="nsl_model" --num_clients=10 --client_partition_idx=${i} --adv_step_size=0.05 --epochs=1 --steps_per_epoch=1 --nsl_reg=True --adv_grad_norm="infinity" &
done

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
wait

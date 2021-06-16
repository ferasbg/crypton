#!/bin/bash

set -e

SERVER_ADDRESS="[::]:8080"
NUM_CLIENTS=1

echo "Starting $NUM_CLIENTS clients."
for ((i = 0; i < $NUM_CLIENTS; i++))
do
    echo "Starting client(cid=$i) with partition $i out of $NUM_CLIENTS clients."
    python3 client.py &
done
echo "Started $NUM_CLIENTS clients."


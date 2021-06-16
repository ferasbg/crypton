#!/bin/bash

python3 server.py & 
sleep 3
python3 client.py --partition=0 &
python3 client.py --partition=1 &
python3 client.py --partition=2 &
python3 client.py --partition=3 &
python3 client.py --partition=4 &
python3 client.py --partition=5 &
python3 client.py --partition=6 &
python3 client.py --partition=7 &
python3 client.py --partition=8 &
python3 client.py --partition=9 &

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT
sleep 86400
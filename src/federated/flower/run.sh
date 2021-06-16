NUM_CLIENTS=100

python3 server.py & 
sleep 4 

for ((i = 0; i < $NUM_CLIENTS; i++))
do
    echo "Starting client(cid=$i) with partition $i out of $NUM_CLIENTS clients."
    python3 client.py & 
done
echo "Started $NUM_CLIENTS clients."

trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT
sleep 86400 # this is 24 hours

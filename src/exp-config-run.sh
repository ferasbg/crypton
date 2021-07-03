#!/bin/bash

NUM_CLIENTS = 10

# setup iterative for loops dependent on the variables you are testing for > nesting particular exp configs where 1 variable is changed, etc;
    # example: you iterate over all the combinations of configs given that 1 variable and the other variables that ofc are not touched are held constant

# base exp config 1: l-infinity epsilon perturbation norm; should the magnitude of perturbation norms change per client? These questions should be ironed out.
python3 server.py --num_rounds=1 --strategy="fedavg" &
python3 client.py --client_partition_idx=0 --adv_grad_norm="infinity" --adv_step_size=0.05 --steps_per_epoch=1 --num_clients=10 --adv_reg=True --epochs=1 --gaussian_layer=False 
python3 client.py --client_partition_idx=1 --adv_grad_norm="infinity" --adv_step_size=0.05 --steps_per_epoch=1 --num_clients=10 --adv_reg=True --epochs=1 --gaussian_layer=False 
python3 client.py --client_partition_idx=2 --adv_grad_norm="infinity" --adv_step_size=0.05 --steps_per_epoch=1 --num_clients=10 --adv_reg=True --epochs=1 --gaussian_layer=False 
python3 client.py --client_partition_idx=3 --adv_grad_norm="infinity" --adv_step_size=0.05 --steps_per_epoch=1 --num_clients=10 --adv_reg=True --epochs=1 --gaussian_layer=False 
python3 client.py --client_partition_idx=4 --adv_grad_norm="infinity" --adv_step_size=0.05 --steps_per_epoch=1 --num_clients=10 --adv_reg=True --epochs=1 --gaussian_layer=False 
python3 client.py --client_partition_idx=5 --adv_grad_norm="infinity" --adv_step_size=0.05 --steps_per_epoch=1 --num_clients=10 --adv_reg=True --epochs=1 --gaussian_layer=False 
python3 client.py --client_partition_idx=6 --adv_grad_norm="infinity" --adv_step_size=0.05 --steps_per_epoch=1 --num_clients=10 --adv_reg=True --epochs=1 --gaussian_layer=False 
python3 client.py --client_partition_idx=7 --adv_grad_norm="infinity" --adv_step_size=0.05 --steps_per_epoch=1 --num_clients=10 --adv_reg=True --epochs=1 --gaussian_layer=False 
python3 client.py --client_partition_idx=8 --adv_grad_norm="infinity" --adv_step_size=0.05 --steps_per_epoch=1 --num_clients=10 --adv_reg=True --epochs=1 --gaussian_layer=False 
python3 client.py --client_partition_idx=9 --adv_grad_norm="infinity" --adv_step_size=0.05 --steps_per_epoch=1 --num_clients=10 --adv_reg=True --epochs=1 --gaussian_layer=False 

# l2 epsilon perturbation norm
python3 server.py --num_rounds=1 --strategy="fedavg" &
python3 client.py --client_partition_idx=0 --adv_grad_norm="l2" --adv_step_size=0.05 --steps_per_epoch=1 --num_clients=10 --adv_reg=True --epochs=1 --gaussian_layer=False 
python3 client.py --client_partition_idx=1 --adv_grad_norm="l2" --adv_step_size=0.05 --steps_per_epoch=1 --num_clients=10 --adv_reg=True --epochs=1 --gaussian_layer=False 
python3 client.py --client_partition_idx=2 --adv_grad_norm="l2" --adv_step_size=0.05 --steps_per_epoch=1 --num_clients=10 --adv_reg=True --epochs=1 --gaussian_layer=False 
python3 client.py --client_partition_idx=3 --adv_grad_norm="l2" --adv_step_size=0.05 --steps_per_epoch=1 --num_clients=10 --adv_reg=True --epochs=1 --gaussian_layer=False 
python3 client.py --client_partition_idx=4 --adv_grad_norm="l2" --adv_step_size=0.05 --steps_per_epoch=1 --num_clients=10 --adv_reg=True --epochs=1 --gaussian_layer=False 
python3 client.py --client_partition_idx=5 --adv_grad_norm="l2" --adv_step_size=0.05 --steps_per_epoch=1 --num_clients=10 --adv_reg=True --epochs=1 --gaussian_layer=False 
python3 client.py --client_partition_idx=6 --adv_grad_norm="l2" --adv_step_size=0.05 --steps_per_epoch=1 --num_clients=10 --adv_reg=True --epochs=1 --gaussian_layer=False 
python3 client.py --client_partition_idx=7 --adv_grad_norm="l2" --adv_step_size=0.05 --steps_per_epoch=1 --num_clients=10 --adv_reg=True --epochs=1 --gaussian_layer=False 
python3 client.py --client_partition_idx=8 --adv_grad_norm="l2" --adv_step_size=0.05 --steps_per_epoch=1 --num_clients=10 --adv_reg=True --epochs=1 --gaussian_layer=False 
python3 client.py --client_partition_idx=9 --adv_grad_norm="l2" --adv_step_size=0.05 --steps_per_epoch=1 --num_clients=10 --adv_reg=True --epochs=1 --gaussian_layer=False 

# process args from client.py and server.py such that they edit the variables passed to the target objects in each "component" file
# essentially run the server.py file but then shut down the server after the client set has been executed. Note that partitions work the same, but the configurations specific to the strategy selected, adversarial hyperparameters, and so on, are all different and may iteratively increase, etc.
# hardcode configurations without abstracting them to exp_config in this shell file
# now you have to explicitly specify what you will keep constant and change in terms of image corruptions, adv_reg, gaussian_layer, norm, norm value, strategy, etc if applicable
# each corruption is itself a transformation/perturbation that should be used in isolation. Then measure by changing adv_grad_norm and measuring within the range of 0-1 with incremental increases in the perturbation intensity.

# for exp_config in exp_config_set:
    # python3 server.py --num_rounds=10 --strategy="fedavg" & 
        # for (int i = 0; i < $NUM_CLIENTS; i++):

            # python3 client.py --num_partitions=$NUM_CLIENTS --client_partition=0 --adv_grad_norm="infinity" --adv_step_size=0.05 --batch_size=32 --epochs=10 --num_clients=$NUM_CLIENTS --adv_reg=True --gaussian_layer=False 
            # python3 client.py --num_partitions=$NUM_CLIENTS --client_partition=1 --adv_grad_norm="infinity" --adv_step_size=0.05 --batch_size=32 --epochs=10 --num_clients=$NUM_CLIENTS --adv_reg=True --gaussian_layer=False 
            # python3 client.py --num_partitions=$NUM_CLIENTS --client_partition=2 --adv_grad_norm="infinity" --adv_step_size=0.05 --batch_size=32 --epochs=10 --num_clients=$NUM_CLIENTS --adv_reg=True --gaussian_layer=False 
            # python3 client.py --num_partitions=$NUM_CLIENTS --client_partition=3 --adv_grad_norm="infinity" --adv_step_size=0.05 --batch_size=32 --epochs=10 --num_clients=$NUM_CLIENTS --adv_reg=True --gaussian_layer=False 
            # python3 client.py --num_partitions=$NUM_CLIENTS --client_partition=4 --adv_grad_norm="infinity" --adv_step_size=0.05 --batch_size=32 --epochs=10 --num_clients=$NUM_CLIENTS --adv_reg=True --gaussian_layer=False 
            # python3 client.py --num_partitions=$NUM_CLIENTS --client_partition=5 --adv_grad_norm="infinity" --adv_step_size=0.05 --batch_size=32 --epochs=10 --num_clients=$NUM_CLIENTS --adv_reg=True --gaussian_layer=False 
            # python3 client.py --num_partitions=$NUM_CLIENTS --client_partition=6 --adv_grad_norm="infinity" --adv_step_size=0.05 --batch_size=32 --epochs=10 --num_clients=$NUM_CLIENTS --adv_reg=True --gaussian_layer=False 
            # python3 client.py --num_partitions=$NUM_CLIENTS --client_partition=7 --adv_grad_norm="infinity" --adv_step_size=0.05 --batch_size=32 --epochs=10 --num_clients=$NUM_CLIENTS --adv_reg=True --gaussian_layer=False 
            # python3 client.py --num_partitions=$NUM_CLIENTS --client_partition=8 --adv_grad_norm="infinity" --adv_step_size=0.05 --batch_size=32 --epochs=10 --num_clients=$NUM_CLIENTS --adv_reg=True --gaussian_layer=False 
            # python3 client.py --num_partitions=$NUM_CLIENTS --client_partition=9 --adv_grad_norm="infinity" --adv_step_size=0.05 --batch_size=32 --epochs=10 --num_clients=$NUM_CLIENTS --adv_reg=True --gaussian_layer=False 



    # python3 server.py --num_rounds=10 --strategy="fedavg" & 
        # for (int i = 0; i < $NUM_CLIENTS; i++):

            # python3 client.py --num_partitions=$NUM_CLIENTS --client_partition=0 --adv_grad_norm="infinity" --adv_step_size=0.05 --batch_size=32 --epochs=10 --num_clients=$NUM_CLIENTS --adv_reg=True --gaussian_layer=False 
            # python3 client.py --num_partitions=$NUM_CLIENTS --client_partition=1 --adv_grad_norm="infinity" --adv_step_size=0.05 --batch_size=32 --epochs=10 --num_clients=$NUM_CLIENTS --adv_reg=True --gaussian_layer=False 
            # python3 client.py --num_partitions=$NUM_CLIENTS --client_partition=2 --adv_grad_norm="infinity" --adv_step_size=0.05 --batch_size=32 --epochs=10 --num_clients=$NUM_CLIENTS --adv_reg=True --gaussian_layer=False 
            # python3 client.py --num_partitions=$NUM_CLIENTS --client_partition=3 --adv_grad_norm="infinity" --adv_step_size=0.05 --batch_size=32 --epochs=10 --num_clients=$NUM_CLIENTS --adv_reg=True --gaussian_layer=False 
            # python3 client.py --num_partitions=$NUM_CLIENTS --client_partition=4 --adv_grad_norm="infinity" --adv_step_size=0.05 --batch_size=32 --epochs=10 --num_clients=$NUM_CLIENTS --adv_reg=True --gaussian_layer=False 
            # python3 client.py --num_partitions=$NUM_CLIENTS --client_partition=5 --adv_grad_norm="infinity" --adv_step_size=0.05 --batch_size=32 --epochs=10 --num_clients=$NUM_CLIENTS --adv_reg=True --gaussian_layer=False 
            # python3 client.py --num_partitions=$NUM_CLIENTS --client_partition=6 --adv_grad_norm="infinity" --adv_step_size=0.05 --batch_size=32 --epochs=10 --num_clients=$NUM_CLIENTS --adv_reg=True --gaussian_layer=False 
            # python3 client.py --num_partitions=$NUM_CLIENTS --client_partition=7 --adv_grad_norm="infinity" --adv_step_size=0.05 --batch_size=32 --epochs=10 --num_clients=$NUM_CLIENTS --adv_reg=True --gaussian_layer=False 
            # python3 client.py --num_partitions=$NUM_CLIENTS --client_partition=8 --adv_grad_norm="infinity" --adv_step_size=0.05 --batch_size=32 --epochs=10 --num_clients=$NUM_CLIENTS --adv_reg=True --gaussian_layer=False 
            # python3 client.py --num_partitions=$NUM_CLIENTS --client_partition=9 --adv_grad_norm="infinity" --adv_step_size=0.05 --batch_size=32 --epochs=10 --num_clients=$NUM_CLIENTS --adv_reg=True --gaussian_layer=False 


    # python3 server.py --num_rounds=10 --strategy="fedavg" & 
        # for (int i = 0; i < $NUM_CLIENTS; i++):

            # python3 client.py --num_partitions=$NUM_CLIENTS --client_partition=0 --adv_grad_norm="infinity" --adv_step_size=0.05 --batch_size=32 --epochs=10 --num_clients=$NUM_CLIENTS --adv_reg=True --gaussian_layer=False 
            # python3 client.py --num_partitions=$NUM_CLIENTS --client_partition=1 --adv_grad_norm="infinity" --adv_step_size=0.05 --batch_size=32 --epochs=10 --num_clients=$NUM_CLIENTS --adv_reg=True --gaussian_layer=False 
            # python3 client.py --num_partitions=$NUM_CLIENTS --client_partition=2 --adv_grad_norm="infinity" --adv_step_size=0.05 --batch_size=32 --epochs=10 --num_clients=$NUM_CLIENTS --adv_reg=True --gaussian_layer=False 
            # python3 client.py --num_partitions=$NUM_CLIENTS --client_partition=3 --adv_grad_norm="infinity" --adv_step_size=0.05 --batch_size=32 --epochs=10 --num_clients=$NUM_CLIENTS --adv_reg=True --gaussian_layer=False 
            # python3 client.py --num_partitions=$NUM_CLIENTS --client_partition=4 --adv_grad_norm="infinity" --adv_step_size=0.05 --batch_size=32 --epochs=10 --num_clients=$NUM_CLIENTS --adv_reg=True --gaussian_layer=False 
            # python3 client.py --num_partitions=$NUM_CLIENTS --client_partition=5 --adv_grad_norm="infinity" --adv_step_size=0.05 --batch_size=32 --epochs=10 --num_clients=$NUM_CLIENTS --adv_reg=True --gaussian_layer=False 
            # python3 client.py --num_partitions=$NUM_CLIENTS --client_partition=6 --adv_grad_norm="infinity" --adv_step_size=0.05 --batch_size=32 --epochs=10 --num_clients=$NUM_CLIENTS --adv_reg=True --gaussian_layer=False 
            # python3 client.py --num_partitions=$NUM_CLIENTS --client_partition=7 --adv_grad_norm="infinity" --adv_step_size=0.05 --batch_size=32 --epochs=10 --num_clients=$NUM_CLIENTS --adv_reg=True --gaussian_layer=False 
            # python3 client.py --num_partitions=$NUM_CLIENTS --client_partition=8 --adv_grad_norm="infinity" --adv_step_size=0.05 --batch_size=32 --epochs=10 --num_clients=$NUM_CLIENTS --adv_reg=True --gaussian_layer=False 
            # python3 client.py --num_partitions=$NUM_CLIENTS --client_partition=9 --adv_grad_norm="infinity" --adv_step_size=0.05 --batch_size=32 --epochs=10 --num_clients=$NUM_CLIENTS --adv_reg=True --gaussian_layer=False 


    # python3 server.py --num_rounds=10 --strategy="fedavg" & 
        # for (int i = 0; i < $NUM_CLIENTS; i++):

            # python3 client.py --num_partitions=$NUM_CLIENTS --client_partition=0 --adv_grad_norm="infinity" --adv_step_size=0.05 --batch_size=32 --epochs=10 --num_clients=$NUM_CLIENTS --adv_reg=True --gaussian_layer=False 
            # python3 client.py --num_partitions=$NUM_CLIENTS --client_partition=1 --adv_grad_norm="infinity" --adv_step_size=0.05 --batch_size=32 --epochs=10 --num_clients=$NUM_CLIENTS --adv_reg=True --gaussian_layer=False 
            # python3 client.py --num_partitions=$NUM_CLIENTS --client_partition=2 --adv_grad_norm="infinity" --adv_step_size=0.05 --batch_size=32 --epochs=10 --num_clients=$NUM_CLIENTS --adv_reg=True --gaussian_layer=False 
            # python3 client.py --num_partitions=$NUM_CLIENTS --client_partition=3 --adv_grad_norm="infinity" --adv_step_size=0.05 --batch_size=32 --epochs=10 --num_clients=$NUM_CLIENTS --adv_reg=True --gaussian_layer=False 
            # python3 client.py --num_partitions=$NUM_CLIENTS --client_partition=4 --adv_grad_norm="infinity" --adv_step_size=0.05 --batch_size=32 --epochs=10 --num_clients=$NUM_CLIENTS --adv_reg=True --gaussian_layer=False 
            # python3 client.py --num_partitions=$NUM_CLIENTS --client_partition=5 --adv_grad_norm="infinity" --adv_step_size=0.05 --batch_size=32 --epochs=10 --num_clients=$NUM_CLIENTS --adv_reg=True --gaussian_layer=False 
            # python3 client.py --num_partitions=$NUM_CLIENTS --client_partition=6 --adv_grad_norm="infinity" --adv_step_size=0.05 --batch_size=32 --epochs=10 --num_clients=$NUM_CLIENTS --adv_reg=True --gaussian_layer=False 
            # python3 client.py --num_partitions=$NUM_CLIENTS --client_partition=7 --adv_grad_norm="infinity" --adv_step_size=0.05 --batch_size=32 --epochs=10 --num_clients=$NUM_CLIENTS --adv_reg=True --gaussian_layer=False 
            # python3 client.py --num_partitions=$NUM_CLIENTS --client_partition=8 --adv_grad_norm="infinity" --adv_step_size=0.05 --batch_size=32 --epochs=10 --num_clients=$NUM_CLIENTS --adv_reg=True --gaussian_layer=False 
            # python3 client.py --num_partitions=$NUM_CLIENTS --client_partition=9 --adv_grad_norm="infinity" --adv_step_size=0.05 --batch_size=32 --epochs=10 --num_clients=$NUM_CLIENTS --adv_reg=True --gaussian_layer=False 


    # python3 server.py --num_rounds=10 --strategy="fedavg" & 
        # for (int i = 0; i < $NUM_CLIENTS; i++):

            # python3 client.py --num_partitions=$NUM_CLIENTS --client_partition=0 --adv_grad_norm="infinity" --adv_step_size=0.05 --batch_size=32 --epochs=10 --num_clients=$NUM_CLIENTS --adv_reg=True --gaussian_layer=False 
            # python3 client.py --num_partitions=$NUM_CLIENTS --client_partition=1 --adv_grad_norm="infinity" --adv_step_size=0.05 --batch_size=32 --epochs=10 --num_clients=$NUM_CLIENTS --adv_reg=True --gaussian_layer=False 
            # python3 client.py --num_partitions=$NUM_CLIENTS --client_partition=2 --adv_grad_norm="infinity" --adv_step_size=0.05 --batch_size=32 --epochs=10 --num_clients=$NUM_CLIENTS --adv_reg=True --gaussian_layer=False 
            # python3 client.py --num_partitions=$NUM_CLIENTS --client_partition=3 --adv_grad_norm="infinity" --adv_step_size=0.05 --batch_size=32 --epochs=10 --num_clients=$NUM_CLIENTS --adv_reg=True --gaussian_layer=False 
            # python3 client.py --num_partitions=$NUM_CLIENTS --client_partition=4 --adv_grad_norm="infinity" --adv_step_size=0.05 --batch_size=32 --epochs=10 --num_clients=$NUM_CLIENTS --adv_reg=True --gaussian_layer=False 
            # python3 client.py --num_partitions=$NUM_CLIENTS --client_partition=5 --adv_grad_norm="infinity" --adv_step_size=0.05 --batch_size=32 --epochs=10 --num_clients=$NUM_CLIENTS --adv_reg=True --gaussian_layer=False 
            # python3 client.py --num_partitions=$NUM_CLIENTS --client_partition=6 --adv_grad_norm="infinity" --adv_step_size=0.05 --batch_size=32 --epochs=10 --num_clients=$NUM_CLIENTS --adv_reg=True --gaussian_layer=False 
            # python3 client.py --num_partitions=$NUM_CLIENTS --client_partition=7 --adv_grad_norm="infinity" --adv_step_size=0.05 --batch_size=32 --epochs=10 --num_clients=$NUM_CLIENTS --adv_reg=True --gaussian_layer=False 
            # python3 client.py --num_partitions=$NUM_CLIENTS --client_partition=8 --adv_grad_norm="infinity" --adv_step_size=0.05 --batch_size=32 --epochs=10 --num_clients=$NUM_CLIENTS --adv_reg=True --gaussian_layer=False 
            # python3 client.py --num_partitions=$NUM_CLIENTS --client_partition=9 --adv_grad_norm="infinity" --adv_step_size=0.05 --batch_size=32 --epochs=10 --num_clients=$NUM_CLIENTS --adv_reg=True --gaussian_layer=False 


    # python3 server.py --num_rounds=10 --strategy="fedavg" & 
        # for (int i = 0; i < $NUM_CLIENTS; i++):

            # python3 client.py --num_partitions=$NUM_CLIENTS --client_partition=0 --adv_grad_norm="infinity" --adv_step_size=0.05 --batch_size=32 --epochs=10 --num_clients=$NUM_CLIENTS --adv_reg=True --gaussian_layer=False 
            # python3 client.py --num_partitions=$NUM_CLIENTS --client_partition=1 --adv_grad_norm="infinity" --adv_step_size=0.05 --batch_size=32 --epochs=10 --num_clients=$NUM_CLIENTS --adv_reg=True --gaussian_layer=False 
            # python3 client.py --num_partitions=$NUM_CLIENTS --client_partition=2 --adv_grad_norm="infinity" --adv_step_size=0.05 --batch_size=32 --epochs=10 --num_clients=$NUM_CLIENTS --adv_reg=True --gaussian_layer=False 
            # python3 client.py --num_partitions=$NUM_CLIENTS --client_partition=3 --adv_grad_norm="infinity" --adv_step_size=0.05 --batch_size=32 --epochs=10 --num_clients=$NUM_CLIENTS --adv_reg=True --gaussian_layer=False 
            # python3 client.py --num_partitions=$NUM_CLIENTS --client_partition=4 --adv_grad_norm="infinity" --adv_step_size=0.05 --batch_size=32 --epochs=10 --num_clients=$NUM_CLIENTS --adv_reg=True --gaussian_layer=False 
            # python3 client.py --num_partitions=$NUM_CLIENTS --client_partition=5 --adv_grad_norm="infinity" --adv_step_size=0.05 --batch_size=32 --epochs=10 --num_clients=$NUM_CLIENTS --adv_reg=True --gaussian_layer=False 
            # python3 client.py --num_partitions=$NUM_CLIENTS --client_partition=6 --adv_grad_norm="infinity" --adv_step_size=0.05 --batch_size=32 --epochs=10 --num_clients=$NUM_CLIENTS --adv_reg=True --gaussian_layer=False 
            # python3 client.py --num_partitions=$NUM_CLIENTS --client_partition=7 --adv_grad_norm="infinity" --adv_step_size=0.05 --batch_size=32 --epochs=10 --num_clients=$NUM_CLIENTS --adv_reg=True --gaussian_layer=False 
            # python3 client.py --num_partitions=$NUM_CLIENTS --client_partition=8 --adv_grad_norm="infinity" --adv_step_size=0.05 --batch_size=32 --epochs=10 --num_clients=$NUM_CLIENTS --adv_reg=True --gaussian_layer=False 
            # python3 client.py --num_partitions=$NUM_CLIENTS --client_partition=9 --adv_grad_norm="infinity" --adv_step_size=0.05 --batch_size=32 --epochs=10 --num_clients=$NUM_CLIENTS --adv_reg=True --gaussian_layer=False 


    # python3 server.py --num_rounds=10 --strategy="fedavg" & 
        # for (int i = 0; i < $NUM_CLIENTS; i++):

            # python3 client.py --num_partitions=$NUM_CLIENTS --client_partition=0 --adv_grad_norm="infinity" --adv_step_size=0.05 --batch_size=32 --epochs=10 --num_clients=$NUM_CLIENTS --adv_reg=True --gaussian_layer=False 
            # python3 client.py --num_partitions=$NUM_CLIENTS --client_partition=1 --adv_grad_norm="infinity" --adv_step_size=0.05 --batch_size=32 --epochs=10 --num_clients=$NUM_CLIENTS --adv_reg=True --gaussian_layer=False 
            # python3 client.py --num_partitions=$NUM_CLIENTS --client_partition=2 --adv_grad_norm="infinity" --adv_step_size=0.05 --batch_size=32 --epochs=10 --num_clients=$NUM_CLIENTS --adv_reg=True --gaussian_layer=False 
            # python3 client.py --num_partitions=$NUM_CLIENTS --client_partition=3 --adv_grad_norm="infinity" --adv_step_size=0.05 --batch_size=32 --epochs=10 --num_clients=$NUM_CLIENTS --adv_reg=True --gaussian_layer=False 
            # python3 client.py --num_partitions=$NUM_CLIENTS --client_partition=4 --adv_grad_norm="infinity" --adv_step_size=0.05 --batch_size=32 --epochs=10 --num_clients=$NUM_CLIENTS --adv_reg=True --gaussian_layer=False 
            # python3 client.py --num_partitions=$NUM_CLIENTS --client_partition=5 --adv_grad_norm="infinity" --adv_step_size=0.05 --batch_size=32 --epochs=10 --num_clients=$NUM_CLIENTS --adv_reg=True --gaussian_layer=False 
            # python3 client.py --num_partitions=$NUM_CLIENTS --client_partition=6 --adv_grad_norm="infinity" --adv_step_size=0.05 --batch_size=32 --epochs=10 --num_clients=$NUM_CLIENTS --adv_reg=True --gaussian_layer=False 
            # python3 client.py --num_partitions=$NUM_CLIENTS --client_partition=7 --adv_grad_norm="infinity" --adv_step_size=0.05 --batch_size=32 --epochs=10 --num_clients=$NUM_CLIENTS --adv_reg=True --gaussian_layer=False 
            # python3 client.py --num_partitions=$NUM_CLIENTS --client_partition=8 --adv_grad_norm="infinity" --adv_step_size=0.05 --batch_size=32 --epochs=10 --num_clients=$NUM_CLIENTS --adv_reg=True --gaussian_layer=False 
            # python3 client.py --num_partitions=$NUM_CLIENTS --client_partition=9 --adv_grad_norm="infinity" --adv_step_size=0.05 --batch_size=32 --epochs=10 --num_clients=$NUM_CLIENTS --adv_reg=True --gaussian_layer=False 

sleep 86400
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

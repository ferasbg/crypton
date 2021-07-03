#!/bin/bash
# process args from client.py and server.py such that they edit the variables passed to the target objects in each "component" file
# essentially run the server.py file but then shut down the server after the client set has been executed. Note that partitions work the same, but the configurations specific to the strategy selected, adversarial hyperparameters, and so on, are all different and may iteratively increase, etc.
# hardcode configurations without abstracting them to exp_config in this shell file
# now you have to explicitly specify what you will keep constant and change in terms of image corruptions, nsl_reg, gaussian_layer, norm, norm value, strategy, etc if applicable
# each corruption is itself a transformation/perturbation that should be used in isolation. Then measure by changing adv_grad_norm and measuring within the range of 0-1 with incremental increases in the perturbation intensity.

NUM_CLIENTS = 10

# setup iterative for loops dependent on the variables you are testing for > nesting particular exp configs where 1 variable is changed, etc;
    # example: you iterate over all the combinations of configs given that 1 variable and the other variables that ofc are not touched are held constant
    # the header loops will be based on strategy (fedavg, fedadagrad) > adversarial regularization technique > default configs with increasing adv_step_size 

# core variables: adv_grad_norm, strategy, adversarial reg. technique; make the assumption that we isolate our regularization techniques
# adversarial regularization techniques: corruption regularization (baseline), gaussian regularization ((baseline) --> can't use more than 1 nsl_reg technique at the same time), neural structured learning
# does it make sense to partition the run.sh files in terms of what variables are held constant? eg each file is for the target reg technique but in terms of all relevant configuration permutations

python3 server.py --num_rounds=1 --strategy="fedavg" &
python3 client.py --client_partition_idx=0 --adv_grad_norm="infinity" --adv_step_size=0.05 --steps_per_epoch=1 --num_clients=10 --nsl_reg=True --epochs=1 
python3 client.py --client_partition_idx=1 --adv_grad_norm="infinity" --adv_step_size=0.05 --steps_per_epoch=1 --num_clients=10 --nsl_reg=True --epochs=1
python3 client.py --client_partition_idx=2 --adv_grad_norm="infinity" --adv_step_size=0.05 --steps_per_epoch=1 --num_clients=10 --nsl_reg=True --epochs=1 
python3 client.py --client_partition_idx=3 --adv_grad_norm="infinity" --adv_step_size=0.05 --steps_per_epoch=1 --num_clients=10 --nsl_reg=True --epochs=1 
python3 client.py --client_partition_idx=4 --adv_grad_norm="infinity" --adv_step_size=0.05 --steps_per_epoch=1 --num_clients=10 --nsl_reg=True --epochs=1 
python3 client.py --client_partition_idx=5 --adv_grad_norm="infinity" --adv_step_size=0.05 --steps_per_epoch=1 --num_clients=10 --nsl_reg=True --epochs=1 
python3 client.py --client_partition_idx=6 --adv_grad_norm="infinity" --adv_step_size=0.05 --steps_per_epoch=1 --num_clients=10 --nsl_reg=True --epochs=1 
python3 client.py --client_partition_idx=7 --adv_grad_norm="infinity" --adv_step_size=0.05 --steps_per_epoch=1 --num_clients=10 --nsl_reg=True --epochs=1 
python3 client.py --client_partition_idx=8 --adv_grad_norm="infinity" --adv_step_size=0.05 --steps_per_epoch=1 --num_clients=10 --nsl_reg=True --epochs=1 
python3 client.py --client_partition_idx=9 --adv_grad_norm="infinity" --adv_step_size=0.05 --steps_per_epoch=1 --num_clients=10 --nsl_reg=True --epochs=1 

# change the adv_step_size so that you can measure for the graph that has the perturbation epsilon norm value against server-side federated accuracy-under-attack or l-inf epsilon-robust accuracy and l2 epsilon-robust accuracy
python3 server.py --num_rounds=1 --strategy="fedavg" &
python3 client.py --client_partition_idx=0 --adv_grad_norm="infinity" --adv_step_size=0.1 --steps_per_epoch=1 --num_clients=10 --nsl_reg=True --epochs=1 
python3 client.py --client_partition_idx=1 --adv_grad_norm="infinity" --adv_step_size=0.1 --steps_per_epoch=1 --num_clients=10 --nsl_reg=True --epochs=1 
python3 client.py --client_partition_idx=2 --adv_grad_norm="infinity" --adv_step_size=0.1 --steps_per_epoch=1 --num_clients=10 --nsl_reg=True --epochs=1 
python3 client.py --client_partition_idx=3 --adv_grad_norm="infinity" --adv_step_size=0.1 --steps_per_epoch=1 --num_clients=10 --nsl_reg=True --epochs=1 
python3 client.py --client_partition_idx=4 --adv_grad_norm="infinity" --adv_step_size=0.1 --steps_per_epoch=1 --num_clients=10 --nsl_reg=True --epochs=1 
python3 client.py --client_partition_idx=5 --adv_grad_norm="infinity" --adv_step_size=0.1 --steps_per_epoch=1 --num_clients=10 --nsl_reg=True --epochs=1 
python3 client.py --client_partition_idx=6 --adv_grad_norm="infinity" --adv_step_size=0.1 --steps_per_epoch=1 --num_clients=10 --nsl_reg=True --epochs=1 
python3 client.py --client_partition_idx=7 --adv_grad_norm="infinity" --adv_step_size=0.1 --steps_per_epoch=1 --num_clients=10 --nsl_reg=True --epochs=1 
python3 client.py --client_partition_idx=8 --adv_grad_norm="infinity" --adv_step_size=0.1 --steps_per_epoch=1 --num_clients=10 --nsl_reg=True --epochs=1 
python3 client.py --client_partition_idx=9 --adv_grad_norm="infinity" --adv_step_size=0.1 --steps_per_epoch=1 --num_clients=10 --nsl_reg=True --epochs=1 

python3 server.py --num_rounds=1 --strategy="fedavg" &
python3 client.py --client_partition_idx=0 --adv_grad_norm="l2" --adv_step_size=0.05 --steps_per_epoch=1 --num_clients=10 --nsl_reg=True --epochs=1 
python3 client.py --client_partition_idx=1 --adv_grad_norm="l2" --adv_step_size=0.05 --steps_per_epoch=1 --num_clients=10 --nsl_reg=True --epochs=1 
python3 client.py --client_partition_idx=2 --adv_grad_norm="l2" --adv_step_size=0.05 --steps_per_epoch=1 --num_clients=10 --nsl_reg=True --epochs=1 
python3 client.py --client_partition_idx=3 --adv_grad_norm="l2" --adv_step_size=0.05 --steps_per_epoch=1 --num_clients=10 --nsl_reg=True --epochs=1 
python3 client.py --client_partition_idx=4 --adv_grad_norm="l2" --adv_step_size=0.05 --steps_per_epoch=1 --num_clients=10 --nsl_reg=True --epochs=1 
python3 client.py --client_partition_idx=5 --adv_grad_norm="l2" --adv_step_size=0.05 --steps_per_epoch=1 --num_clients=10 --nsl_reg=True --epochs=1 
python3 client.py --client_partition_idx=6 --adv_grad_norm="l2" --adv_step_size=0.05 --steps_per_epoch=1 --num_clients=10 --nsl_reg=True --epochs=1 
python3 client.py --client_partition_idx=7 --adv_grad_norm="l2" --adv_step_size=0.05 --steps_per_epoch=1 --num_clients=10 --nsl_reg=True --epochs=1 
python3 client.py --client_partition_idx=8 --adv_grad_norm="l2" --adv_step_size=0.05 --steps_per_epoch=1 --num_clients=10 --nsl_reg=True --epochs=1 
python3 client.py --client_partition_idx=9 --adv_grad_norm="l2" --adv_step_size=0.05 --steps_per_epoch=1 --num_clients=10 --nsl_reg=True --epochs=1 


sleep 86400
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

'''
# process args from client.py and server.py such that they edit the variables passed to the target objects in each "component" file
# essentially run the server.py file but then shut down the server after the client set has been executed. Note that partitions work the same, but the configurations specific to the strategy selected, adversarial hyperparameters, and so on, are all different and may iteratively increase, etc.
# hardcode configurations without abstracting them to exp_config in this shell file
# now you have to explicitly specify what you will keep constant and change in terms of image corruptions, nsl_reg, gaussian_layer, norm, norm value, strategy, etc if applicable
# each corruption is itself a transformation/perturbation that should be used in isolation. Then measure by changing adv_grad_norm and measuring within the range of 0-1 with incremental increases in the perturbation intensity.
# setup iterative for loops dependent on the variables you are testing for > nesting particular exp configs where 1 variable is changed, etc;
    # example: you iterate over all the combinations of configs given that 1 variable and the other variables that ofc are not touched are held constant
    # the header loops will be based on strategy (fedavg, fedadagrad) > adversarial regularization technique > default configs with increasing adv_step_size 

# core variables: adv_grad_norm, strategy, adversarial reg. technique; make the assumption that we isolate our regularization techniques
# adversarial regularization techniques: corruption regularization (baseline), gaussian regularization ((baseline) --> can't use more than 1 nsl_reg technique at the same time), neural structured learning
# does it make sense to partition the run.sh files in terms of what variables are held constant? eg each file is for the target reg technique but in terms of all relevant configuration permutations

NUM_CLIENTS = 10
NUM_NORM_TYPES = 2
NORM_TYPES = ["infinity" "l2"] # l2 hasn't been tested
# the graph is in terms of comm rounds and federated server-side accuracy that iterates over every specific corruption technique in the set
NORM_RANGE = [0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95] # --> 0.05 == {$NORM_RANGE[i]}
NUM_NORM_VALUES = 10
NUM_ROUNDS = 1 # 100 after this is functional
NUM_EPOCHS = 25
NUM_STEPS_PER_EPOCH = 1
# graphs: Corruption Type vs Server-Side Federated Accuracy-Under-Attack (3 with a set of lines per graph), Neural Structured Learning vs. Server-Side Federated Accuracy-Under-Attack
BLUR_CORRUPTION_SET = ["motion_blur" "glass_blur" "zoom_blur" "gaussian_blur" "defocus_blur"] # this corruption type simulates non-clarity of the target features of the image data to act as a uniform perturbation 
DATA_CORRUPTION_SET = ["jpeg_compression" "elastic_transform" "pixelate"] # this corruption type simulates client-side device data that's been corrupted by its own data storage hardware
NOISE_CORRUPTION_SET = ["shot_noise" "impulse_noise" "speckle_noise"] # this corruption type simulates non-uniform noise to act as a natural filter for non-clear image data
ADV_REG_OPTIONS = ["Neural Structured Learning" "Corruption Learning"] 
NUM_ADV_REG_TECHNIQUES = 12 # corruptions + nsl
NUM_FEDERATED_STRATEGIES = 2 # FaultTolerantFedAvg hasn't been tested yet 
FEDERATED_STRATEGY_SET = ["fedavg"]
## EXP CONFIG ALGORITHM: for technique in techniques: for strategy in strategy_set: use the (technique, strategy) pair and execute with ascending adv_step_size for both norm types

# we can do this as a process in a main.py file or through a .sh file
# 480 executions
for (int j = 0; j < $NUM_FEDERATED_STRATEGIES; j++)
    for (int i = 0; i < $NUM_ADV_REG_TECHNIQUES; i++)
        for (int n = 0; n < $NUM_NORM_TYPES; n++)
            # execute in terms of {$NUM_ADV_REG_TECHNIQUES[i] and {$NUM_FEDERATED_STRATEGIES[j]}} and in terms of the NORM_TYPES (then execute ascending adv_step_size inside loop body)
            for (int s = 0; s < $NUM_NORM_VALUES; s++):
                python3 server.py --num_rounds=$NUM_ROUNDS --strategy=$NUM_FEDERATED_STRATEGIES[j] &
                python3 client.py --client_partition_idx=0 --adv_grad_norm={$NUM_NORM_TYPES[n]} --adv_step_size={$NORM_RANGE[s]} --steps_per_epoch=$STEPS_PER_EPOCH --num_clients=$NUM_CLIENTS --nsl_reg=True --epochs=$NUM_EPOCHS 
                python3 client.py --client_partition_idx=1 --adv_grad_norm={$NUM_NORM_TYPES[n]} --adv_step_size={$NORM_RANGE[s]} --steps_per_epoch=$STEPS_PER_EPOCH --num_clients=$NUM_CLIENTS --nsl_reg=True --epochs=$NUM_EPOCHS
                python3 client.py --client_partition_idx=2 --adv_grad_norm={$NUM_NORM_TYPES[n]} --adv_step_size={$NORM_RANGE[s]} --steps_per_epoch=$STEPS_PER_EPOCH --num_clients=$NUM_CLIENTS --nsl_reg=True --epochs=$NUM_EPOCHS 
                python3 client.py --client_partition_idx=3 --adv_grad_norm={$NUM_NORM_TYPES[n]} --adv_step_size={$NORM_RANGE[s]} --steps_per_epoch=$STEPS_PER_EPOCH --num_clients=$NUM_CLIENTS --nsl_reg=True --epochs=$NUM_EPOCHS 
                python3 client.py --client_partition_idx=4 --adv_grad_norm={$NUM_NORM_TYPES[n]} --adv_step_size={$NORM_RANGE[s]} --steps_per_epoch=$STEPS_PER_EPOCH --num_clients=$NUM_CLIENTS --nsl_reg=True --epochs=$NUM_EPOCHS 
                python3 client.py --client_partition_idx=5 --adv_grad_norm={$NUM_NORM_TYPES[n]} --adv_step_size={$NORM_RANGE[s]} --steps_per_epoch=$STEPS_PER_EPOCH --num_clients=$NUM_CLIENTS --nsl_reg=True --epochs=$NUM_EPOCHS 
                python3 client.py --client_partition_idx=6 --adv_grad_norm={$NUM_NORM_TYPES[n]} --adv_step_size={$NORM_RANGE[s]} --steps_per_epoch=$STEPS_PER_EPOCH --num_clients=$NUM_CLIENTS --nsl_reg=True --epochs=$NUM_EPOCHS 
                python3 client.py --client_partition_idx=7 --adv_grad_norm={$NUM_NORM_TYPES[n]} --adv_step_size={$NORM_RANGE[s]} --steps_per_epoch=$STEPS_PER_EPOCH --num_clients=$NUM_CLIENTS --nsl_reg=True --epochs=$NUM_EPOCHS 
                python3 client.py --client_partition_idx=8 --adv_grad_norm={$NUM_NORM_TYPES[n]} --adv_step_size={$NORM_RANGE[s]} --steps_per_epoch=$STEPS_PER_EPOCH --num_clients=$NUM_CLIENTS --nsl_reg=True --epochs=$NUM_EPOCHS 
                python3 client.py --client_partition_idx=9 --adv_grad_norm={$NUM_NORM_TYPES[n]} --adv_step_size={$NORM_RANGE[s]} --steps_per_epoch=$STEPS_PER_EPOCH --num_clients=$NUM_CLIENTS --nsl_reg=True --epochs=$NUM_EPOCHS 

sleep 86400
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT
done


'''
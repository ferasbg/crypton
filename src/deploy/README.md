# `src.deploy`
- Store utilities and configurations for running cloud instance. 
- Additionally will be responsible for storing sequential compute set for all classes and functions. 
- Run network instance, and act as the class that will have end-to-end finite sequence of compute.


`src.deploy` acts as the main system runner e.g. development client. 

## Algorithm Psuedocode for Core Engine Operation 

```python
def compute_trace(Network.ReLU,BoundedNetwork.BoundedReLU, lip, epsilon) {
	Solver.solve() # iterate over each trace
	Trace.checkTrace() # check safety  / robustness traces
	Trace.verifyTraceState() # check trace state before verif	
	Network.encrypt() # encrypt tf.Graph() e.g. network state space and network layers 
	Network.train_adversarial() # train network under adv. constraints for adv. metrics 
	MPCNetwork.checkMPCLayer() # check if all layers are compliant to MPC protocol 
	if MPCNetwork.checkMPCLayer().security_status == False:
		MPCNetwork.encryptNetwork(Network.model())

	Network.evaluate_nominal() # get nominal metrics e.g. IoU, FWIoU, mean pixelwise label acc 
	Network.evaluate_adversarial() # get adversarial robustness metrics
	Network.evaluate_verification() # get verification metrics for specification trace satisfiability state
	
	Verification.compute_reachable_set(BoundedNetwork.network) # compute reachable set
	Verification.create_symbolic_interval(BoundedNetwork.network) # create symbolic interval state for BoundedNetwork (abstract interpretation of network execution state)
	Verification.compute_iterative_interval_refinement(BoundedNetwork.network)

	# iterate over each trace specification
	for trace in SafetyProperties, RobustnessProperties:
			Verification.check_trace()
			if Network.check_trace() == False:
				Network.check_trace().getTrace().setVerificationStatus(False)
			
	}


```


## Class Components
- `src.deploy.main`: store sequential runtime computations
- `src.deploy.deploy_utils`: store utilities to connect with all input functions from server nodes (e.g. `src.verification`, `src.prediction`, `src.crypto`, `src.adversarial`)
- `src.deploy.secure_nn`: link functions from `src.crypto` and `src.prediction.nn` to build privacy-preserving DCNN
- `src.deploy.specification_auth`: use industry-standard signature to authenticate specification computed on neural network at runtime
- `src.deploy.secure_verification`: link sub-class from `src.deploy.secure_nn` and `src.verification.specification`


## Components
- T1: Compute formal specification given safety property p and t and liveness properties, both of which are subsets of trace properties which are subsets of hyperproperties, defined on 3 requirements (cryptographic, adversarial, and the network). Setup `evaluate()` and run all models under privacy-preserving scheme and compute formal specifications on encrypted network to maintain cryptographic properties of system. Compute private inference and semantic image segmentation with deep convolutional neural network and formally prove trace properties of network.


## Checks
- Initialize the monitor traces, load the encrypted DCNN model
- Authenticate verification node to traverse network variables and network object state (access from de-crypted endpoints)
- Run instance of encrypted model node, and execute sequential process of runtime de-cryption and verification through passing retrieved variables of object state and testing against trace policies and checking for violations of specifications (particularly safety, robustness, security policies & properties), and pass logged computed metrics to `src.analytics`


## Usage
- run `python3 -m secure_verification` to run instance of formal specifications on mpc network


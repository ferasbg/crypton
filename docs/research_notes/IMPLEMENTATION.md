# Crypton Core

## Deprecated Psuedocode
```python
@src.deploy('secure-verification-node')
def main() {
	Solver.solve() # setup bmc constraint-solver
	Trace.checkTrace() # check safety traces (trace property state)	
	MPCNetwork.checkMPCLayers() # check if all layers are compliant to MPC protocol 
	if MPCNetwork.checkMPCLayer().security_status == False:
		MPCNetwork.encryptNetwork(Network.model()) # MPCNetwork is sub-node of Network 

    	Verification.getSpecificationAuth(privateKey, verifierNodeState) # get auth to compute trace specifications with MPCNetwork
	# instantiate public network with open-state given trusted local environment
	network = Network()
	# instantiate adversarial threat model with projected gradient descent (perturb gradient updates, perturb input_image with uncorrelated pixelwise guassian noise) 
	projected_gradient_descent = PGD()
    threat_model = Adversarial()
    network.add(perturbation_layer)
    Adversarial.perturb_network() # other perturbation methods besides perturbing input layer 
	# convert type to mpc_network for secure training
	network = MPCNetwork(network) # any interaction (e.g. f.s. abstraction, adversarial attack) is computed on network independent of encrypted type to test robustness either way (authorizing self-inflicted adversarial attack on party nodes computing on network node)
	# initialize and iterate for each safety trace property state
	safety_trace_properties = SafetyTrace()
    CheckTrace()
    VerifyTrace()
	# initialize abstract public_network (type: mpc_network -> type: public_network) for open computational graph state
	abstract_network = AbstractNetwork(network) # if mpc_network, decrypt metadata, input vector norm and execution state necessary to generate abstraction
	# compute BMC and bounded abstract interpretation to verify safety properties on public_network (updates sent to mpc_network also update parent state of public_network)
	model_checker = BoundedNetworkSolver(network)
	# assert metrics for verifiably robust network properties and system behavior with abstract domain
	Metrics.compute_nominal() # compute segmentation metrics
	Metrics.compute_adversarial() # compute adversarial robustness, pgd_attack_metrics
	Metrics.compute_verification() # verification output given bmc and abstract interpretation, violations / sound specifications met, termination_state, counterexample_trace_state
	Metrics.compute_crypto() # kl-divergence (log prob), privacy guarantee

```



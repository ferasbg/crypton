# `src.verification`

## Class Components
- `src.verification.specification`: store safety and liveness properties and specification schema and routes to connect with `src.monitor` and `src.prediction.nn`
- `src.verification.stl`: signal temporal logic code that processes data streams (e.g. signals) in runtime given no ds deadlocks, NP-complete state space
- `src.verification.ibp`: compute verification of adversarial robustness given adversarial protocols and nodes in `src.adversarial` and network layers in `src.prediction.nn`



Compute signal temporal logic specification for runtime verification of deep convolutional network structure, based on parameters and requirements with input bound propagation
## Tasks
- T0: Implement interval bound propagation for verifying adversarial robustness properties of DNN
- T1: compute a differentiable upper bound on the violation of the specification to verify
- T2: design and implement an STL monitor trace for specification for verifying robustness of semantic image segmentation model
- T3: Compute probabilistic guarantees of adversarial robustness and cryptographic scheme

## Major Components
- System Description Formalism S
- Property Specification Formalism, describing sets of acceptable behaviors, those that satisfy the formula ϕ
- Verification Methodology: check whether all behaviors of S tracked via `src.Monitor` satisfy ϕ, via boolean satisfiability for singular behaviors, and given numeric bound propagation given probability distribution of a set of behaviors


## Research
- To summarize, the temporal signals are specific variables of the neural network (either its layers or specific scalar values in the matrices of the weights and nodes/neurons of the network (stored in tensors), need to figure this out). And for bound propagation, what is the inputs we are dealing with and what is the desired or optimal state to measure/compute the metrics for the bounds?
- Invariance properties are conditions that are met throughout the execution of a computation
- Eventuality properties are when under some initial & specific conditions, a certain event must occur. These properties have more to do with the execution and fallback behavior of the computations themselves, rather than the specific computation and its statistical and mathematical significance.
- Precedance properties specify events that are certain to occur before their proceeding events, to formalize the process of sequential execution
- In the specification, there will be a variation of the usage of the properties themselves.
- Predicates are parameterized propositions, for some input set S of constraints / conditions, there is N set of events that occur. Simply just attributes that are semantically the class attributes or member variables respective of a node that computes n functions, and mathematically indicative of the current state of the system, for its signals indicate the edges between nodes that compute n computations/functions, for which the temporal signals signify the state of a machine or program during it's execution with respect to time.
- 

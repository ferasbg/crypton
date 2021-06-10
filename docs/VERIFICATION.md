# `src.verification`
Compute finite-state abstractions for formal verification and specification of deep convolutional neural network. Convert network state into bounded abstraction / interpretation, and convert the relevant input-output norm relationships (vector of network state variables) into a specification problem and check against the trace property's requirements.


## Components
- `src.verification.specification`: store safety and liveness properties and specification schema and routes to connect with `src.monitor` and `src.prediction.nn`, define core logic and formal properties to satisfy, for which `verification.stl` will execute the instance of analyzing the network
- `src.verification.stl`: signal temporal logic code that processes data streams (e.g. signals) in runtime given no ds deadlocks, NP-complete state space
- `src.verification.reachability`: setup reachability analysis to access reachable states of network and perform property inference on de-coupled network, define functions to symbolically represent network state and member variable state of tensor objects relating to network with respect to layers, process state when required at specified timesteps and steps of workflow, check against written property formalisms in `verification.specification` and client node that stores stl in `verification.stl`


## VERIFICATION RESEARCH AND FORMULATION
- "Decision Formulation of Local Robustness: The decision version of this optimization problem states that, given a bound β and input x, the adversarial analysis problem is to find a perturbation δ such that the following formula is satisfied: [µ(δ) < β∧δ ∈ ∆] ⇒ [ fw(x+δ) 6∈ T(x)]. "
- Global Robustness: One can generalize the previous notion of robustness by universally quantifying over all inputs x, to get the following formula, for a fixed β ∀x. ∀δ. ¬ϕ(δ)
- DNN verification amounts to answering the following question: given a DNN N , which maps input vector x to output vector y, and predicates P and Q, does there exist an input x 0 such that P (x 0 ) and Q(N (x 0 )) both hold? In other words, the verification process determines whether there exists a particular input that meets the input criterion P , and that is mapped to an output that meets the output criterion Q.
- Given that the BMC (Bounded Model Checking), Sound Over-Approximation for Inner Maximization Problem, Abstract Interpretation, Symbolic Interval Analysis, are at the faults of scalability and computational cost, I have optimized these methods such that they are under some bounded constraint in terms of space and time complexity. Given that the amount of memory required for sound over-approximation increases significantly with the number of hidden nodes in a network, we had to do [insert x task that was done in order to compensate for computational cost, but still yielding high results]
- With respect to adversarial attacks, verifiable robustness can provide protection against attackers with unlimited computation/information as long as the perturbations are limited within a robustness region B(x). The attacker can use any information, draw input from arbitrary distributions, and launch unknown attacks.
- [Citation et. al] states that in order to gauge the capabilities of unbounded attackers, sound analysis techniques are used to over-estimate the solutions to the inner maximization problem for a given the robustness region $B_{\epsilon}$. The soundness ensures that no successful attack can be constructed within its $B_{\epsilon}$ input range if the analysis found no violations. To summarize at a general level, the verification techniques for adversarial and general robustness all perform a sound transformation T from the input to the output of the network $f\theta$.
- Formally, given input x ∈ X and allowable input range B(x), the transformation T is sound if following condition is true: {fθ(˜x)|x˜ ∈ B(x), ∀x ∈ X} ⊆ T(X)
- The verifiable robustness for an input pair (x, y) is defined as arg maxj∈{1,...,k} d(x) = y.
- It indicates that the output for class y for input x will always be predicted as the largest among all outputs estimated by sound over-approximation T(x). Here, d(x) denotes the worst-case outputs produced by T(x), which corresponds to the estimated worst regular loss value L(d(x), y) within input range B(x). Such sound over-approximations provide the upper bound of the inner maximization problem: max x˜∈B(x) L(fθ(˜x), y) ≤ L(d(x), y)


## Requirements
- access network State --> pass parameters / arguments of network state into each respective function that computes the verification for each trace property (methods and params differ) --> compute if the computed abstraction of the networkState violates the trace property for all trace properties defined in `verification.hyperproperties` and `verification.specification` which stores the state for each neural network property and the functions for the constraints for each trace property.
- need to figure out what specific trace properties must be written and how they will be evaluated and what inputs / parameters are needed to compute the violation of the hyperproperty for all property types (safety, robustness, liveness), need to link functions to compute on DCNN during training, need to also implement all different scenarios to collect data with regards to encrypted network, decrypted base network, and computation time for verification node, track space and time complexity for verification node and trace properties
- implement interval bound propagation for verifying robustness properties of DNN
- compute a differentiable upper and lower bound on the violation of the specification to verify
- store formal specifications as temporal properties (constraints) in `verification.specification`, and run concurrent execution with routes setup with network
- general, but must understand what must be accessed, what inputs are required, and what must be computed for other computations to occur (dependencies)
- schema: class, member functions, arguments (type def, description), return object type  + description, raises, and description of function and neighboring nodes 
- Given that there will training under the constraints of the model being encrypted and decrypted to access the object's state to compute trace properties given specifications for safety, robustness, and liveness properties, there will be tests and all metadata will be collected and tracked.
- Convert network state with abstract interpretation then convert to constraint-satisfaction / optimization problem for property checking.


### Setup
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
- Important to consider the specific methods within temporal logic given the mapping and structure of the problem in terms of verifying the model components and internals. 
- LTL is a modal logic with temporal modalities for describing the possibly infinite behavior of a reactive system
- The method for abstraction given NP-Hard complexity of observable state spaces is to symbolically represent the state of the neural network object and its corresponding variable subsets 
- There will be metrics to evaluate the properties of the network (security, safety verification policies), and it will be important to understand what is important to validate in terms of the symbolic representations of the network in Pytorch, which itself extends Tensorflow.
- Pytorch has much more modality and flexibility in terms of modifying graph state, for which the graph itself represents the nodes that are updated through gradient descent (partial derivative of set of squared difference of cost function to update weights to converge towards general-purpose accurate network, but of course there's little robustness in terms of variance)
- A progress property asserts that it is always the case that an action is eventually executed. Progress is the opposite of starvation, the name given to a concurrent programming situation in which an action is never executed.
- system-level specification —- a property over the entire system that addresses the target application

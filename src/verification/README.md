# crypton.Verification
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


## Notes
To summarize, the temporal signals are specific variables of the neural network (either its layers or specific scalar values in the matrices of the weights and nodes/neurons of the network, need to figure this out). And for bound propagation, what is the inputs we are dealing with and what is the desired or optimal state to measure/compute the metrics for the bounds?


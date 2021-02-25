# `src.prediction`
The core deep convolutional neural network for semantic image segmentation.


## Components
- `src.prediction.network`: network class
- `src.prediction.train`: train & evaluate network
- `src.prediction.abstract_network`: convert tf layers into abstract (bounded) layers to evaluate in `src.verification`


## Requirements
- compute the labels/annotations for each pixel for each input frame given a tensor (set of matrices of images), with metadata including `frame_number`, `num_cores`, `training_time`
- train model, then setup encrypted model that inherits base network from `prediction.network`, perform all operations necessary for training and testing, setup for formal specifications and encrypted training

## Research
- Logits are a probabilistic representation of the classification accuraccy of the model computed with softmax, generally can be symbolically represented with geoemtric abstraction
- Batch normalization for stablization of training network, max pooling for downsampling region of filter iterating over input frame matrix, by taking greatest value of each receptive field (of region / subset of matrix space of pixels), Dropout is meant to optimize model to avoid overfitting, which is where model stores useless generalizations
- Feature map is a representation of the mapping that corresponds to the tensor transformation in Hilbert Space. The map itself is not a representation of the signal but rather the representation of the transformation, the kernel and its learned parameters in their current state. Setup endpoints to track metrics for each timestep, for each updated state of the variables corresponding to the learning of the network.
- The kernel is a matrix that acts a factor for matrix multiplication with input channel, which is a matrix computed from a receptive field, to which there is a set of receptive fields or submatrices that make up a Tensor. The dimensions also define the rank of the representation.
- Important to setup reachable states at different time steps, will just iterate over all sequential computations, and at each timestep, I will call the specification functions to access required object state data to determine if it violates robustness, safety, and liveness properties defined. Now given that the model will just be trained on self-driving data where it's required to map each frame or environment state to maintain trained behavior (to correctly detect and segment input data, we are not concerned with any policy enforcement for control and motion / planning, and only with the perception module in this work)
- To formalize the problem, it is a reachability problem in that we need to make formal and correct abstractions of a combinatorially large state space, and then use symbolic interval analysis and signal temporal logic.
- Given the cryptographic property of an smpc scheme to train and test the model privately, how can we securely and correctly formalize and provide safety and robustness guarantees of our neural network?
- We will be concerned with a subspace or subset of variables or signals relating to the object's state, the object more specifically is the neural network, to which it can extend to maintain / guarantee security and privacy policies / properties.
- We also have to define and upper and lower bound, AND need to understand where it is necessary to access and compute the probabilistic violation of a security, safety, robustness, and liveness property, and collect metrics.
- Defining function to compute upper and lower bounds over values of the function, and that the function itself is Lipschitz continous, which is where for every pair, the absolute value of the slope of the line connecting them is not greater than the real number (the derivative of any point of the function)
- We also need to define a set of properties, and also define a set of arguments necessary to then access from the `src.prediction.nn` 
- Note that ground truth refers to each pixel in a 224x224 image being labeled (multi-label classification) with a state-value. It is a mapping technique for information specific to each 224x224 input frame.
- The key correctness problems can be divided as such: adversarial examples, safety verification, output range analysis, and robustness comparison.
- Past work has been focused on modeling the problem in a reductionist approach, and a boolean satisfiability or probabilistic satisfiability for 0 < k < n, or either it passes properties or not instead of being a distribution of points (indicating metrics regarding its Tensor states) with upper and lower bounds and distance from correctness. 
- During pre-processing, important to initialize weights, and other variables specific to the Tensor object. So the problem also occurs when there's billions of different humans or obstacles, and that in runtime, the abstraction or problem formulation has to be such that there's no lost translation but also that it meets computational constraints.
- A translation relation is a concept in model checking where
- Important to understand signal sets (variables)



## References
- https://github.com/tcwangshiqi-columbia/symbolic_interval


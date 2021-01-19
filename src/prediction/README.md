# `src.prediction`
The core deep convolutional neural network for semantic image segmentation.


## Components
- `src.prediction.nn`: store base class of DCNN
- `src.prediction.train`: train network
- `src.prediction.utils`: store utilities for statistics, math, helper methods to train / test network, routes/endpoints to retrieve/get data from network


## Other Notes
- Feature map is a representation of the mapping that corresponds to the tensor transformation in Hilbert Space. The map itself is not a representation of the signal but rather the representation of the transformation, the kernel and its learned parameters in their current state. Setup endpoints to track metrics for each timestep, for each updated state of the variables corresponding to the learning of the network.

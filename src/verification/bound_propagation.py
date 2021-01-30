# 2021 Copyright Feras Baig


class BoundPropagation():
    """
    Define and compute interval bound propagation to define constraints for trace properties to compute violations of the trace properties for t \epsilon H for H = {safety, robustness, liveness}. Make formal guarantees with upper and lower bounds that maintain reliability of the network's behavior (optimization technique), formalizing it into constraint satisfaction problem.
    Args:
        - self.upperBound
        - self.lowerBound
        - self.state_size

    Raises:

    Returns:

    References:
        - https://github.com/deepmind/interval-bound-propagation/


    """

    pass






class Bounds(BoundPropagation):
    """Compute bounds given ReLU State of Neural Network for Robustness Verification (property inference and checking)"""
    def __init__(self):
        super(Bounds, self).__init__()
        # signed 8-bit ints to signify lipschitz constant
        self.upperBound = 0
        self.lowerBound = 0


    pass

if __name__ == '__main__':
    BoundPropagation()
    Bounds()


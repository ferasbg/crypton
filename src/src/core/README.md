# CORE
- Have more control over the dataset itself before extending the features to partitions etc. Make sure that it can first work on a defined dataset and the tuple in the same script rather than working with more than enough complexity involved with the problem in the first place.
- NSL requires particular data formatting into a dict, but that creates errors with the CifarClient, unless I merge the tuple and access an iterable np.ndarray with xy_test[0] for image set and xy_test[1] for target label set.
- I can also resolve returned values by emulating the history/results dict that is returned for .eval inside a flwr.client sub-class.
- I can reduce complexity by having control over customizing how every process is executed. To an extent, working with other sample code isn't the best way to solve the problem, and especially when all the methods have been exhausted and a re-implementation with more control is a better option.
- server is not under any form of adv. regularization regardless of its initial client state.

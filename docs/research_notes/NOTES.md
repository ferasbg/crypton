# NOTES

## Misc
- there are different regularization techniques, but keep technique constant
- formalize relationship between adversarial input generation for a client and its server-side evaluation-under-attack.
- adversarial regularization is very useful for defining an explicit structure e.g. structural signals rather than single samples.
- nsl-ar structured signals provides more fine-grained information not available in feature inputs.
- Isolate the regularization techniques (Gaussian, Corruptions, Neural Structured Learning) to simplify things. If it makes sense to combine particular regularizations, then go for it. I do think that all nominal regularization techniques should be shared but all the adversarial regularization techniques should be isolated per client-server trial.
- We can assume training with robust adversarial examples makes it robust against adversarial perturbations during inference (eval), but how does this process fair when there's a privacy-specific step of using client models to locally train on this data and use a federated optimization technique for server-side evaluation? How can we utilize unsupervised/semi-supervised learning and these "structured signals" to learn hidden representations in perturbed or otherwise corrupted data (image transformations) with applied gaussian noise (these configurations exist to simulate a real-world scenario). We want to then formalize this phenomenon and the results between each experimental configuration.
- adv reg. --> how does this affect fed optimizer (regularized against adversarial attacks) and how would differences in fed optimizer affect adv. reg model? Seems like FedAdagrad is better on het. data, so if it was regularized anyway with adv. perturbation attacks, it should perform well against any uniform of non-uniform or non-bounded real-world or fixed norm perturbations.
- Note that the focus of the paper (after pivot) is not on the uniqueness of the certification with a unique system design, but rather applying formalized tests but for a particular component of the federated system eg the server model as mentioned many times before.


# ROBUSTNESS
Notes on robustness.

## Assumptions of Input Function (Neural Network)
- We assume that there is a weighted function (CNN) passing in an input image __x__ with a perturbation S, for the x itself can be a product of random transformations and Gaussian Noise.
- federated setting involves parent and child models e.g. server-side trusted aggregator and client-side models that are distillations of the parent model; understanding adv. regularization and fed. optimization to improve the server-side model as much as possible (which IS formalized through its adv. robustness) is crucial for a robust federated ML system in general 
- We want to generate robust adversarial examples as well as a robust defense by applying adversarial regularization paired with GaussianNoise and random image transformations specific to modelling real-world data. 
- Purpose: formalizations are meant to assert or certify specific adversarial robustness properties of the client models and the trusted aggregator model (server model), so we'd fit these same equations in the context of client networks and client-specific hyper-parameters etc (L = (theta, x,y)), and the value is in fitting the optimization formulations of adversarial robustness to the federated setup (as stated many times before)


## Formulas
- Typically, the problem of finding an adversarial example x ∗ for a model f at a given input x ∈ X as formulated above, can be formulated as an optimization problem in one of two ways: • Minimizing perturbation: find the closest x ∗ that alters f’s prediction. This can be encoded in constraint (2) as µ(x, x∗ ) ≤ α; • Maximizing the loss: find x ∗ which maximizes false classification. This can be encoded in the constraint (3) as L(f(x), f(x ∗ )) ≥ β.
- is it more relevant to certify robustness of the model to a set of perturbed/adversarial inputs than to focus on whether adversarial examples exist?


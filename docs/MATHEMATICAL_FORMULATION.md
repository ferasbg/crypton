# MATHEMATICAL FORMULATION
Store the mathematical formulations extrapolated from background research.

## Neural Network
- neural network function (map to an output probability with a softmax)
- sampling algorithm for clients for federated strategy
- adaptive federated optimization algorithm given adversarially regularized client models (corruption with imagedegrade/imagecorruptions, perturbation with nsl that perturbs data and converts feature representation into graph of explicit and implicit structured signals)
- neural structured learning algorithm given neural network (formalize feature decomposition, regularization --> federated strategy --> server-side parameter evaluation-under-attack (robustness))
- server-side parameter evaluation algorithm in federated setting
- checking server-side model (trusted aggregator) against server-side specifications for certification of adversarial robustness

## Specifications / Certifications
- A specification is a decision formulation that in this case, represents the state of an adversarial robustness property for a server-side trusted aggregator model in a federated environment, with respect to the adversarial examples used within the adversarial regularization technique (corruption regularization, perturbation-regularized learning, neural structured learning). 
- Variables responsible involve the federated strategy, the adversarial regularization technique, and the model architecture at the client-level.
- We can concretely make conclusions after testing our experimental configurations, and we can analyze our system from both the formal systems, complexity, and neural network interpretability perspective, but we will focus most of the discussion on reasoning and creating justifications that are as concrete as possible, and resolve as many uncertainties given all the permutations tested.
- Relationship between federated strategy and server-side model matters a lot as well as the adversarial regularization techniques used. There's many different variables to test, and it's worth keeping that in mind when running each configuration in order to measure for the effects of particular algorithms used in the system (eg strategy, adv. reg technique).
- The mathematical formulation will encompass the variables stated above, and will be formalized such that nuances and little details will be negligible in majority of the discussion, albeit analysis of each of the experimental configurations will be evaluated, and the variables held constant will be compared against each other (eg the trials per exp config).

## Neural Structured Learning
- The original image is converted into a set of nodes in which it's converted into a structured signal (implicitly defined given adversarial example) such that there's the supervised loss and the adversarial neighbor loss. The reason this architecture makes sense is that it forces the target neural network to learn accurate predictions with supervised loss while maintaining the similarity of the inputs from the same structure (of nodes of neighbors of the source image, thus minimizing the neighbor loss). 
- There's a set of samples and each sample has a set of neighbors to generate an input structure. Then the batch of labeled samples with neighbors is passed into the target neural network such that there's the sample features and the neighbor features. It seems that each neighbor is generated based on the l2 distance metric, adv_grad_norm (perturbation norm), and adv_step_size (magnitude of perturbation). The adversarial perturbation acts a function to generate the adversarial neighbors that are used to regularize the target network.
- It's still worth validating the approach to how adversarial neighbors are created by reading the code specific to the `nsl.AdversarialRegularization` class.

## Corruptions as Adversarial Regularization
- This is a baseline adversarial regularization technique used to measure how the changes in surface variations and the image corruptions used affect the client and server models. it seems that there's 10 corruptions in question that are relevant, but we also want to measure for the change in the `adv_step_size` so that being said, either way it is implied that there will be many graphs but we will combine them in terms of their class (eg in terms of corruptions, measure federated accuracy-under-attack given corruption technique applied where each line in the graph is a different corruption technique, and the adv_grad_norm is held constant but the adv_step_size is increasing on the y-axis).
- The reference for the Neural Structured Learning algorithm can be found [here](https://dl.acm.org/doi/pdf/10.1145/3437963.3441666).
- Speaking about neural network interpretability should be restricted to discussion and extrapolated in general terms rather than formal and direct extrapolations of the certification algorithms (based on the specifications for each adversarial robustness property).

## Adversarial Robustness Properties
- The properties below will evaluate and certify the adversarial robustness of the server-side trusted aggregator model only.
- I am unsure on why synthesizing formulations to certify the robustness of an adversarial example matters as much as measuring the server-side model's global and semantic robustness to norm-bounded adversarial inputs.
- We are also making the assumption that the adversary in this environment is restricted to model poisoning and model spoofing rather than maximizing the loss with access to client and/or server-side model gradients. We act with the assumption to modify our client-level architecture and our data processed to the client-level architecture rather than allowing the attacks during regularization depend on more attacks dependent on vital data (model gradients, model parameters).

### Definition 1
The first formal statement and definition.
#### Property 1: Admissibility Constraint
- "The Admissibility Constraint (1) ensures that the adversarial input x∗ belongs to the space of admissible perturbed inputs."
- Eq: x∗ ∈ X˜
- Comments: How do you know what's an "admissible" input? An input that maximizes loss? Remember how optimization desires for the opposite? Formulations here are contradicting each other, but these are general statements not included in any of the functions in this file.

#### Property 2: Distance Constraint
- "The Distance Constraint (2) constrains x∗ to be no more distant from x than α."
- Eq: D(µ(x, x∗), α)

#### Property 3: Target Behavior Constraint
- The Target Behavior Constraint (3) captures the target behavior of the adversary as a predicate A(x, x∗, β) which is true if the adversary changes the behavior of the ML model by at least β modifying x to x∗. If the three constraints hold, then we say that the ML model has failed for input x. We note that this is a so called “local” robustness property for a specific input x, as opposed to other notions of “global” robustness to changes to a population of inputs (see Dreossi et al. [2018b]; Seshia et al. [2018].
- Eq: A(x, x∗, β)


## Mathematical Notes
 

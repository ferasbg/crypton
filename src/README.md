# Crypton: Adversarially Robust Federated Optimization with Neural Structured Learning
The purpose of this work is to build an adversarially robust federated system by utilizing adaptive and non-adaptive federated optimization techniques, as well as optimizing the adversarial regularization techniques used at the client-level in order to build a robust trusted aggregator model. More specifically, we aim to use neural structured learning and adaptive federated optimization together to build an adversarially robust federated learning system that can adapt to heterogeneous, sparse, and perturbed data that would be faced in a production-level federated environment.

The connection can dilute the focus of what is being changed and what the system aims to solve. Simply put, adaptivity is assumed to reduce the number of model updates, and adversarial regularization with respect to what's being tested, exists for the purpose of meshing adv. augmentation of sorts with an adaptive strategy that aggregates the adversarial client model gradients to make the server model robust against surface variations and variations of corruptions and perturbations as a whole.


## Objective
- Execute experimental configurations to assess the most optimal configuration for a production-level federated system that is robust against adversarial attacks fixed by norm type and norm values. Though it's not fixed to norm type nor norm value. Attack vectors aren't limited to adversarial examples (adv. input generation).

## Technical Overview
- In `server.py`, the component at the server-level is the model used for server-side parameter evaluation. This is executed after the client-level process is computed in terms of its args-based configs in `simulation.py`.
- In `client.py`, the component at the client-level is split into the following subcomponents: `AdvRegClient` which is the client that uses the model of type `nsl.AdversarialRegularization`. `Client` which uses the base `tf.keras.models.Model` object for its model. 
- `DatasetConfig`, which creates the client partitions depending on the argument for the total number of clients, and passes the loaded client train & test partitions to the `Client` or `AdvRegClient` object respectively. 
- By default, the `HParams` utilities object is also used, which defines the adversarial and base network hyperparameters for the models used by the `AdvRegClient` and `Client` objects.
- In `simulation.py`, the system is formalized in terms of the server and client execution process defined with `multiprocessing.Process`. The experimental configuration defined by `create_args_parser` configures the `server_args : ArgumentParser` and `client_args : ArgumentParser` for their respective processes. 
- When executing the experimental configurations, we iterate over every combination and sequentially utilize `client.DatasetConfig` to handle data processing, `simulation.PlotsConfig` to ingest and configure the metrics for the current experimental config and creating the plots with `utils.Plot` given the `simulation.PlotConfig` object. 
- In `certify.py`, we utilize the `certify.Specification` class which defines all the adversarial robustness properties to target the subcomponents of the federated system, eg the server-side trusted aggregator model in `server.py`. The objective of the properties and their checking methods are to certify, assert, and measure the formalized adversarial robustness of the federated system and testing what system-level (adaptive/non-adaptive federated strategy, adversarial regularization technique(s)) configurations map best to its real-world production-level stability, dependability, and reliability.

## Context
- It's important to utilize adversarial regularization whether or not the data is non-IID or IID for machine learning models in production systems.

## Adversarial Regularization Techniques
We want to measure for the best combination between adversarial regularization technique and strategy.

- Target Technique: Neural Structured Learning
- Baseline 1: Data Corruption-Regularized Learning
- Baseline 2: Noise Corruption-Regularized Learning
- Baseline 3: Blur Corruption-Regularized Learning
- Control: Nominal Regularization (L2 Weight Reg, Dropout, BatchNormalization)

## Figures
We need to construct several tables that illustrate the client-side model regularization and then map that to a server-side parameter evaluation under the constraint of various strategies.


 Method | Client-Side Adversarially-Regularized Accuracy     | Client-Side Adversarially-Regularized Loss | Adversarial Step Size
| --- | ---| ---|---|
| FedAvg + Neural Structured Learning           | Null | X%                                |  Y%
| FedAvg + Gaussian Noise | Null | X%                                |  Y%
| FedAvg + Shot Noise | Null | X%                                |  Y%
| FedAvg + Impulse Noise | Null | X%                                |  Y%
| FedAvg + Speckle Noise | Null | X%                                |  Y%
| FedAvg + Motion Blur| Null | X%                                |  Y%
| FedAvg + Glass Blur | Null | X%                                |  Y%
| FedAvg + Zoom Blur | Null | X%                                |  Y%
| FedAvg + Gaussian Blur | Null | X%                                |  Y%
| FedAvg + Defocus Blur | Null | X%                                |  Y%
| FedAvg + Jpeg Compression | Null | X%                                |  Y%
| FedAvg + Elastic Transform | Null | X%                                |  Y%
| FedAvg + Pixelation | Null | X%                                |  Y%
| FedAdagrad + Neural Structured Learning           | Null | X%                                |  Y%
| FedAdagrad + Gaussian Noise | Null | X%                                |  Y%
| FedAdagrad + Shot Noise | Null | X%                                |  Y%
| FedAdagrad + Impulse Noise | Null | X%                                |  Y%
| FedAdagrad + Speckle Noise | Null | X%                                |  Y%
| FedAdagrad + Motion Blur| Null | X%                                |  Y%
| FedAdagrad + Glass Blur | Null | X%                                |  Y%
| FedAdagrad + Zoom Blur | Null | X%                                |  Y%
| FedAdagrad + Gaussian Blur | Null | X%                                |  Y%
| FedAdagrad + Defocus Blur | Null | X%                                |  Y%
| FedAdagrad + Jpeg Compression | Null | X%                                |  Y%
| FedAdagrad + Elastic Transform | Null | X%                                |  Y%
| FedAdagrad + Pixelation | Null | X%                                |  Y%

Note that we are measuring in terms of client-side adversarial regularization performance at the client-level, and the server-side parameter evaluation via adaptive and non-adaptive server-side optimization. We want to connect the idea of the two in order to converge on a robust large-scale federated system for semi-supervised computer vision algorithms that operate with real-world data. Independent of adversarial and optimization-wise adaptivity, it serves to the benefit of any large-scale computer vision tasks, independent of the medium (eg. browser, self-driving cars, etc). Let's first start with FedAdagrad and FedAvg before editing the strategies supported at the server-side, which is relatively straightforward and has minimum turnaround time.

Break up how the data is organized in order to illustrate key ideas in a compartmentalized manner. In order to simplify the experiments, make sure to account for the constant norm value, norm type, client learning rate, server learning rate, model hyperparameter size, data state, model-wise sgd momentum, l2 weight reg, epochs/rounds, partitio size, non-iid/iid data state

Note how you minimized experimental design error with respect to communication costs, model architecture, iid/non-iid data state, constants held, conditions per experiment, number of control variables



## Todos
- get plot data to work on each of the tests/experiments for the final plots
- test the .sh file that automates all the plot creation / tests and round iteration / testing
- create all the final plots given 10 rounds, then do given 1000 rounds
- make the data non-iid, and support the rest of the strategies that will be used given arg parameter (trials.sh file)
- formally note all the mathematical formulations specific to the paper, and get this checked by research-wise discord frens as well as other mentors
- make all the diagrams & tables (system architecture, algorithm set, mathematical equations, algorithm latex form, sample code)

Do this before doing any patent research and figuring out the other "parts" e.g. math + component dependencies, and so on.
Do this before figuring out technical details specific to Perseus and Neuralark MVP.
Do this before solving USACO Silver/Gold Problems.
Do this before writing any of the patent document. The patent document requires research-specific context, and that requires reading. That comes after this.

## Process
process: server-side eval (acc, loss) can be done through flwr; client-side eval (acc,loss) can be done by writing in data pre-flwr and pre-aggregation
plots to create: comm. rounds vs accuracy (default strategy: fedavg; dependent variable: reg. techniques), rounds vs server-side eval loss, rounds vs client-side loss
- add. strategies to support: fedadagrad, fedyogi, fedavgM, FedAdam
- we can compare the SGD vs adaptive server-side optimization to acknowledge more effective aggregation. We can always note down that the data was IID and other conditions that may of been more effective, but kept out of the focus for the paper.
- run trials.sh in terms of initial data with 10 clients and 10 rounds with low data (proper batch size relative to epochs and steps per epoch)
- Collect the data from the print stream in a way that you can insert the data with a fixed number of rounds for the plot

## Conditions
- Constants: norm value, norm type, and server/client learning rate are all constant

Note that the "defense" against the real-world nature of sparse & corrupted data is the reason for adversarial regularization. Evaluating between various adaptive and non-adaptive strategies has to do with efficiently updating the main model with each of the local client updates. The attack itself is the nature of the data itself, and we measure with various regularization techniques so that our model can converge well on adversarial data and also optimize at the server-side model-level given the federated/decentralized nature of the training/evaluation.

- We will record in terms of 10-round iterations in terms of the total communication rounds (4000, eg 400 epochs). Let's do 100 rounds first.
- We are changing the strategy used and the adversarial regularization technique, thus forming 13 methods per strategy, given the net attacks given all corruption types
- When measuring the "client-side" accuracy, I will take the average of the clients' accuracies at each epoch instance.
- Note that clients are also partitioned in terms of which account for the evaluation round and the fit round, eg which clients are used for client-side evaluation. How do we factor this in to how we collect the training-wise accuracy for the clients?

## Plots
- Comm Rounds vs Client Adversarially-Regularized Accuracy (comparing reg. techniques + control/nominal)
- Comm Rounds vs. Server-Side Accuracy-Under-Attack 
- Comm Rounds vs. Client-Side Adversarially-Regularized Loss (measuring for convergence behavior)
- Comm Rounds vs. Server-Side Loss-Under-Attack (measure for server-side strategy optimization against adaptive/non-adaptive strategies to aggregate client grads)
- Comm Rounds vs. Server-Side Accuracy-Under-Attack (Strategy + Adv. Reg. Technique)

## Notes
- the batch size and epoch affect the partition size and the accuracy/loss values stored in the History object for both the fit and eval function since they depend on the partitioned dataset.
- Label your outputs so that the plot creation process is smooth.
- access history object for the client-side object and compare it against the history object within the eval and fit functions of the client object
- each client will have a set of accuracy values for each accuracy value is based on the epoch which is accuracy over a subset of its partitioned dataset. We will take the average accuracies from this.
- note the distinction related to the data that flwr returns given the sampling/eval rounds from metrics.
- Paper: If you don't need many aggregation updates, and you can operate under corrupt and sparse data, then your machine learning system is robust.
- Before worrying about changing control variables (eg client, server learning rate, norm val/type, etc...), hold more variables constant and run the exp. Establish a basis before addressing further nested variations.
- The accuracy of the clients on such a low-volume partitioned dataset is creating an issue with the client model's accuracy vector calculated from the fit and eval function. What difference are we applying other than the dataset partition size? Is it properly partitioned? Is there an issue with validation steps because of the batch size? Are we forgetting a step?
- We posit that client-side optimizations via regularization will proportionally improve the adaptive server-side optimization. Regardless of the adaptivity of the strategy used, the aggregation process to optimize the server-side aggregator model diminishes in return of investment given the gradients of the clients that map to poor convergence times relative to the data sparsity and corruption states. We want to then measure for how surface variations within the corruptions affect how the client models converge and regularize, and how that affects the server-side model's ability to converge under an adversarial attack relative to similar corruptions that occur in the client-side device-wise data collection process within federated machine learning systems. 
- model under-fitting is a risk both due to the epoch size (1 due to round equivalency given partition) and given adversarial input data state
- SGD and FedAvg are non-adaptive strategies.
- We hold certain variables constant to measure the effectiveness of certain algorithms within our methods. This is more useful to keep in mind of when writing the observations/understandings in the discussion.
- Secure gRPC to secure the server-client channels, apply differential privacy to the data across all the corruption types, apply certification algorithms, then apply formal verification algorithm that unifies graph technique 
- Feedback about my explanations: they are too wordy and detract away from the cohesive balance between persuasive expression (why is this even useful, and how can this really create a trillion dollar impact). Reviewers operate on dynamical responses.... emotional invocation backed by logical expressivity.
- Verify certification techniques, minimize I.V's by focusing on the client-server connection with respect to minimizing aggregation updates while also converging with adversarial data.
- Measure for the effect of client/server lr and the IID state of the data relative to the strategy.

## Tables
- Hyperparameters (Configuration)
- Method-Wise Feature Comparison

## Comments
- if the vectors change, then we can store the average during each fit iteration. Given that we are using the .fit() function approximately for 50 iterations over 100 batches during 1 .fit() iteration
- The evaluate() function uses the test dataset and computes a regularization loss. Thus the more clients under fraction_fit, the better the client evaluation loss.
- First use the first 10 rounds to get data at the epoch-level and then average the accuracy and loss data at the client-level. 80% are used for fit, and 20% are used for evaluation. Keep this in mind when storing information and then averaging the net evaluation losses/accuracy values for each client since 10 clients run 1 epoch 10 times because it's 1 epoch per 10 round iterations 
- 50 iterations does not mean that the vector cardinality is 50 either. 
- End goal is to test the system through 1000 rounds. So 100 10 round iterations or v.v.

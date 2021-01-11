# Crypton

## Overview
The purpose of this work is to build a secure, privacy-preserving formal verification framework that can formally verify granular hyperproperties and system behavior of each deep neural network in a set of ensemble deep neural networks inclusive of all components of an autonomous system and its neighboring set of autonomous agents in a real-time, secure zero-trust approach with respect to computational efficiency.

## System Components

### `Core`

- [`Crypton.Agent`](#)
- [`Crypton.Config`](#)
- [`Crypton.Utils`](#)
- [`Crypton.Logger`](#)
- [`Crypton.Build`](#)

### `Crypton.Agent`

- [`Agent.Perception`](#)
- [`Agent.Prediction`](#)
- [`Agent.Planning`](#)
- [`Agent.Control`](#)
- [`Agent.Adversarial`](#)
- [`Agent.Async`](#)
- [`Agent.Sync`](#)
- [`Agent.Verification`](#)
- [`Agent.Validation`](#)
- [`Agent.Byzantine`](#)
- [`Agent.Analytics`](#)
- [`Agent.Crypto`](#)
- [`Agent.MultiAgent`](#)
- [`Agent.Processing`](#)
- [`Agent.Config`](#)
- [`Agent.Utils`](#)
- [`Agent.Deploy`](#)
- [`Agent.Cluster`](#)
- [`Agent.Environment`](#)
- [`Agent.Fallback`](#)



## Subcomponents


### `Crypton.Agent`

#### `Agent.Perception`
- `Agent.Perception.Localization`
- `Agent.Perception.Sensors`

In order to make sure that the system can be tested against real-world inputs, the input data each neural network n in each ensemble N in the set of ensemble networks N\epsilon A for A is the Agent, must go through empirical verification as in the ability for it to translate to a production-level system. It is vitally important for empirical testing that the data is trained and tested on real-world sensor data.

#### `Agent.Prediction`
In order to build prediction network, must integrate localization and geomapping, and computed real-time output streams from perception network given ensemble input streams from sensor fusion. 

#### `Agent.Planning`
Given the ensemble classification and recognition of environment space (traffic, neighboring agents), design an optimal driving strategy for motion planning that optimizes for local minima for least traffic congestion.  

#### `Agent.Control`
Design a set of control policies that guide the agent's angular motion, throttle, speed, and optimization techniques given state estimation from geomapping input streams, and executing the optimal observation space with respect to its neighboring agents and 3d environment space. Optimize for generalization ability. 

#### `Agent.Adversarial`
- `Agent.Adversarial.Perturbation`
- `Agent.Adversarial.InputRectification`
- `Agent.Adversarial.AdaptiveStress`
- `Agent.Adversarial.Protocols`
- `Agent.Adversarial.Robustness`

For the intent of ensuring that the agent is robust to adversarial threat models external to the local network that it is running in, design protocols to "attack" the agent's components with methods such as adversarial perturbation. Verify adversarial robustness with proof-based methods, run adaptive stress testing by optimizing for randomness and system complexity (and thus parallelizing models for scale and usage for the real-world), and handle perturbations with automatic input rectification.

#### `Agent.Async`
Handle all asynchronous, off-chain, pre-computed algorithms during initialization in `Agent.Config`.

#### `Agent.Sync`
Handle all synchronous, real-time computations necessary at runtime during agent node running on distributed cluster instance for real-time feedback. Handle all on-chain, real-time synchronous computations with respect to testing all models, and passing all output streams of output variables to different models with respect to asynchronously computed output streams.

#### `Agent.Verification`
- `Agent.Verification.Routing`
- `Agent.Verification.Adversarial`
- `Agent.Verification.FaultTolerance`
- `Agent.Verification.Specification`
    - `Agent.Verification.Specification.Trace`
        - `Agent.Verification.Specification.Trace.Perception`
        - `Agent.Verification.Specification.Trace.Prediction`
        - `Agent.Verification.Specification.Trace.Planning`
        - `Agent.Verification.Specification.Trace.Control`
- `Agent.Verification.SimulationTesting`
- `Agent.Satisfiability`

- Given each trace property node in a set for all hyperproperties, compute all specifications for all net properties and components, match against specifications/requirements/desired system behavior. Build all hyperproperties to verify each deep neural network in `Crypton.Agent`. Compute for model satisfiability for each respective deep neural network separate from written formal specifications.

#### `Agent.Validation`
Validate the computations, primarily during training phase for all the deep neural networks of `Crypton.Agent`. Continously collect logger data for tracking learning and model performance over each trial. 

#### `Agent.Byzantine`
Guarantee that there are no single points of failure when sharing privately computed nodes (represent each computation) between neighboring agents in a multi-agent network of agent nodes. Verify byzantine fault tolerance given the components and distributed cluster setting.

#### `Agent.Analytics`
Collect metrics in order to evaluate `Crypton.Agent`, as well as porting to Crypton's web client, for use cases involved in real-time, distributed network metadata to evaluate performance in real-time. Asynchronous and synchronous computations and applied statistical methods to evaluate every module in `Crypton.Agent`.   

#### `Agent.Crypto`
- `Agent.Crypto.MPC`
- `Agent.Crypto.OT`
- `Agent.Crypto.Solvency`
- `Agent.Crypto.Auth`

Encrypting each deep neural network is separate than authenticated distributed data sharing for multi-agent interaction between neighboring agent nodes K. Besides industry-standard encryption techniques for all traffic flows and data serialization, ensure that all input/output streams are secure and oblivious. Include written proof-of-security algorithm for verifying cryptographic scheme iterating over each submodule of `Crypton.Agent`. Module to handle zero-trust agent authentication for sending oblivious protocol buffers and output streams between each agent node. Solvency is for proving solvency of specific output stream without revealing output stream, ensuring zero-trust multi-agent interaction.

#### `Agent.MultiAgent`
Difference between `Agent.MultiAgent` and `Agent.Cluster.MultiAgent` is that `Agent.Cluster.MultiAgent` will communicate with Kubernetes, Apache Spark, for deployment and testing, while `Agent.MultiAgent` will define the specific algorithms for multi-agent interaction. Two separate tasks.

#### `Agent.Processing`
- setup integration of agent node instance with `Crypton.Agent.Deploy` with Apache Kafka handling data processing on Kubernetes Cluster, as well as during multi-agent interaction (sharing n computations).
- In order to ensure fault tolerance and policy enforcement algorithm for optimal data traffic flow, implement distributed load balancers for handling and sending data either between each agent node or amongst it's modules.
- Data processing & data streaming for all oblivious input/outputs of each neural net / algo for each network N.

#### `Agent.Config`
- `Agent.Config.Setup`: Automatic Scripts for Initialization & Setup of Various Experiments 
- `Agent.Config.Helpers`: Helper Methods for Config Service
    - Initialization of all agent components, and running the deployment instance, et cetera. Helpers meant to simplify and conserve readability.

#### `Agent.Utils`
Crypton's utility service for handling repetitive tasks and automating initialization and scripts for running specific operations. Static methods without instantiation.

#### `Agent.Deploy`
Middleware for Running Cluster Instance & Deployment Module Given Simulation Apparatus.
#### `Agent.Cluster`
- `Agent.Cluster.Multiagent`
Operating in concurrence to all defined actions in synchronous runtime operation, specifically defining all of the consensus and zero-trust, multi-agent interaction algorithms necessary for testing network.

#### `Agent.Environment`
- `Agent.Environment.Initialization`: Initialize environment space and variable set for each respective experiment 
- `Agent.Environment.Simulation`: Define environment space required for formal specification and agent

- Environmental variables defined for agents in multi-agent space and specifically for each component (environmental conditions).
- Work adjacently with dependencies from  `Crypton.Agent.Deploy` and `Crypton.Agent.Verification` and `Crypton.Agent`, and `Crypton.Agent.MultiAgent`
- Compartmentalize various testing environments for agent-based scenarios with respect to `Agent.Verification`, multi-agent zero-trust scenarios with `Agent.MultiAgent` and `Crypton.Agent.Deploy` with agent cluster (sharing compute amongst n agents).

#### `Agent.Fallback`
`Agent.Fallback` specifically handles the instance of the errors in order to make sure that agents are fault-tolerant and still follow the given constraints to run the tests correctly, and handling the edge cases given the factor of probabilistic randomness.

### `Crypton.Tests`
- `Crypton.Tests.Agent`

Crypton's automated tester microservice for real-time subcomponent testing for checking system behavior. Not inductive tests, but rather verifying proper functionality for meta-operations that are run at runtime and during compile time to ensure all components work correctly.

### `Crypton.Config`
Crypton's configuration service for interconnected dependency graph of all components.
### `Crypton.Utils`
Crypton's utility service for handling tasks for other microservices or modules or subcomponents. 
### `Crypton.Logger`
Crypton's logger microservice ingests ensemble traffic flow from all output streams of every respective method executed in each subcomponent. Will additionally use Datadog for automated logging for extensive logging.
### `Crypton.Build`
Crypton's automated build system for running automated tests defined in tester microservice.
### `Crypton.Client`
Crypton's custom web client for storing all synchronous and asynchronous computations, results, metrics, necessary for empirical testing and system performance evaluation. This can be productionized given infrastructure and dashboard for tracking metrics and continual learning given feed of all metrics. 

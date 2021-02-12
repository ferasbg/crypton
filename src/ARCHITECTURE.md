# ARCHITECTURE
Defined specs

## Table of Contents
- [Core Components](#core-components)
- [Mathematical Formulation](#mathematical-formulation)
- [Technical Workflow](#)

## Core Components
- [`src.deploy`](#src-deploy)
    - [secure_verification.py](#)
    - [main.py](#)
- [`src.verification`](#src-verification)
    - [specification.py](#)
    - [verification.py](#)
    - [reachability.py](#)
    - [stl.py](#)
- [`src.prediction`](#src-prediction)
    - [network.py](#)
    - [train.py](#)
    - [convert.py](#)
- [`src.crypto`](#src-crypto)
    - [mpc.py](#)
    - [mpc_net.py](#)
    - [mpc_convert.py](#)
- [`src.adversarial`](#src-adversarial)

## Todo
In each core component, define the specification for programming model and implementation, 2) the mathematical formulation of each class and function, so have a table for each class for each file in each component, and define the dependency in terms of input params and functions and classes, when they are used and how (e.g. client/entry point, finite-state abstraction, compute verification, check trace property, encrypt network)


## `src.deploy`






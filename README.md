# Variational Quantum Eigensolver with Machine Learning


##  Overview
This repository explores the intersection of Machine Learning (ML) and quantum computing by applying ML techniques to the Variational Quantum Eigensolver (VQE). VQE is a hybrid quantum-classical algorithm used to estimate the ground state energy of quantum systems, particularly useful for solving complex Hamiltonians in quantum chemistry and physics.


Our primary objective is to estimate the ground state energy for different Hamiltonians. This approach is tailored for experimental scenarios where prior knowledge of the quantum system is available (in this case, a quantum circuit represented by a set of initial parameters $\theta_{0}$​).


The workflow involves conducting experiments iteratively during the optimisation process to maximise or minimise a figure of merit, which could be energy. The overall process is illustrated in the diagram below:


![Diagram](https://github.com/babulab/QuantumOptimalControl-ML/blob/main/figures/diagram_exp.jpg?raw=true)



Currently, the repository implements optimisation techniques based on Bayesian Optimisation, Reinforcement Learning, L_BFGS_B and SPSA.

A key goal of this repository is to provide a flexible and accessible platform for research. To achieve this, the components of the algorithm are lightly packaged, allowing for easy customization and modification.

If you are interested in visiting the final results of this repository you can check out the jupyter notebook [Results](https://github.com/babulab/VariationalQuantumEigensolver-ML/blob/main/notebooks/Results.ipynb).


## Future Updates 

We plan to enhance the repository with the following features:

    - A larger benchmark to compare different optimisation techniques.
    - Improvements to the existing Bayesian Optimisation framework, including:
        - Parallelize training of the GPs associated with the observables
        - Enhancing the evaluation of GP performance
    

This repository is based on my research project,‘Observable-Guided Bayesian Optimisation for Quantum Circuits Fidelity and Ground State Energy Estimation’, conducted during my Master's degree at Imperial College London. I would like to extend my gratitude to Florian M. for his supervision throughout this project.


## Requirements

- qiskit    >=1.2 
- qiskit_algorithms    >=0.3
- gpytorch  >=1.12
- botorch   >= 0.11
- SMT: Surrogate Modeling Toolbox   >=2.6
- matplotlib    >=3.9
- seaborn   >=3.9
- gym   >=0.26
- stable_baselines3    >=2.3
- mlflow    >=2.16 


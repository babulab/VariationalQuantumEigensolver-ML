# Variational Quantum Eigensolver with Machine Learning


##  Overview
This repository explores the intersection of **Machine Learning (ML)** and quantum computing by applying ML techniques to the **Variational Quantum Eigensolver (VQE)*. VQE is a hybrid quantum-classical algorithm used to estimate the ground state energy of quantum systems, particularly useful for solving complex Hamiltonians in quantum chemistry and physics.

The focus here is on estimating the ground state energy of two key molecules: hydrogen $H_2$ and water $H_2O$. This approach reflects experimental realities where the ansatz is fixed, meaning the number of parameters remains constant throughout the VQE process.

A central motivation for this work is practical applicability in resource-constrained quantum experiments, where performance must be maximised using a limited number of iterations rather than relying on convergence of the objective function.

The general optimisation framework is outlined in the figure below. While Bayesian Optimisation is shown, the architecture supports other strategies with minimal changes:

![Diagram](https://github.com/babulab/VariationalQuantumEigensolver-ML/blob/main/figures/diagram_exp.jpg?raw=true)



## ✅ Currently Implemented Optimisation Techniques

- Bayesian Optimisation
- Reinforcement Learning
- Simultaneous Perturbation Stochastic Approximation (SPSA)

A modular and flexible design ensures that components are loosely coupled, enabling easy customisation and experimentation. This makes the repository suitable for both research and prototyping of hybrid quantum-classical optimisation strategies.

---

If you're interested in the results, the good news is that they are summarised below. The plot below compares the performance of several optimisers for estimating the ground state energy of $H_2$. Most methods converge to similar values, though none reach the true ground state due to the limited expressiveness of the fixed ansatz.


For more advanced systems like $H_2O$, convergence becomes even more challenging due to the larger Hamiltonian and more complex optimisation landscape. However, PPO-based methods show promising results under these constraints.

> ⚠️ Note: These findings are problem-specific and should not be generalised without considering the constraints, computational limitations, and further discussion in the accompanying [results notebook](https://github.com/babulab/VariationalQuantumEigensolver-ML/blob/main/notebooks/results.ipynb).

![Figure](https://github.com/babulab/VariationalQuantumEigensolver-ML/blob/main/figures/results_h2_eigenvalue.png?raw=true)

---

## 🚀 Getting Started

### Prerequisites

To install the required dependencies, it is recommended to use the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```


### Quick Start

To easily explore how the library works, run one of the basic training notebooks:

- `notebooks/train_bo.ipynb` – Run a simple Bayesian Optimisation example
- `notebooks/train_rl.ipynb` – Run a basic Reinforcement Learning example

These notebooks provide a hands-on way to understand the optimization process.

### Advanced Usage

If you want to include **hyperparameter optimization**, you can run the Python scripts directly:

- `src/bo/train_bo.py` – Bayesian Optimisation with hyperparameter tuning
- `src/rl/train_rl.py` – Reinforcement Learning with hyperparameter tuning
- `src/others/train_others.py` – Classical Optimisers (SPSA, L-BFGS-B)

---

## 🔮 Future Updates

A major planned enhancement is the implementation of an Adaptive Ansatz approach (e.g., ADAPT-VQE), which can dynamically grow the ansatz to improve expressiveness. Additional directions include:

- Integration of gradient-based VQE variants
- Noisy simulations and real-hardware interfacing
---

## Acknowledgments

This repository is based on my Master's research project:  
*Observable-Guided Bayesian Optimisation for Quantum Circuits Fidelity and Ground State Energy Estimation*  
conducted at **Imperial College London**.

Special thanks to **Florian M.** for his supervision and guidance throughout the project.




## Requirements

- qiskit    >=1.2 
- qiskit_algorithms    >=0.3
- qiskit-nature    >=0.7.2
- qiskit-aer    >=0.15.0
- gpytorch  >=1.12
- botorch   >= 0.11
- SMT: Surrogate Modeling Toolbox   >=2.6
- matplotlib    >=3.9
- seaborn   >=3.9
- gym   >=0.26
- stable_baselines3    >=2.3
- mlflow    >=2.16 
- optuna    >=4.2 

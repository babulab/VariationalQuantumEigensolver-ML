import numpy as np
from optimisation_others import *
import yaml
import optuna
from optuna.samplers import TPESampler
import os
import json
import sys
sys.path.insert(1, '../')
from utils.functions import CustomEncoder, load_yaml_config
from utils.circuit_utils import create_circuit_ansatz
from utils.molecule_utils import setup_molecule_and_hamiltonian
from utils.ansatz_utils import build_ansatz

from scipy.linalg import eigh
from qiskit.quantum_info import Statevector
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.operators import PolynomialTensor
from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock
from qiskit.circuit.library import TwoLocal




def objective(trial):
    if config["vqe"]["type_optimiser"]== 'spsa':
        optimiser_fn = SPSA
        # Sample hyperparameters from Optuna
        hyperparams = {
            "blocking": trial.suggest_categorical("blocking", [True, False]),
            "trust_region": trial.suggest_categorical("trust_region", [True, False]),
            "learning_rate": trial.suggest_loguniform("learning_rate", 1e-3, 1.0),
            "allowed_increase": trial.suggest_uniform("allowed_increase", 0.01, 0.1),
            "perturbation": trial.suggest_loguniform("perturbation", 1e-3, 1.0),
            "resamplings": trial.suggest_int("resamplings", 1, 3),
            "maxiter": config["vqe"]["num_iters"],  # Read from YAML
            }

    elif  config["vqe"]["type_optimiser"]== 'l_bfgs_b':
        optimiser_fn = L_BFGS_B
        hyperparams = {
            "maxiter": config["vqe"]["num_iters"],  # Read from YAML
            "ftol": trial.suggest_loguniform("ftol", 1e-10, 1e-2),  
            "maxfun": trial.suggest_int("maxfun", 100, 5000),      
            "iprint": -1,  
            }
    else:
        raise ValueError(f"Unsupported optimiser: {config['vqe']['type_optimiser']}")

    # Number of optimiser runs per hyperparameter set
    n_run = config["vqe"]["n_run"]  
    metrics = {'history_best_eigenvalue_runs':[],
           'history_best_thetas_runs':[],
           'best_eigenvalue_runs':[],
           'best_thetas_runs': []}
    
    for iter_run in range(n_run):
        print(f"Run {iter_run+1}/{n_run}")

        # Initialize Optimiser
        optimiser = ClassicOptimiser(
                optimiser_fn=optimiser_fn,
                circuit_fn=create_circuit_ansatz,
                ansatz=ansatz,
                n_thetas=config["vqe"]["n_thetas"],
                initial_state=initial_state,
                H=H,
                num_qubits=config["vqe"]["num_qubits"],
                hyperparams=hyperparams,
                )

        # Run optimisation
        optimal_params, eigenvalue, info = optimiser.optimise()

        print("Optimal parameters found:", optimal_params)
        print("Optimal eigenvalue found:", eigenvalue)

        log_best_thetas = np.array(info['best_log']['thetas'] )
        log_best_eigenvalue = np.array(info['best_log']['eigenvalue'] )

        metrics['history_best_thetas_runs'].append(log_best_thetas)
        metrics['history_best_eigenvalue_runs'].append(log_best_eigenvalue)
        metrics['best_thetas_runs'].append(optimal_params)
        metrics['best_eigenvalue_runs'].append(eigenvalue)

    # Compute the average best eigenvalue over n_run executions
    avg_eigenvalue = np.mean(metrics["best_eigenvalue_runs"])
    trial.set_user_attr("metrics", metrics)

    return avg_eigenvalue  # minimize this


if __name__ == "__main__":
    config = load_yaml_config("../../configs/config_train_others.yaml")  # Load YAML config

    H, mapper, problem = setup_molecule_and_hamiltonian(config)
    ansatz, initial_state = build_ansatz(config, mapper, problem)

    eigenvalues, eigenvectors = eigh(H.to_matrix())
    ground_state_energy = np.min(eigenvalues)
    ground_state = eigenvectors[:, np.argmin(eigenvalues)]
    #ground_state_energy = -74.96 #H2O

    study = optuna.create_study(direction="maximize", sampler=TPESampler(multivariate=True))  # Maximize fidelity
    study.optimize(objective, n_trials=config["vqe"]["n_trials"])

    print("Best Hyperparameters:", study.best_params)
    print("Best Eigenvalue:", study.best_value)

    # Get best trial
    best_trial = study.best_trial
    best_params = best_trial.params
    best_eigenvalue = best_trial.value
    best_metrics = best_trial.user_attrs["metrics"]


    print("Best Hyperparameters:", best_params)
    print("Best Average Eigenvalue:", best_eigenvalue)

    best_results = {
        "best_hyperparameters": best_params,
        "best_eigenvalue": best_eigenvalue,
        "best_metrics": best_metrics,
        "ground_state_energy":ground_state_energy,
        "ansatz":config["ansatz"]["type"],
    }


    # Save best trial results
    output_path = config["output_path_results"]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(best_results, f, cls=CustomEncoder)


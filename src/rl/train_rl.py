import numpy as np
from optimisation_rl import RLQuantumOptimiserPPO, RLQuantumOptimiserTD3
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


def objective(trial):
    # Sample hyperparameters from Optuna
    type_optimiser = config["vqe"]["type_optimiser"]
    if type_optimiser =="ppo":
        # Predefined valid (n_steps, batch_size) pairs where n_steps % batch_size == 0
        VALID_PPO_COMBINATIONS = [
            {"n_steps": 5,  "batch_size": 5},
            {"n_steps": 10, "batch_size": 5},
            {"n_steps": 10, "batch_size": 5},
            {"n_steps": 10, "batch_size": 10},
            {"n_steps": 20, "batch_size": 2},
            {"n_steps": 20, "batch_size": 5},
            {"n_steps": 20, "batch_size": 20}]
        
        combo = trial.suggest_categorical("combo", VALID_PPO_COMBINATIONS)
        n_steps = combo["n_steps"]
        batch_size = combo["batch_size"]
        hyperparams = {
            "n_steps": n_steps,
            "batch_size": batch_size,
            "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1e-2),
            "gamma": trial.suggest_uniform("gamma", 0.95, 0.99),
            "gae_lambda": trial.suggest_uniform("gae_lambda", 0.9, 0.99),
            "clip_range": trial.suggest_uniform("clip_range", 0.1, 0.3),
            "n_epochs": trial.suggest_int("n_epochs",10, 30),
            "ent_coef": trial.suggest_uniform("ent_coef", 0.1, 0.3),
            "verbose":0,
            "total_timesteps": config["vqe"]["total_timesteps"],  # Read from YAML
            }
        optimiser_fn = RLQuantumOptimiserPPO
    elif type_optimiser=="td3":
        hyperparams = {
            "batch_size": trial.suggest_categorical("batch_size", [64,128,256,512]),
            "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1e-2),
            "gamma": trial.suggest_uniform("gamma", 0.9, 0.999),
            "tau": trial.suggest_uniform("tau", 0.001, 0.02),
            "noise_std": trial.suggest_uniform("noise_std", 0.001, 0.5),
            "policy_delay": trial.suggest_categorical("policy_delay", [2, 3, 4]),
            "verbose":0,
            "total_timesteps": config["vqe"]["total_timesteps"],
            }
 
        optimiser_fn = RLQuantumOptimiserTD3

    # Number of RL runs per hyperparameter set
    n_run = config["vqe"]["n_run"]   
    metrics = {'history_best_eigenvalue_runs':[],
           'history_best_thetas_runs':[],
           'best_eigenvalue_runs':[],
           'best_thetas_runs': []}
    for iter_run in range(n_run):
        print(f"Run {iter_run+1}/{n_run}")

        # Initialize RL Optimiser
        optimiser = optimiser_fn(
                circuit_fn=create_circuit_ansatz,
                ansatz = ansatz,
                n_thetas=config["vqe"]["n_thetas"],
                initial_state=initial_state,
                H=H,
                num_qubits=config["vqe"]["num_qubits"],
                hyperparams=hyperparams,
                )
        
        # Run optimisation
        optimal_params, eigenvalue, info = optimiser.optimise()

        #print("Optimal parameters found:", optimal_params)
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
    config = load_yaml_config("../../configs/config_train_rl.yaml")  # Load YAML config

    H, mapper, problem = setup_molecule_and_hamiltonian(config)
    ansatz, initial_state = build_ansatz(config, mapper, problem)

    eigenvalues, eigenvectors = eigh(H.to_matrix())
    ground_state_energy = np.min(eigenvalues)
    ground_state = eigenvectors[:, np.argmin(eigenvalues)]
    #ground_state_energy = -74.96 #H2O


    study = optuna.create_study(direction="minimize", sampler=TPESampler(multivariate=True))  # Maximize fidelity
    study.optimize(objective, n_trials=config["vqe"]["n_trials"])

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

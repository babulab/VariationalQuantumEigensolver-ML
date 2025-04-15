import torch
import gpytorch
from botorch.models import SingleTaskGP
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf
import numpy as np
import sys
sys.path.insert(1, '..')
import os
parallel_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(parallel_folder_path)

from utils.functions import *
from models_bo import *


class BayesianOptimiserEigenvalue:
    def __init__(self, circuit, ansatz, n_thetas, initial_state, H, param_bounds, num_qubits, hyperparams, num_initial_samples=50):
        self.circuit = circuit
        self.ansatz = ansatz
        self.n_thetas = n_thetas
        self.initial_state = initial_state
        self.H = H
        self.param_bounds = torch.tensor(param_bounds, dtype=torch.double).T
        self.num_qubits = num_qubits
        self.num_initial_samples = num_initial_samples
        self.info =  {"best_eigenvalue": None, "best_params": None, "best_log": {'eigenvalue':[], 'thetas':[]}}
      
        # Generate initial training data
        #X_thetas, y_expectation_values, y_eigenvalues
        self.X_train, _, self.y_train = get_real_samples_vqe(circuit, ansatz, num_initial_samples, n_thetas, initial_state, H,)
        self.best_eigenvalue = np.inf
        self.best_thetas = np.zeros(n_thetas)

        # Train GP Surrogate Model
        self.hyperparams = hyperparams.copy()
        self.num_iters = self.hyperparams.pop('num_iters', 100)
        self.model, self.likelihood, _, _ = train_surrogate_models_gp((self.X_train, self.y_train, [], [], self.hyperparams))


    def optimise(self):
        """Runs Bayesian optimisation to find optimal quantum parameters."""
        for i in range(self.num_iters):
            wrapped_gp_model = WrappedGPyTorchModelMonoGP(self.model) 
            # Define acquisition function (UCB)
            acqf = UpperConfidenceBound(wrapped_gp_model, beta=self.hyperparams['beta']
                                        ,maximize=False)
            acqf.model.eval()
            # Optimise acquisition function to propose new sample
            candidate, _ = optimize_acqf(
                acq_function=acqf,
                bounds=self.param_bounds,
                q=1,
                num_restarts=self.hyperparams['num_restarts'],
                raw_samples=self.hyperparams['raw_samples'],
            )
            
            # Evaluate new sample
            new_x = np.array(candidate).squeeze(0)
            _, new_y_eigenvalue = get_expectation_values_hamiltonian(self.circuit, self.ansatz, new_x[np.newaxis,:], self.initial_state, self.H)


            # Update training data
            self.X_train = np.vstack((self.X_train, new_x[np.newaxis, :]))
            self.y_train = np.hstack((self.y_train, new_y_eigenvalue))

            # Update best eigenvalue log
            if new_y_eigenvalue < self.best_eigenvalue:
                self.best_eigenvalue = new_y_eigenvalue
                self.best_thetas = new_x
                self.info["best_eigenvalue"] = new_y_eigenvalue
                self.info["best_params"] = new_x

            self.info["best_log"]['eigenvalue'].append(self.info["best_eigenvalue"])
            self.info["best_log"]['thetas'].append(self.info["best_params"])

            # Retrain GP Model
            self.model, self.likelihood, _, _ = train_surrogate_models_gp((self.X_train, self.y_train, [], [], self.hyperparams))

        # Return best parameters found
        best_idx = np.argmin(self.y_train)
        return self.X_train[best_idx], self.y_train[best_idx], self.info



class BayesianOptimiserObservables:
    def __init__(self, circuit, ansatz, n_thetas, initial_state, H, param_bounds, num_qubits, hyperparams, num_initial_samples=50):
        self.circuit = circuit
        self.ansatz = ansatz
        self.n_thetas = n_thetas
        self.initial_state = initial_state
        self.H = H
        self.n_surrogates_models = len(H.coeffs)
        self.param_bounds = torch.tensor(param_bounds, dtype=torch.double).T
        self.num_qubits = num_qubits
        self.num_initial_samples = num_initial_samples
        self.info =  {"best_eigenvalue": None, "best_params": None, "best_log": {'eigenvalue':[], 'thetas':[]}}

        # Generate initial training data
        #X_thetas, y_expectation_values, y_eigenvalues
        self.X_train, self.y_train, self.y_train_eigenvalues = get_real_samples_vqe(circuit, ansatz, num_initial_samples, n_thetas, initial_state, H,)
        self.best_eigenvalue = np.inf
        self.best_thetas = np.zeros(n_thetas)
        # Train GP Surrogate Model
        self.hyperparams = hyperparams.copy()
        self.num_iters = self.hyperparams.pop('num_iters', 100)

        self.surrogate_models = {'gp_model':[], 'likelihood':[], 'scaler_target':[]}
        for n_sm in range(self.n_surrogates_models):
            model, ll, _, scaler_y = train_surrogate_models_gp((self.X_train, self.y_train[:, n_sm], [], [], self.hyperparams))
            self.surrogate_models['gp_model'].append(model)
            self.surrogate_models['likelihood'].append(ll)
            self.surrogate_models['scaler_target'].append(scaler_y)



    def optimise(self):
        """Runs Bayesian optimisation to find optimal quantum parameters."""
        for i in range(self.num_iters):
            wrapped_gp_model = WrappedGPyTorchModelMultiGPHamiltonian(self.surrogate_models, np.ones(self.n_surrogates_models), self.num_qubits) 
            # Define acquisition function (UCB)
            acqf = UpperConfidenceBound(wrapped_gp_model, beta=self.hyperparams['beta']
                                        ,maximize=False)
            acqf.model.eval()
            # Optimise acquisition function to propose new sample
            candidate, _ = optimize_acqf(
                acq_function=acqf,
                bounds=self.param_bounds,
                q=1,
                num_restarts=self.hyperparams['num_restarts'],
                raw_samples=self.hyperparams['raw_samples'],
            )
            
            # Evaluate new sample
            new_x = np.array(candidate).squeeze(0)
            #new_y, new_y_fid, _ = get_expectation_values(self.circuit, new_x[np.newaxis,:], self.initial_state, self.target_state, self.num_qubits)
            new_y, new_y_eigenvalue = get_expectation_values_hamiltonian(self.circuit, self.ansatz, new_x[np.newaxis,:], self.initial_state, self.H)
            
            # Update training data
            self.X_train = np.vstack((self.X_train, new_x[np.newaxis, :]))
            self.y_train = np.vstack((self.y_train, new_y))
            self.y_train_eigenvalues = np.hstack((self.y_train_eigenvalues, new_y_eigenvalue))

            # Update best fidelity log
            if new_y_eigenvalue < self.best_eigenvalue:
                self.best_eigenvalue = new_y_eigenvalue
                self.best_thetas = new_x
                self.info["best_eigenvalue"] = new_y_eigenvalue
                self.info["best_params"] = new_x

            self.info["best_log"]['eigenvalue'].append(self.info["best_eigenvalue"])
            self.info["best_log"]['thetas'].append(self.info["best_params"])

            self.surrogate_models = {'gp_model':[], 'likelihood':[], 'scaler_target':[]}
            for n_sm in range(self.n_surrogates_models):
                model, ll, _, scaler_y = train_surrogate_models_gp((self.X_train, self.y_train[:, n_sm], [], [], self.hyperparams))
                self.surrogate_models['gp_model'].append(model)
                self.surrogate_models['likelihood'].append(ll)            
                self.surrogate_models['scaler_target'].append(scaler_y)

        # Return best parameters found
        best_idx = np.argmin(self.y_train_eigenvalues)
        return self.X_train[best_idx], self.y_train_eigenvalues[best_idx], self.info

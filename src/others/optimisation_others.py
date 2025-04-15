from qiskit_algorithms.optimizers import SPSA, L_BFGS_B, SLSQP
import numpy as np
import sys
import os
utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../utils'))
sys.path.append(utils_path)
from functions import *

class ClassicOptimiser:
    def __init__(self, optimiser_fn, circuit_fn, ansatz, n_thetas, initial_state, H, num_qubits, hyperparams):
        """
        classic-based optimiser for quantum circuit optimisation.
        """
        self.circuit_fn = circuit_fn
        self.ansatz = ansatz
        self.n_thetas = n_thetas
        self.initial_state = initial_state
        self.H = H
        self.num_qubits = num_qubits
        self.hyperparams = hyperparams.copy()
        self.model = optimiser_fn(**self.hyperparams) #SPSA(**self.hyperparams)
        self.info = {}
        self.X_train = []
        self.y_train = []
        self.best_eigenvalue = float("inf")
        self.best_thetas = np.zeros(self.n_thetas)
        self.eigenvalue_history = []
        self.thetas_history = []

    def _compute_eigenvalue(self, thetas):
        """Compute the eigenvalue of the circuit with given parameters."""
        _, new_eigenvalue = get_expectation_values_hamiltonian(self.circuit_fn, self.ansatz, thetas[np.newaxis,:], self.initial_state, self.H)
        return new_eigenvalue
    

    def _objective_function(self, thetas):
        """
        Wrapper function for the objective function that tracks fidelity and updates best parameters.
        """
        eigenvalue = self._compute_eigenvalue(thetas)
        
        self.X_train.append(thetas[0])
        self.y_train.append(eigenvalue)

        if eigenvalue < self.best_eigenvalue:
            self.best_eigenvalue = eigenvalue
            self.best_thetas = thetas

        self.eigenvalue_history.append(self.best_eigenvalue)
        self.thetas_history.append(self.best_thetas)

        return eigenvalue 
     

    def optimise(self):
        """
        Runs optimisation.
        """

        initial_thetas = np.random.uniform(0, 2 * np.pi, self.n_thetas)#[np.newaxis,:]

        results = self.model.minimize(
                self._objective_function,
                x0=initial_thetas
                )
        
        #LBFGS sometimes stop before n_total_timesteps
        if np.array(self.eigenvalue_history).shape[0]<=self.hyperparams['maxiter']:
            optimal_params =  np.array(self.thetas_history)[-1]
            optimal_eigenvalue =  np.abs(np.array(self.eigenvalue_history)[-1])
            n_missing = self.hyperparams['maxiter'] - np.array(self.eigenvalue_history).shape[0]
            n_params = np.shape(np.array(self.thetas_history))[1]
            fill_nan = np.nan*np.zeros(n_missing)
            self.eigenvalue_history = np.hstack((np.array(self.eigenvalue_history),fill_nan))    
            matrix_nan = (np.nan*np.zeros(n_missing*n_params)).reshape(n_missing,n_params) 
            self.thetas_history =  np.vstack((np.array(self.thetas_history),matrix_nan))


        else:
            optimal_params =  np.array(self.thetas_history)[self.hyperparams['maxiter']]
            optimal_eigenvalue =  np.abs(np.array(self.eigenvalue_history)[self.hyperparams['maxiter']])
        self.info = {'best_eigenvalue': optimal_eigenvalue,
                     'best_params': optimal_params,
                     'best_log':{'eigenvalue':np.array(self.eigenvalue_history)[:self.hyperparams['maxiter']],
                                 'thetas': np.array(self.thetas_history)[:self.hyperparams['maxiter']]}}
   
        return optimal_params, optimal_eigenvalue, self.info   
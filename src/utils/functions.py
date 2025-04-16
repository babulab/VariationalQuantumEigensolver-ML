#Functions to be used

from qiskit import QuantumCircuit, transpile, assemble
from qiskit_aer import Aer
import numpy as np
import qiskit.quantum_info as qi
from math import pi
import math
import matplotlib.pylab as plt
from itertools import product
from qiskit.quantum_info import Statevector, Operator, DensityMatrix, state_fidelity, SparsePauliOp
import itertools
import warnings
from smt.sampling_methods import LHS
warnings.filterwarnings("ignore")
import json
import uuid
from qiskit.primitives import Estimator
import yaml


#%% VQE functions

def expectation2eigenvalue(expectations,coefficients_array, num_qubits):
    #Calculate the Eigenvalue using the observables       
    eigenvalue = np.sum( expectations *coefficients_array, axis=1)  
    #eigenvalue = np.sum(((expectations *coefficients_array)/(2**num_qubits)), axis=1)
    return eigenvalue
  


def get_expectation_values_hamiltonian(circuit, ansatz, thetas, initial_state, H):
    #Given a circuit, Hamiltonian, initial_state and theta angles. The expectation values ​​are returned

    circ = circuit(ansatz, thetas, initial_state)
    statevector = Statevector.from_instruction(circ)
    expectation_values = []
    for pauli_op in H:
        coefficient = pauli_op.coeffs
        observable = pauli_op.paulis 
        observable = SparsePauliOp(observable, coefficient)
        expectation = statevector.expectation_value(observable)

        expectation_values.append( expectation.real )
    expectation_values = -np.array(expectation_values) #Minus because the source code (QOC) was made to maximise, with the minus the code minimise
    return expectation_values, np.sum(expectation_values)


def get_real_samples_vqe(circuit, ansatz, n_samples, n_thetas, initial_state, H, type_sampling = 'Random_Uniform'):
   
    if type_sampling=='Random_Uniform':
        X_out = get_new_samples(n_samples, n_thetas)
    elif type_sampling == 'LHS':
        X_out = get_new_samples_lhs(n_samples, n_thetas)
    else:
      raise Exception("Sorry, Sampling method unavailable")   
    expectations_values = []
    for j in range(n_samples):
        thetas_new = X_out[j]
        expectation_values, _ = get_expectation_values_hamiltonian(circuit, ansatz, thetas_new[np.newaxis,:], initial_state, H)
        expectations_values.append(expectation_values)

    expectations_values =  np.array(expectations_values)

    return X_out, np.real(expectations_values), np.sum(expectations_values,1)   


def get_expectation_values_hamiltonian_estimator(circuit, thetas, initial_state, H):
    # Given a circuit, Hamiltonian, initial_state and theta angles. The expectation values are returned

    # Create the circuit
    circ = circuit(thetas, initial_state)
    # Initialize the Estimator
    estimator = Estimator()
    # Prepare observables
    observables = [SparsePauliOp(pauli_op.paulis, pauli_op.coeffs) for pauli_op in H]
    # Run the estimator
    job = estimator.run([circ] * len(observables), observables)
    result = job.result()
    # Extract expectation values
    expectation_values = -np.array(result.values)  # Minus for minimization
    return expectation_values, np.sum(expectation_values)

#%% General functions
# Generate samples using latin hypercube sampling
def get_new_samples_lhs(n_samples, n_thetas):
    xlimits = np.array([[0,2*pi]]*n_thetas )
    sampling = LHS(xlimits=xlimits)
    XNewOut = sampling(n_samples)

    return XNewOut

def get_new_samples(n_samples, n_thetas):
    XNewOut = []
    for _ in range(n_samples):
        thetas_new = np.random.uniform(0,2*pi, n_thetas)
        XNewOut.append(thetas_new)
    return np.array(XNewOut)

#Enconder json
class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert numpy array to list
        elif isinstance(obj, uuid.UUID):
            return str(obj)  # Convert UUID to string
        elif isinstance(obj, complex):
            return {"__complex__": True, "real": obj.real, "imag": obj.imag}
        elif isinstance(obj, np.float32):
             return float(obj)
       
        return json.JSONEncoder.default(self, obj)

def load_yaml_config(yaml_file):
    with open(yaml_file, "r") as f:
        config = yaml.safe_load(f)

    return config





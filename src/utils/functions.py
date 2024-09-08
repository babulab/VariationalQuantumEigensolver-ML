#Functions to be used

from qiskit import QuantumCircuit, transpile, assemble
from qiskit_aer import Aer
import numpy as np
import qiskit.quantum_info as qi
from math import pi
import math
import matplotlib.pylab as plt
from itertools import product
from qiskit.quantum_info import Statevector, Operator, DensityMatrix, state_fidelity
import itertools
import warnings
from smt.sampling_methods import LHS
warnings.filterwarnings("ignore")
import json
import uuid


#%% QOC functions

# Function to calculate fidelity
def calculate_fidelity(real_state, pred_state):
    return qi.state_fidelity(real_state, pred_state)

def get_expectation_values(circuit, thetas,  initial_state, target_state):
    
    circ = circuit(thetas, initial_state)
    n_qubits = circ.num_qubits

    #Operator to use
    coefficients,_ = get_coefficients(target_state, n_qubits)
    operators = [ Operator.from_label(i) for i in coefficients.keys()]

    simulator_contr = Aer.get_backend('statevector_simulator')
    compiled_circuit_contr = transpile(circ, simulator_contr)
    #result_contr = simulator_contr.run(assemble(compiled_circuit_contr)).result()
    result_contr = simulator_contr.run(compiled_circuit_contr).result()

    state_vector = result_contr.get_statevector()

    # Calculate fidelity
    real_fidelity = calculate_fidelity(target_state, state_vector.data)

    unnormalized_state_vector = state_vector.data*np.linalg.norm(state_vector)
    expectation_values = []
    for op in operators:
        expectation_value = (np.dot(np.conj(unnormalized_state_vector), np.dot(op.data, unnormalized_state_vector))).real
        expectation_values.append(expectation_value)   

    return np.array(expectation_values), real_fidelity, coefficients


def get_new_samples(n_samples, n_thetas):
    XNewOut = []
    for _ in range(n_samples):
        thetas_new = np.random.uniform(0,2*pi, n_thetas)
        XNewOut.append(thetas_new)
    return np.array(XNewOut)



def get_real_samples(circuit, n_samples, n_thetas, initial_state, target_state, type_sampling = 'Random_Uniform'):

    if type_sampling=='Random_Uniform':
        X_out = get_new_samples(n_samples, n_thetas)
    elif type_sampling == 'LHS':
        X_out = get_new_samples_lhs(n_samples, n_thetas)
    else:
      raise Exception("Sorry, Sampling method unavailable")         
    
    fidelities_real = []
    expectations_values = []
    for j in range(n_samples):
        thetas_new = X_out[j]
        expectation_values, fid_real, _ = get_expectation_values(circuit, thetas_new[np.newaxis,:], initial_state, target_state)
        expectations_values.append(expectation_values)
        fidelities_real.append(fid_real)

    expectations_values =  np.array(expectations_values)
    fidelities_real = np.array(fidelities_real)

    return X_out, np.real(expectations_values), fidelities_real




def expectation2fidelity(expectations,coefficients_array, num_qubits):
    #Calculate the Fidelity using the observables       
    fid = np.abs(np.sum((np.hstack((np.ones((np.shape(expectations)[0],1)), expectations *coefficients_array))/(2**num_qubits)), axis=1))    

    return fid



def get_coefficients(target_state, n_qubits):
    #Get the coefficients and observables, functions used to estimate the fidelity using observables

    str_list = [''.join(item) for item in product(['I','X','Y', 'Z'], repeat=n_qubits)]
    operators = [ Operator.from_label(i) for i in str_list]
    all_coef, coef = {}, {}
    for i in range(len(operators)):
        val = (np.dot(np.conj(target_state), np.dot(operators[i],target_state))).real
        all_coef[str_list[i]] = val
        if np.abs(val)>1e-8:
            coef[str_list[i]] = val

    del coef['I'*n_qubits] #Remove the Identity

    return coef, n_qubits


#%% VQE functions


def expectation2eigenvalue(expectations,coefficients_array, num_qubits):
    #Calculate the Eigenvalue using the observables       
    eigenvalue = np.sum( expectations *coefficients_array, axis=1)  
    #eigenvalue = np.sum(((expectations *coefficients_array)/(2**num_qubits)), axis=1)
    return eigenvalue
  


def get_expectation_values_hamiltonian(circuit, thetas, initial_state, H):
    #Given a circuit, Hamiltonian, initial_state and theta angles. The expectation values ​​are returned

    circ = circuit(thetas, initial_state)
    statevector = Statevector.from_instruction(circ)
    expectation_values = []
    for pauli_op in H:
        coefficient = pauli_op.coeffs
        observable = pauli_op.to_matrix()
        expectation = coefficient * statevector.expectation_value(observable)
        expectation_values.append( expectation.real )
    expectation_values = -np.array(expectation_values) #Minus because the source code (QOC) was made to maximise, with the minus the code minimise
    expectation_values = expectation_values.squeeze(-1)
    return expectation_values, np.sum(expectation_values)


def get_real_samples_vqe(circuit, n_samples, n_thetas, initial_state, H, type_sampling = 'Random_Uniform'):
   
    if type_sampling=='Random_Uniform':
        X_out = get_new_samples(n_samples, n_thetas)
    elif type_sampling == 'LHS':
        X_out = get_new_samples_lhs(n_samples, n_thetas)
    else:
      raise Exception("Sorry, Sampling method unavailable")   
    expectations_values = []
    for j in range(n_samples):
        thetas_new = X_out[j]
        expectation_values, _ = get_expectation_values_hamiltonian(circuit, thetas_new[np.newaxis,:], initial_state, H)
        expectations_values.append(expectation_values)

    expectations_values =  np.array(expectations_values)

    return X_out, np.real(expectations_values), np.sum(expectations_values,1)   



#%% General functions


# Generate samples using latin hypercube sampling
def get_new_samples_lhs(n_samples, n_thetas):
    xlimits = np.array([[0,2*pi]]*n_thetas )
    sampling = LHS(xlimits=xlimits)
    XNewOut = sampling(n_samples)

    return XNewOut



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






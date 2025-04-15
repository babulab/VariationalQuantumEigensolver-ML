from qiskit import QuantumCircuit
import numpy as np


def create_circuit_ansatz(ansatz, thetas, initial_state):

    # Create circuit
    circ = QuantumCircuit(ansatz.num_qubits)
    # Apply the initial state (HartreeFock)
    circ.compose(initial_state, inplace=True)
    # Apply the UCCSD ansatz with the generated parameters
    circ = ansatz.assign_parameters(thetas.squeeze()).compose(circ)
    return circ
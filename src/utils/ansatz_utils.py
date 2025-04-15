
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.circuit.library import UCCSD, HartreeFock
from qiskit.circuit.library import TwoLocal, EfficientSU2
from qiskit_nature.second_q.mappers import JordanWignerMapper


def build_ansatz(config, mapper, problem):
    ansatz_type = config["ansatz"]["type"]
    options = config["ansatz"].get("options", {})
    num_spatial_orbitals = problem.num_spatial_orbitals
    num_particles = problem.num_particles

    mapper = JordanWignerMapper()
    initial_state = HartreeFock(
        num_spatial_orbitals=num_spatial_orbitals,
        num_particles=num_particles,
        qubit_mapper=mapper
    )

    if ansatz_type == "UCCSD":
        ansatz = UCCSD(
            num_particles=num_particles,
            num_spatial_orbitals=num_spatial_orbitals,
            qubit_mapper=mapper,
            initial_state=initial_state
        )
    elif ansatz_type == "TwoLocal":
        ansatz = TwoLocal(
            num_qubits=H.num_qubits,
            rotation_blocks=options.get("rotation_blocks", "ry"),
            entanglement_blocks=options.get("entanglement_blocks", "cz"),
            reps=options.get("reps", 3),
            initial_state=initial_state
        )

    else:
        raise ValueError(f"Unsupported ansatz type: {ansatz_type}")

    num_qubits = ansatz.num_qubits
    config["vqe"]["num_qubits"] = num_qubits  # save for reuse
    config["vqe"]["n_thetas"] = ansatz.num_parameters

    return ansatz, initial_state

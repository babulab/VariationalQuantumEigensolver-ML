from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.operators import PolynomialTensor


def setup_molecule_and_hamiltonian(config):
    mol_cfg = config["molecule"]
    driver = PySCFDriver(
        atom=mol_cfg["atoms"],
        basis=mol_cfg["basis"],
        charge=mol_cfg["charge"],
        spin=mol_cfg["spin"],
        unit=DistanceUnit[mol_cfg["unit"]]
    )
    problem = driver.run()

    #Obtaining the Hamiltonian
    hamiltonian = problem.hamiltonian    
    hamiltonian.electronic_integrals.alpha += PolynomialTensor({"": hamiltonian.nuclear_repulsion_energy})

    mapper = JordanWignerMapper()
    second_q_op = hamiltonian.second_q_op()
    H = mapper.map(second_q_op)

    return H, mapper, problem



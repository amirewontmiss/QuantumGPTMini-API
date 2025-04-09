import pennylane as qml
import torch
from config import QUBITS_PER_HEAD, QUANTUM_LAYERS

# Set up the quantum device with the number of qubits per head.
dev = qml.device("default.qubit", wires=QUBITS_PER_HEAD)

@qml.qnode(dev, interface="torch")
def quantum_transformer_circuit(angles, weights):
    """
    Quantum circuit:
      - Encodes input angles using AngleEmbedding.
      - Applies a variational circuit with BasicEntanglerLayers.
      - Returns expectation values of PauliZ for each qubit.
    """
    qml.templates.AngleEmbedding(angles, wires=range(QUBITS_PER_HEAD))
    qml.templates.BasicEntanglerLayers(weights, wires=range(QUBITS_PER_HEAD))
    return [qml.expval(qml.PauliZ(i)) for i in range(QUBITS_PER_HEAD)]


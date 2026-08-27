import pennylane as qml
import numpy as np

n_qubits = 3
dev = qml.device("default.qubit", wires=n_qubits)

def qnn_circuit(state, weights):
    # Encoding dello stato normalizzato
    for i in range(n_qubits):
        qml.RY(state[i], wires=i)

    # Layer quantistico semplice e stabile
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i+1])

    for i in range(n_qubits):
        qml.RY(weights[i], wires=i)

    return qml.expval(qml.PauliZ(0))

@qml.qnode(dev)
def qnn_forward(state, weights):
    return qnn_circuit(state, weights)

def qnn_policy(state, weights):
    p = qnn_forward(state, weights)
    return p

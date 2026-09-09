import pennylane as qml

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

    qml.CNOT(wires=[2, 1])

    return qml.probs(wires=[0, 1])

@qml.qnode(dev, interface="autograd")
def qnn_forward(state, weights):
    return qnn_circuit(state, weights)

def qnn_policy(state, weights):
    return qnn_forward(state, weights)

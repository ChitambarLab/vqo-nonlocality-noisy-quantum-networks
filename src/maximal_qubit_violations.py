from pennylane import numpy as np
import pennylane as qml


def chsh_violation_criterion(operator):

    X = np.array([[0,1],[1,0]])
    Y = np.array([[0,-1j],[1j, 0]])
    Z = np.array([[1,0],[0,-1]])

    paulis = [X, Y, Z]

    corr_mat = np.zeros((3,3))
    for i in range(3):
        pauli_i = paulis[i]
        for j in range(3):
            pauli_j = paulis[j]

            pauli_op = np.kron(pauli_i, pauli_j)

            corr_mat[i,j] = np.real(np.trace(operator @ pauli_op))

    U = corr_mat.T @ corr_mat

    eigenvals = np.sort(np.linalg.eigvals(U))

    return corr_mat, U, eigenvals

def chsh_max_violation(rho):
    (T, U, eigvals) = chsh_violation_criterion(rho)

    return 2 * np.sqrt(eigvals[-1] + eigvals[-2])


def bilocal_max_violation_chsh_prod(rho1, rho2):

    (T1, U1, eigvals1) = chsh_violation_criterion(rho1)
    (T2, U2, eigvals2) = chsh_violation_criterion(rho2)

    return np.sqrt(
        np.sqrt(eigvals1[-1] + eigvals1[-2]) * np.sqrt(eigvals2[-1] + eigvals2[-2])
    )

def bilocal_max_violation(rho1, rho2):

    (T1, U1, eigvals1) = chsh_violation_criterion(rho1)
    (T2, U2, eigvals2) = chsh_violation_criterion(rho2)

    return np.sqrt(
        np.sqrt(eigvals1[-1] * eigvals2[-1]) + np.sqrt(eigvals1[-2] * eigvals2[-2])
    )

def star_max_violation_chsh_prod(states):

    n = len(states)

    chsh_violations = [
        chsh_max_violation(state) / 2 
        for state in states
    ]

    return np.power((np.prod(chsh_violations)), 1/n )

def star_max_violation(states):

    n = len(states)

    states_eigvals = [
        chsh_violation_criterion(state)[2]
        for state in states
    ]

    states_eigvals1 = [
        eigvals[-1]
        for eigvals in states_eigvals
    ]

    states_eigvals2 = [
        eigvals[-2]
        for eigvals in states_eigvals
    ]

    return np.sqrt(
        np.power( np.prod(states_eigvals1), 1/n) + np.power( np.prod(states_eigvals2), 1/n)
    )


def chain_max_violation_chsh_prod(states):
    n = len(states)

    chsh_violations = [
        chsh_max_violation(state) / 2 
        for state in states
    ]

    return np.power((np.prod(chsh_violations)), 1/n)

def chain_max_violation_chsh_prod2(states):
    n = len(states)

    chsh_violations = [
        chsh_max_violation(state) / 2 
        for state in states
    ]

    S_bilocal = bilocal_max_violation_chsh_prod(states[0], states[-1])

    return np.power((S_bilocal**2) * np.prod(chsh_violations), 1/n)

def chain_classical_interior_max_violation(states):
    n = len(states)

    S_bilocal = bilocal_max_violation_chsh_prod(states[0], states[-1])

    interior_states_max_eigvals = [
        chsh_violation_criterion(state)[2][-1]
        for state in states[1:n]
    ]

    return S_bilocal*np.sqrt(np.prod(interior_states_max_eigvals))


def chain_max_violation(states):

    n = len(states)

    states_eigvals = [
        chsh_violation_criterion(state)[2]
        for state in states
    ]

    states_eigvals1 = [
        eigvals[-1]
        for eigvals in states_eigvals
    ]

    states_eigvals2 = [
        eigvals[-2]
        for eigvals in states_eigvals
    ]

    return np.sqrt(
        np.sqrt( np.prod(states_eigvals1)) + np.sqrt( np.prod(states_eigvals2))
    )



if __name__ == "__main__":

    @qml.qnode(qml.device("default.mixed", wires=[0,1]))
    def circ():
        # qml.Hadamard(wires=[0])
        # qml.CNOT(wires=[0,1])

        # qml.Rot(0.1,0.2,0.3, wires=[0])

        # qml.AmplitudeDamping(0.3, wires=[0])
        qml.PhaseDamping(0.3, wires=[0])


        return qml.state()

    print(chsh_violation_criterion(np.array([[4/3,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])/2 + np.eye(4)/12))


    # state = np.array([[1,0,0,1],[0,0,0,0],[0,0,0,0],[1,0,0,1]])/2
    state = circ()


    print(state)
    (T, U, eigenvals) = chsh_violation_criterion(state)

    print(T)
    print(U)
    print(eigenvals)

    print(chsh_max_violation(state))

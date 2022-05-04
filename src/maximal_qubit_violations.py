from pennylane import numpy as np
import pennylane as qml


def chsh_violation_criterion(operator):
    """Collect the data needed to use the necessary conditions
    for violation of the CHSH inequality.

    :param operator: A matrix representing a two-qubit density operator.
    :type operator: np.array

    :returns: A triple containing the correlation matrix `corr_mat`, the symmetric 
              correlation matrix product `U`, an the eigenvalues of U
    """
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])

    paulis = [X, Y, Z]

    corr_mat = np.zeros((3, 3))
    for i in range(3):
        pauli_i = paulis[i]
        for j in range(3):
            pauli_j = paulis[j]

            pauli_op = np.kron(pauli_i, pauli_j)

            corr_mat[i, j] = np.real(np.trace(operator @ pauli_op))

    U = corr_mat.T @ corr_mat

    eigenvals = np.sort(np.linalg.eigvals(U))

    return corr_mat, U, eigenvals


def chsh_max_violation(rho):
    """Returns the maximal violation of the density operator rho.
    """
    (T, U, eigvals) = chsh_violation_criterion(rho)

    return 2 * np.sqrt(eigvals[-1] + eigvals[-2])


def bilocal_max_violation_chsh_prod(rho1, rho2):
    """Returns the square root of the product
    of CHSH violations for state rho1 and rho2.
    """

    (T1, U1, eigvals1) = chsh_violation_criterion(rho1)
    (T2, U2, eigvals2) = chsh_violation_criterion(rho2)

    return np.sqrt(np.sqrt(eigvals1[-1] + eigvals1[-2]) * np.sqrt(eigvals2[-1] + eigvals2[-2]))


def bilocal_max_violation(rho1, rho2):
    """Returns the max bilocal violation for states rho1 and rho2
    using the Horodecki-like violation criterion.
    """

    (T1, U1, eigvals1) = chsh_violation_criterion(rho1)
    (T2, U2, eigvals2) = chsh_violation_criterion(rho2)

    return np.sqrt(np.sqrt(eigvals1[-1] * eigvals2[-1]) + np.sqrt(eigvals1[-2] * eigvals2[-2]))


def star_max_violation_chsh_prod(states):
    """Returns the n-th root of the product of CHSH violations for all
    provided state.
    """

    n = len(states)

    chsh_violations = [chsh_max_violation(state) / 2 for state in states]

    return np.power((np.prod(chsh_violations)), 1 / n)


def star_max_violation(states):
    """Returns the max star violation for the set of states using the
    Horodecki-like violation criterion.
    """

    n = len(states)

    states_eigvals = [chsh_violation_criterion(state)[2] for state in states]

    states_eigvals1 = [eigvals[-1] for eigvals in states_eigvals]

    states_eigvals2 = [eigvals[-2] for eigvals in states_eigvals]

    return np.sqrt(
        np.power(np.prod(states_eigvals1), 1 / n) + np.power(np.prod(states_eigvals2), 1 / n)
    )


def chain_max_violation_chsh_prod(states):
    """Returns the n-th root of the product of CHSH violations for all
    provided states.
    """

    n = len(states)

    chsh_violations = [chsh_max_violation(state) / 2 for state in states]

    return np.power((np.prod(chsh_violations)), 1 / n)


def chain_classical_interior_max_violation(states):
    """Returns the bilocal score for the first and last states
    computed as the sqrt of CHSH violations. The remaining interior 
    states are evaluated as classical strategies using only their largest
    eigenvalue.
    """

    n = len(states)

    S_bilocal = bilocal_max_violation_chsh_prod(states[0], states[-1])

    interior_states_max_eigvals = [chsh_violation_criterion(state)[2][-1] for state in states[1:n]]

    return S_bilocal * np.sqrt(np.prod(interior_states_max_eigvals))


def chain_max_violation(states):
    """Returns the max chain violation for the set of states using the
    Horodecki-like violation criterion.
    """

    n = len(states)

    states_eigvals = [chsh_violation_criterion(state)[2] for state in states]

    states_eigvals1 = [eigvals[-1] for eigvals in states_eigvals]

    states_eigvals2 = [eigvals[-2] for eigvals in states_eigvals]

    return np.sqrt(np.sqrt(np.prod(states_eigvals1)) + np.sqrt(np.prod(states_eigvals2)))

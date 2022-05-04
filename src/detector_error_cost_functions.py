from pennylane import numpy as np
from pennylane import math
import qnetvo as qnet


def detector_error_chsh_cost_fn(
    chsh_ansatz, error_rates, error_map=np.array([[1, 1], [0, 0]]), **qnode_kwargs
):
    """Constructs an ansatz-specific cost function for maximizing
    nonlocality with respect to the CHSH inequality against detector
    errors.

    The detector errors are treated as classical post-processing maps
    applied to the probability distribution output from the quantum
    circuit executions.

    :param chsh_ansatz: Ansatz for the CHSH scenario.
    :type chsh_ansatz: qnetvo.NetworkAnsatz

    :param error_rates: Two values ranging from 0 to 1 that describe the
                        probability that each detector errors.
    :type error_rates: List[Float]

    :param error_map: A column stochast matrix with positive elements and
                      columns summing to one. The default is to output 0
                      with certainty if an error occurs.
    :type error_map: np.array[Float]

    :param qnode_kwargs: Keyword arguments passed through to the qnode constructors.
    :type qnode_kwargs: Dictionary
    """
    print("error rates : ", error_rates)
    p1, p2 = error_rates

    error_map1 = (1 - p1) * np.eye(2) + p1 * error_map
    error_map2 = (1 - p2) * np.eye(2) + p2 * error_map

    detectors_error = np.kron(error_map1, error_map2)

    chsh_probs = qnet.joint_probs_qnode(chsh_ansatz, **qnode_kwargs)

    def cost(network_settings):

        chsh_score = 0
        for x, y in [[0, 0], [0, 1], [1, 0], [1, 1]]:

            settings = chsh_ansatz.qnode_settings(network_settings, [0], [x, y])
            probs = detectors_error @ chsh_probs(settings)
            correlator = np.sum(probs * np.array([1, -1, -1, 1]))

            chsh_score += (-1) ** (x * y) * correlator

        return -(chsh_score)

    return cost


def detector_error_chain_cost_fn(
    chain_ansatz, error_rates, error_map=np.array([[1, 1], [0, 0]]), **qnode_kwargs
):
    """Constructs an ansatz-specific cost function for maximizing
    nonlocality with respect to the n-local chain inequality against detector
    errors.

    The detector errors are treated as classical post-processing maps
    applied to the probability distribution output from the quantum
    circuit executions. The quantum circuit behavior is post-processed to two
    outcomes before the detector errors are applied.

    :param chsh_ansatz: Ansatz for the CHSH scenario.
    :type chsh_ansatz: qnetvo.NetworkAnsatz

    :param error_rates: Two values ranging from 0 to 1 that describe the
                        probability that each detector errors.
    :type error_rates: List[Float]

    :param error_map: A column stochast matrix with positive elements and
                      columns summing to one. The default is to output 0
                      with certainty if an error occurs.
    :type error_map: np.array[Float]

    :param qnode_kwargs: Keyword arguments passed through to the qnode constructors.
    :type qnode_kwargs: Dictionary
    """
    n = len(chain_ansatz.prepare_nodes)
    print("error rates : ", error_rates)
    error_maps = [(1 - gamma) * np.eye(2) + gamma * error_map for gamma in error_rates]

    detector_errors = np.array([[1]])
    for detector_error in error_maps:
        detector_errors = np.kron(detector_errors, detector_error)

    chain_probs = qnet.joint_probs_qnode(chain_ansatz, **qnode_kwargs)

    prep_inputs = [0] * n
    xy_inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]

    I22_xy_inputs = [[x_a] + [0] * (n - 1) + [x_b] for x_a, x_b in xy_inputs]
    J22_xy_inputs = [[x_a] + [1] * (n - 1) + [x_b] for x_a, x_b in xy_inputs]

    post_map = np.eye(2)
    for i in range(n - 1):
        post_map = np.kron(post_map, np.array([[1, 0, 0, 1], [0, 1, 1, 0]]))

    post_map = np.kron(post_map, np.eye(2))

    parity_vec = qnet.parity_vector(n + 1)

    def cost(network_settings):

        I22_xy_settings = [
            chain_ansatz.qnode_settings(network_settings, prep_inputs, meas_inputs)
            for meas_inputs in I22_xy_inputs
        ]
        J22_xy_settings = [
            chain_ansatz.qnode_settings(network_settings, prep_inputs, meas_inputs)
            for meas_inputs in J22_xy_inputs
        ]

        I22_score = 0
        for I22_settings in I22_xy_settings:
            I22_probs = detector_errors @ post_map @ chain_probs(I22_settings)

            I22_correlator = math.sum(I22_probs * parity_vec)

            I22_score += I22_correlator

        J22_score = 0
        J22_scalars = [1, -1, -1, 1]
        for i in range(len(J22_xy_settings)):
            J22_scalar = J22_scalars[i]
            J22_settings = J22_xy_settings[i]
            J22_probs = detector_errors @ post_map @ chain_probs(J22_settings)

            J22_correlator = math.sum(J22_probs * parity_vec)
            J22_score += J22_scalar * J22_correlator

        chain_score = math.sqrt(math.abs(I22_score) / 4) + math.sqrt(math.abs(J22_score) / 4)

        return -(chain_score)

    return cost


def detector_error_star_cost_fn(
    star_ansatz, error_rates, error_map=np.array([[1, 1], [0, 0]]), **qnode_kwargs
):
    """Constructs an ansatz-specific cost function for maximizing
    nonlocality with respect to the n-local star inequality against detector
    errors.

    The detector errors are treated as classical post-processing maps
    applied to the probability distribution output from the quantum
    circuit executions. The quantum circuit behavior is post-processed to two
    outcomes before the detector errors are applied.

    :param chsh_ansatz: Ansatz for the CHSH scenario.
    :type chsh_ansatz: qnetvo.NetworkAnsatz

    :param error_rates: Two values ranging from 0 to 1 that describe the
                        probability that each detector errors.
    :type error_rates: List[Float]

    :param error_map: A column stochast matrix with positive elements and
                      columns summing to one. The default is to output 0
                      with certainty if an error occurs.
    :type error_map: np.array[Float]

    :param qnode_kwargs: Keyword arguments passed through to the qnode constructors.
    :type qnode_kwargs: Dictionary
    """
    n = len(star_ansatz.prepare_nodes)

    error_maps = [(1 - gamma) * np.eye(2) + gamma * error_map for gamma in error_rates]

    detector_errors = np.array([[1]])
    for detector_error in error_maps:
        detector_errors = np.kron(detector_errors, detector_error)

    star_probs = qnet.joint_probs_qnode(star_ansatz, **qnode_kwargs)

    prep_inputs = [0] * n
    I22_x_inputs = [[int(bit) for bit in np.binary_repr(x, width=n) + "0"] for x in range(2 ** n)]
    J22_x_inputs = [[int(bit) for bit in np.binary_repr(x, width=n) + "1"] for x in range(2 ** n)]

    post_map = np.eye(2)
    for i in range(n - 1):
        post_map = np.kron(post_map, np.eye(2))

    central_parity_vec = qnet.parity_vector(n)

    central_post_map = np.zeros((2, 2 ** n))
    for i in range(len(central_parity_vec)):
        val = central_parity_vec[i]
        if val == 1:
            central_post_map[0, i] = 1
        else:
            central_post_map[1, i] = 1

    post_map = np.kron(post_map, central_post_map)

    parity_vec = qnet.parity_vector(n + 1)

    def cost(network_settings):

        I22_x_settings = [
            star_ansatz.qnode_settings(network_settings, prep_inputs, meas_inputs)
            for meas_inputs in I22_x_inputs
        ]
        J22_x_settings = [
            star_ansatz.qnode_settings(network_settings, prep_inputs, meas_inputs)
            for meas_inputs in J22_x_inputs
        ]

        I22_score = 0
        for I22_settings in I22_x_settings:
            I22_probs = detector_errors @ post_map @ star_probs(I22_settings)

            I22_correlator = math.sum(I22_probs * parity_vec)

            I22_score += I22_correlator

        J22_score = 0
        for i in range(len(J22_x_settings)):

            J22_scalar = (-1) ** (math.sum(J22_x_inputs[i][0:n]))

            J22_settings = J22_x_settings[i]
            J22_probs = detector_errors @ post_map @ star_probs(J22_settings)

            J22_correlator = math.sum(J22_probs * parity_vec)
            J22_score += J22_scalar * J22_correlator

        star_score = (
            np.power(math.abs(I22_score), 1 / n) / 2 + np.power(math.abs(J22_score), 1 / n) / 2
        )

        return -(star_score)

    return cost

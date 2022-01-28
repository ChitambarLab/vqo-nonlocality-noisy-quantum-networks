from pennylane import numpy as np
from context import QNetOptimizer as QNopt


def detector_error_chsh_cost_fn(
    chsh_ansatz, error_rates, error_map=np.array([[0, 0], [1, 1]]), **qnode_kwargs
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
                      columns summing to one. The default is to output 1
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

    chsh_probs = QNopt.joint_probs_qnode(chsh_ansatz, **qnode_kwargs)

    def cost(network_settings):

        chsh_score = 0
        for x, y in [[0, 0], [0, 1], [1, 0], [1, 1]]:

            settings = chsh_ansatz.qnode_settings(network_settings, [0], [x, y])
            probs = detectors_error @ chsh_probs(settings)
            correlator = np.sum(probs * np.array([1, -1, -1, 1]))

            chsh_score += (-1) ** (x * y) * correlator

        return -(chsh_score)

    return cost

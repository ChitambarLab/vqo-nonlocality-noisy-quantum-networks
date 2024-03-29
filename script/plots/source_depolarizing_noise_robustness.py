import qnetvo as qnet
from context import src

from pennylane import numpy as np
import pennylane as qml


"""
This script aggregates data and plots the noise robustness results qubit
depolarizing noise.
"""


@qml.qnode(qml.device("default.mixed", wires=[0, 1]))
def bell_state_noise(gamma):
    qml.Hadamard(wires=[0])
    qml.CNOT(wires=[0, 1])

    qnet.two_qubit_depolarizing(gamma, wires=[0, 1])

    return qml.state()


if __name__ == "__main__":
    num_samples = 21

    theory_noise_params = np.arange(0, 1.0001, 0.001)

    bell_state_noise_states = [bell_state_noise(gamma) for gamma in np.arange(0, 1.01, 0.05)]

    bell_state = np.array([[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]]) / 2

    """
    Loading CHSH Data
    """
    chsh_dep_regexes = [
        r"max_ent_local_rot_.*",
        r"arb_local_rot.*",
        r"phi_plus_local_ry_.*",
        r"phi_plus_local_rot_.*",
    ]

    chsh_dep_dir = "./data/chsh/source_depolarizing/"

    chsh_dep_data = [
        src.analyze_data_one_param_scan(src.get_data_files(chsh_dep_dir, regex))
        for regex in chsh_dep_regexes
    ]
    max_chsh_dep = [
        max(map(lambda opt_data: opt_data["max_scores"][i], chsh_dep_data)) / 2
        for i in range(num_samples)
    ]

    # theoretical_bell_state_chsh = [
    #     src.chsh_max_violation(state) / 2
    #     for state in bell_state_noise_states
    # ]

    theoretical_max_chsh = [
        np.sqrt(2) * np.abs((1 - gamma * 16 / 15)) for gamma in theory_noise_params
    ]

    """
    Loading Bilocal Data
    """
    bilocal_uniform_dep_dir = "./data/bilocal/uniform_source_depolarizing/"

    bilocal_dep_regexes = [
        r"phi_plus_local_ry_.*",
        r"max_ent_local_rot_.*",
        r"phi_plus_arb_.*",
        r"max_ent_arb_.*",
        r"arb_arb_.*",
    ]
    bilocal_uniform_dep_data = [
        src.analyze_data_one_param_scan(src.get_data_files(bilocal_uniform_dep_dir, regex))
        for regex in bilocal_dep_regexes
    ]

    max_bilocal_uniform_dep = [
        max(map(lambda opt_data: opt_data["max_scores"][i], bilocal_uniform_dep_data))
        for i in range(num_samples)
    ]

    bilocal_single_dep_dir = "./data/bilocal/single_source_depolarizing/"

    bilocal_single_dep_data = [
        src.analyze_data_one_param_scan(src.get_data_files(bilocal_single_dep_dir, regex))
        for regex in bilocal_dep_regexes
    ]

    max_bilocal_single_dep = [
        max(map(lambda opt_data: opt_data["max_scores"][i], bilocal_single_dep_data))
        for i in range(num_samples)
    ]

    # theoretical_bell_state_uniform_bilocal = [
    #     src.bilocal_max_violation(state, state)
    #     for state in bell_state_noise_states
    # ]

    # theoretical_bell_state_single_bilocal = [
    #     src.bilocal_max_violation(state, bell_state)
    #     for state in bell_state_noise_states
    # ]

    theoretical_max_uniform_bilocal = [
        np.sqrt(2) * np.sqrt(np.abs((1 - gamma * 16 / 15) ** 2)) for gamma in theory_noise_params
    ]

    theoretical_max_single_bilocal = [
        np.sqrt(2) * np.sqrt(np.abs((1 - gamma * 16 / 15))) for gamma in theory_noise_params
    ]

    """
    Loading n-Chain Data
    """

    chain_uniform_dep_dir = "./data/n-chain/uniform_source_depolarizing/"

    n3_chain_dep_regexes = [r"phi_plus_local_ry_n-3_.*"]

    n3_chain_uniform_dep_data = [
        src.analyze_data_one_param_scan(src.get_data_files(chain_uniform_dep_dir, regex))
        for regex in n3_chain_dep_regexes
    ]

    max_n3_chain_uniform_dep = [
        max(map(lambda opt_data: opt_data["max_scores"][i], n3_chain_uniform_dep_data))
        for i in range(num_samples)
    ]

    chain_single_dep_dir = "./data/n-chain/single_source_depolarizing/"

    n3_chain_single_dep_data = [
        src.analyze_data_one_param_scan(src.get_data_files(chain_single_dep_dir, regex))
        for regex in n3_chain_dep_regexes
    ]

    max_n3_chain_single_dep = [
        max(map(lambda opt_data: opt_data["max_scores"][i], n3_chain_single_dep_data))
        for i in range(num_samples)
    ]

    # theoretical_bell_state_uniform_n3_chain = [
    #     src.chain_max_violation([state, state, state])
    #     for state in bell_state_noise_states
    # ]

    # theoretical_bell_state_single_n3_chain = [
    #     src.chain_max_violation([state, bell_state, bell_state])
    #     for state in bell_state_noise_states
    # ]

    theoretical_max_uniform_n3_chain = [
        np.sqrt(2) * np.sqrt(np.abs((1 - gamma * 16 / 15) ** 3)) for gamma in theory_noise_params
    ]

    theoretical_max_single_n3_chain = [
        np.sqrt(2) * np.sqrt(np.abs((1 - gamma * 16 / 15))) for gamma in theory_noise_params
    ]

    """
    Loading n-Star Data
    """

    star_uniform_dep_dir = "./data/n-star/uniform_source_depolarizing/"

    n3_star_dep_regexes = [r"phi_plus_local_ry_n-3_.*"]

    n3_star_uniform_dep_data = [
        src.analyze_data_one_param_scan(src.get_data_files(star_uniform_dep_dir, regex))
        for regex in n3_star_dep_regexes
    ]

    max_n3_star_uniform_dep = [
        max(map(lambda opt_data: opt_data["max_scores"][i], n3_star_uniform_dep_data))
        for i in range(num_samples)
    ]

    star_single_dep_dir = "./data/n-star/single_source_depolarizing/"

    n3_star_single_dep_data = [
        src.analyze_data_one_param_scan(src.get_data_files(star_single_dep_dir, regex))
        for regex in n3_star_dep_regexes
    ]

    max_n3_star_single_dep = [
        max(map(lambda opt_data: opt_data["max_scores"][i], n3_star_single_dep_data))
        for i in range(num_samples)
    ]

    # theoretical_bell_state_uniform_n3_star = [
    #     src.star_max_violation([state, state, state])
    #     for state in bell_state_noise_states
    # ]

    # theoretical_bell_state_single_n3_star = [
    #     src.star_max_violation([state, bell_state, bell_state])
    #     for state in bell_state_noise_states
    # ]

    theoretical_max_single_n3_star = [
        np.sqrt(2) * np.power(np.abs((1 - gamma * 16 / 15)), 1 / 3) for gamma in theory_noise_params
    ]

    theoretical_max_uniform_n3_star = [
        np.sqrt(2) * np.power(np.abs((1 - gamma * 16 / 15) ** 3), 1 / 3)
        for gamma in theory_noise_params
    ]

    """
    Verifying Data
    """

    def verify_data(theoretical_score, vqo_score):
        return theoretical_score >= vqo_score or np.isclose(theoretical_score, vqo_score)

    for u in range(21):
        assert verify_data(theoretical_max_chsh[50 * u], max_chsh_dep[u])
        assert verify_data(theoretical_max_uniform_bilocal[50 * u], max_bilocal_uniform_dep[u])
        assert verify_data(theoretical_max_uniform_n3_chain[50 * u], max_n3_chain_uniform_dep[u])
        assert verify_data(theoretical_max_uniform_n3_star[50 * u], max_n3_star_uniform_dep[u])

        assert verify_data(theoretical_max_chsh[50 * u], max_chsh_dep[u])
        assert verify_data(theoretical_max_single_bilocal[50 * u], max_bilocal_single_dep[u])
        assert verify_data(theoretical_max_single_n3_chain[50 * u], max_n3_chain_single_dep[u])
        assert verify_data(theoretical_max_single_n3_star[50 * u], max_n3_star_single_dep[u])

    """
    Plotting Data
    """

    src.plot_unital_single_and_uniform_max_scores_data(
        fig_title="Source Depolarizing Noise Robustness",
        ax_titles=["Single Source Noise", "Uniform Source Noise"],
        noise_params=chsh_dep_data[0]["noise_params"],
        quantum_bound=np.sqrt(2),
        classical_bound=1,
        single_max_scores=[
            max_chsh_dep,
            max_bilocal_single_dep,
            max_n3_chain_single_dep,
            max_n3_star_single_dep,
        ],
        single_theoretical_scores=[
            theoretical_max_chsh,
            theoretical_max_single_bilocal,
            theoretical_max_single_n3_chain,
            theoretical_max_single_n3_star,
        ],
        single_match_scores=[],
        uniform_max_scores=[
            max_chsh_dep,
            max_bilocal_uniform_dep,
            max_n3_chain_uniform_dep,
            max_n3_star_uniform_dep,
        ],
        uniform_theoretical_scores=[
            theoretical_max_chsh,
            theoretical_max_uniform_bilocal,
            theoretical_max_uniform_n3_chain,
            theoretical_max_uniform_n3_star,
        ],
        uniform_match_scores=[],
        data_labels=["CHSH", "Bilocal", "3-Local Chain", "3-Local Star"],
        plot_dir="./data/plots/source_depolarizing_noise_robustness/",
        # bottom_padding=0.4,
        ncol_legend=3,
        theory_params=theory_noise_params,
    )

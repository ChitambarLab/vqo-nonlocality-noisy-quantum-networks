import qnetvo as qnet
from context import src

from pennylane import numpy as np
import pennylane as qml


"""
This script aggregates data and plots the noise robustness results source
colored noise.
"""

@qml.qnode(qml.device("default.mixed", wires=[0,1]))
def bell_state_noise(gamma):
    qml.Hadamard(wires=[0])
    qml.CNOT(wires=[0,1])
    qml.PauliX(wires=[0])

    qnet.colored_noise(gamma, wires=[0,1])

    return qml.state()

if __name__ == "__main__":
    num_samples = 21

    bell_state_noise_states = [
        bell_state_noise(gamma)
        for gamma in np.arange(0, 1.01, 0.05)
    ]

    bell_state = np.array([[1,0,0,1],[0,0,0,0],[0,0,0,0],[1,0,0,1]])/2

    """
    Loading CHSH Data
    """
    chsh_colored_regexes = [r"max_ent_local_rot_.*", r"arb_local_rot.*", r"phi_plus_local_ry_.*", r"phi_plus_local_rot_.*", r"psi_plus_local_ry_.*"]

    chsh_colored_dir = "./data/chsh/source_colored_noise/"

    chsh_colored_data = [
        src.analyze_data_one_param_scan(
            src.get_data_files(chsh_colored_dir, regex)
        )
        for regex in chsh_colored_regexes
    ]
    max_chsh_colored = [
        max(map(lambda opt_data: opt_data["max_scores"][i], chsh_colored_data)) / 2
        for i in range(num_samples)
    ]

    theoretical_bell_state_chsh = [
        src.chsh_max_violation(state) / 2
        for state in bell_state_noise_states
    ]

    """
    Loading Bilocal Data
    """
    bilocal_uniform_colored_dir = "./data/bilocal/uniform_source_colored_noise/"

    bilocal_colored_regexes = [r"phi_plus_local_ry_.*", r"max_ent_local_rot_.*", r"phi_plus_arb_.*", r"psi_plus_local_ry_.*", r"arb_arb_.*", r"max_ent_arb_.*"]
    bilocal_uniform_colored_data = [
        src.analyze_data_one_param_scan(
            src.get_data_files(bilocal_uniform_colored_dir, regex)
        )
        for regex in bilocal_colored_regexes
    ]

    max_bilocal_uniform_colored = [
        max(map(lambda opt_data: opt_data["max_scores"][i], bilocal_uniform_colored_data))
        for i in range(num_samples)
    ]

    bilocal_single_colored_dir = "./data/bilocal/single_source_colored_noise/"

    bilocal_single_colored_data = [
        src.analyze_data_one_param_scan(
            src.get_data_files(bilocal_single_colored_dir, regex)
        )
        for regex in bilocal_colored_regexes
    ]

    max_bilocal_single_colored = [
        max(map(lambda opt_data: opt_data["max_scores"][i], bilocal_single_colored_data))
        for i in range(num_samples)
    ]

    theoretical_bell_state_uniform_bilocal = [
        src.bilocal_max_violation(state, state)
        for state in bell_state_noise_states
    ]

    theoretical_bell_state_single_bilocal = [
        src.bilocal_max_violation(state, bell_state)
        for state in bell_state_noise_states
    ]

    """
    Loading n-Chain Data
    """

    chain_uniform_colored_dir = "./data/n-chain/uniform_source_colored_noise/"

    n3_chain_colored_regexes = [r"phi_plus_local_ry_n-3_.*", r"psi_plus_local_ry_n-3_.*"]

    n3_chain_uniform_colored_data = [
        src.analyze_data_one_param_scan(
            src.get_data_files(chain_uniform_colored_dir, regex)
        )
        for regex in n3_chain_colored_regexes
    ]

    max_n3_chain_uniform_colored = [
        max(map(lambda opt_data: opt_data["max_scores"][i], n3_chain_uniform_colored_data))
        for i in range(num_samples)
    ]

    chain_single_colored_dir = "./data/n-chain/single_source_colored_noise/"

    n3_chain_single_colored_data = [
        src.analyze_data_one_param_scan(
            src.get_data_files(chain_single_colored_dir, regex)
        )
        for regex in n3_chain_colored_regexes
    ]

    max_n3_chain_single_colored = [
        max(map(lambda opt_data: opt_data["max_scores"][i], n3_chain_single_colored_data))
        for i in range(num_samples)
    ]

    n4_chain_colored_regexes = [r"phi_plus_local_ry_n-4_.*", r"psi_plus_local_ry_n-4_.*"]

    n4_chain_uniform_colored_data = [
        src.analyze_data_one_param_scan(
            src.get_data_files(chain_uniform_colored_dir, regex)
        )
        for regex in n4_chain_colored_regexes
    ]

    max_n4_chain_uniform_colored = [
        max(map(lambda opt_data: opt_data["max_scores"][i], n4_chain_uniform_colored_data))
        for i in range(num_samples)
    ]

    n4_chain_single_colored_data = [
        src.analyze_data_one_param_scan(
            src.get_data_files(chain_single_colored_dir, regex)
        )
        for regex in n4_chain_colored_regexes
    ]

    max_n4_chain_single_colored = [
        max(map(lambda opt_data: opt_data["max_scores"][i], n4_chain_single_colored_data))
        for i in range(num_samples)
    ]


    theoretical_bell_state_uniform_n3_chain = [
        src.chain_classical_interior_max_violation([state, state, state])
        for state in bell_state_noise_states
    ]

    theoretical_bell_state_single_n3_chain = [
        src.chain_classical_interior_max_violation([state, bell_state, bell_state])
        for state in bell_state_noise_states
    ]

    theoretical_bell_state_uniform_n4_chain = [
        src.chain_classical_interior_max_violation([state, state, state, state])
        for state in bell_state_noise_states
    ]

    theoretical_bell_state_single_n4_chain = [
        src.chain_classical_interior_max_violation([state, bell_state, bell_state, bell_state])
        for state in bell_state_noise_states
    ]

    """
    Loading n-Star Data
    """

    star_uniform_colored_dir = "./data/n-star/uniform_source_colored_noise/"

    n3_star_colored_regexes = [r"psi_plus_local_ry_n-3_.*"]

    n3_star_uniform_colored_data = [
        src.analyze_data_one_param_scan(
            src.get_data_files(star_uniform_colored_dir, regex)
        )
        for regex in n3_star_colored_regexes
    ]

    max_n3_star_uniform_colored = [
        max(map(lambda opt_data: opt_data["max_scores"][i], n3_star_uniform_colored_data))
        for i in range(num_samples)
    ]

    star_single_colored_dir = "./data/n-star/single_source_colored_noise/"

    n3_star_single_colored_data = [
        src.analyze_data_one_param_scan(
            src.get_data_files(star_single_colored_dir, regex)
        )
        for regex in n3_star_colored_regexes
    ]

    max_n3_star_single_colored = [
        max(map(lambda opt_data: opt_data["max_scores"][i], n3_star_single_colored_data))
        for i in range(num_samples)
    ]

    theoretical_bell_state_uniform_n3_star = [
        src.star_max_violation([state, state, state])
        for state in bell_state_noise_states
    ]

    theoretical_bell_state_single_n3_star = [
        src.star_max_violation([state, bell_state, bell_state])
        for state in bell_state_noise_states
    ]

    """
    Plotting Data
    """

    src.plot_single_and_uniform_max_scores_data(
        fig_title = "Source Colored Noise Robustness",
        ax_titles = ["Single Source Noise", "Uniform Source Noise"],
        noise_params = chsh_colored_data[0]["noise_params"],
        quantum_bound = np.sqrt(2),
        classical_bound = 1,
        single_max_scores = [
            max_chsh_colored, max_bilocal_single_colored, max_n3_chain_single_colored,
            max_n4_chain_single_colored, max_n3_star_single_colored
        ],
        single_theoretical_scores = [
            theoretical_bell_state_chsh,
            theoretical_bell_state_single_bilocal,
            theoretical_bell_state_single_n3_chain,
            theoretical_bell_state_single_n4_chain,
            theoretical_bell_state_single_n3_star,
        ],
        uniform_max_scores = [
            max_chsh_colored, max_bilocal_uniform_colored, max_n3_chain_uniform_colored,
            max_n3_chain_uniform_colored, max_n3_star_uniform_colored,
        ],
        uniform_theoretical_scores = [
            theoretical_bell_state_chsh,
            theoretical_bell_state_uniform_bilocal,
            theoretical_bell_state_uniform_n3_chain,
            theoretical_bell_state_uniform_n4_chain,
            theoretical_bell_state_uniform_n3_star,
        ],
        data_labels = ["CHSH", "Bilocal", "3-Local Chain", "4-Local Chain", "3-Local Star"],
        plot_dir =  "./data/plots/source_colored_noise_robustness/" 
    )


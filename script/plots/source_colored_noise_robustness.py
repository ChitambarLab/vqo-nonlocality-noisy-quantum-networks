import qnetvo as qnet
from context import src

from pennylane import numpy as np
import pennylane as qml


"""
This script aggregates data and plots the noise robustness results source
colored noise.
"""


@qml.qnode(qml.device("default.mixed", wires=[0, 1]))
def psi_plus_state_noise(gamma):
    qml.Hadamard(wires=[0])
    qml.CNOT(wires=[0, 1])
    qml.PauliX(wires=[0])

    qnet.colored_noise(gamma, wires=[0, 1])

    return qml.state()


@qml.qnode(qml.device("default.mixed", wires=[0, 1]))
def phi_plus_state_noise(gamma):
    qml.Hadamard(wires=[0])
    qml.CNOT(wires=[0, 1])

    qnet.colored_noise(gamma, wires=[0, 1])

    return qml.state()


if __name__ == "__main__":
    num_samples = 21

    psi_plus_noise_states = [psi_plus_state_noise(gamma) for gamma in np.arange(0, 1.01, 0.05)]
    phi_plus_noise_states = [phi_plus_state_noise(gamma) for gamma in np.arange(0, 1.01, 0.05)]

    phi_plus_state = np.array([[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]]) / 2
    psi_plus_state = np.array([[0, 0, 0, 0], [0, 1, -1, 0], [0, -1, 1, 0], [0, 0, 0, 0]]) / 2

    """
    Loading CHSH Data
    """
    psi_plus_chsh_colored_regexes = [
        r"max_ent_local_rot_.*",
        r"arb_local_rot.*",
        r"psi_plus_local_ry_.*",
    ]
    phi_plus_chsh_colored_regexes = [r"phi_plus_local_ry_.*", r"phi_plus_local_rot_.*"]

    chsh_colored_dir = "./data/chsh/source_colored_noise/"

    psi_plus_chsh_colored_data = [
        src.analyze_data_one_param_scan(src.get_data_files(chsh_colored_dir, regex))
        for regex in psi_plus_chsh_colored_regexes
    ]
    psi_plus_max_chsh_colored = [
        max(map(lambda opt_data: opt_data["max_scores"][i], psi_plus_chsh_colored_data)) / 2
        for i in range(num_samples)
    ]

    phi_plus_chsh_colored_data = [
        src.analyze_data_one_param_scan(src.get_data_files(chsh_colored_dir, regex))
        for regex in phi_plus_chsh_colored_regexes
    ]
    phi_plus_max_chsh_colored = [
        max(map(lambda opt_data: opt_data["max_scores"][i], phi_plus_chsh_colored_data)) / 2
        for i in range(num_samples)
    ]

    phi_plus_theoretical_bell_state_chsh = [
        src.chsh_max_violation(state) / 2 for state in phi_plus_noise_states
    ]

    psi_plus_theoretical_bell_state_chsh = [
        src.chsh_max_violation(state) / 2 for state in psi_plus_noise_states
    ]

    """
    Loading Bilocal Data
    """
    bilocal_uniform_colored_dir = "./data/bilocal/uniform_source_colored_noise/"

    psi_plus_bilocal_colored_regexes = [
        r"max_ent_local_rot_.*",
        r"phi_plus_arb_.*",
        r"psi_plus_local_ry_.*",
        r"arb_arb_.*",
        r"max_ent_arb_.*",
    ]
    phi_plus_bilocal_colored_regexes = [r"phi_plus_local_ry_.*", r"phi_plus_local_rot_.*"]

    psi_plus_bilocal_uniform_colored_data = [
        src.analyze_data_one_param_scan(src.get_data_files(bilocal_uniform_colored_dir, regex))
        for regex in psi_plus_bilocal_colored_regexes
    ]

    psi_plus_max_bilocal_uniform_colored = [
        max(map(lambda opt_data: opt_data["max_scores"][i], psi_plus_bilocal_uniform_colored_data))
        for i in range(num_samples)
    ]

    phi_plus_bilocal_uniform_colored_data = [
        src.analyze_data_one_param_scan(src.get_data_files(bilocal_uniform_colored_dir, regex))
        for regex in phi_plus_bilocal_colored_regexes
    ]

    phi_plus_max_bilocal_uniform_colored = [
        max(map(lambda opt_data: opt_data["max_scores"][i], phi_plus_bilocal_uniform_colored_data))
        for i in range(num_samples)
    ]

    bilocal_single_colored_dir = "./data/bilocal/single_source_colored_noise/"

    phi_plus_bilocal_single_colored_data = [
        src.analyze_data_one_param_scan(src.get_data_files(bilocal_single_colored_dir, regex))
        for regex in phi_plus_bilocal_colored_regexes
    ]

    phi_plus_max_bilocal_single_colored = [
        max(map(lambda opt_data: opt_data["max_scores"][i], phi_plus_bilocal_single_colored_data))
        for i in range(num_samples)
    ]

    psi_plus_bilocal_single_colored_data = [
        src.analyze_data_one_param_scan(src.get_data_files(bilocal_single_colored_dir, regex))
        for regex in psi_plus_bilocal_colored_regexes
    ]

    psi_plus_max_bilocal_single_colored = [
        max(map(lambda opt_data: opt_data["max_scores"][i], psi_plus_bilocal_single_colored_data))
        for i in range(num_samples)
    ]

    phi_plus_theoretical_bell_state_single_bilocal = [
        src.bilocal_max_violation_chsh_prod(state, phi_plus_state)
        for state in phi_plus_noise_states
    ]

    phi_plus_theoretical_bell_state_uniform_bilocal = [
        src.bilocal_max_violation(state, state) for state in phi_plus_noise_states
    ]

    psi_plus_theoretical_bell_state_uniform_bilocal = [
        src.bilocal_max_violation(state, state) for state in psi_plus_noise_states
    ]

    psi_plus_theoretical_bell_state_single_bilocal = [
        src.bilocal_max_violation_chsh_prod(state, psi_plus_state)
        for state in psi_plus_noise_states
    ]

    """
    Loading n-Chain Data
    """

    chain_uniform_colored_dir = "./data/n-chain/uniform_source_colored_noise/"

    psi_plus_n3_chain_colored_regexes = [r"psi_plus_local_ry_n-3_.*"]
    phi_plus_n3_chain_colored_regexes = [r"phi_plus_local_ry_n-3_.*", r"phi_plus_local_rot_n-3_.*"]

    psi_plus_n3_chain_uniform_colored_data = [
        src.analyze_data_one_param_scan(src.get_data_files(chain_uniform_colored_dir, regex))
        for regex in psi_plus_n3_chain_colored_regexes
    ]

    psi_plus_max_n3_chain_uniform_colored = [
        max(map(lambda opt_data: opt_data["max_scores"][i], psi_plus_n3_chain_uniform_colored_data))
        for i in range(num_samples)
    ]

    phi_plus_n3_chain_uniform_colored_data = [
        src.analyze_data_one_param_scan(src.get_data_files(chain_uniform_colored_dir, regex))
        for regex in phi_plus_n3_chain_colored_regexes
    ]

    phi_plus_max_n3_chain_uniform_colored = [
        max(map(lambda opt_data: opt_data["max_scores"][i], phi_plus_n3_chain_uniform_colored_data))
        for i in range(num_samples)
    ]

    chain_single_colored_dir = "./data/n-chain/single_source_colored_noise/"

    phi_plus_n3_chain_single_colored_data = [
        src.analyze_data_one_param_scan(src.get_data_files(chain_single_colored_dir, regex))
        for regex in phi_plus_n3_chain_colored_regexes
    ]

    phi_plus_max_n3_chain_single_colored = [
        max(map(lambda opt_data: opt_data["max_scores"][i], phi_plus_n3_chain_single_colored_data))
        for i in range(num_samples)
    ]

    psi_plus_n3_chain_single_colored_data = [
        src.analyze_data_one_param_scan(src.get_data_files(chain_single_colored_dir, regex))
        for regex in psi_plus_n3_chain_colored_regexes
    ]

    psi_plus_max_n3_chain_single_colored = [
        max(map(lambda opt_data: opt_data["max_scores"][i], psi_plus_n3_chain_single_colored_data))
        for i in range(num_samples)
    ]

    phi_plus_n4_chain_colored_regexes = [r"phi_plus_local_ry_n-4_.*", r"phi_plus_local_rot_n-4_.*"]
    psi_plus_n4_chain_colored_regexes = [r"psi_plus_local_ry_n-4_.*"]

    phi_plus_n4_chain_uniform_colored_data = [
        src.analyze_data_one_param_scan(src.get_data_files(chain_uniform_colored_dir, regex))
        for regex in phi_plus_n4_chain_colored_regexes
    ]

    phi_plus_max_n4_chain_uniform_colored = [
        max(map(lambda opt_data: opt_data["max_scores"][i], phi_plus_n4_chain_uniform_colored_data))
        for i in range(num_samples)
    ]

    psi_plus_n4_chain_uniform_colored_data = [
        src.analyze_data_one_param_scan(src.get_data_files(chain_uniform_colored_dir, regex))
        for regex in psi_plus_n4_chain_colored_regexes
    ]

    psi_plus_max_n4_chain_uniform_colored = [
        max(map(lambda opt_data: opt_data["max_scores"][i], psi_plus_n4_chain_uniform_colored_data))
        for i in range(num_samples)
    ]

    phi_plus_n4_chain_single_colored_data = [
        src.analyze_data_one_param_scan(src.get_data_files(chain_single_colored_dir, regex))
        for regex in phi_plus_n4_chain_colored_regexes
    ]

    phi_plus_max_n4_chain_single_colored = [
        max(map(lambda opt_data: opt_data["max_scores"][i], phi_plus_n4_chain_single_colored_data))
        for i in range(num_samples)
    ]

    psi_plus_n4_chain_single_colored_data = [
        src.analyze_data_one_param_scan(src.get_data_files(chain_single_colored_dir, regex))
        for regex in psi_plus_n4_chain_colored_regexes
    ]

    psi_plus_max_n4_chain_single_colored = [
        max(map(lambda opt_data: opt_data["max_scores"][i], psi_plus_n4_chain_single_colored_data))
        for i in range(num_samples)
    ]

    phi_plus_theoretical_bell_state_single_n3_chain = [
        src.chain_classical_interior_max_violation([state, phi_plus_state, phi_plus_state])
        for state in phi_plus_noise_states
    ]

    phi_plus_theoretical_bell_state_single_n4_chain = [
        src.chain_classical_interior_max_violation(
            [state, phi_plus_state, phi_plus_state, phi_plus_state]
        )
        for state in phi_plus_noise_states
    ]

    phi_plus_theoretical_bell_state_uniform_n3_chain = [
        src.chain_max_violation([state, state, state]) for state in phi_plus_noise_states
    ]

    phi_plus_theoretical_bell_state_uniform_n4_chain = [
        src.chain_max_violation([state, state, state, state]) for state in phi_plus_noise_states
    ]

    psi_plus_theoretical_bell_state_uniform_n3_chain = [
        src.chain_classical_interior_max_violation([state, state, state])
        for state in psi_plus_noise_states
    ]

    psi_plus_theoretical_bell_state_single_n3_chain = [
        src.chain_classical_interior_max_violation([state, psi_plus_state, psi_plus_state])
        for state in psi_plus_noise_states
    ]

    psi_plus_theoretical_bell_state_uniform_n4_chain = [
        src.chain_classical_interior_max_violation([state, state, state, state])
        for state in psi_plus_noise_states
    ]

    psi_plus_theoretical_bell_state_single_n4_chain = [
        src.chain_classical_interior_max_violation(
            [state, psi_plus_state, psi_plus_state, psi_plus_state]
        )
        for state in psi_plus_noise_states
    ]

    """
    Loading n-Star Data
    """

    star_uniform_colored_dir = "./data/n-star/uniform_source_colored_noise/"

    psi_plus_n3_star_colored_regexes = [r"psi_plus_local_ry_n-3_.*"]
    phi_plus_n3_star_colored_regexes = [r"phi_plus_local_ry_n-3_.*", r"phi_plus_local_rot_n-3_.*"]

    psi_plus_n3_star_uniform_colored_data = [
        src.analyze_data_one_param_scan(src.get_data_files(star_uniform_colored_dir, regex))
        for regex in psi_plus_n3_star_colored_regexes
    ]

    psi_plus_max_n3_star_uniform_colored = [
        max(map(lambda opt_data: opt_data["max_scores"][i], psi_plus_n3_star_uniform_colored_data))
        for i in range(num_samples)
    ]

    phi_plus_n3_star_uniform_colored_data = [
        src.analyze_data_one_param_scan(src.get_data_files(star_uniform_colored_dir, regex))
        for regex in phi_plus_n3_star_colored_regexes
    ]

    phi_plus_max_n3_star_uniform_colored = [
        max(map(lambda opt_data: opt_data["max_scores"][i], phi_plus_n3_star_uniform_colored_data))
        for i in range(num_samples)
    ]

    star_single_colored_dir = "./data/n-star/single_source_colored_noise/"

    psi_plus_n3_star_single_colored_data = [
        src.analyze_data_one_param_scan(src.get_data_files(star_single_colored_dir, regex))
        for regex in psi_plus_n3_star_colored_regexes
    ]

    psi_plus_max_n3_star_single_colored = [
        max(map(lambda opt_data: opt_data["max_scores"][i], psi_plus_n3_star_single_colored_data))
        for i in range(num_samples)
    ]

    phi_plus_n3_star_single_colored_data = [
        src.analyze_data_one_param_scan(src.get_data_files(star_single_colored_dir, regex))
        for regex in phi_plus_n3_star_colored_regexes
    ]

    phi_plus_max_n3_star_single_colored = [
        max(map(lambda opt_data: opt_data["max_scores"][i], phi_plus_n3_star_single_colored_data))
        for i in range(num_samples)
    ]

    psi_plus_theoretical_bell_state_single_n3_star = [
        src.star_max_violation_chsh_prod([state, psi_plus_state, psi_plus_state])
        for state in psi_plus_noise_states
    ]

    phi_plus_theoretical_bell_state_single_n3_star = [
        src.star_max_violation_chsh_prod([state, psi_plus_state, psi_plus_state])
        for state in phi_plus_noise_states
    ]

    psi_plus_theoretical_bell_state_uniform_n3_star = [
        src.star_max_violation_chsh_prod([state, state, state]) for state in psi_plus_noise_states
    ]

    phi_plus_theoretical_bell_state_uniform_n3_star = [
        src.star_max_violation_chsh_prod([state, state, state]) for state in phi_plus_noise_states
    ]

    """
    Plotting Data
    """

    src.plot_nonunital_single_and_uniform_max_scores_data(
        fig_title="Source Colored Noise Robustness",
        ax_titles=["Single Source Noise", "Uniform Source Noise"],
        noise_params=psi_plus_chsh_colored_data[0]["noise_params"],
        quantum_bound=np.sqrt(2),
        classical_bound=1,
        row1_single_max_scores=[
            phi_plus_max_chsh_colored,
            phi_plus_max_bilocal_single_colored,
            phi_plus_max_n3_chain_single_colored,
            phi_plus_max_n4_chain_single_colored,
            phi_plus_max_n3_star_single_colored,
        ],
        row1_single_theoretical_scores=[
            phi_plus_theoretical_bell_state_chsh,
            phi_plus_theoretical_bell_state_single_bilocal,
            phi_plus_theoretical_bell_state_single_n3_chain,
            phi_plus_theoretical_bell_state_single_n4_chain,
            phi_plus_theoretical_bell_state_single_n3_star,
        ],
        row1_uniform_max_scores=[
            phi_plus_max_chsh_colored,
            phi_plus_max_bilocal_uniform_colored,
            phi_plus_max_n3_chain_uniform_colored,
            phi_plus_max_n4_chain_uniform_colored,
            phi_plus_max_n3_star_uniform_colored,
        ],
        row1_uniform_theoretical_scores=[
            phi_plus_theoretical_bell_state_chsh,
            phi_plus_theoretical_bell_state_uniform_bilocal,
            phi_plus_theoretical_bell_state_uniform_n3_chain,
            phi_plus_theoretical_bell_state_uniform_n4_chain,
            phi_plus_theoretical_bell_state_uniform_n3_star,
        ],
        row2_single_max_scores=[
            psi_plus_max_chsh_colored,
            psi_plus_max_bilocal_single_colored,
            psi_plus_max_n3_chain_single_colored,
            psi_plus_max_n4_chain_single_colored,
            psi_plus_max_n3_star_single_colored,
        ],
        row2_single_theoretical_scores=[
            psi_plus_theoretical_bell_state_chsh,
            psi_plus_theoretical_bell_state_single_bilocal,
            psi_plus_theoretical_bell_state_single_n3_chain,
            psi_plus_theoretical_bell_state_single_n4_chain,
            psi_plus_theoretical_bell_state_single_n3_star,
        ],
        row2_uniform_max_scores=[
            psi_plus_max_chsh_colored,
            psi_plus_max_bilocal_uniform_colored,
            psi_plus_max_n3_chain_uniform_colored,
            psi_plus_max_n3_chain_uniform_colored,
            psi_plus_max_n3_star_uniform_colored,
        ],
        row2_uniform_theoretical_scores=[
            psi_plus_theoretical_bell_state_chsh,
            psi_plus_theoretical_bell_state_uniform_bilocal,
            psi_plus_theoretical_bell_state_uniform_n3_chain,
            psi_plus_theoretical_bell_state_uniform_n4_chain,
            psi_plus_theoretical_bell_state_uniform_n3_star,
        ],
        data_labels=["CHSH", "Bilocal", "3-Local Chain", "4-Local Chain", "3-Local Star"],
        row_labels=[r"$|\Phi^+\rangle$ State", r"$|\Psi^+\rangle$ State"],
        plot_dir="./data/plots/source_colored_noise_robustness/",
    )

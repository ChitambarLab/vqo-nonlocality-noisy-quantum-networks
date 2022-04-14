import qnetvo as qnet
from context import src

from pennylane import numpy as np
import pennylane as qml

"""
This script aggregates and plots data for qubit phase damping noise.
"""

@qml.qnode(qml.device("default.mixed", wires=[0,1]))
def bell_state_uniform_noise(gamma):
    qml.Hadamard(wires=[0])
    qml.CNOT(wires=[0,1])

    qml.PhaseDamping(gamma, wires=[0])
    qml.PhaseDamping(gamma, wires=[1])

    return qml.state()

@qml.qnode(qml.device("default.mixed", wires=[0,1]))
def bell_state_single_noise(gamma):
    qml.Hadamard(wires=[0])
    qml.CNOT(wires=[0,1])

    qml.PhaseDamping(gamma, wires=[0])

    return qml.state()


if __name__ == "__main__":
    num_samples = 21

    bell_state_uniform_noise_states = [
        bell_state_uniform_noise(gamma)
        for gamma in np.arange(0, 1.01, 0.05)
    ]
    bell_state_single_noise_states = [
        bell_state_single_noise(gamma)
        for gamma in np.arange(0, 1.01, 0.05)
    ]

    bell_state = np.array([[1,0,0,1],[0,0,0,0],[0,0,0,0],[1,0,0,1]])/2

    """
    Loading CHSH Data
    """
    chsh_pd_regexes = [r"max_ent_local_rot_.*", r"arb_local_rot.*", r"ghz_local_ry_.*"]

    chsh_uniform_pd_dir = "./data/chsh/uniform_qubit_phase_damping/"

    chsh_uniform_pd_data = [
        src.analyze_data_one_param_scan(
            src.get_data_files(chsh_uniform_pd_dir, regex)
        )
        for regex in chsh_pd_regexes
    ]
    max_chsh_uniform_pd = [
        max(map(lambda opt_data: opt_data["max_scores"][i], chsh_uniform_pd_data)) / 2
        for i in range(num_samples)
    ]

    chsh_single_pd_dir = "./data/chsh/single_qubit_phase_damping/"

    chsh_single_pd_regexes = [r"arb_local_rot_.*", r"max_ent_local_rot_.*", r"ghz_local_ry_.*"]
    chsh_single_pd_data = [
        src.analyze_data_one_param_scan(
            src.get_data_files(chsh_single_pd_dir, regex)
        )
        for regex in chsh_single_pd_regexes
    ]
    max_chsh_single_pd = [
        max(map(lambda opt_data: opt_data["max_scores"][i], chsh_single_pd_data)) / 2
        for i in range(num_samples)
    ]

    theoretical_bell_state_uniform_chsh = [
        src.chsh_max_violation(state) / 2
        for state in bell_state_uniform_noise_states
    ]

    theoretical_bell_state_single_chsh = [
        src.chsh_max_violation(state) / 2
        for state in bell_state_single_noise_states
    ]

    """
    Loading Bilocal Data
    """
    bilocal_uniform_pd_dir = "./data/bilocal/uniform_qubit_phase_damping/"

    bilocal_uniform_pd_regexes = [r"ghz_local_ry_.*", r"arb_arb_.*", r"max_ent_arb_.*", r"max_ent_local_rot_.*"]
    bilocal_uniform_pd_data = [
        src.analyze_data_one_param_scan(
            src.get_data_files(bilocal_uniform_pd_dir, regex)
        )
        for regex in bilocal_uniform_pd_regexes
    ]

    max_bilocal_uniform_pd = [
        max(map(lambda opt_data: opt_data["max_scores"][i], bilocal_uniform_pd_data))
        for i in range(num_samples)
    ]

    bilocal_single_pd_dir = "./data/bilocal/single_qubit_phase_damping/"

    bilocal_single_pd_regexes = [r"max_ent_local_rot_.*", r"ghz_local_ry_.*", r"max_ent_arb_.*", r"arb_arb_.*"]
    bilocal_single_pd_data = [
        src.analyze_data_one_param_scan(
            src.get_data_files(bilocal_single_pd_dir, regex)
        )
        for regex in bilocal_single_pd_regexes
    ]

    max_bilocal_single_pd = [
        max(map(lambda opt_data: opt_data["max_scores"][i], bilocal_single_pd_data))
        for i in range(num_samples)
    ]

    theoretical_bell_state_uniform_bilocal = [
        src.bilocal_max_violation(state, state)
        for state in bell_state_uniform_noise_states
    ]

    theoretical_bell_state_single_bilocal = [
        src.bilocal_max_violation(state, bell_state)
        for state in bell_state_single_noise_states
    ]

    """
    Loading n-Chain Data
    """

    chain_uniform_pd_dir = "./data/n-chain/uniform_qubit_phase_damping/"

    n3_chain_uniform_pd_regexes = [
        r"ghz_local_ry_n-3_.*",
        r"arb_arb_n-3_.*",
        r"arb_local_rot_n-3_.*",
        r"max_entangled_arb_n-3_.*",
        r"max_entangled_local_rot_n-3_.*"
    ]

    n3_chain_uniform_pd_data = [
        src.analyze_data_one_param_scan(
            src.get_data_files(chain_uniform_pd_dir, regex)
        )
        for regex in n3_chain_uniform_pd_regexes
    ]

    max_n3_chain_uniform_pd = [
        max(map(lambda opt_data: opt_data["max_scores"][i], n3_chain_uniform_pd_data))
        for i in range(num_samples)
    ]

    n4_chain_uniform_pd_regexes = [
        r"ghz_local_ry_n-4_.*",
        r"arb_arb_n-4_.*",
        r"arb_local_rot_n-4_.*",
        r"max_entangled_arb_n-4_.*",
        r"max_entangled_local_rot_n-4_.*"
    ]

    n4_chain_uniform_pd_data = [
        src.analyze_data_one_param_scan(
            src.get_data_files(chain_uniform_pd_dir, regex)
        )
        for regex in n4_chain_uniform_pd_regexes
    ]

    max_n4_chain_uniform_pd = [
        max(map(lambda opt_data: opt_data["max_scores"][i], n4_chain_uniform_pd_data))
        for i in range(num_samples)
    ]

    chain_single_pd_dir = "./data/n-chain/single_qubit_phase_damping/"

    n3_chain_single_pd_regexes = [
        r"ghz_local_ry_n-3_.*",
        r"arb_arb_n-3_.*",
        r"arb_local_rot_n-3_.*",
        r"max_entangled_arb_n-3_.*",
        r"max_entangled_local_rot_n-3_.*",
        r"ryrz_cnot_local_ry_n-3_.*",
    ]

    n3_chain_single_pd_data = [
        src.analyze_data_one_param_scan(
            src.get_data_files(chain_single_pd_dir, regex)
        )
        for regex in n3_chain_single_pd_regexes
    ]

    max_n3_chain_single_pd = [
        max(map(lambda opt_data: opt_data["max_scores"][i], n3_chain_single_pd_data))
        for i in range(num_samples)
    ]

    n4_chain_single_pd_regexes = [
        r"ghz_local_ry_n-4_.*",
        r"arb_arb_n-4_.*",
        r"arb_local_rot_n-4_.*",
        r"max_entangled_arb_n-4_.*",
        r"max_entangled_local_rot_n-4_.*",
        r"ryrz_cnot_local_ry_n-4_.*",
    ]
    n4_chain_single_pd_data = [
        src.analyze_data_one_param_scan(
            src.get_data_files(chain_single_pd_dir, regex)
        )
        for regex in n4_chain_single_pd_regexes
    ]

    max_n4_chain_single_pd = [
        max(map(lambda opt_data: opt_data["max_scores"][i], n4_chain_single_pd_data))
        for i in range(num_samples)
    ]

    theoretical_bell_state_uniform_n3_chain = [
        src.chain_classical_interior_max_violation([state, state, state])
        for state in bell_state_uniform_noise_states
    ]

    theoretical_bell_state_single_n3_chain = [
        src.chain_classical_interior_max_violation([state, bell_state, bell_state])
        for state in bell_state_single_noise_states
    ]

    theoretical_bell_state_uniform_n4_chain = [
        src.chain_classical_interior_max_violation([state, state, state, state])
        for state in bell_state_uniform_noise_states
    ]

    theoretical_bell_state_single_n4_chain = [
        src.chain_classical_interior_max_violation([state, bell_state, bell_state, bell_state])
        for state in bell_state_single_noise_states
    ]

    """
    Loading n-Star Data
    """

    star_uniform_pd_dir = "./data/n-star/uniform_qubit_phase_damping/"

    n3_star_pd_uniform_regexes = [
        r"arb_ghz_rot_n-3_.*",
        r"arb_local_rot_n-3_.*",
        r"max_entangled_ghz_rot_n-3_.*",
        r"max_entangled_local_rot_n-3_.*",
        r"ghz_local_ry_n-3_.*",
    ]

    n3_star_uniform_pd_data = [
        src.analyze_data_one_param_scan(
            src.get_data_files(star_uniform_pd_dir, regex)
        )
        for regex in n3_star_pd_uniform_regexes
    ]

    max_n3_star_uniform_pd = [
        max(map(lambda opt_data: opt_data["max_scores"][i], n3_star_uniform_pd_data))
        for i in range(num_samples)
    ]

    n4_star_pd_uniform_regexes = [
        r"arb_ghz_rot_n-4_.*",
        r"arb_local_rot_n-4_.*",
        r"max_entangled_ghz_rot_n-4_.*",
        r"max_entangled_local_rot_n-4_.*",
        r"ghz_local_ry_n-4_.*",
    ]

    n4_star_uniform_pd_data = [
        src.analyze_data_one_param_scan(
            src.get_data_files(star_uniform_pd_dir, regex)
        )
        for regex in n4_star_pd_uniform_regexes
    ]

    max_n4_star_uniform_pd = [
        max(map(lambda opt_data: opt_data["max_scores"][i], n4_star_uniform_pd_data))
        for i in range(num_samples)
    ]

    star_single_pd_dir = "./data/n-star/single_qubit_phase_damping/"

    n3_star_pd_single_regexes = [
        r"arb_ghz_rot_n-3_.*",
        r"arb_local_rot_n-3_.*",
        r"max_entangled_ghz_rot_n-3_.*",
        r"max_entangled_local_rot_n-3_.*",
        r"ghz_local_ry_n-3_.*",
    ]

    n3_star_single_pd_data = [
        src.analyze_data_one_param_scan(
            src.get_data_files(star_single_pd_dir, regex)
        )
        for regex in n3_star_pd_single_regexes
    ]

    max_n3_star_single_pd = [
        max(map(lambda opt_data: opt_data["max_scores"][i], n3_star_single_pd_data))
        for i in range(num_samples)
    ]

    n4_star_pd_single_regexes = [
        r"arb_ghz_rot_n-4_.*",
        r"arb_local_rot_n-4_.*",
        r"max_entangled_ghz_rot_n-4_.*",
        r"max_entangled_local_rot_n-4_.*",
        r"ghz_local_ry_n-4_.*",
    ]

    n4_star_single_pd_data = [
        src.analyze_data_one_param_scan(
            src.get_data_files(star_single_pd_dir, regex)
        )
        for regex in n4_star_pd_single_regexes
    ]

    max_n4_star_single_pd = [
        max(map(lambda opt_data: opt_data["max_scores"][i], n4_star_single_pd_data))
        for i in range(num_samples)
    ]

    theoretical_bell_state_uniform_n3_star = [
        src.star_max_violation([state, state, state])
        for state in bell_state_uniform_noise_states
    ]

    theoretical_bell_state_single_n3_star = [
        src.star_max_violation([state, bell_state, bell_state])
        for state in bell_state_single_noise_states
    ]

    theoretical_bell_state_uniform_n4_star = [
        src.star_max_violation([state, state, state, state])
        for state in bell_state_uniform_noise_states
    ]

    theoretical_bell_state_single_n4_star = [
        src.star_max_violation([state, bell_state, bell_state, bell_state])
        for state in bell_state_single_noise_states
    ]

    """
    Plotting Data
    """

    src.plot_single_and_uniform_max_scores_data(
        fig_title = "Qubit Phase Damping Noise Robustness",
        ax_titles = ["Single Qubit Noise", "Uniform Qubit Noise"],
        noise_params = chsh_uniform_pd_data[0]["noise_params"],
        quantum_bound = np.sqrt(2),
        classical_bound = 1,
        single_max_scores = [
            max_chsh_single_pd, max_bilocal_single_pd, max_n3_chain_single_pd,
            max_n4_chain_single_pd, max_n3_star_single_pd, max_n4_star_single_pd
        ],
        single_theoretical_scores = [
            theoretical_bell_state_single_chsh,
            theoretical_bell_state_single_bilocal,
            theoretical_bell_state_single_n3_chain,
            theoretical_bell_state_single_n4_chain,
            theoretical_bell_state_single_n3_star,
            theoretical_bell_state_single_n4_star,
        ],
        uniform_max_scores = [
            max_chsh_uniform_pd, max_bilocal_uniform_pd, max_n3_chain_uniform_pd,
            max_n4_chain_uniform_pd, max_n3_star_uniform_pd, max_n4_star_uniform_pd
        ],
        uniform_theoretical_scores = [
            theoretical_bell_state_uniform_chsh,
            theoretical_bell_state_uniform_bilocal,
            theoretical_bell_state_uniform_n3_chain,
            theoretical_bell_state_uniform_n4_chain,
            theoretical_bell_state_uniform_n3_star,
            theoretical_bell_state_uniform_n4_star,
        ],
        data_labels = ["CHSH", "Bilocal", "3-Local Chain", "4-Local Chain", "3-Local Star", "4-Local Star"],
        plot_dir =  "./data/plots/qubit_phase_damping_noise_robustness/" 
    )


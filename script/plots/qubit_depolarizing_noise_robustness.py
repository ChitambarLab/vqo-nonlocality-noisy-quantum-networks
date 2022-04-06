import qnetvo as qnet
from context import src

from pennylane import numpy as np


"""
This script aggregates data and plots the noise robustness results qubit
depolarizing noise.
"""
if __name__ == "__main__":
    num_samples = 21

    """
    Loading CHSH Data
    """
    chsh_dep_regexes = [r"max_ent_local_rot_.*", r"arb_local_rot.*", r"ghz_local_ry_.*"]

    chsh_uniform_dep_dir = "./data/chsh/uniform_qubit_depolarizing/"

    chsh_uniform_dep_data = [
        src.analyze_data_one_param_scan(
            src.get_data_files(chsh_uniform_dep_dir, regex)
        )
        for regex in chsh_dep_regexes
    ]
    max_chsh_uniform_dep = [
        max(map(lambda opt_data: opt_data["max_scores"][i], chsh_uniform_dep_data)) / 2
        for i in range(num_samples)
    ]

    chsh_single_dep_dir = "./data/chsh/single_qubit_depolarizing/"

    chsh_single_dep_regexes = [r"arb_local_rot_.*", r"max_ent_local_rot_.*", r"ghz_local_ry_.*"]
    chsh_single_dep_data = [
        src.analyze_data_one_param_scan(
            src.get_data_files(chsh_single_dep_dir, regex)
        )
        for regex in chsh_single_dep_regexes
    ]
    max_chsh_single_dep = [
        max(map(lambda opt_data: opt_data["max_scores"][i], chsh_single_dep_data)) / 2
        for i in range(num_samples)
    ]

    """
    Loading Bilocal Data
    """
    bilocal_uniform_dep_dir = "./data/bilocal/uniform_qubit_depolarizing/"

    bilocal_dep_regexes = [r"ghz_local_ry_.*"]
    bilocal_uniform_dep_data = [
        src.analyze_data_one_param_scan(
            src.get_data_files(bilocal_uniform_dep_dir, regex)
        )
        for regex in bilocal_dep_regexes
    ]

    max_bilocal_uniform_dep = [
        max(map(lambda opt_data: opt_data["max_scores"][i], bilocal_uniform_dep_data))
        for i in range(num_samples)
    ]

    bilocal_single_dep_dir = "./data/bilocal/single_qubit_depolarizing/"

    bilocal_dep_regexes = [r"max_ent_local_rot_out.*", r"ghz_local_ry_out_.*", r"max_ent_arb_out_.*"]#, r"arb_arb_.*"]
    bilocal_single_dep_data = [
        src.analyze_data_one_param_scan(
            src.get_data_files(bilocal_single_dep_dir, regex)
        )
        for regex in bilocal_dep_regexes
    ]

    max_bilocal_single_dep = [
        max(map(lambda opt_data: opt_data["max_scores"][i], bilocal_single_dep_data))
        for i in range(num_samples)
    ]

    """
    Loading n-Chain Data
    """

    chain_uniform_dep_dir = "./data/n-chain/uniform_qubit_depolarizing/"

    n3_chain_uniform_dep_regexes = [r"ghz_local_ry_n-3_.*"]

    n3_chain_uniform_dep_data = [
        src.analyze_data_one_param_scan(
            src.get_data_files(chain_uniform_dep_dir, regex)
        )
        for regex in n3_chain_uniform_dep_regexes
    ]

    max_n3_chain_uniform_dep = [
        max(map(lambda opt_data: opt_data["max_scores"][i], n3_chain_uniform_dep_data))
        for i in range(num_samples)
    ]

    n4_chain_uniform_dep_regexes = [r"ghz_local_ry_n-4_.*"]

    n4_chain_uniform_dep_data = [
        src.analyze_data_one_param_scan(
            src.get_data_files(chain_uniform_dep_dir, regex)
        )
        for regex in n4_chain_uniform_dep_regexes
    ]

    max_n4_chain_uniform_dep = [
        max(map(lambda opt_data: opt_data["max_scores"][i], n4_chain_uniform_dep_data))
        for i in range(num_samples)
    ]

    chain_single_dep_dir = "./data/n-chain/single_qubit_depolarizing/"

    n3_chain_single_dep_regexes = [r"ghz_local_ry_out_n-3_.*"]

    n3_chain_single_dep_data = [
        src.analyze_data_one_param_scan(
            src.get_data_files(chain_single_dep_dir, regex)
        )
        for regex in n3_chain_single_dep_regexes
    ]

    max_n3_chain_single_dep = [
        max(map(lambda opt_data: opt_data["max_scores"][i], n3_chain_single_dep_data))
        for i in range(num_samples)
    ]

    n4_chain_single_dep_regexes = [r"ghz_local_ry_out_n-4_.*"]

    n4_chain_single_dep_data = [
        src.analyze_data_one_param_scan(
            src.get_data_files(chain_single_dep_dir, regex)
        )
        for regex in n4_chain_single_dep_regexes
    ]

    max_n4_chain_single_dep = [
        max(map(lambda opt_data: opt_data["max_scores"][i], n4_chain_single_dep_data))
        for i in range(num_samples)
    ]

    """
    Loading n-Star Data
    """

    star_uniform_dep_dir = "./data/n-star/uniform_qubit_depolarizing/"

    n3_star_dep_uniform_regexes = [r"ghz_local_ry_n-3_.*"]

    n3_star_uniform_dep_data = [
        src.analyze_data_one_param_scan(
            src.get_data_files(star_uniform_dep_dir, regex)
        )
        for regex in n3_star_dep_uniform_regexes
    ]

    max_n3_star_uniform_dep = [
        max(map(lambda opt_data: opt_data["max_scores"][i], n3_star_uniform_dep_data))
        for i in range(num_samples)
    ]

    n4_star_dep_uniform_regexes = [r"ghz_local_ry_n-4_.*"]

    n4_star_uniform_dep_data = [
        src.analyze_data_one_param_scan(
            src.get_data_files(star_uniform_dep_dir, regex)
        )
        for regex in n4_star_dep_uniform_regexes
    ]

    max_n4_star_uniform_dep = [
        max(map(lambda opt_data: opt_data["max_scores"][i], n4_star_uniform_dep_data))
        for i in range(num_samples)
    ]

    star_single_dep_dir = "./data/n-star/single_qubit_depolarizing/"

    n3_star_dep_single_regexes = [r"ghz_local_ry_out_n-3_.*"]

    n3_star_single_dep_data = [
        src.analyze_data_one_param_scan(
            src.get_data_files(star_single_dep_dir, regex)
        )
        for regex in n3_star_dep_single_regexes
    ]

    max_n3_star_single_dep = [
        max(map(lambda opt_data: opt_data["max_scores"][i], n3_star_single_dep_data))
        for i in range(num_samples)
    ]

    n4_star_dep_single_regexes = [r"ghz_local_ry_out_n-4_.*"]

    n4_star_single_dep_data = [
        src.analyze_data_one_param_scan(
            src.get_data_files(star_single_dep_dir, regex)
        )
        for regex in n4_star_dep_single_regexes
    ]

    max_n4_star_single_dep = [
        max(map(lambda opt_data: opt_data["max_scores"][i], n4_star_single_dep_data))
        for i in range(num_samples)
    ]

    """
    Plotting Data
    """

    src.plot_single_and_uniform_max_scores_data(
        fig_title = "Qubit Depolarizing Noise Robustness",
        ax_titles = ["Single Qubit Noise", "Uniform Qubit Noise"],
        noise_params = chsh_uniform_dep_data[0]["noise_params"],
        quantum_bound = np.sqrt(2),
        classical_bound = 1,
        single_max_scores = [
            max_chsh_single_dep, max_bilocal_single_dep, max_n3_chain_single_dep,
            max_n4_chain_single_dep, max_n3_star_single_dep, max_n4_star_single_dep
        ],
        uniform_max_scores = [
            max_chsh_uniform_dep, max_bilocal_uniform_dep, max_n3_chain_uniform_dep,
            max_n4_chain_uniform_dep, max_n3_star_uniform_dep, max_n4_star_uniform_dep
        ],
        data_labels = ["CHSH", "Bilocal", "3-Local Chain", "4-Local Chain", "3-Local Star", "4-Local Star"],
        plot_dir =  "./data/plots/qubit_depolarizing_noise_robustness/" 
    )


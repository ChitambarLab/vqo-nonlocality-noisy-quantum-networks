import qnetvo as qnet
from context import src

from pennylane import numpy as np


"""
This script aggregates data and plots the noise robustness results qubit
depolarizing noise.
"""

if __name__ == "__main__":
    num_samples = 21

    noise_params = np.arange(0, 1.01, 0.05)

    """
    Loading CHSH Data
    """
    chsh_biased_regexes = [r"max_ent_local_rot_.*", r"ghz_local_rot_.*", r"ryrz_cnot_local_ry_.*", r"arb_local_rot_.*"]

    chsh_single_biased_dir = "./data/chsh/single_detector_biased_noise/"

    chsh_single_biased_data = [
        src.analyze_data_one_param_scan(
            src.get_data_files(chsh_single_biased_dir, regex)
        )
        for regex in chsh_biased_regexes
    ]
    max_chsh_single_biased = [
        max(map(lambda opt_data: opt_data["max_scores"][i], chsh_single_biased_data)) / 2
        for i in range(num_samples)
    ]

    chsh_uniform_biased_dir = "./data/chsh/uniform_detector_biased_noise/"

    chsh_uniform_biased_data = [
        src.analyze_data_one_param_scan(
            src.get_data_files(chsh_uniform_biased_dir, regex)
        )
        for regex in chsh_biased_regexes
    ]
    max_chsh_uniform_biased = [
        max(map(lambda opt_data: opt_data["max_scores"][i], chsh_uniform_biased_data)) / 2
        for i in range(num_samples)
    ]

    theoretical_max_chsh_uniform_score = [
        np.sqrt(2) * (1 - gamma) ** 2
        for gamma in noise_params
    ]

    theoretical_max_chsh_single_score = [
        np.sqrt(2) * (1 - gamma)
        for gamma in noise_params
    ]

    """
    Loading Bilocal Data
    """
    bilocal_biased_regexes = [r"max_ent_local_rot_.*", r"arb_arb_.*", r"ghz_arb_.*", r"ghz_local_ry_.*", r"ryrz_cnot_local_ry_.*", r"ryrz_cnot_arb_.*",   r"max_ent_arb_.*", r"arb_local_rot_.*"]

    bilocal_uniform_biased_dir = "./data/bilocal/uniform_detector_biased_noise/"

    bilocal_uniform_biased_data = [
        src.analyze_data_one_param_scan(
            src.get_data_files(bilocal_uniform_biased_dir, regex)
        )
        for regex in bilocal_biased_regexes
    ]

    max_bilocal_uniform_biased = [
        max(map(lambda opt_data: opt_data["max_scores"][i], bilocal_uniform_biased_data))
        for i in range(num_samples)
    ]

    bilocal_single_biased_dir = "./data/bilocal/single_detector_biased_noise/"

    bilocal_single_biased_data = [
        src.analyze_data_one_param_scan(
            src.get_data_files(bilocal_single_biased_dir, regex)
        )
        for regex in bilocal_biased_regexes
    ]

    max_bilocal_single_biased = [
        max(map(lambda opt_data: opt_data["max_scores"][i], bilocal_single_biased_data))
        for i in range(num_samples)
    ]

    theoretical_max_bilocal_single_score = [
        np.sqrt(2) * np.sqrt((1 - gamma))
        for gamma in noise_params
    ]
    theoretical_max_bilocal_uniform_score = [
        np.sqrt(2) * np.sqrt((1 - gamma) ** 3)
        for gamma in noise_params
    ]

    """
    Loading n-Chain Data
    """

    chain_uniform_biased_dir = "./data/n-chain/uniform_detector_biased_noise/"

    n3_chain_biased_regexes = [r"arb_arb_n-3_.*", r"arb_local_rot_n-3_.*", r"ghz_local_ry_n-3_.*", r"max_ent_arb_n-3_.*", r"ryrz_cnot_local_ry_n-3_.*"]

    n3_chain_uniform_biased_data = [
        src.analyze_data_one_param_scan(
            src.get_data_files(chain_uniform_biased_dir, regex)
        )
        for regex in n3_chain_biased_regexes
    ]

    max_n3_chain_uniform_biased = [
        max(map(lambda opt_data: opt_data["max_scores"][i], n3_chain_uniform_biased_data))
        for i in range(num_samples)
    ]

    chain_single_biased_dir = "./data/n-chain/single_detector_biased_noise/"

    n3_chain_single_biased_data = [
        src.analyze_data_one_param_scan(
            src.get_data_files(chain_single_biased_dir, regex)
        )
        for regex in n3_chain_biased_regexes
    ]

    max_n3_chain_single_biased = [
        max(map(lambda opt_data: opt_data["max_scores"][i], n3_chain_single_biased_data))
        for i in range(num_samples)
    ]

    n4_chain_biased_regexes = [r"arb_arb_n-4_.*", r"arb_local_rot_n-4_.*", r"ghz_local_ry_n-4_.*", r"max_ent_arb_n-4_.*", r"ryrz_cnot_local_ry_n-4_.*"]

    n4_chain_uniform_biased_data = [
        src.analyze_data_one_param_scan(
            src.get_data_files(chain_uniform_biased_dir, regex)
        )
        for regex in n4_chain_biased_regexes
    ]

    max_n4_chain_uniform_biased = [
        max(map(lambda opt_data: opt_data["max_scores"][i], n4_chain_uniform_biased_data))
        for i in range(num_samples)
    ]

    chain_single_biased_dir = "./data/n-chain/single_detector_biased_noise/"

    n4_chain_single_biased_data = [
        src.analyze_data_one_param_scan(
            src.get_data_files(chain_single_biased_dir, regex)
        )
        for regex in n4_chain_biased_regexes
    ]

    max_n4_chain_single_biased = [
        max(map(lambda opt_data: opt_data["max_scores"][i], n4_chain_single_biased_data))
        for i in range(num_samples)
    ]

    theoretical_max_n3_chain_single_score = [
        np.sqrt(2) * np.sqrt((1 - gamma))
        for gamma in noise_params
    ]

    theoretical_max_n4_chain_single_score = [
        np.sqrt(2) * np.sqrt((1 - gamma))
        for gamma in noise_params
    ]

    theoretical_max_n3_chain_uniform_score = [
        np.sqrt(2) * np.sqrt((1 - gamma) ** 4)
        for gamma in noise_params
    ]

    theoretical_max_n4_chain_uniform_score = [
        np.sqrt(2) * np.sqrt((1 - gamma) ** 5)
        for gamma in noise_params
    ]

    """
    Loading n-Star Data
    """

    star_uniform_biased_dir = "./data/n-star/uniform_detector_biased_noise/"

    n3_star_biased_regexes = [r"arb_arb_n-3_.*", r"arb_local_rot_n-3_.*", r"ghz_local_ry_n-3_.*", r"max_ent_arb_n-3_.*", r"ryrz_cnot_local_ry_n-3_.*"]

    n3_star_uniform_biased_data = [
        src.analyze_data_one_param_scan(
            src.get_data_files(star_uniform_biased_dir, regex)
        )
        for regex in n3_star_biased_regexes
    ]

    max_n3_star_uniform_biased = [
        max(map(lambda opt_data: opt_data["max_scores"][i], n3_star_uniform_biased_data))
        for i in range(num_samples)
    ]

    star_single_biased_dir = "./data/n-star/single_detector_biased_noise/"

    n3_star_single_biased_data = [
        src.analyze_data_one_param_scan(
            src.get_data_files(star_single_biased_dir, regex)
        )
        for regex in n3_star_biased_regexes
    ]

    max_n3_star_single_biased = [
        max(map(lambda opt_data: opt_data["max_scores"][i], n3_star_single_biased_data))
        for i in range(num_samples)
    ]

    n4_star_biased_regexes = [r"arb_local_rot_n-4_.*", r"ghz_local_ry_n-4_.*", r"ryrz_cnot_local_ry_n-4_.*"]

    n4_star_uniform_biased_data = [
        src.analyze_data_one_param_scan(
            src.get_data_files(star_uniform_biased_dir, regex)
        )
        for regex in n4_star_biased_regexes
    ]

    max_n4_star_uniform_biased = [
        max(map(lambda opt_data: opt_data["max_scores"][i], n4_star_uniform_biased_data))
        for i in range(num_samples)
    ]

    n4_star_single_biased_data = [
        src.analyze_data_one_param_scan(
            src.get_data_files(star_single_biased_dir, regex)
        )
        for regex in n4_star_biased_regexes
    ]

    max_n4_star_single_biased = [
        max(map(lambda opt_data: opt_data["max_scores"][i], n4_star_single_biased_data))
        for i in range(num_samples)
    ]

    theoretical_max_n3_star_single_score = [
        np.sqrt(2) * np.power((1 - gamma), 1/3)
        for gamma in noise_params
    ]

    theoretical_max_n4_star_single_score = [
        np.sqrt(2) * np.power((1 - gamma), 1/4)
        for gamma in noise_params
    ]

    theoretical_max_n3_star_uniform_score = [
        np.sqrt(2) * np.power((1 - gamma) ** 4, 1/3)
        for gamma in noise_params
    ]

    theoretical_max_n4_star_uniform_score = [
        np.sqrt(2) * np.power((1 - gamma) ** 5, 1/4)
        for gamma in noise_params
    ]

    """
    Plotting Data
    """

    src.plot_single_and_uniform_max_scores_data(
        fig_title = "Detector Biased Noise Robustness",
        ax_titles = ["Single Detector Noise", "Uniform Detector Noise"],
        noise_params = chsh_single_biased_data[0]["noise_params"],
        quantum_bound = np.sqrt(2),
        classical_bound = 1,
        single_max_scores = [
            max_chsh_single_biased, max_bilocal_single_biased, max_n3_chain_single_biased,
            max_n4_chain_single_biased, max_n3_star_single_biased, max_n4_star_single_biased
        ],
        single_theoretical_scores = [
            theoretical_max_chsh_single_score,
            theoretical_max_bilocal_single_score,
            theoretical_max_n3_chain_single_score,
            theoretical_max_n4_chain_single_score,
            theoretical_max_n3_star_single_score,
            theoretical_max_n4_star_single_score,
        ],
        uniform_max_scores = [
            max_chsh_uniform_biased, max_bilocal_uniform_biased, max_n3_chain_uniform_biased,
            max_n4_chain_uniform_biased, max_n3_star_uniform_biased, max_n4_star_uniform_biased,
        ],
        uniform_theoretical_scores = [
            theoretical_max_chsh_uniform_score,
            theoretical_max_bilocal_uniform_score,
            theoretical_max_n3_chain_uniform_score,
            theoretical_max_n4_chain_uniform_score,
            theoretical_max_n3_star_uniform_score,
            theoretical_max_n4_star_uniform_score,
        ],
        data_labels = ["CHSH", "Bilocal", "3-Local Chain", "4-Local Chain", "3-Local Star", "4-Local Chain"],
        plot_dir =  "./data/plots/detector_biased_noise_robustness/" 
    )


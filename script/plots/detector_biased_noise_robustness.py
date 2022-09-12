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
    chsh_single_biased_dir = "./data/chsh/single_detector_biased_noise/"

    ent_chsh_biased_regexes = [r"max_ent_local_rot_.*", r"ghz_local_rot_.*"]
    arb_chsh_biased_regexes = [
        r"ryrz_cnot_local_ry_.*",
        r"arb_local_rot_.*",
        r"max_ent_local_rot_.*",
        r"ghz_local_rot_.*",
    ]

    ent_chsh_single_biased_data = [
        src.analyze_data_one_param_scan(src.get_data_files(chsh_single_biased_dir, regex))
        for regex in ent_chsh_biased_regexes
    ]
    ent_max_chsh_single_biased = [
        max(map(lambda opt_data: opt_data["max_scores"][i], ent_chsh_single_biased_data)) / 2
        for i in range(num_samples)
    ]

    arb_chsh_single_biased_data = [
        src.analyze_data_one_param_scan(src.get_data_files(chsh_single_biased_dir, regex))
        for regex in arb_chsh_biased_regexes
    ]
    arb_max_chsh_single_biased = [
        max(map(lambda opt_data: opt_data["max_scores"][i], arb_chsh_single_biased_data)) / 2
        for i in range(num_samples)
    ]

    chsh_uniform_biased_dir = "./data/chsh/uniform_detector_biased_noise/"

    arb_chsh_uniform_biased_data = [
        src.analyze_data_one_param_scan(src.get_data_files(chsh_uniform_biased_dir, regex))
        for regex in arb_chsh_biased_regexes
    ]
    arb_max_chsh_uniform_biased = [
        max(map(lambda opt_data: opt_data["max_scores"][i], arb_chsh_uniform_biased_data)) / 2
        for i in range(num_samples)
    ]

    ent_chsh_uniform_biased_data = [
        src.analyze_data_one_param_scan(src.get_data_files(chsh_uniform_biased_dir, regex))
        for regex in ent_chsh_biased_regexes
    ]
    ent_max_chsh_uniform_biased = [
        max(map(lambda opt_data: opt_data["max_scores"][i], ent_chsh_uniform_biased_data)) / 2
        for i in range(num_samples)
    ]

    """
    Loading Bilocal Data
    """
    ent_bilocal_biased_regexes = [r"max_ent_local_rot_.*", r"ghz_local_ry_.*"]
    arb_bilocal_biased_regexes = [
        r"arb_arb_.*",
        r"ghz_arb_.*",
        r"ryrz_cnot_arb_.*",
        r"max_ent_arb_.*",
        r"max_ent_local_rot_.*",
        r"ghz_local_ry_.*",
        r"ryrz_cnot_local_ry_.*",
        r"arb_local_rot_.*",
    ]

    bilocal_uniform_biased_dir = "./data/bilocal/uniform_detector_biased_noise/"

    ent_bilocal_uniform_biased_data = [
        src.analyze_data_one_param_scan(src.get_data_files(bilocal_uniform_biased_dir, regex))
        for regex in ent_bilocal_biased_regexes
    ]

    ent_max_bilocal_uniform_biased = [
        max(map(lambda opt_data: opt_data["max_scores"][i], ent_bilocal_uniform_biased_data))
        for i in range(num_samples)
    ]

    arb_bilocal_uniform_biased_data = [
        src.analyze_data_one_param_scan(src.get_data_files(bilocal_uniform_biased_dir, regex))
        for regex in arb_bilocal_biased_regexes
    ]

    arb_max_bilocal_uniform_biased = [
        max(map(lambda opt_data: opt_data["max_scores"][i], arb_bilocal_uniform_biased_data))
        for i in range(num_samples)
    ]

    bilocal_single_biased_dir = "./data/bilocal/single_detector_biased_noise/"

    ent_bilocal_single_biased_data = [
        src.analyze_data_one_param_scan(src.get_data_files(bilocal_single_biased_dir, regex))
        for regex in ent_bilocal_biased_regexes
    ]

    ent_max_bilocal_single_biased = [
        max(map(lambda opt_data: opt_data["max_scores"][i], ent_bilocal_single_biased_data))
        for i in range(num_samples)
    ]

    arb_bilocal_single_biased_data = [
        src.analyze_data_one_param_scan(src.get_data_files(bilocal_single_biased_dir, regex))
        for regex in arb_bilocal_biased_regexes
    ]

    arb_max_bilocal_single_biased = [
        max(map(lambda opt_data: opt_data["max_scores"][i], arb_bilocal_single_biased_data))
        for i in range(num_samples)
    ]

    """
    Loading n-Chain Data
    """

    chain_uniform_biased_dir = "./data/n-chain/uniform_detector_biased_noise/"

    ent_n3_chain_biased_regexes = [r"ghz_local_ry_n-3_.*"]
    arb_n3_chain_biased_regexes = [
        r"arb_arb_n-3_.*",
        r"max_ent_arb_n-3_.*",
        r"arb_local_rot_n-3_.*",
        r"ryrz_cnot_local_ry_n-3_.*",
        r"ghz_local_ry_n-3_.*",
    ]

    ent_n3_chain_uniform_biased_data = [
        src.analyze_data_one_param_scan(src.get_data_files(chain_uniform_biased_dir, regex))
        for regex in ent_n3_chain_biased_regexes
    ]

    ent_max_n3_chain_uniform_biased = [
        max(map(lambda opt_data: opt_data["max_scores"][i], ent_n3_chain_uniform_biased_data))
        for i in range(num_samples)
    ]

    arb_n3_chain_uniform_biased_data = [
        src.analyze_data_one_param_scan(src.get_data_files(chain_uniform_biased_dir, regex))
        for regex in arb_n3_chain_biased_regexes
    ]

    arb_max_n3_chain_uniform_biased = [
        max(map(lambda opt_data: opt_data["max_scores"][i], arb_n3_chain_uniform_biased_data))
        for i in range(num_samples)
    ]

    chain_single_biased_dir = "./data/n-chain/single_detector_biased_noise/"

    ent_n3_chain_single_biased_data = [
        src.analyze_data_one_param_scan(src.get_data_files(chain_single_biased_dir, regex))
        for regex in ent_n3_chain_biased_regexes
    ]

    ent_max_n3_chain_single_biased = [
        max(map(lambda opt_data: opt_data["max_scores"][i], ent_n3_chain_single_biased_data))
        for i in range(num_samples)
    ]

    arb_n3_chain_single_biased_data = [
        src.analyze_data_one_param_scan(src.get_data_files(chain_single_biased_dir, regex))
        for regex in arb_n3_chain_biased_regexes
    ]

    arb_max_n3_chain_single_biased = [
        max(map(lambda opt_data: opt_data["max_scores"][i], arb_n3_chain_single_biased_data))
        for i in range(num_samples)
    ]

    ent_n4_chain_biased_regexes = [r"ghz_local_ry_n-4_.*"]
    arb_n4_chain_biased_regexes = [
        r"arb_arb_n-4_.*",
        r"arb_local_rot_n-4_.*",
        r"max_ent_arb_n-4_.*",
        r"ryrz_cnot_local_ry_n-4_.*",
        r"ghz_local_ry_n-4_.*",
    ]

    ent_n4_chain_uniform_biased_data = [
        src.analyze_data_one_param_scan(src.get_data_files(chain_uniform_biased_dir, regex))
        for regex in ent_n4_chain_biased_regexes
    ]

    ent_max_n4_chain_uniform_biased = [
        max(map(lambda opt_data: opt_data["max_scores"][i], ent_n4_chain_uniform_biased_data))
        for i in range(num_samples)
    ]

    arb_n4_chain_uniform_biased_data = [
        src.analyze_data_one_param_scan(src.get_data_files(chain_uniform_biased_dir, regex))
        for regex in arb_n4_chain_biased_regexes
    ]

    arb_max_n4_chain_uniform_biased = [
        max(map(lambda opt_data: opt_data["max_scores"][i], arb_n4_chain_uniform_biased_data))
        for i in range(num_samples)
    ]

    ent_n4_chain_single_biased_data = [
        src.analyze_data_one_param_scan(src.get_data_files(chain_single_biased_dir, regex))
        for regex in ent_n4_chain_biased_regexes
    ]

    ent_max_n4_chain_single_biased = [
        max(map(lambda opt_data: opt_data["max_scores"][i], ent_n4_chain_single_biased_data))
        for i in range(num_samples)
    ]

    arb_n4_chain_single_biased_data = [
        src.analyze_data_one_param_scan(src.get_data_files(chain_single_biased_dir, regex))
        for regex in arb_n4_chain_biased_regexes
    ]

    arb_max_n4_chain_single_biased = [
        max(map(lambda opt_data: opt_data["max_scores"][i], arb_n4_chain_single_biased_data))
        for i in range(num_samples)
    ]

    """
    Loading n-Star Data
    """

    star_uniform_biased_dir = "./data/n-star/uniform_detector_biased_noise/"

    ent_n3_star_biased_regexes = [r"ghz_local_ry_n-3_.*"]
    arb_n3_star_biased_regexes = [
        r"arb_arb_n-3_.*",
        r"arb_local_rot_n-3_.*",
        r"max_ent_arb_n-3_.*",
        r"ryrz_cnot_local_ry_n-3_.*",
        r"ghz_local_ry_n-3_.*",
    ]

    ent_n3_star_uniform_biased_data = [
        src.analyze_data_one_param_scan(src.get_data_files(star_uniform_biased_dir, regex))
        for regex in ent_n3_star_biased_regexes
    ]

    ent_max_n3_star_uniform_biased = [
        max(map(lambda opt_data: opt_data["max_scores"][i], ent_n3_star_uniform_biased_data))
        for i in range(num_samples)
    ]

    arb_n3_star_uniform_biased_data = [
        src.analyze_data_one_param_scan(src.get_data_files(star_uniform_biased_dir, regex))
        for regex in arb_n3_star_biased_regexes
    ]

    arb_max_n3_star_uniform_biased = [
        max(map(lambda opt_data: opt_data["max_scores"][i], arb_n3_star_uniform_biased_data))
        for i in range(num_samples)
    ]

    star_single_biased_dir = "./data/n-star/single_detector_biased_noise/"

    ent_n3_star_single_biased_data = [
        src.analyze_data_one_param_scan(src.get_data_files(star_single_biased_dir, regex))
        for regex in ent_n3_star_biased_regexes
    ]

    ent_max_n3_star_single_biased = [
        max(map(lambda opt_data: opt_data["max_scores"][i], ent_n3_star_single_biased_data))
        for i in range(num_samples)
    ]

    arb_n3_star_single_biased_data = [
        src.analyze_data_one_param_scan(src.get_data_files(star_single_biased_dir, regex))
        for regex in arb_n3_star_biased_regexes
    ]

    arb_max_n3_star_single_biased = [
        max(map(lambda opt_data: opt_data["max_scores"][i], arb_n3_star_single_biased_data))
        for i in range(num_samples)
    ]

    ent_n4_star_biased_regexes = [r"ghz_local_ry_n-4_.*"]
    arb_n4_star_biased_regexes = [r"arb_local_rot_n-4_.*", r"ryrz_cnot_local_ry_n-4_.*"]

    ent_n4_star_uniform_biased_data = [
        src.analyze_data_one_param_scan(src.get_data_files(star_uniform_biased_dir, regex))
        for regex in ent_n4_star_biased_regexes
    ]

    ent_max_n4_star_uniform_biased = [
        max(map(lambda opt_data: opt_data["max_scores"][i], ent_n4_star_uniform_biased_data))
        for i in range(num_samples)
    ]

    arb_n4_star_uniform_biased_data = [
        src.analyze_data_one_param_scan(src.get_data_files(star_uniform_biased_dir, regex))
        for regex in arb_n4_star_biased_regexes
    ]

    arb_max_n4_star_uniform_biased = [
        max(map(lambda opt_data: opt_data["max_scores"][i], arb_n4_star_uniform_biased_data))
        for i in range(num_samples)
    ]

    arb_n4_star_single_biased_data = [
        src.analyze_data_one_param_scan(src.get_data_files(star_single_biased_dir, regex))
        for regex in arb_n4_star_biased_regexes
    ]

    arb_max_n4_star_single_biased = [
        max(map(lambda opt_data: opt_data["max_scores"][i], arb_n4_star_single_biased_data))
        for i in range(num_samples)
    ]

    ent_n4_star_single_biased_data = [
        src.analyze_data_one_param_scan(src.get_data_files(star_single_biased_dir, regex))
        for regex in ent_n4_star_biased_regexes
    ]

    ent_max_n4_star_single_biased = [
        max(map(lambda opt_data: opt_data["max_scores"][i], ent_n4_star_single_biased_data))
        for i in range(num_samples)
    ]

    """
    Plotting Data
    """

    src.plot_unital_single_and_uniform_max_scores_data(
        fig_title="Detector Biased Noise Robustness",
        ax_titles=["Single Detector Noise", "Uniform Detector Noise"],
        noise_params=arb_chsh_single_biased_data[0]["noise_params"],
        quantum_bound=np.sqrt(2),
        classical_bound=1,
        single_max_scores=[
            arb_max_chsh_single_biased,
            arb_max_bilocal_single_biased,
            arb_max_n3_chain_single_biased,
            arb_max_n4_chain_single_biased,
            arb_max_n3_star_single_biased,
            arb_max_n4_star_single_biased,
        ],
        single_theoretical_scores=[
            ent_max_chsh_single_biased,
            ent_max_bilocal_single_biased,
            ent_max_n3_chain_single_biased,
            ent_max_n4_chain_single_biased,
            ent_max_n3_star_single_biased,
            ent_max_n4_star_single_biased,
        ],
        single_match_scores=[],
        uniform_max_scores=[
            arb_max_chsh_uniform_biased,
            arb_max_bilocal_uniform_biased,
            arb_max_n3_chain_uniform_biased,
            arb_max_n4_chain_uniform_biased,
            arb_max_n3_star_uniform_biased,
            arb_max_n4_star_uniform_biased,
        ],
        uniform_theoretical_scores=[
            ent_max_chsh_uniform_biased,
            ent_max_bilocal_uniform_biased,
            ent_max_n3_chain_uniform_biased,
            ent_max_n4_chain_uniform_biased,
            ent_max_n3_star_uniform_biased,
            ent_max_n4_star_uniform_biased,
        ],
        uniform_match_scores=[],
        data_labels=[
            "CHSH",
            "Bilocal",
            "3-Local Chain",
            "4-Local Chain",
            "3-Local Star",
            "4-Local Star",
        ],
        plot_dir="./data/plots/detector_biased_noise_robustness/",
        legend_labels=["Arbitrary VQO", "Max Entangled VQO"],
        ncol_legend=4,
        bottom_padding=0.3,
        fig_height=5,
    )

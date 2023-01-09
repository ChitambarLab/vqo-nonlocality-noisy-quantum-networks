import qnetvo as qnet
from context import src

import matplotlib.pyplot as plt
from pennylane import numpy as np
from datetime import datetime

if __name__ == "__main__":

    prefix = ""

    """
	Loading CHSH Data
	"""

    chsh_dir = prefix + "data/chsh/ibm_hardware_simple_parameter_shift_opt/"
    chsh_files = src.get_data_files(chsh_dir, r".*")
    chsh_opt_dicts = [qnet.read_optimization_json(file) for file in chsh_files]

    print("num chsh optimizations : ", len(chsh_files))

    chsh_ansatz = qnet.NetworkAnsatz(
        [qnet.PrepareNode(1, [0, 1], qnet.ghz_state, 0),],
        [
            qnet.MeasureNode(2, 2, [0], qnet.local_RY, 1),
            qnet.MeasureNode(2, 2, [1], qnet.local_RY, 1),
        ],
    )
    chsh_cost = qnet.chsh_inequality_cost(chsh_ansatz)

    num_chsh_steps = 17

    chsh_data = src.opt_dicts_mean_stderr(chsh_opt_dicts, num_chsh_steps)
    chsh_ideal_scores = [-(chsh_cost(chsh_data["opt_settings"][i])) for i in range(num_chsh_steps)]

    """
	Loading Bilocal Data
	"""

    bilocal_dir = prefix + "data/bilocal/ibm_hardware_simple_parameter_shift_opt/"
    bilocal_files = src.get_data_files(bilocal_dir, r".*")
    bilocal_opt_dicts = [qnet.read_optimization_json(file) for file in bilocal_files]

    print("num bilocal optimizations : ", len(bilocal_files))

    bilocal_ansatz = qnet.NetworkAnsatz(
        [
            qnet.PrepareNode(1, [0, 1], qnet.ghz_state, 0),
            qnet.PrepareNode(1, [2, 3], qnet.ghz_state, 0),
        ],
        [
            qnet.MeasureNode(2, 2, [0], qnet.local_RY, 1),
            qnet.MeasureNode(2, 2, [1, 2], qnet.local_RY, 2),
            qnet.MeasureNode(2, 2, [3], qnet.local_RY, 1),
        ],
    )
    bilocal_cost = qnet.nlocal_chain_cost_22(bilocal_ansatz)

    num_bilocal_steps = 10

    bilocal_data = src.opt_dicts_mean_stderr(bilocal_opt_dicts, num_bilocal_steps)
    bilocal_ideal_scores = [
        -(bilocal_cost(bilocal_data["opt_settings"][i])) for i in range(num_bilocal_steps)
    ]

    """
	Loading Trilocal Chain Data
	"""

    chain_dir = prefix + "data/n-chain/ibm_hardware_simple_trilocal_parameter_shift_opt/"
    chain_files = src.get_data_files(chain_dir, r".*")
    chain_opt_dicts = [qnet.read_optimization_json(file) for file in chain_files]

    print("num chain optimizations : ", len(chain_files))

    chain_ansatz = qnet.NetworkAnsatz(
        [
            qnet.PrepareNode(1, [0, 1], qnet.ghz_state, 0),
            qnet.PrepareNode(1, [2, 3], qnet.ghz_state, 0),
            qnet.PrepareNode(1, [4, 5], qnet.ghz_state, 0),
        ],
        [
            qnet.MeasureNode(2, 2, [0], qnet.local_RY, 1),
            qnet.MeasureNode(2, 2, [1, 2], qnet.local_RY, 2),
            qnet.MeasureNode(2, 2, [3, 4], qnet.local_RY, 2),
            qnet.MeasureNode(2, 2, [5], qnet.local_RY, 1),
        ],
    )
    chain_cost = qnet.nlocal_chain_cost_22(chain_ansatz)

    num_chain_steps = 11

    chain_data = src.opt_dicts_mean_stderr(chain_opt_dicts, num_chain_steps)
    chain_ideal_scores = [
        -(chain_cost(chain_data["opt_settings"][i])) for i in range(num_chain_steps)
    ]

    """
	Loading Trilocal Star Data
	"""

    star_dir = prefix + "data/n-star/ibm_hardware_simple_trilocal_parameter_shift_opt/"
    star_files = src.get_data_files(star_dir, r".*")
    star_opt_dicts = [qnet.read_optimization_json(file) for file in star_files]

    print("num star optimizations : ", len(star_files))

    star_ansatz = qnet.NetworkAnsatz(
        [
            qnet.PrepareNode(1, [0, 3], qnet.ghz_state, 0),
            qnet.PrepareNode(1, [1, 4], qnet.ghz_state, 0),
            qnet.PrepareNode(1, [2, 5], qnet.ghz_state, 0),
        ],
        [
            qnet.MeasureNode(2, 2, [0], qnet.local_RY, 1),
            qnet.MeasureNode(2, 2, [1], qnet.local_RY, 1),
            qnet.MeasureNode(2, 2, [2], qnet.local_RY, 1),
            qnet.MeasureNode(2, 2, [3, 4, 5], qnet.local_RY, 3),
        ],
    )
    star_cost = qnet.nlocal_star_22_cost_fn(star_ansatz)

    num_star_steps = 11

    star_data = src.opt_dicts_mean_stderr(star_opt_dicts, num_star_steps)
    star_ideal_scores = [-(star_cost(star_data["opt_settings"][i])) for i in range(num_star_steps)]

    """
	Plotting Data
	"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 8))

    fig.suptitle(r"VQO of Non-$n$-Locality on IBM Quantum Computers", size=24, fontweight="bold")

    axes = [ax1, ax2, ax3, ax4]
    titles = [
        "CHSH Network",
        "Bilocal Network",
        "\nTrilocal Chain Network",
        "\nTrilocal Star Network",
    ]
    ylabels = [
        r"Bell Score ($S_{\mathrm{CHSH}}$)",
        r"Bell Score ($S_{\mathrm{Bilocal}}$)",
        r"Bell Score ($S_{3\mathrm{-Chain}}$)",
        r"Bell Score ($S_{3\mathrm{-Star}}$)",
    ]
    num_steps_list = [num_chsh_steps, num_bilocal_steps, num_chain_steps, num_star_steps]
    quantum_bounds = [2 * np.sqrt(2), np.sqrt(2), np.sqrt(2), np.sqrt(2)]
    classical_bounds = [2, 1, 1, 1]
    ideal_data_sets = [
        chsh_ideal_scores,
        bilocal_ideal_scores,
        chain_ideal_scores,
        star_ideal_scores,
    ]
    data_sets = [chsh_data, bilocal_data, chain_data, star_data]
    for i in range(4):
        ax = axes[i]
        num_steps = num_steps_list[i]

        ax.plot(
            range(num_steps),
            [quantum_bounds[i]] * num_steps,
            "-",
            linewidth=3,
            label="Quantum\nBound",
            color="C0",
        )
        ax.plot(
            range(num_steps),
            [classical_bounds[i]] * num_steps,
            "--",
            linewidth=3,
            label="Classical\nBound",
            color="C1",
        )
        ax.plot(
            range(num_steps),
            [data_sets[i]["max_theoretical_score"]] * num_steps,
            linestyle=(0, (3, 2, 1, 2, 1, 2)),
            linewidth=3,
            label="Theoretical\nMax Score",
            color="C6",
        )
        ax.plot(
            range(num_steps),
            [data_sets[i]["mean_theoretical_score"]] * num_steps,
            "-.",
            label="Theoretical\nMean Score",
            linewidth=3,
            color="C5",
        )
        ax.plot(
            range(num_steps),
            ideal_data_sets[i],
            ":d",
            markersize=8,
            linewidth=3,
            label="Noiseless\nOptimized\nMax Score",
            color="C2",
        )
        ax.plot(
            range(num_steps),
            data_sets[i]["max_scores"],
            ":o",
            markersize=7,
            linewidth=3,
            label="Optimized\nMax Score",
            color="C3",
        )
        ax.errorbar(
            range(num_steps),
            data_sets[i]["mean_scores"],
            data_sets[i]["stderr_scores"],
            linestyle=":",
            linewidth=3,
            label="Optimized\nMean Score",
            color="C4",
        )

        ax.set_title(titles[i], size=20)
        if i >= 2:
            ax.set_xlabel("Gradient Descent Step", size=18)

        ax.set_ylabel(ylabels[i], size=18)

        if i == 0:
            plt.figlegend(
                ncol=1,
                loc="center right",
                fontsize=18,
                bbox_to_anchor=(0, 0, 1, 1),
                labelspacing=1.1,
            )  # , bbox_to_anchor=(0, 0,1,1,))

    fig.subplots_adjust(left=0.05, right=0.82)

    datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
    filename = "simple_ansatzes_" + datetime_ext
    plot_dir = "data/plots/vqo_ibm_hardware/"


    # """
    # semilog plot of data
    # """

    # fig, (ax2) = plt.subplots(1, 1, figsize=(25, 18))

    # fig.suptitle("VQO of Star Network Nonlocality on\nNoisy IBM Quantum Computers", size=80, fontweight="bold")

    # num_steps = num_star_steps
    # quantum_bound = np.sqrt(2)
    # classical_bound = 1
    # ideal_data_set = star_ideal_scores
    # data_set = star_data

    # ax2.semilogy(
    #     range(num_steps),
    #     np.sqrt(2) - data_set["mean_scores"],
    #     linestyle=":",
    #     marker="s",
    #     markersize=32,
    #     linewidth=8,
    #     label="VQO Mean",
    #     color="C4",
    # )
    # ax2.semilogy(
    #     range(num_steps),
    #     np.sqrt(2) - data_set["max_scores"],
    #     ":o",
    #     markersize=32,
    #     linewidth=10,
    #     label="VQO Max",
    #     color="C3",
    # )
    # ax2.semilogy(
    #     range(num_steps),
    #     np.sqrt(2) - ideal_data_set,
    #     ":d",
    #     markersize=32,
    #     linewidth=10,
    #     label="Noiseless VQO Max",
    #     color="C2",
    # )

    # ax2.semilogy(
    #     range(num_steps),
    #     np.sqrt(2) - [classical_bound] * num_steps,
    #     "--",
    #     linewidth=10,
    #     label="Classical Bound",
    #     color="C0",
    # )
    # ax2.semilogy(
    #     range(num_steps),
    #     np.sqrt(2) - [data_set["mean_theoretical_score"]] * num_steps,
    #     "-.",
    #     label="Noisy Quantum Bound",
    #     linewidth=10,
    #     color="C1",
    # )
    # ax2.legend(
    #     ncol=1,
    #     fontsize=56,
    #     loc="lower left",
    # )
    # ax2.tick_params(axis='both', which='major', labelsize=56)

    # ax2.set_xlabel("Gradient Descent Step", fontsize=72)

    # ax2.set_ylabel("Distance to Max Bell Violation", fontsize=72)

    # plt.grid(linewidth=4)
    # fig.tight_layout(pad=2)


    # datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
    # filename = "simple_ansatzes_" + datetime_ext
    # plot_dir = "data/plots/vqo_ibm_hardware/"

    # plt.savefig(plot_dir + filename)

import qnetvo as qnet
from context import src

from pennylane import numpy as np
import pennylane as qml
import matplotlib.pyplot as plt
from datetime import datetime


"""
This script aggregates and plots data for qubit phase damping noise.
"""

# @qml.qnode(qml.device("default.mixed", wires=[0,1]))
# def bell_state_uniform_noise(gamma):
#     qml.Hadamard(wires=[0])
#     qml.CNOT(wires=[0,1])

#     qml.AmplitudeDamping(gamma, wires=[0])
#     qml.AmplitudeDamping(gamma, wires=[1])

#     return qml.state()

# @qml.qnode(qml.device("default.mixed", wires=[0,1]))
# def bell_state_single_noise(gamma):
#     qml.Hadamard(wires=[0])
#     qml.CNOT(wires=[0,1])

#     qml.AmplitudeDamping(gamma, wires=[0])

#     return qml.state()

if __name__ == "__main__":
    num_samples = 101
    noise_params = np.arange(0, 1.001, 0.01)
    noise_params_inset = np.arange(0.25, 0.35001, 0.001)

    # bell_state_uniform_noise_states = [
    #     bell_state_uniform_noise(gamma)
    #     for gamma in np.arange(0, 1.01, 0.05)
    # ]
    # bell_state_single_noise_states = [
    #     bell_state_single_noise(gamma)
    #     for gamma in np.arange(0, 1.01, 0.05)
    # ]

    # bell_state = np.array([[1,0,0,1],[0,0,0,0],[0,0,0,0],[1,0,0,1]])/2

    """
    Loading CHSH Data
    """
    chsh_uniform_ad_dir = "./data/chsh/uniform_qubit_amplitude_damping_high_res/"
    chsh_uniform_ad_inset_dir = "./data/chsh/uniform_qubit_amplitude_damping_high_res_inset/"

    ghz_chsh_ad_regexes = [r"ghz_ry_.*", r"ghz_local_rot_.*"]
    ryrz_cnot_chsh_ad_regexes = [
        r"ryrz_cnot_ry_.*",
        r"ryrz_cnot_local_rot_.*",
        r"ghz_ry_.*",
        r"ghz_local_rot_.*",
    ]

    ghz_chsh_ad_inset_regexes = [r"ghz_local_rot_.*"]
    ryrz_cnot_chsh_ad_inset_regexes = [r"ryrz_cnot_local_rot_.*", r"ghz_local_rot_.*"]

    ghz_chsh_uniform_ad_data = [
        src.analyze_data_one_param_scan(src.get_data_files(chsh_uniform_ad_dir, regex))
        for regex in ghz_chsh_ad_regexes
    ]
    ghz_max_chsh_uniform_ad = [
        max(map(lambda opt_data: opt_data["max_scores"][i], ghz_chsh_uniform_ad_data))
        for i in range(num_samples)
    ]

    ghz_chsh_uniform_ad_inset_data = [
        src.analyze_data_one_param_scan(src.get_data_files(chsh_uniform_ad_inset_dir, regex))
        for regex in ghz_chsh_ad_inset_regexes
    ]

    ghz_max_chsh_uniform_ad_inset = [
        max(map(lambda opt_data: opt_data["max_scores"][i], ghz_chsh_uniform_ad_inset_data))
        for i in range(num_samples)
    ]

    ryrz_cnot_chsh_uniform_ad_data = [
        src.analyze_data_one_param_scan(src.get_data_files(chsh_uniform_ad_dir, regex))
        for regex in ryrz_cnot_chsh_ad_regexes
    ]
    ryrz_cnot_max_chsh_uniform_ad = [
        max(map(lambda opt_data: opt_data["max_scores"][i], ryrz_cnot_chsh_uniform_ad_data))
        for i in range(num_samples)
    ]
    ryrz_cnot_chsh_uniform_ad_inset_data = [
        src.analyze_data_one_param_scan(src.get_data_files(chsh_uniform_ad_inset_dir, regex))
        for regex in ryrz_cnot_chsh_ad_inset_regexes
    ]
    ryrz_cnot_max_chsh_uniform_ad_inset = [
        max(map(lambda opt_data: opt_data["max_scores"][i], ryrz_cnot_chsh_uniform_ad_inset_data))
        for i in range(num_samples)
    ]

    """
    Theoretical Scores
    """

    # def max_entangled_score(gamma):
    #     return 2 * np.sqrt(2 * (1 - gamma) ** 2)

    def max_entangled_score(gamma):
        return max(
            2 * np.sqrt(2 * (1 - gamma) ** 2),
            2 * np.sqrt((1 - gamma) ** 2 + (gamma ** 2 + (1 - gamma) ** 2) ** 2)
        )

    def max_entangled_score2(gamma):
        return 2 * np.sqrt((1 - gamma) ** 2 + (gamma ** 2 + (1 - gamma) ** 2) ** 2)

    def lambda_star_score(gamma):
        lambda_star = 0
        if gamma >= 0.5:
            lambda_star = 1
        else:
            lambda_star = (
                1
                - (gamma ** 2 + (1 - gamma) ** 2)
                * (2 * gamma * (1 - gamma))
                / ((2 * gamma * (1 - gamma)) ** 2 - (1 - gamma) ** 2)
            ) / 2

        return min(1, lambda_star)

    def _nonmax_entangled_score(gamma, lambda_star):
        a = 4 * (1 - gamma) ** 2 * lambda_star * (1 - lambda_star)
        b = (gamma ** 2 + (1 - gamma) ** 2 + (2 * lambda_star - 1) * (2 * gamma * (1 - gamma))) ** 2
        return 2 * np.sqrt(a + b)

    def nonmax_entangled_score(gamma):
        lambda_star = lambda_star_score(gamma)
        # return _nonmax_entangled_score(gamma, lambda_star)
        return max(max_entangled_score(gamma), _nonmax_entangled_score(gamma, lambda_star))


    max_entangled_theory = [max_entangled_score(gamma) for gamma in noise_params]
    # max_entangled_theory2 = [max_entangled_score2(gamma) for gamma in noise_params]

    nonmax_entangled_theory = [nonmax_entangled_score(gamma) for gamma in noise_params]

    max_entangled_theory_inset = [max_entangled_score(gamma) for gamma in noise_params_inset]
    nonmax_entangled_theory_inset = [nonmax_entangled_score(gamma) for gamma in noise_params_inset]

    # crossover of max entangled and nonmax entangled optimality
    crit_gamma = (
        4
        + (3 * np.sqrt(114) - 32) ** (1 / 3) / (2 ** (2 / 3))
        - 1 / (6 * np.sqrt(114) - 64) ** (1 / 3)
    ) / 6

    """
    Plotting Data
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 7))

    fig.suptitle(
        "Maximal CHSH Scores in the Presence of\nUniform Qubit Amplitude Damping Noise",
        size=24,
        fontweight="bold",
    )

    ylabel = r"$S^\star_{\mathrm{CHSH}}(\tilde{\rho}_{\lambda,\gamma})$"

    (qbound_plot,) = ax1.plot(noise_params, [np.sqrt(2) * 2] * num_samples, "-", linewidth=2, label="Quantum Bound")
    (cbound_plot,) = ax1.plot(noise_params, [2] * num_samples, "-.", linewidth=2, label="Classical Bound")
    (max_ent_theory_plot,) = ax1.plot(
        noise_params,
        max_entangled_theory,
        "-",
        color="C2",
        linewidth=2,
        label=r"$S^\star_{\mathrm{CHSH}}(\tilde{\rho}_{\frac{1}{2},\gamma})$",
    )
    # ax1.plot(
    #     noise_params,
    #     max_entangled_theory2,
    #     "--",
    #     color="C2",
    #     linewidth=2,
    #     label="Max Entangled " + r"($\sqrt{\tau^2_x + \tau^2_z}$)",
    # )
    (nonmax_ent_theory_plot,) = ax1.plot(
        noise_params,
        nonmax_entangled_theory,
        "-",
        color="C3",
        linewidth=2,
        label=r"$S^\star_{\mathrm{CHSH}}(\tilde{\rho}_{\lambda^\star,\gamma})$",
    )
    (gamma_c_nonmax_ent_plot,) = ax1.plot(
        [crit_gamma, crit_gamma],
        [1, 2 * np.sqrt(2)],
        ":",
        linewidth=2,
        color="C3",
        label=r"$\gamma_{c,\lambda^\star}$",
    )
    (gamma_0_max_ent_plot,) = ax1.plot(
        [1 - 1 / np.sqrt(2), 1 - 1 / np.sqrt(2)],
        [1, 2 * np.sqrt(2)],
        "-.",
        linewidth=2,
        color="C2",
        label=r"$\gamma_{0,\frac{1}{2}}$",
    )
    (gamma_0_non_max_ent_plot,) = ax1.plot(
        [1 / 3, 1 / 3],
        [1, 2 * np.sqrt(2)],
        "-.",
        linewidth=2,
        color="C3",
        label=r"$\gamma_{0,\lambda^\star}$",
    )
    (gamma_c_max_ent_plot,) = ax1.plot(
        [1 / 2, 1 / 2],
        [1, 2 * np.sqrt(2)],
        ":",
        linewidth=2,
        color="C2",
        label=r"$\gamma_{c,\frac{1}{2}}$",
    )
    (vqo_max_ent_plot,) = ax1.plot(
        noise_params,
        ghz_max_chsh_uniform_ad,
        marker="d",
        color="C2",
        markersize=6,
        markerfacecolor="None",
        linewidth=2,
        linestyle="None",
        label="VQO Max Entangled",
        alpha=0.6,
    )
    (vqo_nonmax_ent_plot,) = ax1.plot(
        noise_params,
        ryrz_cnot_max_chsh_uniform_ad,
        color="C3",
        marker="o",
        markersize=6,
        markerfacecolor="None",
        linewidth=2,
        linestyle="None",
        label="VQO Nonmax Entangled",
        alpha=0.6,
    )

    ax1.set_xlabel(r"Noise Parameter ($\gamma$)", size=18)
    ax1.set_ylabel(ylabel, size=18)

    plt.figlegend(
        [
            qbound_plot,
            cbound_plot,
            (max_ent_theory_plot, vqo_max_ent_plot),
            (nonmax_ent_theory_plot, vqo_nonmax_ent_plot),
            gamma_c_max_ent_plot,
            gamma_c_nonmax_ent_plot,
            gamma_0_max_ent_plot,
            gamma_0_non_max_ent_plot,
        ],
        [
            "Quantum Bound", "Classical Bound",
            r"Max CHSH Score $S^\star_{\mathrm{CHSH}}(\tilde{\rho}_{\frac{1}{2},\gamma})$",
            r"Max CHSH Score $S^\star_{\mathrm{CHSH}}(\tilde{\rho}_{\lambda^\star,\gamma})$",
            r"Crossover Parameter $\gamma_{c,\frac{1}{2}}$",
            r"Crossover Parameter $\gamma_{c,\lambda^\star}$",
            r"Nonlocality Broken $\gamma_{0,\frac{1}{2}}$",
            r"Nonlocality Broken $\gamma_{0,\lambda^\star}$",
        ],
        ncol=2, loc="lower center", fontsize=18, bbox_to_anchor=(0, 0.05, 1, 1,)
    )

    ax2.plot(
        noise_params_inset,
        [2] * num_samples,
        "-.",
        color="C1",
        linewidth=2,
        label="Classical Bound",
    )
    ax2.plot(
        noise_params_inset,
        max_entangled_theory_inset,
        "-",
        color="C2",
        linewidth=2,
        label=r"Bell State Preparation ($|\Phi^+\rangle$)",
    )
    ax2.plot(
        noise_params_inset,
        nonmax_entangled_theory_inset,
        "-",
        color="C3",
        linewidth=2,
        label="Nonmaximally Entangled\nState Preparation",
    )
    ax2.plot(
        [1 - 1 / np.sqrt(2), 1 - 1 / np.sqrt(2)],
        [1.84, 2.12],
        "-.",
        color="C2",
        label="Max Entangled\nNonlocality Broken",
        linewidth=2,
    )
    ax2.plot(
        [1 / 3, 1 / 3],
        [1.84, 2.12],
        "-.",
        color="C3",
        label="Nonmax Entanglement\nNonlocality Broken",
        linewidth=2,
    )
    ax2.plot(
        [crit_gamma, crit_gamma],
        [1.84, 2.12],
        ":",
        color="C3",
        label="Nonmax Entanglement\nNonlocality Broken",
        linewidth=2,
    )
    ax2.plot(
        noise_params_inset,
        ghz_max_chsh_uniform_ad_inset,
        color="C2",
        marker="d",
        markersize=6,
        markerfacecolor="None",
        label=r"Bell State Preparation ($|\Phi^+\rangle$)",
        alpha=0.6,
    )
    ax2.plot(
        noise_params_inset,
        ryrz_cnot_max_chsh_uniform_ad_inset,
        color="C3",
        marker="o",
        markersize=6,
        markerfacecolor="None",
        linewidth=1,
        label="Nonmaximally Entangled\nState Preparation",
        alpha=0.6,
    )

    ax2.set_xlabel(r"Noise Parameter ($\gamma$)", size=18)
    ax2.set_xticks([0.25, 0.27, 0.29, 0.31, 0.33, 0.35])

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.45)

    datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
    filename = "chsh_uniform_qubit_amplitude_damping_high_res_" + datetime_ext
    plot_dir = "data/plots/chsh_uniform_amplitude_damping_high_res/"

    plt.savefig(plot_dir + filename)

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
    noise_params_inset = np.arange(0.25,0.35001,0.001)



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
    ryrz_cnot_chsh_ad_regexes = [ r"ryrz_cnot_ry_.*", r"ryrz_cnot_local_rot_.*", r"ghz_ry_.*", r"ghz_local_rot_.*"]

    ghz_chsh_ad_inset_regexes = [r"ghz_local_rot_.*"]
    ryrz_cnot_chsh_ad_inset_regexes = [r"ryrz_cnot_local_rot_.*", r"ghz_local_rot_.*"]

    ghz_chsh_uniform_ad_data = [
        src.analyze_data_one_param_scan(
            src.get_data_files(chsh_uniform_ad_dir, regex)
        )
        for regex in ghz_chsh_ad_regexes
    ]
    ghz_max_chsh_uniform_ad = [
        max(map(lambda opt_data: opt_data["max_scores"][i], ghz_chsh_uniform_ad_data))
        for i in range(num_samples)
    ]

    ghz_chsh_uniform_ad_inset_data = [
        src.analyze_data_one_param_scan(
            src.get_data_files(chsh_uniform_ad_inset_dir, regex)
        )
        for regex in ghz_chsh_ad_inset_regexes
    ]

    ghz_max_chsh_uniform_ad_inset = [
        max(map(lambda opt_data: opt_data["max_scores"][i], ghz_chsh_uniform_ad_inset_data))
        for i in range(num_samples)
    ]

    ryrz_cnot_chsh_uniform_ad_data = [
        src.analyze_data_one_param_scan(
            src.get_data_files(chsh_uniform_ad_dir, regex)
        )
        for regex in ryrz_cnot_chsh_ad_regexes
    ]
    ryrz_cnot_max_chsh_uniform_ad = [
        max(map(lambda opt_data: opt_data["max_scores"][i], ryrz_cnot_chsh_uniform_ad_data))
        for i in range(num_samples)
    ]
    ryrz_cnot_chsh_uniform_ad_inset_data = [
        src.analyze_data_one_param_scan(
            src.get_data_files(chsh_uniform_ad_inset_dir, regex)
        )
        for regex in ryrz_cnot_chsh_ad_inset_regexes
    ]
    ryrz_cnot_max_chsh_uniform_ad_inset = [
        max(map(lambda opt_data: opt_data["max_scores"][i], ryrz_cnot_chsh_uniform_ad_inset_data))
        for i in range(num_samples)
    ]

    """
    Plotting Data
    """
    fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(12,6)) 

    fig.suptitle("CHSH Violations in the Presence of\nUniform Qubit Amplitude Damping Noise", size=24, fontweight="bold")

    ylabel = r"Bell Score ($S_{\mathrm{CHSH}}$)"


    ax1.plot(noise_params, [np.sqrt(2)*2]*num_samples, "-", linewidth=2, label="Quantum Bound")
    ax1.plot(noise_params, [2]*num_samples, "-.", linewidth=2, label="Classical Bound")
    ax1.plot(
        noise_params,
        ghz_max_chsh_uniform_ad,
        "-",
        marker="d",
        markersize=4,
        markerfacecolor="None",
        linewidth=1,
        label=r"Bell State Preparation ($|\Phi^+\rangle$)",
        alpha=0.8
    )
    ax1.plot(
        noise_params,
        ryrz_cnot_max_chsh_uniform_ad,
        "-",
        marker="o",
        markersize=4,
        markerfacecolor="None",
        linewidth=1,
        label="Nonmaximally Entangled\nState Preparation",
        alpha=0.8
    )
    ax1.plot([1-1/np.sqrt(2)], [2], marker="*", markersize=6, linestyle="None", color="b", label="CHSH Breaking Threshold")

    ax1.set_xlabel(r"Noise Parameter ($\gamma$)", size=18)
    ax1.set_ylabel(ylabel, size=18)

    plt.figlegend(ncol=2, loc="lower center", fontsize=16, bbox_to_anchor=(0,-0.01,1,1,))


    ax2.plot(noise_params_inset, [2]*num_samples, "-.", color="C1", linewidth=2, label="Classical Bound")
    ax2.plot(
        noise_params_inset,
        ghz_max_chsh_uniform_ad_inset,
        "-",
        color="C2",
        marker="d",
        markersize=4,
        markerfacecolor="None",
        linewidth=1,
        label=r"Bell State Preparation ($|\Phi^+\rangle$)",
        alpha=0.8
    )
    ax2.plot(
        noise_params_inset,
        ryrz_cnot_max_chsh_uniform_ad_inset,
        "-",
        color="C3",
        marker="o",
        markersize=4,
        markerfacecolor="None",
        linewidth=1,
        label="Nonmaximally Entangled\nState Preparation",
        alpha=0.8
    )
    ax2.plot([1-1/np.sqrt(2)], [2], marker="*", markersize=6, color="b")

    ax2.set_xlabel(r"Noise Parameter ($\gamma$)", size=18)
    ax2.set_xticks([0.25,0.27,0.29,0.31,0.33,0.35])

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.35)

    datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
    filename =  "chsh_uniform_qubit_amplitude_damping_high_res_" + datetime_ext
    plot_dir = "data/plots/chsh_uniform_amplitude_damping_high_res/"

    plt.savefig(plot_dir + filename)


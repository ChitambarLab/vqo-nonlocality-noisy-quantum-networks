import qnetvo as qnet
from context import src

from pennylane import numpy as np
import pennylane as qml

"""
This script aggregates and plots data for qubit phase damping noise.
"""


@qml.qnode(qml.device("default.mixed", wires=[0, 1]))
def bell_state_uniform_noise(gamma):
    qml.Hadamard(wires=[0])
    qml.CNOT(wires=[0, 1])

    qml.AmplitudeDamping(gamma, wires=[0])
    qml.AmplitudeDamping(gamma, wires=[1])

    return qml.state()


@qml.qnode(qml.device("default.mixed", wires=[0, 1]))
def bell_state_single_noise(gamma):
    qml.Hadamard(wires=[0])
    qml.CNOT(wires=[0, 1])

    qml.AmplitudeDamping(gamma, wires=[0])

    return qml.state()


def uniform_max_entangled_theoretical_star_score1(gamma):
    return np.sqrt(2 * (1 - gamma) ** 2)


def uniform_max_entangled_theoretical_star_score2(gamma):
    return np.sqrt((1 - gamma) ** 2 + (gamma ** 2 + (1 - gamma) ** 2) ** 2)


def uniform_max_entangled_theoretical_star_score(gamma):
    return max(
        uniform_max_entangled_theoretical_star_score1(gamma),
        uniform_max_entangled_theoretical_star_score2(gamma),
    )


def uniform_max_entangled_theoretical_chain_score1(gamma, n):
    return np.sqrt(2 * (1 - gamma) ** 2) * np.sqrt(1 - gamma) ** (n - 2)


def uniform_max_entangled_theoretical_chain_score2(gamma, n):
    return np.sqrt((1 - gamma) ** 2 + (gamma ** 2 + (1 - gamma) ** 2) ** 2) * np.sqrt(
        gamma ** 2 + (1 - gamma) ** 2
    ) ** (n - 2)


def uniform_max_entangled_theoretical_chain_score(gamma, n):
    return max(
        uniform_max_entangled_theoretical_chain_score1(gamma, n),
        uniform_max_entangled_theoretical_chain_score2(gamma, n),
    )


def single_max_entangled_theoretical_star_score(gamma, n):
    return np.power(np.sqrt(2) ** (n - 1) * np.sqrt(2 * (1 - gamma)), 1 / n)


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
    return np.sqrt(a + b)


def uniform_nonmax_entangled_theoretical_star_score(gamma):
    lambda_star = lambda_star_score(gamma)
    return max(
        uniform_max_entangled_theoretical_star_score1(gamma),
        _nonmax_entangled_score(gamma, lambda_star),
    )


def single_nonmax_entangled_theoretical_star_score(gamma, n):
    return max(
        np.power(np.sqrt(2), (n - 1) / n),
        np.power(np.sqrt(2), (n - 1) / n) * np.power(np.sqrt(2 * (1 - gamma)), 1 / n),
    )


gamma_range = np.arange(0, 1.01, 0.05)

if __name__ == "__main__":
    num_samples = 21

    bell_state_uniform_noise_states = [
        bell_state_uniform_noise(gamma) for gamma in np.arange(0, 1.01, 0.05)
    ]
    bell_state_single_noise_states = [
        bell_state_single_noise(gamma) for gamma in np.arange(0, 1.01, 0.05)
    ]

    bell_state = np.array([[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]]) / 2

    """
    Loading CHSH Data
    """
    chsh_uniform_ad_dir = "./data/chsh/uniform_qubit_amplitude_damping/"

    ent_chsh_ad_regexes = [r"max_ent_.*"]
    arb_single_chsh_ad_regexes = [r"arb_.*", r"ryrz_cnot_local_ry_.*", r"max_ent_.*"]
    arb_uniform_chsh_ad_regexes = [
        r"arb_.*",
        r"ryrz_cnot_ry_.*",
        r"max_ent_.*",
        r"ryrz_cnot_local_rot_.*",
    ]

    ent_chsh_uniform_ad_data = [
        src.analyze_data_one_param_scan(src.get_data_files(chsh_uniform_ad_dir, regex))
        for regex in ent_chsh_ad_regexes
    ]
    ent_max_chsh_uniform_ad = [
        max(map(lambda opt_data: opt_data["max_scores"][i], ent_chsh_uniform_ad_data)) / 2
        for i in range(num_samples)
    ]

    arb_chsh_uniform_ad_data = [
        src.analyze_data_one_param_scan(src.get_data_files(chsh_uniform_ad_dir, regex))
        for regex in arb_uniform_chsh_ad_regexes
    ]
    arb_max_chsh_uniform_ad = [
        max(map(lambda opt_data: opt_data["max_scores"][i], arb_chsh_uniform_ad_data)) / 2
        for i in range(num_samples)
    ]

    chsh_single_ad_dir = "./data/chsh/single_qubit_amplitude_damping/"

    ent_chsh_single_ad_data = [
        src.analyze_data_one_param_scan(src.get_data_files(chsh_single_ad_dir, regex))
        for regex in ent_chsh_ad_regexes
    ]
    ent_max_chsh_single_ad = [
        max(map(lambda opt_data: opt_data["max_scores"][i], ent_chsh_single_ad_data)) / 2
        for i in range(num_samples)
    ]

    arb_chsh_single_ad_data = [
        src.analyze_data_one_param_scan(src.get_data_files(chsh_single_ad_dir, regex))
        for regex in arb_single_chsh_ad_regexes
    ]
    arb_max_chsh_single_ad = [
        max(map(lambda opt_data: opt_data["max_scores"][i], arb_chsh_single_ad_data)) / 2
        for i in range(num_samples)
    ]

    # theoretical_bell_state_uniform_chsh = [
    #     src.chsh_max_violation(state) / 2 for state in bell_state_uniform_noise_states
    # ]

    # theoretical_bell_state_single_chsh = [
    #     src.chsh_max_violation(state) / 2 for state in bell_state_single_noise_states
    # ]

    theoretical_bell_state_uniform_chsh = [
        uniform_max_entangled_theoretical_star_score(gamma) for gamma in gamma_range
    ]

    theoretical_bell_state_single_chsh = [
        single_max_entangled_theoretical_star_score(gamma, n=1) for gamma in gamma_range
    ]

    theoretical_nonmax_uniform_chsh = [
        uniform_nonmax_entangled_theoretical_star_score(gamma) for gamma in gamma_range
    ]

    theoretical_nonmax_single_chsh = [
        single_nonmax_entangled_theoretical_star_score(gamma, n=1) for gamma in gamma_range
    ]

    """
    Loading Bilocal Data
    """
    bilocal_uniform_ad_dir = "./data/bilocal/uniform_amplitude_damping/"

    arb_bilocal_uniform_ad_regexes = [
        r"ryrz_cnot_local_ry_.*",
        r"ryrz_cnot_local_rot_.*",
        r"arb_arb_.*",
        r"max_ent_arb_.*",
        r"max_ent_local_rot_.*",
    ]
    ent_bilocal_uniform_ad_regexes = [r"max_ent_local_rot_.*"]

    ent_bilocal_uniform_ad_data = [
        src.analyze_data_one_param_scan(src.get_data_files(bilocal_uniform_ad_dir, regex))
        for regex in ent_bilocal_uniform_ad_regexes
    ]

    ent_max_bilocal_uniform_ad = [
        max(map(lambda opt_data: opt_data["max_scores"][i], ent_bilocal_uniform_ad_data))
        for i in range(num_samples)
    ]

    arb_bilocal_uniform_ad_data = [
        src.analyze_data_one_param_scan(src.get_data_files(bilocal_uniform_ad_dir, regex))
        for regex in arb_bilocal_uniform_ad_regexes
    ]

    arb_max_bilocal_uniform_ad = [
        max(map(lambda opt_data: opt_data["max_scores"][i], arb_bilocal_uniform_ad_data))
        for i in range(num_samples)
    ]

    bilocal_single_ad_dir = "./data/bilocal/single_qubit_amplitude_damping/"

    arb_bilocal_single_ad_regexes = [
        r"ryrz_cnot_local_ry_out_.*",
        r"max_ent_arb_out_.*",
        r"arb_arb_out_.*",
    ]
    ent_bilocal_single_ad_regexes = [r"max_ent_local_rot_.*"]

    ent_bilocal_single_ad_data = [
        src.analyze_data_one_param_scan(src.get_data_files(bilocal_single_ad_dir, regex))
        for regex in ent_bilocal_single_ad_regexes
    ]

    ent_max_bilocal_single_ad = [
        max(map(lambda opt_data: opt_data["max_scores"][i], ent_bilocal_single_ad_data))
        for i in range(num_samples)
    ]

    arb_bilocal_single_ad_data = [
        src.analyze_data_one_param_scan(src.get_data_files(bilocal_single_ad_dir, regex))
        for regex in arb_bilocal_single_ad_regexes
    ]

    arb_max_bilocal_single_ad = [
        max(map(lambda opt_data: opt_data["max_scores"][i], arb_bilocal_single_ad_data))
        for i in range(num_samples)
    ]

    # theoretical_bell_state_uniform_bilocal = [
    #     src.bilocal_max_violation(state, state) for state in bell_state_uniform_noise_states
    # ]

    # theoretical_bell_state_single_bilocal = [
    #     src.bilocal_max_violation(state, bell_state) for state in bell_state_single_noise_states
    # ]

    theoretical_bell_state_uniform_bilocal = [
        uniform_max_entangled_theoretical_star_score(gamma) for gamma in gamma_range
    ]

    theoretical_bell_state_single_bilocal = [
        single_max_entangled_theoretical_star_score(gamma, n=2) for gamma in gamma_range
    ]

    theoretical_nonmax_uniform_bilocal = [
        uniform_nonmax_entangled_theoretical_star_score(gamma) for gamma in gamma_range
    ]

    theoretical_nonmax_single_bilocal = [
        single_nonmax_entangled_theoretical_star_score(gamma, n=2) for gamma in gamma_range
    ]

    """
    Loading n-Chain Data
    """

    chain_uniform_ad_dir = "./data/n-chain/uniform_amplitude_damping/"

    arb_n3_chain_uniform_ad_regexes = [
        r"ryrz_cnot_local_ry_n-3_.*",
        r"arb_arb_n-3_.*",
        r"arb_local_rot_n-3_.*",
        r"max_entangled_local_rot_n-3_.*",
        r"ghz_local_rot_n-3_.*",
        r"ryrz_cnot_local_rot_n-3_.*",
    ]

    ent_n3_chain_uniform_ad_regexes = [
        r"max_entangled_local_rot_n-3_.*",
        r"ghz_local_rot_n-3_.*",
    ]

    arb_n3_chain_uniform_ad_data = [
        src.analyze_data_one_param_scan(src.get_data_files(chain_uniform_ad_dir, regex))
        for regex in arb_n3_chain_uniform_ad_regexes
    ]

    arb_max_n3_chain_uniform_ad = [
        max(map(lambda opt_data: opt_data["max_scores"][i], arb_n3_chain_uniform_ad_data))
        for i in range(num_samples)
    ]

    ent_n3_chain_uniform_ad_data = [
        src.analyze_data_one_param_scan(src.get_data_files(chain_uniform_ad_dir, regex))
        for regex in ent_n3_chain_uniform_ad_regexes
    ]

    ent_max_n3_chain_uniform_ad = [
        max(map(lambda opt_data: opt_data["max_scores"][i], ent_n3_chain_uniform_ad_data))
        for i in range(num_samples)
    ]

    arb_n4_chain_uniform_ad_regexes = [
        r"ryrz_cnot_local_ry_n-4_.*",
        r"arb_arb_n-4_.*",
        r"arb_local_rot_n-4_.*",
        r"max_entangled_local_rot_n-4_.*",
        r"ghz_local_rot_n-4_.*",
        r"ryrz_cnot_local_rot_n-4_.*",
    ]
    ent_n4_chain_uniform_ad_regexes = [
        r"max_entangled_local_rot_n-4_.*",
        r"ghz_local_rot_n-4_.*",
    ]

    ent_n4_chain_uniform_ad_data = [
        src.analyze_data_one_param_scan(src.get_data_files(chain_uniform_ad_dir, regex))
        for regex in ent_n4_chain_uniform_ad_regexes
    ]

    ent_max_n4_chain_uniform_ad = [
        max(map(lambda opt_data: opt_data["max_scores"][i], ent_n4_chain_uniform_ad_data))
        for i in range(num_samples)
    ]

    arb_n4_chain_uniform_ad_data = [
        src.analyze_data_one_param_scan(src.get_data_files(chain_uniform_ad_dir, regex))
        for regex in arb_n4_chain_uniform_ad_regexes
    ]

    arb_max_n4_chain_uniform_ad = [
        max(map(lambda opt_data: opt_data["max_scores"][i], arb_n4_chain_uniform_ad_data))
        for i in range(num_samples)
    ]

    chain_single_ad_dir = "./data/n-chain/single_qubit_amplitude_damping/"

    arb_n3_chain_single_ad_regexes = [
        r"arb_arb_n-3_.*",
        r"arb_local_rot_n-3_.*",
        r"max_entangled_arb_n-3_.*",
        r"ryrz_cnot_local_ry_n-3_.*",
        r"max_entangled_local_rot_n-3_.*",
    ]
    ent_n3_chain_single_ad_regexes = [
        r"max_entangled_local_rot_n-3_.*",
    ]

    ent_n3_chain_single_ad_data = [
        src.analyze_data_one_param_scan(src.get_data_files(chain_single_ad_dir, regex))
        for regex in ent_n3_chain_single_ad_regexes
    ]

    ent_max_n3_chain_single_ad = [
        max(map(lambda opt_data: opt_data["max_scores"][i], ent_n3_chain_single_ad_data))
        for i in range(num_samples)
    ]

    arb_n3_chain_single_ad_data = [
        src.analyze_data_one_param_scan(src.get_data_files(chain_single_ad_dir, regex))
        for regex in arb_n3_chain_single_ad_regexes
    ]

    arb_max_n3_chain_single_ad = [
        max(map(lambda opt_data: opt_data["max_scores"][i], arb_n3_chain_single_ad_data))
        for i in range(num_samples)
    ]

    arb_n4_chain_single_ad_regexes = [
        r"ryrz_cnot_local_ry_n-4_.*",
        r"arb_arb_n-4_.*",
        r"arb_local_rot_n-4_.*",
        r"max_entangled_arb_n-4_.*",
        r"max_entangled_local_rot_n-4_.*",
    ]
    ent_n4_chain_single_ad_regexes = [
        r"max_entangled_local_rot_n-4_.*",
    ]

    ent_n4_chain_single_ad_data = [
        src.analyze_data_one_param_scan(src.get_data_files(chain_single_ad_dir, regex))
        for regex in ent_n4_chain_single_ad_regexes
    ]

    ent_max_n4_chain_single_ad = [
        max(map(lambda opt_data: opt_data["max_scores"][i], ent_n4_chain_single_ad_data))
        for i in range(num_samples)
    ]

    arb_n4_chain_single_ad_data = [
        src.analyze_data_one_param_scan(src.get_data_files(chain_single_ad_dir, regex))
        for regex in arb_n4_chain_single_ad_regexes
    ]

    arb_max_n4_chain_single_ad = [
        max(map(lambda opt_data: opt_data["max_scores"][i], arb_n4_chain_single_ad_data))
        for i in range(num_samples)
    ]

    theoretical_bell_state_uniform_n3_chain = [
        uniform_max_entangled_theoretical_chain_score(gamma, n=3) for gamma in gamma_range
    ]

    theoretical_bell_state_single_n3_chain = [
        single_max_entangled_theoretical_star_score(gamma, n=2) for gamma in gamma_range
    ]

    theoretical_nonmax_uniform_n3_chain = [
        uniform_nonmax_entangled_theoretical_star_score(gamma) for gamma in gamma_range
    ]

    theoretical_nonmax_single_n3_chain = [
        single_nonmax_entangled_theoretical_star_score(gamma, n=2) for gamma in gamma_range
    ]

    theoretical_bell_state_uniform_n4_chain = [
        uniform_max_entangled_theoretical_chain_score(gamma, n=4) for gamma in gamma_range
    ]

    theoretical_bell_state_single_n4_chain = [
        single_max_entangled_theoretical_star_score(gamma, n=2) for gamma in gamma_range
    ]

    theoretical_nonmax_uniform_n4_chain = [
        uniform_nonmax_entangled_theoretical_star_score(gamma) for gamma in gamma_range
    ]

    theoretical_nonmax_single_n4_chain = [
        single_nonmax_entangled_theoretical_star_score(gamma, n=2) for gamma in gamma_range
    ]

    """
    Loading n-Star Data
    """

    star_uniform_ad_dir = "./data/n-star/uniform_amplitude_damping/"

    ent_n3_star_ad_uniform_regexes = [
        r"max_entangled_local_rot_n-3_.*",
        r"ghz_local_rot_n-3_.*",
    ]
    arb_n3_star_ad_uniform_regexes = [
        r"arb_ghz_rot_n-3_.*",
        r"arb_local_rot_n-3_.*",
        r"max_entangled_ghz_rot_n-3_.*",
        r"ryrz_cnot_local_ry_n-3_.*",
        r"max_entangled_local_rot_n-3_.*",
        r"ryrz_cnot_local_rot_n-3_.*",
        r"ghz_local_rot_n-3_.*",
    ]

    ent_n3_star_uniform_ad_data = [
        src.analyze_data_one_param_scan(src.get_data_files(star_uniform_ad_dir, regex))
        for regex in ent_n3_star_ad_uniform_regexes
    ]

    ent_max_n3_star_uniform_ad = [
        max(map(lambda opt_data: opt_data["max_scores"][i], ent_n3_star_uniform_ad_data))
        for i in range(num_samples)
    ]

    arb_n3_star_uniform_ad_data = [
        src.analyze_data_one_param_scan(src.get_data_files(star_uniform_ad_dir, regex))
        for regex in arb_n3_star_ad_uniform_regexes
    ]

    arb_max_n3_star_uniform_ad = [
        max(map(lambda opt_data: opt_data["max_scores"][i], arb_n3_star_uniform_ad_data))
        for i in range(num_samples)
    ]

    arb_n4_star_ad_uniform_regexes = [
        r"arb_ghz_rot_n-4_.*",
        r"arb_local_rot_n-4_.*",
        r"max_entangled_ghz_rot_n-4_.*",
        r"ryrz_cnot_local_ry_n-4_.*",
        r"ryrz_cnot_local_rot_n-4_.*",
        r"max_entangled_local_rot_n-4_.*",
        r"ghz_local_rot_n-4_.*",
    ]
    ent_n4_star_ad_uniform_regexes = [
        r"max_entangled_local_rot_n-4_.*",
        r"ghz_local_rot_n-4_.*",
    ]

    ent_n4_star_uniform_ad_data = [
        src.analyze_data_one_param_scan(src.get_data_files(star_uniform_ad_dir, regex))
        for regex in ent_n4_star_ad_uniform_regexes
    ]

    ent_max_n4_star_uniform_ad = [
        max(map(lambda opt_data: opt_data["max_scores"][i], ent_n4_star_uniform_ad_data))
        for i in range(num_samples)
    ]

    arb_n4_star_uniform_ad_data = [
        src.analyze_data_one_param_scan(src.get_data_files(star_uniform_ad_dir, regex))
        for regex in arb_n4_star_ad_uniform_regexes
    ]

    arb_max_n4_star_uniform_ad = [
        max(map(lambda opt_data: opt_data["max_scores"][i], arb_n4_star_uniform_ad_data))
        for i in range(num_samples)
    ]

    star_single_ad_dir = "./data/n-star/single_qubit_amplitude_damping/"

    ent_n3_star_ad_single_regexes = [
        r"max_entangled_local_rot_n-3_.*",
    ]
    arb_n3_star_ad_single_regexes = [
        r"arb_ghz_rot_n-3_.*",
        r"arb_local_rot_n-3_.*",
        r"max_entangled_ghz_rot_n-3_.*",
        r"ryrz_cnot_local_ry_n-3_.*",
        r"max_entangled_local_rot_n-3_.*",
    ]

    ent_n3_star_single_ad_data = [
        src.analyze_data_one_param_scan(src.get_data_files(star_single_ad_dir, regex))
        for regex in ent_n3_star_ad_single_regexes
    ]

    ent_max_n3_star_single_ad = [
        max(map(lambda opt_data: opt_data["max_scores"][i], ent_n3_star_single_ad_data))
        for i in range(num_samples)
    ]

    arb_n3_star_single_ad_data = [
        src.analyze_data_one_param_scan(src.get_data_files(star_single_ad_dir, regex))
        for regex in arb_n3_star_ad_single_regexes
    ]

    arb_max_n3_star_single_ad = [
        max(map(lambda opt_data: opt_data["max_scores"][i], arb_n3_star_single_ad_data))
        for i in range(num_samples)
    ]

    ent_n4_star_ad_single_regexes = [
        r"max_entangled_local_rot_n-4_.*",
    ]
    arb_n4_star_ad_single_regexes = [
        r"arb_ghz_rot_n-4_.*",
        r"arb_local_rot_n-4_.*",
        r"max_entangled_ghz_rot_n-4_.*",
        r"ryrz_cnot_local_ry_n-4_.*",
        r"max_entangled_local_rot_n-4_.*",
    ]

    ent_n4_star_single_ad_data = [
        src.analyze_data_one_param_scan(src.get_data_files(star_single_ad_dir, regex))
        for regex in ent_n4_star_ad_single_regexes
    ]

    ent_max_n4_star_single_ad = [
        max(map(lambda opt_data: opt_data["max_scores"][i], ent_n4_star_single_ad_data))
        for i in range(num_samples)
    ]

    arb_n4_star_single_ad_data = [
        src.analyze_data_one_param_scan(src.get_data_files(star_single_ad_dir, regex))
        for regex in arb_n4_star_ad_single_regexes
    ]

    arb_max_n4_star_single_ad = [
        max(map(lambda opt_data: opt_data["max_scores"][i], arb_n4_star_single_ad_data))
        for i in range(num_samples)
    ]

    theoretical_bell_state_uniform_n3_star = [
        uniform_max_entangled_theoretical_star_score(gamma) for gamma in gamma_range
    ]

    theoretical_bell_state_single_n3_star = [
        single_max_entangled_theoretical_star_score(gamma, n=3) for gamma in gamma_range
    ]

    theoretical_nonmax_uniform_n3_star = [
        uniform_nonmax_entangled_theoretical_star_score(gamma) for gamma in gamma_range
    ]

    theoretical_nonmax_single_n3_star = [
        single_nonmax_entangled_theoretical_star_score(gamma, n=3) for gamma in gamma_range
    ]

    theoretical_bell_state_uniform_n4_star = [
        uniform_max_entangled_theoretical_star_score(gamma) for gamma in gamma_range
    ]

    theoretical_bell_state_single_n4_star = [
        single_max_entangled_theoretical_star_score(gamma, n=4) for gamma in gamma_range
    ]

    theoretical_nonmax_uniform_n4_star = [
        uniform_nonmax_entangled_theoretical_star_score(gamma) for gamma in gamma_range
    ]

    theoretical_nonmax_single_n4_star = [
        single_nonmax_entangled_theoretical_star_score(gamma, n=4) for gamma in gamma_range
    ]

    """
    Verifying Data
    """

    def verify_data(theoretical_score, vqo_score, atol=1e-8):
        return theoretical_score >= vqo_score or np.isclose(theoretical_score, vqo_score, atol=atol)

    for u in range(21):
        assert verify_data(theoretical_bell_state_single_chsh[u], ent_max_chsh_single_ad[u])
        assert verify_data(
            theoretical_bell_state_single_bilocal[u], ent_max_bilocal_single_ad[u], atol=1e-7
        )
        assert verify_data(
            theoretical_bell_state_single_n3_chain[u], ent_max_n3_chain_single_ad[u], atol=1e-6
        )
        assert verify_data(
            theoretical_bell_state_single_n4_chain[u], ent_max_n4_chain_single_ad[u], atol=1e-5
        )
        assert verify_data(
            theoretical_bell_state_single_n3_star[u], ent_max_n3_star_single_ad[u], atol=1e-5
        )
        assert verify_data(
            theoretical_bell_state_single_n4_star[u], ent_max_n4_star_single_ad[u], atol=1e-3
        )

        assert verify_data(theoretical_bell_state_uniform_chsh[u], ent_max_chsh_uniform_ad[u])
        assert verify_data(theoretical_bell_state_uniform_bilocal[u], ent_max_bilocal_uniform_ad[u])
        assert verify_data(
            theoretical_bell_state_uniform_n3_chain[u], ent_max_n3_chain_uniform_ad[u]
        )
        assert verify_data(
            theoretical_bell_state_uniform_n4_chain[u], ent_max_n4_chain_uniform_ad[u]
        )
        assert verify_data(theoretical_bell_state_uniform_n3_star[u], ent_max_n3_star_uniform_ad[u])
        assert verify_data(theoretical_bell_state_uniform_n4_star[u], ent_max_n4_star_uniform_ad[u])

        assert verify_data(theoretical_nonmax_single_chsh[u], arb_max_chsh_single_ad[u])
        assert verify_data(theoretical_nonmax_single_bilocal[u], arb_max_bilocal_single_ad[u])
        assert verify_data(theoretical_nonmax_single_n3_chain[u], arb_max_n3_chain_single_ad[u])
        assert verify_data(theoretical_nonmax_single_n4_chain[u], arb_max_n4_chain_single_ad[u])
        assert verify_data(theoretical_nonmax_single_n3_star[u], arb_max_n3_star_single_ad[u])
        assert verify_data(theoretical_nonmax_single_n4_star[u], arb_max_n4_star_single_ad[u])

        assert verify_data(theoretical_nonmax_uniform_chsh[u], arb_max_chsh_uniform_ad[u])
        assert verify_data(theoretical_nonmax_uniform_bilocal[u], arb_max_bilocal_uniform_ad[u])
        assert verify_data(theoretical_nonmax_uniform_n3_chain[u], arb_max_n3_chain_uniform_ad[u])
        assert verify_data(theoretical_nonmax_uniform_n4_chain[u], arb_max_n4_chain_uniform_ad[u])
        assert verify_data(theoretical_nonmax_uniform_n3_star[u], arb_max_n3_star_uniform_ad[u])
        assert verify_data(theoretical_nonmax_uniform_n4_star[u], arb_max_n4_star_uniform_ad[u])

    """
    Plotting Data
    """

    src.plot_nonunital_single_and_uniform_max_scores_data(
        fig_title="Qubit Amplitude Damping Noise Robustness",
        ax_titles=["Single Qubit Noise", "Uniform Qubit Noise"],
        noise_params=ent_chsh_uniform_ad_data[0]["noise_params"],
        quantum_bound=np.sqrt(2),
        classical_bound=1,
        row1_single_max_scores=[
            ent_max_chsh_single_ad,
            ent_max_bilocal_single_ad,
            ent_max_n3_chain_single_ad,
            ent_max_n4_chain_single_ad,
            ent_max_n3_star_single_ad,
            ent_max_n4_star_single_ad,
        ],
        row1_single_theoretical_scores=[
            theoretical_bell_state_single_chsh,
            theoretical_bell_state_single_bilocal,
            theoretical_bell_state_single_n3_chain,
            theoretical_bell_state_single_n4_chain,
            theoretical_bell_state_single_n3_star,
            theoretical_bell_state_single_n4_star,
        ],
        row1_uniform_max_scores=[
            ent_max_chsh_uniform_ad,
            ent_max_bilocal_uniform_ad,
            ent_max_n3_chain_uniform_ad,
            ent_max_n4_chain_uniform_ad,
            ent_max_n3_star_uniform_ad,
            ent_max_n4_star_uniform_ad,
        ],
        row1_uniform_theoretical_scores=[
            theoretical_bell_state_uniform_chsh,
            theoretical_bell_state_uniform_bilocal,
            theoretical_bell_state_uniform_n3_chain,
            theoretical_bell_state_uniform_n4_chain,
            theoretical_bell_state_uniform_n3_star,
            theoretical_bell_state_uniform_n4_star,
        ],
        row2_single_max_scores=[
            arb_max_chsh_single_ad,
            arb_max_bilocal_single_ad,
            arb_max_n3_chain_single_ad,
            arb_max_n4_chain_single_ad,
            arb_max_n3_star_single_ad,
            arb_max_n4_star_single_ad,
        ],
        row2_single_theoretical_scores=[
            theoretical_nonmax_single_chsh,
            theoretical_nonmax_single_bilocal,
            theoretical_nonmax_single_n3_chain,
            theoretical_nonmax_single_n4_chain,
            theoretical_nonmax_single_n3_star,
            theoretical_nonmax_single_n4_star,
        ],
        row2_uniform_max_scores=[
            arb_max_chsh_uniform_ad,
            arb_max_bilocal_uniform_ad,
            arb_max_n3_chain_uniform_ad,
            arb_max_n4_chain_uniform_ad,
            arb_max_n3_star_uniform_ad,
            arb_max_n4_star_uniform_ad,
        ],
        row2_uniform_theoretical_scores=[
            theoretical_nonmax_uniform_chsh,
            theoretical_nonmax_uniform_bilocal,
            theoretical_nonmax_uniform_n3_chain,
            theoretical_nonmax_uniform_n4_chain,
            theoretical_nonmax_uniform_n3_star,
            theoretical_nonmax_uniform_n4_star,
        ],
        data_labels=[
            "CHSH",
            "Bilocal",
            "3-Local Chain",
            "4-Local Chain",
            "3-Local Star",
            "4-Local Star",
        ],
        row_labels=["Max Entangled", "General"],
        plot_dir="./data/plots/qubit_amplitude_damping_noise_robustness/",
        bottom_padding=0.2,
        ncol_legend=4,
    )

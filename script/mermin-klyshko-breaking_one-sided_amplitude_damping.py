from dask.distributed import Client
import time
from pennylane import numpy as np
import pennylane as qml

import math
from context import qnetvo as qnet
import sys

import utilities
import network_ansatzes as ansatzes


"""
This script collects data about the robustness of the Mermin-Klyshko inequality
with respect to one-sided amplitude damping channels.

This script compares the performance of the GHZ state against arrbitrary state
preparations.
"""


def pure_amplitude_damping(noise_params, wires):
    ry_setting = 2 * math.asin(math.sqrt(noise_params[0]))

    qml.ctrl(qml.RY, control=wires[0])(ry_setting, wires=wires[1])
    qml.CNOT(wires=[wires[1], wires[0]])


def single_qubit_amplitude_damping_nodes_fn(n, wires=[0]):
    def noise_nodes(noise_args):
        return [
            qnet.NoiseNode(
                [wires[0], n],
                # lambda settings, wires: qnet.pure_amplitude_damping([noise_args], wires=wires),
                lambda settings, wires: pure_amplitude_damping([noise_args], wires=wires),
            ),
        ]

    return noise_nodes


if __name__ == "__main__":

    client = Client(processes=True, n_workers=5, threads_per_worker=1)

    step_sizes = [0.1, 0.03, 0.0075]
    num_steps = [45, 65, 105]

    for i, n in enumerate([3, 4, 5]):
        print("n = ", n)
        param_range = np.arange(0, 1.01, 0.05)

        step_size = step_sizes[i]

        # ghz state preparations
        time_start = time.time()

        ghz_optimization = utilities.noisy_net_opt_fn(
            ansatzes.ghz_prep_node(n),
            ansatzes.local_rot_meas_nodes(n),
            single_qubit_amplitude_damping_nodes_fn(n, wires=[0]),
            qnet.mermin_klyshko_cost_fn,
            ansatz_kwargs={
                "dev_kwargs": {
                    "name": "default.qubit",
                },
            },
            opt_kwargs={
                "sample_width": 5,
                "step_size": step_size,
                "num_steps": num_steps[i],
                "verbose": False,
            },
        )
        ghz_jobs = client.map(ghz_optimization, param_range)
        ghz_opt_dicts = client.gather(ghz_jobs)

        utilities.save_optimizations_one_param_scan(
            "script/data/mermin-klyshko-breaking_one-sided_amplitude_damping/",
            "ghz_n-" + str(n) + "_",
            param_range,
            ghz_opt_dicts,
            quantum_bound=qnet.mermin_klyshko_quantum_bound(n),
            classical_bound=qnet.mermin_klyshko_classical_bound(n),
        )

        time_elapsed = time.time() - time_start
        print("\nelapsed time : ", time_elapsed, "\n")

        client.restart()

        # arb state preparations
        time_start = time.time()
        arb_optimization = utilities.noisy_net_opt_fn(
            ansatzes.arb_prep_node(n),
            ansatzes.local_rot_meas_nodes(n),
            single_qubit_amplitude_damping_nodes_fn(n, wires=[0]),
            qnet.mermin_klyshko_cost_fn,
            ansatz_kwargs={
                "dev_kwargs": {
                    "name": "default.qubit",
                },
            },
            opt_kwargs={
                "sample_width": 5,
                "step_size": step_size,
                "num_steps": num_steps[i],
                "verbose": False,
            },
        )
        arb_jobs = client.map(arb_optimization, param_range)
        arb_opt_dicts = client.gather(arb_jobs)

        utilities.save_optimizations_one_param_scan(
            "script/data/mermin-klyshko-breaking_one-sided_amplitude_damping/",
            "arb_n-" + str(n) + "_",
            param_range,
            arb_opt_dicts,
            quantum_bound=qnet.mermin_klyshko_quantum_bound(n),
            classical_bound=qnet.mermin_klyshko_classical_bound(n),
        )

        time_elapsed = time.time() - time_start
        print("\nelapsed time : ", time_elapsed, "\n")

        client.restart()

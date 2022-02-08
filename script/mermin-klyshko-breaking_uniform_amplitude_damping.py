from dask.distributed import Client
import time
from pennylane import numpy as np
import pennylane as qml

import math
from context import QNetOptimizer as QNopt

import utilities
import network_ansatzes as ansatzes


"""
This script collects data about the robustness of the Mermin-Klyshko inequality
with respect to one-sided amplitude damping channels.

This script compares the performance of the GHZ state against arrbitrary state
preparations.
"""

def pure_amplitude_damping_nodes_fn(n):
    def noise_nodes(noise_args):
        return [
            QNopt.NoiseNode(
                [i, n + i],
                lambda settings, wires: QNopt.pure_amplitude_damping([noise_args], wires=wires),
            )
            for i in range(n)
        ]

    return noise_nodes

if __name__ == "__main__":


    data_dir = "script/data/mermin-klyshko-breaking_uniform_amplitude_damping/"

    step_sizes = [0.1, 0.03, 0.0075]
    num_steps = [50, 85, 125]

    for i, n in enumerate([3, 4, 5]):
        client = Client(processes=True, n_workers=5, threads_per_worker=1)

        print("n = ", n)
        param_range = np.arange(0, 1.01, 0.05)

        step_size = step_sizes[i]

        # ghz state preparations
        time_start = time.time()

        ghz_optimization = utilities.noisy_net_opt_fn(
            ansatzes.ghz_prep_node(n),
            ansatzes.local_rot_meas_nodes(n),
            pure_amplitude_damping_nodes_fn(n),
            QNopt.mermin_klyshko_cost_fn,
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

        print("about to create jobs")
        ghz_jobs = client.map(ghz_optimization, param_range)
        ghz_opt_dicts = client.gather(ghz_jobs)

        utilities.save_optimizations_one_param_scan(
            data_dir,
            "ghz_n-" + str(n) + "_",
            param_range,
            ghz_opt_dicts,
            quantum_bound=QNopt.mermin_klyshko_quantum_bound(n),
            classical_bound=QNopt.mermin_klyshko_classical_bound(n),
        )

        time_elapsed = time.time() - time_start
        print("\nelapsed time : ", time_elapsed, "\n")

        client.restart()

        # arb state preparations
        time_start = time.time()
        arb_optimization = utilities.noisy_net_opt_fn(
            ansatzes.arb_prep_node(n),
            ansatzes.local_rot_meas_nodes(n),
            pure_amplitude_damping_nodes_fn(n),
            QNopt.mermin_klyshko_cost_fn,
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
            data_dir,
            "arb_n-" + str(n) + "_",
            param_range,
            arb_opt_dicts,
            quantum_bound=QNopt.mermin_klyshko_quantum_bound(n),
            classical_bound=QNopt.mermin_klyshko_classical_bound(n),
        )

        time_elapsed = time.time() - time_start
        print("\nelapsed time : ", time_elapsed, "\n")

from dask.distributed import Client
import time
from pennylane import numpy as np
import pennylane as qml
from context import qnetvo as qnet

import utilities
import network_ansatzes as ansatzes

import tensorflow as tf


"""
This script collects data from noisy n-local star optimizations.
The considered noise model is an amplitude damping channel on a
single qubit in the star.

The scan range is gamma in [0,1] with an interval of 0.05.
Stars with n = 3 are considered.

An arbitrary maximally entangled state is prepared and measured
with local qubit rotations and a arbitrary two-qubit unitary on the central
measurement node. 

The script accepts the command line argument "inside" to specify that the
noisy qubit acts upon wires=[1] (the interior measurement node).
Otherwise, the noisy qubit acts upon wires=[0] (the end node measurement).
"""


def uniform_depolarizing_nodes_fn(n):
    def noise_nodes(noise_args):
        return [
            qnet.NoiseNode(
                [i], lambda settings, wires: qml.DepolarizingChannel(noise_args, wires=wires)
            )
            for i in range(2 * n)
        ]

    return noise_nodes


if __name__ == "__main__":

    n = 4

    client = Client(processes=True, n_workers=3, threads_per_worker=1)

    param_range = np.arange(0, 1.01, 0.5)

    # local qubit rotation measurements and max entangled states
    time_start = time.time()

    local_ry_opt = utilities.noisy_net_opt_fn(
        ansatzes.star_nlocal_max_entangled_prep_nodes(n),
        ansatzes.star_22_local_rot_meas_nodes(n),
        uniform_depolarizing_nodes_fn(n),
        qnet.nlocal_star_22_cost_fn,
        opt_kwargs={
            "sample_width": 5,
            "step_size": 1.2,
            "num_steps": 30,
            "verbose": True,
        },
    )
    local_ry_jobs = client.map(local_ry_opt, param_range)
    local_ry_opt_dicts = client.gather(local_ry_jobs)

    utilities.save_optimizations_one_param_scan(
        "script/data/star_n-local_uniform_qubit_depolarizing/",
        "max_entangled_local_rot_n-" + str(n) + "_",
        param_range,
        local_ry_opt_dicts,
        quantum_bound=np.sqrt(2),
        classical_bound=1,
    )

    time_elapsed = time.time() - time_start
    print("\nelapsed time : ", time_elapsed, "\n")

    # client.restart()

    # # local qubit rotation measurements and arb states
    # time_start = time.time()

    # arb_opt = utilities.noisy_net_opt_fn(
    #     ansatzes.star_nlocal_arb_prep_nodes(n),
    #     ansatzes.star_22_local_rot_meas_nodes(n),
    #     uniform_amplitude_damping_nodes_fn(n),
    #     qnet.nlocal_star_22_cost_fn,
    #     ansatz_kwargs={
    #         "dev_kwargs": {
    #             "name": "default.qubit",
    #         },
    #     },
    #     opt_kwargs={
    #         "sample_width": 5,
    #         "step_size": 1.2,
    #         "num_steps": 30,
    #         # "interface": "tf",
    #         "verbose": True,
    #     },
    #     qnode_kwargs={
    #         # "interface": "tf",
    #         # "diff_method": "backprop",
    #     },
    # )
    # arb_jobs = client.map(arb_opt, param_range)
    # arb_opt_dicts = client.gather(arb_jobs)

    # utilities.save_optimizations_one_param_scan(
    #     "script/data/star_n-local_uniform_amplitude_damping/",
    #     "arb_local_rot_n-" + str(n) + "_",
    #     param_range,
    #     arb_opt_dicts,
    #     quantum_bound=np.sqrt(2),
    #     classical_bound=1,
    # )

    # time_elapsed = time.time() - time_start
    # print("\nelapsed time : ", time_elapsed, "\n")

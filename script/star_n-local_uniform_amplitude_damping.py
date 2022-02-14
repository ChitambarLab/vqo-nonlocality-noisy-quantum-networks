from dask.distributed import Client
import time
from pennylane import numpy as np
import pennylane as qml
from context import qnetvo as qnet

import utilities
import network_ansatzes as ansatzes


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


def uniform_amplitude_damping_nodes_fn(n):
    def noise_nodes(noise_args):
        return [
            qnet.NoiseNode(
                [i,2*n+i], lambda settings, wires: qnet.pure_amplitude_damping([noise_args], wires=wires)
            )
            for i in range(2*n)
        ]

    return noise_nodes


if __name__ == "__main__":

    n = 5

    client = Client(processes=True, n_workers=5, threads_per_worker=1)

    param_range = np.arange(0, 1.01, 0.5)

    # local qubit rotation measurements
    time_start = time.time()

    local_ry_opt = utilities.noisy_net_opt_fn(
        ansatzes.star_nlocal_max_entangled_prep_nodes(n),
        ansatzes.star_22_local_rot_meas_nodes(n),
        uniform_amplitude_damping_nodes_fn(n),
        qnet.nlocal_star_22_cost_fn,
        ansatz_kwargs={
            "dev_kwargs": {
                "name": "default.qubit",
            },
        },
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
        "script/data/star_n-local_uniform_amplitude_damping/",
        "max_entangled_local_rot_n-" + str(n) + "_",
        param_range,
        local_ry_opt_dicts,
        quantum_bound=np.sqrt(2),
        classical_bound=1,
    )

    time_elapsed = time.time() - time_start
    print("\nelapsed time : ", time_elapsed, "\n")

        # client.restart()

        # # ghz rotation measurements
        # time_start = time.time()

        # arb_optimization = utilities.noisy_net_opt_fn(
        #     ansatzes.star_nlocal_max_entangled_prep_nodes(n),
        #     ansatzes.star_22_ghz_rot_meas_nodes(n),
        #     single_qubit_amplitude_damping_nodes_fn(n, wires=noisy_wire),
        #     qnet.nlocal_star_22_cost_fn,
        #     opt_kwargs = {
        #         "sample_width" : 5,
        #         "step_size" : 1.2,
        #         "num_steps" : 30,
        #         "verbose" : False,
        #     }
        # )
        # arb_jobs = client.map(arb_optimization, param_range)
        # arb_opt_dicts = client.gather(arb_jobs)

        # utilities.save_optimizations_one_param_scan(
        #     "script/data/star_n-local_1-qubit_amplitude_damping/" + dir_ext,
        #     "max_entangled_ghz_rot_n-" + str(n) + "_",
        #     param_range,
        #     arb_opt_dicts,
        #     quantum_bound = np.sqrt(2),
        #     classical_bound = 1,
        # )

        # time_elapsed = time.time() - time_start
        # print("\nelapsed time : ", time_elapsed, "\n")

        # client.restart()

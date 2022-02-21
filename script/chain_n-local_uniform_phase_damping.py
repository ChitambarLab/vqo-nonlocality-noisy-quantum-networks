from dask.distributed import Client
import time
from pennylane import numpy as np
import pennylane as qml
from context import qnetvo as qnet

import utilities
import network_ansatzes as ansatzes


"""
This script collects data from noisy n-local chain optimizations.
The considered noise model is a phase damping channel applied
equally to each qubit in the chain.

The scan range is gamma in [0,1] with an interval of 0.05.
Chains with n = 3, 4 are considered.

Arbitrary state preparations and measurements are considered along with
local qubit measurements and maximally entangled state preparations.
"""


def uniform_phase_damping_nodes_fn(n):
    def noise_nodes(noise_args):
        return [
            qnet.NoiseNode(
                [i, 2 * n + i],
                lambda settings, wires: qnet.pure_phase_damping([noise_args], wires=wires),
            )
            for i in range(2 * n)
        ]

    return noise_nodes


if __name__ == "__main__":

    data_dir = "script/data/chain_n-local_uniform_phase_damping/"
    param_range = np.arange(0, 1.01, 0.05)

    for n in [3, 4]:

        client = Client(processes=True, n_workers=5, threads_per_worker=1)

        # local qubit rotation measurements and max entangled states
        time_start = time.time()

        max_ent_local_rot_opt = utilities.noisy_net_opt_fn(
            ansatzes.chain_nlocal_max_entangled_prep_nodes(n),
            ansatzes.chain_local_rot_meas_nodes(n),
            uniform_phase_damping_nodes_fn(n),
            qnet.nlocal_chain_cost_22,
            ansatz_kwargs={
                "dev_kwargs": {
                    "name": "default.qubit",
                },
            },
            opt_kwargs={
                "sample_width": 5,
                "step_size": 1.3,
                "num_steps": 60,
                "verbose": True,
            },
        )
        max_ent_local_rot_jobs = client.map(max_ent_local_rot_opt, param_range)
        max_ent_local_rot_opt_dicts = client.gather(max_ent_local_rot_jobs)

        utilities.save_optimizations_one_param_scan(
            data_dir,
            "max_entangled_local_rot_n-" + str(n) + "_",
            param_range,
            max_ent_local_rot_opt_dicts,
            quantum_bound=np.sqrt(2),
            classical_bound=1,
        )

        time_elapsed = time.time() - time_start
        print("\nelapsed time : ", time_elapsed, "\n")

        client.restart()

        # local qubit rotation measurements and arb states
        time_start = time.time()

        arb_local_rot_opt = utilities.noisy_net_opt_fn(
            ansatzes.chain_nlocal_arbitrary_prep_nodes(n),
            ansatzes.chain_local_rot_meas_nodes(n),
            uniform_phase_damping_nodes_fn(n),
            qnet.nlocal_chain_cost_22,
            ansatz_kwargs={
                "dev_kwargs": {
                    "name": "default.qubit",
                },
            },
            opt_kwargs={
                "sample_width": 5,
                "step_size": 1.4,
                "num_steps": 80,
                "verbose": True,
            },
        )
        arb_local_rot_jobs = client.map(arb_local_rot_opt, param_range)
        arb_local_rot_opt_dicts = client.gather(arb_local_rot_jobs)

        utilities.save_optimizations_one_param_scan(
            data_dir,
            "arb_local_rot_n-" + str(n) + "_",
            param_range,
            arb_local_rot_opt_dicts,
            quantum_bound=np.sqrt(2),
            classical_bound=1,
        )

        time_elapsed = time.time() - time_start
        print("\nelapsed time : ", time_elapsed, "\n")

        client.restart()

        # local qubit rotation measurements and arb states
        time_start = time.time()

        arb_opt = utilities.noisy_net_opt_fn(
            ansatzes.chain_nlocal_arbitrary_prep_nodes(n),
            ansatzes.chain_arb_meas_nodes(n),
            uniform_phase_damping_nodes_fn(n),
            qnet.nlocal_chain_cost_22,
            ansatz_kwargs={
                "dev_kwargs": {
                    "name": "default.qubit",
                },
            },
            opt_kwargs={
                "sample_width": 5,
                "step_size": 1,
                "num_steps": 80,
                "verbose": True,
            },
        )
        arb_jobs = client.map(arb_opt, param_range)
        arb_opt_dicts = client.gather(arb_jobs)

        utilities.save_optimizations_one_param_scan(
            data_dir,
            "arb_arb_n-" + str(n) + "_",
            param_range,
            arb_opt_dicts,
            quantum_bound=np.sqrt(2),
            classical_bound=1,
        )

        time_elapsed = time.time() - time_start
        print("\nelapsed time : ", time_elapsed, "\n")

        client.restart()

        # ghz rotation measurements and arb states
        time_start = time.time()

        max_entangled_opt = utilities.noisy_net_opt_fn(
            ansatzes.chain_nlocal_max_entangled_prep_nodes(n),
            ansatzes.chain_arb_meas_nodes(n),
            uniform_phase_damping_nodes_fn(n),
            qnet.nlocal_chain_cost_22,
            ansatz_kwargs={
                "dev_kwargs": {
                    "name": "default.qubit",
                },
            },
            opt_kwargs={
                "sample_width": 5,
                "step_size": 1,
                "num_steps": 80,
                "verbose": True,
            },
        )
        max_entangled_jobs = client.map(max_entangled_opt, param_range)
        max_entangled_opt_dicts = client.gather(max_entangled_jobs)

        utilities.save_optimizations_one_param_scan(
            data_dir,
            "max_entangled_arb_n-" + str(n) + "_",
            param_range,
            max_entangled_opt_dicts,
            quantum_bound=np.sqrt(2),
            classical_bound=1,
        )

        time_elapsed = time.time() - time_start
        print("\nelapsed time : ", time_elapsed, "\n")

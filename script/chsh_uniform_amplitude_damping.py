from dask.distributed import Client
import time
from pennylane import numpy as np
import pennylane as qml
from context import qnetvo as qnet

import utilities
import network_ansatzes


"""
This script collects data from noisy chsh optimizations.
The considered noise model is an amplitude damping channel applied
equally to each qubit.

The scan range is gamma in [0,1] with an interval of 0.05.

Arbitrary state preparations and maximally entanggled state prepaarations
are considered.
"""


def uniform_amplitude_damping_nodes_fn():
    def noise_nodes(noise_args):
        return [
            qnet.NoiseNode(
                [0,2], lambda settings, wires: qnet.pure_amplitude_damping([noise_args], wires=wires)
            ),
            qnet.NoiseNode(
                [1, 3], lambda settings, wires: qnet.pure_amplitude_damping([noise_args], wires=wires)
            )
        ]

    return noise_nodes


if __name__ == "__main__":

    data_dir = "script/data/chsh_uniform_amplitude_damping/"
    param_range = np.arange(0, 1.01, 0.05)

    max_ent_prep_nodes = [
        qnet.PrepareNode(1, [0,1], qnet.max_entangled_state, 3)
    ]
    arb_prep_nodes = [
        qnet.PrepareNode(1, [0, 1], qml.ArbitraryStatePreparation, 6)
    ]
    ryrz_cnot_prep_nodes = [
        qnet.PrepareNode(1, [0, 1], network_ansatzes.ryrz_cnot, 2)
    ]
    meas_nodes = [
        qnet.MeasureNode(2, 2, [0], lambda settings, wires: qml.Rot(*settings, wires=wires), 3),
        qnet.MeasureNode(2, 2, [1], lambda settings, wires: qml.Rot(*settings, wires=wires), 3)
    ]
    ry_meas_nodes = [
        qnet.MeasureNode(2, 2, [0], qnet.local_RY, 1),
        qnet.MeasureNode(2, 2, [1], qnet.local_RY, 1)
    ]

    client = Client(processes=True, n_workers=5, threads_per_worker=1)

    # local qubit rotation measurements and max entangled states
    time_start = time.time()

    max_ent_opt = utilities.noisy_net_opt_fn(
        max_ent_prep_nodes,
        meas_nodes,
        uniform_amplitude_damping_nodes_fn(),
        qnet.chsh_inequality_cost,
        ansatz_kwargs={
            "dev_kwargs": {
                "name": "default.qubit",
            },
        },
        opt_kwargs={
            "sample_width": 5,
            "step_size": 0.3,
            "num_steps": 50,
            "verbose": False,
        },
    )
    max_ent_jobs = client.map(max_ent_opt, param_range)
    max_ent_opt_dicts = client.gather(max_ent_jobs)

    utilities.save_optimizations_one_param_scan(
        data_dir,
        "max_ent_",
        param_range,
        max_ent_opt_dicts,
        quantum_bound=2*np.sqrt(2),
        classical_bound=2,
    )

    time_elapsed = time.time() - time_start
    print("\nelapsed time : ", time_elapsed, "\n")

    # minimal ryrz_cnot prep ansatz for optimal strategy
    time_start = time.time()

    ryrz_cnot_ry_opt = utilities.noisy_net_opt_fn(
        ryrz_cnot_prep_nodes,
        ry_meas_nodes,
        uniform_amplitude_damping_nodes_fn(),
        qnet.chsh_inequality_cost,
        ansatz_kwargs={
            "dev_kwargs": {
                "name": "default.qubit",
            },
        },
        opt_kwargs={
            "sample_width": 5,
            "step_size": 0.3,
            "num_steps": 50,
            "verbose": False,
        },
    )
    ryrz_cnot_ry_jobs = client.map(ryrz_cnot_ry_opt, param_range)
    ryrz_cnot_ry_opt_dicts = client.gather(ryrz_cnot_ry_jobs)

    utilities.save_optimizations_one_param_scan(
        data_dir,
        "ryrz_cnot_ry_",
        param_range,
        ryrz_cnot_ry_opt_dicts,
        quantum_bound=2*np.sqrt(2),
        classical_bound=2,
    )

    time_elapsed = time.time() - time_start
    print("\nelapsed time : ", time_elapsed, "\n")

    # local qubit rotation measurements and arb states
    time_start = time.time()

    arb_opt = utilities.noisy_net_opt_fn(
        arb_prep_nodes,
        meas_nodes,
        uniform_amplitude_damping_nodes_fn(),
        qnet.chsh_inequality_cost,
        ansatz_kwargs={
            "dev_kwargs": {
                "name": "default.qubit",
            },
        },
        opt_kwargs={
            "sample_width": 5,
            "step_size": 0.15,
            "num_steps": 60,
            "verbose": False,
        },
    )
    arb_jobs = client.map(arb_opt, param_range)
    arb_opt_dicts = client.gather(arb_jobs)

    utilities.save_optimizations_one_param_scan(
        data_dir,
        "arb_",
        param_range,
        arb_opt_dicts,
        quantum_bound=2*np.sqrt(2),
        classical_bound=2,
    )

    time_elapsed = time.time() - time_start
    print("\nelapsed time : ", time_elapsed, "\n")
from dask.distributed import Client
import time
from pennylane import numpy as np
import pennylane as qml

from context import qnetvo as qnet
from context import src


"""
This script collects data from noisy chsh optimizations.
The considered noise model is an depolarizing channel applied
equally to each qubit.

The scan range is gamma in [0,1] with an interval of 0.05.

Arbitrary state preparations and maximally entangled state prepaarations
are considered.
"""


def uniform_depolarizing_nodes_fn():
    def noise_nodes(noise_args):
        return [
            qnet.NoiseNode(
                [0], lambda settings, wires: qml.DepolarizingChannel(noise_args, wires=wires)
            ),
            qnet.NoiseNode(
                [1], lambda settings, wires: qml.DepolarizingChannel(noise_args, wires=wires)
            ),
        ]

    return noise_nodes


if __name__ == "__main__":

    data_dir = "data/chsh/uniform_qubit_depolarizing/"
    param_range = np.arange(0, 1.01, 0.05)

    max_ent_prep_nodes = [qnet.PrepareNode(1, [0, 1], qnet.max_entangled_state, 3)]
    arb_prep_nodes = [qnet.PrepareNode(1, [0, 1], qml.ArbitraryStatePreparation, 6)]
    ghz_prep_nodes = [qnet.PrepareNode(1, [1, 0], qnet.ghz_state, 2)]
    meas_nodes = [
        qnet.MeasureNode(2, 2, [0], lambda settings, wires: qml.Rot(*settings, wires=wires), 3),
        qnet.MeasureNode(2, 2, [1], lambda settings, wires: qml.Rot(*settings, wires=wires), 3),
    ]
    ry_meas_nodes = [
        qnet.MeasureNode(2, 2, [0], qnet.local_RY, 1),
        qnet.MeasureNode(2, 2, [1], qnet.local_RY, 1),
    ]

    client = Client(processes=True, n_workers=5, threads_per_worker=1)

    # local qubit rotation measurements and max entangled states
    time_start = time.time()

    max_ent_opt = src.noisy_net_opt_fn(
        max_ent_prep_nodes,
        meas_nodes,
        uniform_depolarizing_nodes_fn(),
        qnet.chsh_inequality_cost,
        opt_kwargs={
            "sample_width": 5,
            "step_size": 0.3,
            "num_steps": 50,
            "verbose": False,
        },
    )
    max_ent_jobs = client.map(max_ent_opt, param_range)
    max_ent_opt_dicts = client.gather(max_ent_jobs)

    src.save_optimizations_one_param_scan(
        data_dir,
        "max_ent_local_rot_",
        param_range,
        max_ent_opt_dicts,
        quantum_bound=2 * np.sqrt(2),
        classical_bound=2,
    )

    time_elapsed = time.time() - time_start
    print("\nelapsed time : ", time_elapsed, "\n")

    # minimal ryrz_cnot prep ansatz for optimal strategy
    time_start = time.time()

    ghz_local_ry_opt = src.noisy_net_opt_fn(
        ghz_prep_nodes,
        ry_meas_nodes,
        uniform_depolarizing_nodes_fn(),
        qnet.chsh_inequality_cost,
        opt_kwargs={
            "sample_width": 5,
            "step_size": 0.3,
            "num_steps": 70,
            "verbose": False,
        },
    )
    ghz_local_ry_jobs = client.map(ghz_local_ry_opt, param_range)
    ghz_local_ry_opt_dicts = client.gather(ghz_local_ry_jobs)

    src.save_optimizations_one_param_scan(
        data_dir,
        "ghz_local_ry_",
        param_range,
        ghz_local_ry_opt_dicts,
        quantum_bound=2 * np.sqrt(2),
        classical_bound=2,
    )

    time_elapsed = time.time() - time_start
    print("\nelapsed time : ", time_elapsed, "\n")

    # local qubit rotation measurements and arb states
    time_start = time.time()

    arb_opt = src.noisy_net_opt_fn(
        arb_prep_nodes,
        meas_nodes,
        uniform_depolarizing_nodes_fn(),
        qnet.chsh_inequality_cost,
        opt_kwargs={
            "sample_width": 5,
            "step_size": 0.15,
            "num_steps": 60,
            "verbose": False,
        },
    )
    arb_jobs = client.map(arb_opt, param_range)
    arb_opt_dicts = client.gather(arb_jobs)

    src.save_optimizations_one_param_scan(
        data_dir,
        "arb_local_rot_",
        param_range,
        arb_opt_dicts,
        quantum_bound=2 * np.sqrt(2),
        classical_bound=2,
    )

    time_elapsed = time.time() - time_start
    print("\nelapsed time : ", time_elapsed, "\n")

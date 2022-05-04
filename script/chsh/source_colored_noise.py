from dask.distributed import Client
import time
from pennylane import numpy as np
import pennylane as qml

import qnetvo as qnet
from context import src


"""
This script collects data from noisy chsh optimizations.
The considered noise model is a colored noise channel applied
to the two-qubit source.

The scan range is gamma in [0,1] with an interval of 0.05.

Arbitrary state preparations and maximally entangled state prepaarations
are considered.
"""


def source_colored_noise_nodes_fn():
    def noise_nodes(noise_args):
        return [
            qnet.NoiseNode(
                [0, 1], lambda settings, wires: qnet.colored_noise(noise_args, wires=wires)
            )
        ]

    return noise_nodes


if __name__ == "__main__":

    data_dir = "data/chsh/source_colored_noise/"
    param_range = np.arange(0, 1.01, 0.05)

    def psi_plus_state(settings, wires):
        qnet.ghz_state(settings, wires=wires)
        qml.PauliX(wires=wires[0])

    phi_plus_prep_nodes = [qnet.PrepareNode(1, [1, 0], qnet.ghz_state, 0)]
    psi_plus_prep_nodes = [qnet.PrepareNode(1, [1, 0], psi_plus_state, 0)]
    arb_prep_nodes = [qnet.PrepareNode(1, [0, 1], qml.ArbitraryStatePreparation, 6)]
    max_ent_prep_nodes = [qnet.PrepareNode(1, [0, 1], qnet.max_entangled_state, 3)]

    rot_meas_nodes = [
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
        rot_meas_nodes,
        source_colored_noise_nodes_fn(),
        qnet.chsh_inequality_cost,
        opt_kwargs={"sample_width": 5, "step_size": 0.3, "num_steps": 50, "verbose": False,},
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

    # minimal prep ansatz for nonoptimal phi_plus strategy
    time_start = time.time()

    phi_plus_local_ry_opt = src.noisy_net_opt_fn(
        phi_plus_prep_nodes,
        ry_meas_nodes,
        source_colored_noise_nodes_fn(),
        qnet.chsh_inequality_cost,
        opt_kwargs={"sample_width": 5, "step_size": 0.3, "num_steps": 70, "verbose": False,},
    )
    phi_plus_local_ry_jobs = client.map(phi_plus_local_ry_opt, param_range)
    phi_plus_local_ry_opt_dicts = client.gather(phi_plus_local_ry_jobs)

    src.save_optimizations_one_param_scan(
        data_dir,
        "phi_plus_local_ry_",
        param_range,
        phi_plus_local_ry_opt_dicts,
        quantum_bound=2 * np.sqrt(2),
        classical_bound=2,
    )

    time_elapsed = time.time() - time_start
    print("\nelapsed time : ", time_elapsed, "\n")

    # min prep ansatz and general measurement for nonoptimal phi_plus strategy
    time_start = time.time()

    phi_plus_local_rot_opt = src.noisy_net_opt_fn(
        phi_plus_prep_nodes,
        rot_meas_nodes,
        source_colored_noise_nodes_fn(),
        qnet.chsh_inequality_cost,
        opt_kwargs={"sample_width": 5, "step_size": 0.3, "num_steps": 70, "verbose": False,},
    )
    phi_plus_local_rot_jobs = client.map(phi_plus_local_rot_opt, param_range)
    phi_plus_local_rot_opt_dicts = client.gather(phi_plus_local_rot_jobs)

    src.save_optimizations_one_param_scan(
        data_dir,
        "phi_plus_local_rot_",
        param_range,
        phi_plus_local_rot_opt_dicts,
        quantum_bound=2 * np.sqrt(2),
        classical_bound=2,
    )

    time_elapsed = time.time() - time_start
    print("\nelapsed time : ", time_elapsed, "\n")

    # minimal prep ansatz for optimal psi_plus strategy
    time_start = time.time()

    psi_plus_local_ry_opt = src.noisy_net_opt_fn(
        psi_plus_prep_nodes,
        ry_meas_nodes,
        source_colored_noise_nodes_fn(),
        qnet.chsh_inequality_cost,
        opt_kwargs={"sample_width": 5, "step_size": 0.3, "num_steps": 70, "verbose": False,},
    )
    psi_plus_local_ry_jobs = client.map(psi_plus_local_ry_opt, param_range)
    psi_plus_local_ry_opt_dicts = client.gather(psi_plus_local_ry_jobs)

    src.save_optimizations_one_param_scan(
        data_dir,
        "psi_plus_local_ry_",
        param_range,
        psi_plus_local_ry_opt_dicts,
        quantum_bound=2 * np.sqrt(2),
        classical_bound=2,
    )

    time_elapsed = time.time() - time_start
    print("\nelapsed time : ", time_elapsed, "\n")

    # local qubit rotation measurements and arb states
    time_start = time.time()

    arb_opt = src.noisy_net_opt_fn(
        arb_prep_nodes,
        rot_meas_nodes,
        source_colored_noise_nodes_fn(),
        qnet.chsh_inequality_cost,
        opt_kwargs={"sample_width": 5, "step_size": 0.2, "num_steps": 60, "verbose": False,},
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

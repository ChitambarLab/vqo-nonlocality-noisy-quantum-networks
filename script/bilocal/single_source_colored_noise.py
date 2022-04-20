from dask.distributed import Client
import time
from pennylane import numpy as np
import pennylane as qml

import qnetvo as qnet
from context import src


"""
This script collects data from noisy bilocal optimizations.
The considered noise model is a colored noise channel applied
to the first source.

The scan range is gamma in [0,1] with an interval of 0.05.

Arbitrary state preparations and maximally entanggled state preparations
are considered.
"""


def single_colored_noise_nodes_fn():
    def noise_nodes(noise_args):
        return [
            qnet.NoiseNode(
                [0, 1], lambda settings, wires: qnet.colored_noise(noise_args, wires=wires)
            )
        ]

    return noise_nodes


if __name__ == "__main__":

    data_dir = "data/bilocal/single_source_colored_noise/"
    param_range = np.arange(0, 1.01, 0.05)

    def psi_plus_state(settings, wires):
        qnet.ghz_state(settings, wires=wires)
        qml.PauliX(wires=wires[0])

    max_ent_prep_nodes = [
        qnet.PrepareNode(1, [0,1], qnet.max_entangled_state, 3),
        qnet.PrepareNode(1, [2,3], qnet.max_entangled_state, 3)
    ]
    arb_prep_nodes = [
        qnet.PrepareNode(1, [0,1], qml.ArbitraryStatePreparation, 6),
        qnet.PrepareNode(1, [2,3], qml.ArbitraryStatePreparation, 6)
    ]
    phi_plus_prep_nodes = [
        qnet.PrepareNode(1, [0, 1], qnet.ghz_state, 0),
        qnet.PrepareNode(1, [2, 3], qnet.ghz_state, 0),
    ]
    psi_plus_prep_nodes = [
        qnet.PrepareNode(1, [0, 1], psi_plus_state, 0),
        qnet.PrepareNode(1, [2, 3], psi_plus_state, 0),
    ]

    arb_meas_nodes = [
        qnet.MeasureNode(2, 2, [0], lambda settings, wires: qml.Rot(*settings, wires=wires), 3),
        qnet.MeasureNode(2, 2, [1, 2], qml.ArbitraryUnitary, 15),
        qnet.MeasureNode(2, 2, [3], lambda settings, wires: qml.Rot(*settings, wires=wires), 3)
    ]
    local_rot_meas_nodes = [
        qnet.MeasureNode(2, 2, [0], src.local_rot, 3),
        qnet.MeasureNode(2, 2, [1, 2], src.local_rot, 6),
        qnet.MeasureNode(2, 2, [3], src.local_rot, 3)
    ]
    local_ry_meas_nodes = [
        qnet.MeasureNode(2, 2, [0], qnet.local_RY, 1),
        qnet.MeasureNode(2, 2, [1, 2], qnet.local_RY, 2),
        qnet.MeasureNode(2, 2, [3], qnet.local_RY, 1),
    ]

    client = Client(processes=True, n_workers=5, threads_per_worker=1)

    # """
    # local qubit rotation measurements and max entangled states
    # """
    # time_start = time.time()

    # max_ent_local_rot_opt = src.noisy_net_opt_fn(
    #     max_ent_prep_nodes,
    #     local_rot_meas_nodes,
    #     single_colored_noise_nodes_fn(),
    #     qnet.nlocal_chain_cost_22,
    #     opt_kwargs={
    #         "sample_width": 5,
    #         "step_size": 1.4,
    #         "num_steps": 50,
    #         "verbose": False,
    #     },
    # )
    # max_ent_local_rot_jobs = client.map(max_ent_local_rot_opt, param_range)
    # max_ent_local_rot_opt_dicts = client.gather(max_ent_local_rot_jobs)

    # src.save_optimizations_one_param_scan(
    #     data_dir,
    #     "max_ent_local_rot_",
    #     param_range,
    #     max_ent_local_rot_opt_dicts,
    #     quantum_bound=np.sqrt(2),
    #     classical_bound=1,
    # )

    # time_elapsed = time.time() - time_start
    # print("\nelapsed time : ", time_elapsed, "\n")

    # client = Client(processes=True, n_workers=5, threads_per_worker=1)

    # """
    # Minimal Optimal Ansatz, psi_plus state 
    # and local ry measurements.
    # """
    # time_start = time.time()

    # psi_plus_local_ry_opt = src.noisy_net_opt_fn(
    #     psi_plus_prep_nodes,
    #     local_ry_meas_nodes,
    #     single_colored_noise_nodes_fn(),
    #     qnet.nlocal_chain_cost_22,
    #     opt_kwargs={
    #         "sample_width": 5,
    #         "step_size": 1.4,
    #         "num_steps": 60,
    #         "verbose": False,
    #     },
    # )
    # psi_plus_local_ry_jobs = client.map(psi_plus_local_ry_opt, param_range)
    # psi_plus_local_ry_opt_dicts = client.gather(psi_plus_local_ry_jobs)

    # src.save_optimizations_one_param_scan(
    #     data_dir,
    #     "psi_plus_local_ry_",
    #     param_range,
    #     psi_plus_local_ry_opt_dicts,
    #     quantum_bound=np.sqrt(2),
    #     classical_bound=1,
    # )

    # time_elapsed = time.time() - time_start
    # print("\nelapsed time : ", time_elapsed, "\n")

    # client = Client(processes=True, n_workers=5, threads_per_worker=1)

    # """
    # Bad choice minimal Ansatz, phi_plus state 
    # and local ry measurements.
    # """
    # time_start = time.time()

    # phi_plus_local_ry_opt = src.noisy_net_opt_fn(
    #     phi_plus_prep_nodes,
    #     local_ry_meas_nodes,
    #     single_colored_noise_nodes_fn(),
    #     qnet.nlocal_chain_cost_22,
    #     opt_kwargs={
    #         "sample_width": 5,
    #         "step_size": 1.4,
    #         "num_steps": 60,
    #         "verbose": False,
    #     },
    # )
    # phi_plus_local_ry_jobs = client.map(phi_plus_local_ry_opt, param_range)
    # phi_plus_local_ry_opt_dicts = client.gather(phi_plus_local_ry_jobs)

    # src.save_optimizations_one_param_scan(
    #     data_dir,
    #     "phi_plus_local_ry_",
    #     param_range,
    #     phi_plus_local_ry_opt_dicts,
    #     quantum_bound=np.sqrt(2),
    #     classical_bound=1,
    # )

    # time_elapsed = time.time() - time_start
    # print("\nelapsed time : ", time_elapsed, "\n")

    # client = Client(processes=True, n_workers=5, threads_per_worker=1)

    """
    Bad choice minimal Ansatz, phi_plus state 
    and local ry measurements.
    """
    time_start = time.time()

    phi_plus_local_rot_opt = src.noisy_net_opt_fn(
        phi_plus_prep_nodes,
        local_rot_meas_nodes,
        single_colored_noise_nodes_fn(),
        qnet.nlocal_chain_cost_22,
        opt_kwargs={
            "sample_width": 5,
            "step_size": 1.5,
            "num_steps": 60,
            "verbose": False,
        },
    )
    phi_plus_local_rot_jobs = client.map(phi_plus_local_rot_opt, param_range)
    phi_plus_local_rot_opt_dicts = client.gather(phi_plus_local_rot_jobs)

    src.save_optimizations_one_param_scan(
        data_dir,
        "phi_plus_local_rot_",
        param_range,
        phi_plus_local_rot_opt_dicts,
        quantum_bound=np.sqrt(2),
        classical_bound=1,
    )

    time_elapsed = time.time() - time_start
    print("\nelapsed time : ", time_elapsed, "\n")

    # client = Client(processes=True, n_workers=5, threads_per_worker=1)

    # """
    # Bad choice minimal Ansatz, phi_plus state 
    # and arbitrary measurements.
    # """
    # time_start = time.time()

    # phi_plus_arb_opt = src.noisy_net_opt_fn(
    #     phi_plus_prep_nodes,
    #     arb_meas_nodes,
    #     single_colored_noise_nodes_fn(),
    #     qnet.nlocal_chain_cost_22,
    #     opt_kwargs={
    #         "sample_width": 5,
    #         "step_size": 1.4,
    #         "num_steps": 60,
    #         "verbose": False,
    #     },
    # )
    # phi_plus_arb_jobs = client.map(phi_plus_arb_opt, param_range)
    # phi_plus_arb_opt_dicts = client.gather(phi_plus_arb_jobs)

    # src.save_optimizations_one_param_scan(
    #     data_dir,
    #     "phi_plus_arb_",
    #     param_range,
    #     phi_plus_local_ry_opt_dicts,
    #     quantum_bound=np.sqrt(2),
    #     classical_bound=1,
    # )

    # time_elapsed = time.time() - time_start
    # print("\nelapsed time : ", time_elapsed, "\n")

    # client = Client(processes=True, n_workers=5, threads_per_worker=1)

    # """
    # Maximally entangled states and Arbitrary Measurements
    # """
    # time_start = time.time()

    # max_ent_arb_opt = src.noisy_net_opt_fn(
    #     max_ent_prep_nodes,
    #     arb_meas_nodes,
    #     single_colored_noise_nodes_fn(),
    #     qnet.nlocal_chain_cost_22,
    #     opt_kwargs={
    #         "sample_width": 5,
    #         "step_size": 1.2,
    #         "num_steps": 70,
    #         "verbose": False,
    #     },
    # )
    # max_ent_arb_jobs = client.map(max_ent_arb_opt, param_range)
    # max_ent_arb_opt_dicts = client.gather(max_ent_arb_jobs)

    # src.save_optimizations_one_param_scan(
    #     data_dir,
    #     "max_ent_arb_",
    #     param_range,
    #     max_ent_arb_opt_dicts,
    #     quantum_bound=np.sqrt(2),
    #     classical_bound=1,
    # )

    # time_elapsed = time.time() - time_start
    # print("\nelapsed time : ", time_elapsed, "\n")

    # client = Client(processes=True, n_workers=5, threads_per_worker=1)

    # """
    # Arbitrary state preparations and measurements
    # """
    # time_start = time.time()

    # arb_arb_opt = src.noisy_net_opt_fn(
    #     arb_prep_nodes,
    #     arb_meas_nodes,
    #     single_colored_noise_nodes_fn(),
    #     qnet.nlocal_chain_cost_22,
    #     opt_kwargs={
    #         "sample_width": 5,
    #         "step_size": 1.1,
    #         "num_steps": 80,
    #         "verbose": False,
    #     },
    # )
    # arb_arb_jobs = client.map(arb_arb_opt, param_range)
    # arb_arb_opt_dicts = client.gather(arb_arb_jobs)

    # src.save_optimizations_one_param_scan(
    #     data_dir,
    #     "arb_arb_",
    #     param_range,
    #     arb_arb_opt_dicts,
    #     quantum_bound=np.sqrt(2),
    #     classical_bound=1,
    # )

    # time_elapsed = time.time() - time_start
    # print("\nelapsed time : ", time_elapsed, "\n")

from dask.distributed import Client
import time
from pennylane import numpy as np
import pennylane as qml

import qnetvo as qnet
from context import src


"""
This script collects data from noisy n-local star optimizations.
Colored noise is applied to the first sources in the n-local star.

The scan range is gamma in [0,1] with an interval of 0.05.
Stars with n = 3,4 are considered.

An arbitrary maximally entangled state is prepared and measured
with local qubit rotations and a arbitrary two-qubit unitary on the central
measurement node. 

The script accepts the command line argument "inside" to specify that the
noisy qubit acts upon wires=[1] (the interior measurement node).
Otherwise, the noisy qubit acts upon wires=[0] (the end node measurement).
"""


def single_colored_noise_nodes_fn(n):
    def noise_nodes(noise_args):
        return [
            qnet.NoiseNode(
                [0, n], lambda settings, wires: qnet.colored_noise(noise_args, wires=wires)
            )
        ]

    return noise_nodes


if __name__ == "__main__":

    data_dir = "data/n-star/single_source_colored_noise/"
    param_range = np.arange(0, 1.01, 0.05)

    for n in [3]:

        # client = Client(processes=True, n_workers=5, threads_per_worker=1)

        # """
        # Minimal optimal ansatz for colored_noise
        # """
        # time_start = time.time()

        # psi_plus_local_ry_opt = src.noisy_net_opt_fn(
        #     src.star_psi_plus_prep_nodes(n),
        #     src.star_22_local_ry_meas_nodes(n),
        #     single_colored_noise_nodes_fn(n),
        #     qnet.nlocal_star_22_cost_fn,
        #     opt_kwargs={
        #         "sample_width": 5,
        #         "step_size": 2,
        #         "num_steps": 40,
        #         "verbose": True,
        #     },
        # )
        # psi_plus_local_ry_jobs = client.map(psi_plus_local_ry_opt, param_range)
        # psi_plus_local_ry_opt_dicts = client.gather(psi_plus_local_ry_jobs)

        # src.save_optimizations_one_param_scan(
        #     data_dir,
        #     "psi_plus_local_ry_n-" + str(n) + "_",
        #     param_range,
        #     psi_plus_local_ry_opt_dicts,
        #     quantum_bound=np.sqrt(2),
        #     classical_bound=1,
        # )

        # time_elapsed = time.time() - time_start
        # print("\nelapsed time : ", time_elapsed, "\n")

        client = Client(processes=True, n_workers=5, threads_per_worker=1)

        # """
        # Minimal optimal ansatz for colored_noise
        # """
        # time_start = time.time()

        # psi_plus_local_ry_opt = src.noisy_net_opt_fn(
        #     src.star_psi_plus_prep_nodes(n),
        #     src.star_22_local_ry_meas_nodes(n),
        #     single_colored_noise_nodes_fn(n),
        #     qnet.nlocal_star_22_cost_fn,
        #     opt_kwargs={
        #         "sample_width": 5,
        #         "step_size": 2,
        #         "num_steps": 40,
        #         "verbose": True,
        #     },
        # )
        # psi_plus_local_ry_jobs = client.map(psi_plus_local_ry_opt, param_range)
        # psi_plus_local_ry_opt_dicts = client.gather(psi_plus_local_ry_jobs)

        # src.save_optimizations_one_param_scan(
        #     data_dir,
        #     "psi_plus_local_ry_n-" + str(n) + "_",
        #     param_range,
        #     psi_plus_local_ry_opt_dicts,
        #     quantum_bound=np.sqrt(2),
        #     classical_bound=1,
        # )

        # time_elapsed = time.time() - time_start
        # print("\nelapsed time : ", time_elapsed, "\n")

        # client.restart()

        # """
        # Minimal nonoptimal ansatz for phi plus bell state
        # """
        time_start = time.time()

        phi_plus_local_ry_opt = src.noisy_net_opt_fn(
            src.star_ghz_prep_nodes(n),
            src.star_22_local_ry_meas_nodes(n),
            single_colored_noise_nodes_fn(n),
            qnet.nlocal_star_22_cost_fn,
            opt_kwargs={
                "sample_width": 5,
                "step_size": 2,
                "num_steps": 40,
                "verbose": True,
            },
        )
        phi_plus_local_ry_jobs = client.map(phi_plus_local_ry_opt, param_range)
        phi_plus_local_ry_opt_dicts = client.gather(phi_plus_local_ry_jobs)

        src.save_optimizations_one_param_scan(
            data_dir,
            "phi_plus_local_ry_n-" + str(n) + "_",
            param_range,
            phi_plus_local_ry_opt_dicts,
            quantum_bound=np.sqrt(2),
            classical_bound=1,
        )

        time_elapsed = time.time() - time_start
        print("\nelapsed time : ", time_elapsed, "\n")

        # client.restart()


        # # local qubit rotation measurements and max entangled states
        # time_start = time.time()

        # max_ent_local_rot_opt = src.noisy_net_opt_fn(
        #     src.star_nlocal_max_entangled_prep_nodes(n),
        #     src.star_22_local_rot_meas_nodes(n),
        #     single_colored_noise_nodes_fn(n),
        #     qnet.nlocal_star_22_cost_fn,
        #     opt_kwargs={
        #         "sample_width": 5,
        #         "step_size": 1.8,
        #         "num_steps": 40,
        #         "verbose": True,
        #     },
        # )
        # max_ent_local_rot_jobs = client.map(max_ent_local_rot_opt, param_range)
        # max_ent_local_rot_opt_dicts = client.gather(max_ent_local_rot_jobs)

        # src.save_optimizations_one_param_scan(
        #     data_dir,
        #     "max_entangled_local_rot_n-" + str(n) + "_",
        #     param_range,
        #     max_ent_local_rot_opt_dicts,
        #     quantum_bound=np.sqrt(2),
        #     classical_bound=1,
        # )

        # time_elapsed = time.time() - time_start
        # print("\nelapsed time : ", time_elapsed, "\n")

        # client.restart()

        # # local qubit rotation measurements and arb states
        # time_start = time.time()

        # max_ent_ghz_opt = src.noisy_net_opt_fn(
        #     src.star_nlocal_max_entangled_prep_nodes(n),
        #     src.star_22_ghz_rot_meas_nodes(n),
        #     single_colored_noise_nodes_fn(n),
        #     qnet.nlocal_star_22_cost_fn,
        #     opt_kwargs={
        #         "sample_width": 5,
        #         "step_size": 1.8,
        #         "num_steps": 40,
        #         "verbose": True,
        #     },
        # )
        # max_ent_ghz_jobs = client.map(max_ent_ghz_opt, param_range)
        # max_ent_ghz_opt_dicts = client.gather(max_ent_ghz_jobs)

        # src.save_optimizations_one_param_scan(
        #     data_dir,
        #     "max_entangled_ghz_rot_n-" + str(n) + "_",
        #     param_range,
        #     max_ent_ghz_opt_dicts,
        #     quantum_bound=np.sqrt(2),
        #     classical_bound=1,
        # )

        # time_elapsed = time.time() - time_start
        # print("\nelapsed time : ", time_elapsed, "\n")

        # n_workers = 5 if n == 3 else 3
        # client = Client(processes=True, n_workers=n_workers, threads_per_worker=1)

        # # local qubit rotation measurements and arb states
        # time_start = time.time()

        # arb_opt = src.noisy_net_opt_fn(
        #     src.star_nlocal_arb_prep_nodes(n),
        #     src.star_22_local_rot_meas_nodes(n),
        #     single_colored_noise_nodes_fn(n),
        #     qnet.nlocal_star_22_cost_fn,
        #     opt_kwargs={
        #         "sample_width": 5,
        #         "step_size": 1.8,
        #         "num_steps": 40,
        #         "verbose": True,
        #     },
        # )
        # arb_jobs = client.map(arb_opt, param_range)
        # arb_opt_dicts = client.gather(arb_jobs)

        # src.save_optimizations_one_param_scan(
        #     data_dir,
        #     "arb_local_rot_n-" + str(n) + "_",
        #     param_range,
        #     arb_opt_dicts,
        #     quantum_bound=np.sqrt(2),
        #     classical_bound=1,
        # )

        # time_elapsed = time.time() - time_start
        # print("\nelapsed time : ", time_elapsed, "\n")

        # client.restart()

        # # ghz rotation measurements and arb states
        # time_start = time.time()

        # arb_ghz_opt = src.noisy_net_opt_fn(
        #     src.star_nlocal_arb_prep_nodes(n),
        #     src.star_22_ghz_rot_meas_nodes(n),
        #     single_colored_noise_nodes_fn(n),
        #     qnet.nlocal_star_22_cost_fn,
        #     opt_kwargs={
        #         "sample_width": 5,
        #         "step_size": 1.8,
        #         "num_steps": 40,
        #         "verbose": True,
        #     },
        # )
        # arb_ghz_jobs = client.map(arb_ghz_opt, param_range)
        # arb_ghz_opt_dicts = client.gather(arb_ghz_jobs)

        # src.save_optimizations_one_param_scan(
        #     data_dir,
        #     "arb_ghz_rot_n-" + str(n) + "_",
        #     param_range,
        #     arb_ghz_opt_dicts,
        #     quantum_bound=np.sqrt(2),
        #     classical_bound=1,
        # )

        # time_elapsed = time.time() - time_start
        # print("\nelapsed time : ", time_elapsed, "\n")

from dask.distributed import Client
import time
from pennylane import numpy as np
import pennylane as qml

import qnetvo as qnet
from context import src


"""
This script collects data from noisy n-local chain optimizations.
The considered colored noise applied
to the first source in the chain.

The scan range is gamma in [0,1] with an interval of 0.05.
Chains with n = 3, 4 are considered.

Arbitrary state preparations and measurements are considered along with
local qubit measurements and maximally entangled state preparations.
"""

def single_source_colored_noise_nodes_fn(n):
    def noise_nodes(noise_args):
        return [
            qnet.NoiseNode(
                [0, 1], lambda settings, wires: qnet.colored_noise(noise_args, wires=wires)
            )
        ]

    return noise_nodes


if __name__ == "__main__":

    data_dir = "data/n-chain/single_source_colored_noise/"
    param_range = np.arange(0, 1.01, 0.05)

    for n in [3, 4]:

        client = Client(processes=True, n_workers=5, threads_per_worker=1)


        # """
        # minimal optimal ansatz
        # """
        # time_start = time.time()

        # psi_plus_local_ry_opt = src.noisy_net_opt_fn(
        #     src.chain_psi_plus_prep_nodes(n),
        #     src.chain_local_ry_meas_nodes(n),
        #     single_source_colored_noise_nodes_fn(n),
        #     qnet.nlocal_chain_cost_22,
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


        # """
        # Minimal non-optimal ansatz
        # """
        # time_start = time.time()

        # phi_plus_local_ry_opt = src.noisy_net_opt_fn(
        #     src.chain_ghz_prep_nodes(n),
        #     src.chain_local_ry_meas_nodes(n),
        #     single_source_colored_noise_nodes_fn(n),
        #     qnet.nlocal_chain_cost_22,
        #     opt_kwargs={
        #         "sample_width": 5,
        #         "step_size": 2,
        #         "num_steps": 40,
        #         "verbose": True,
        #     },
        # )
        # phi_plus_local_ry_jobs = client.map(phi_plus_local_ry_opt, param_range)
        # phi_plus_local_ry_opt_dicts = client.gather(phi_plus_local_ry_jobs)

        # src.save_optimizations_one_param_scan(
        #     data_dir,
        #     "phi_plus_local_ry_n-" + str(n) + "_",
        #     param_range,
        #     phi_plus_local_ry_opt_dicts,
        #     quantum_bound=np.sqrt(2),
        #     classical_bound=1,
        # )

        # time_elapsed = time.time() - time_start
        # print("\nelapsed time : ", time_elapsed, "\n")

        # client.restart()

        """
        Minimal non-optimal ansatz
        """
        time_start = time.time()

        phi_plus_local_rot_opt = src.noisy_net_opt_fn(
            src.chain_ghz_prep_nodes(n),
            src.chain_local_rot_meas_nodes(n),
            single_source_colored_noise_nodes_fn(n),
            qnet.nlocal_chain_cost_22,
            opt_kwargs={
                "sample_width": 5,
                "step_size": 2.1,
                "num_steps": 50,
                "verbose": True,
            },
        )
        phi_plus_local_rot_jobs = client.map(phi_plus_local_rot_opt, param_range)
        phi_plus_local_rot_opt_dicts = client.gather(phi_plus_local_rot_jobs)

        src.save_optimizations_one_param_scan(
            data_dir,
            "phi_plus_local_rot_n-" + str(n) + "_",
            param_range,
            phi_plus_local_rot_opt_dicts,
            quantum_bound=np.sqrt(2),
            classical_bound=1,
        )

        time_elapsed = time.time() - time_start
        print("\nelapsed time : ", time_elapsed, "\n")


        # # local qubit rotation measurements and max entangled states
        # time_start = time.time()

        # max_ent_local_rot_opt = src.noisy_net_opt_fn(
        #     src.chain_nlocal_max_entangled_prep_nodes(n),
        #     src.chain_local_rot_meas_nodes(n),
        #     single_source_colored_noise_nodes_fn(n),
        #     qnet.nlocal_chain_cost_22,
        #     ansatz_kwargs={
        #         "dev_kwargs": {
        #             "name": "default.qubit",
        #         },
        #     },
        #     opt_kwargs={
        #         "sample_width": 5,
        #         "step_size": 1.3,
        #         "num_steps": 80,
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

        # arb_local_rot_opt = src.noisy_net_opt_fn(
        #     src.chain_nlocal_arbitrary_prep_nodes(n),
        #     src.chain_local_rot_meas_nodes(n),
        #     single_source_colored_noise_nodes_fn(n),
        #     qnet.nlocal_chain_cost_22,
        #     ansatz_kwargs={
        #         "dev_kwargs": {
        #             "name": "default.qubit",
        #         },
        #     },
        #     opt_kwargs={
        #         "sample_width": 5,
        #         "step_size": 1.4,
        #         "num_steps": 100,
        #         "verbose": True,
        #     },
        # )
        # arb_local_rot_jobs = client.map(arb_local_rot_opt, param_range)
        # arb_local_rot_opt_dicts = client.gather(arb_local_rot_jobs)

        # src.save_optimizations_one_param_scan(
        #     data_dir,
        #     "arb_local_rot_n-" + str(n) + "_",
        #     param_range,
        #     arb_local_rot_opt_dicts,
        #     quantum_bound=np.sqrt(2),
        #     classical_bound=1,
        # )

        # time_elapsed = time.time() - time_start
        # print("\nelapsed time : ", time_elapsed, "\n")

        # client.restart()

        # # local qubit rotation measurements and arb states
        # time_start = time.time()

        # arb_opt = src.noisy_net_opt_fn(
        #     src.chain_nlocal_arbitrary_prep_nodes(n),
        #     src.chain_arb_meas_nodes(n),
        #     single_source_colored_noise_nodes_fn(n),
        #     qnet.nlocal_chain_cost_22,
        #     ansatz_kwargs={
        #         "dev_kwargs": {
        #             "name": "default.qubit",
        #         },
        #     },
        #     opt_kwargs={
        #         "sample_width": 5,
        #         "step_size": 1,
        #         "num_steps": 110,
        #         "verbose": True,
        #     },
        # )
        # arb_jobs = client.map(arb_opt, param_range)
        # arb_opt_dicts = client.gather(arb_jobs)

        # src.save_optimizations_one_param_scan(
        #     data_dir,
        #     "arb_arb_n-" + str(n) + "_",
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

        # max_entangled_opt = src.noisy_net_opt_fn(
        #     src.chain_nlocal_max_entangled_prep_nodes(n),
        #     src.chain_arb_meas_nodes(n),
        #     single_source_colored_noise_nodes_fn(n),
        #     qnet.nlocal_chain_cost_22,
        #     ansatz_kwargs={
        #         "dev_kwargs": {
        #             "name": "default.qubit",
        #         },
        #     },
        #     opt_kwargs={
        #         "sample_width": 5,
        #         "step_size": 1.2,
        #         "num_steps": 100,
        #         "verbose": True,
        #     },
        # )
        # max_entangled_jobs = client.map(max_entangled_opt, param_range)
        # max_entangled_opt_dicts = client.gather(max_entangled_jobs)

        # src.save_optimizations_one_param_scan(
        #     data_dir,
        #     "max_entangled_arb_n-" + str(n) + "_",
        #     param_range,
        #     max_entangled_opt_dicts,
        #     quantum_bound=np.sqrt(2),
        #     classical_bound=1,
        # )

        # time_elapsed = time.time() - time_start
        # print("\nelapsed time : ", time_elapsed, "\n")

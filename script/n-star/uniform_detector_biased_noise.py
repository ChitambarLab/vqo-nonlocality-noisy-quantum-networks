from dask.distributed import Client
import time
from pennylane import numpy as np
import pennylane as qml
import qnetvo as qnet

# local imports
from context import src

"""
This script considers the bilocal scenario with biased noise detectors errors applied
to each detector.

We consider errors where the constant value 0/1 are output with equal probability when
the detector errors occur.
We scan through the values [0,1] for errors on each detector.
"""

if __name__ == "__main__":

    white_noise_error_map = np.array([[0.5, 0.5], [0.5, 0.5]])
    scan_range = np.arange(0, 1.001, 0.05)
    data_filepath = "data/n-star/uniform_detector_biased_noise/"

    for n in [3, 4]:
         # preparing noise parameters for use with dask.distributed
        params_range = np.zeros((n+1, len(scan_range)))
        for i, gamma in enumerate(scan_range):
            params_range[:, i] = [gamma] * (n+1)
        
        """
        # ryrz_cnot local ry optimal ansatz
        """
        client = Client(processes=True, n_workers=5, threads_per_worker=1)

        time_start = time.time()
        ryrz_cnot_local_ry_state_optimization = src.detector_error_opt_fn(
            qnet.NetworkAnsatz(
                src.star_ryrz_cnot_prep_nodes(n),
                src.star_22_local_ry_meas_nodes(n)
            ),
            src.detector_error_star_cost_fn,
            cost_kwargs={
                "error_map": white_noise_error_map,
            },
            opt_kwargs={
                "step_size": 1.4,
                "num_steps": 60,
                "sample_width": 5,
                "verbose": False,
            },
        )

        ryrz_cnot_local_ry_opt_jobs = client.map(ryrz_cnot_local_ry_state_optimization, *params_range)
        ryrz_cnot_local_ry_opt_dicts = client.gather(ryrz_cnot_local_ry_opt_jobs)

        print("optimization time : ", time.time() - time_start)

        src.save_optimizations_one_param_scan(
            data_filepath,
            "ryrz_cnot_local_ry_n-" + str(n) + "_",
            scan_range,
            ryrz_cnot_local_ry_opt_dicts,
            quantum_bound=np.sqrt(2),
            classical_bound=1,
        )

        """
        # ghz local ry optimal ansatz
        """
        client = Client(processes=True, n_workers=5, threads_per_worker=1)

        time_start = time.time()
        ghz_local_ry_state_optimization = src.detector_error_opt_fn(
            qnet.NetworkAnsatz(
                src.star_ghz_prep_nodes(n),
                src.star_22_local_ry_meas_nodes(n)
            ),
            src.detector_error_star_cost_fn,
            cost_kwargs={
                "error_map": white_noise_error_map,
            },
            opt_kwargs={
                "step_size": 1.4,
                "num_steps": 60,
                "sample_width": 5,
                "verbose": False,
            },
        )

        ghz_local_ry_opt_jobs = client.map(ghz_local_ry_state_optimization, *params_range)
        ghz_local_ry_opt_dicts = client.gather(ghz_local_ry_opt_jobs)

        print("optimization time : ", time.time() - time_start)

        src.save_optimizations_one_param_scan(
            data_filepath,
            "ghz_local_ry_n-" + str(n) + "_",
            scan_range,
            ghz_local_ry_opt_dicts,
            quantum_bound=np.sqrt(2),
            classical_bound=1,
        )

        """
        # arb preparations local rot
        """
        time_start = time.time()
        arb_local_rot_state_optimization = src.detector_error_opt_fn(
            qnet.NetworkAnsatz(
                src.star_nlocal_arb_prep_nodes(n),
                src.star_22_local_rot_meas_nodes(n)
            ),
            src.detector_error_star_cost_fn,
            cost_kwargs={
                "error_map": white_noise_error_map,
            },
            opt_kwargs={
                "step_size": 1,
                "num_steps": 60,
                "sample_width": 5,
                "verbose": False,
            },
        )

        arb_local_rot_opt_jobs = client.map(arb_local_rot_state_optimization, *params_range)
        arb_local_rot_opt_dicts = client.gather(arb_local_rot_opt_jobs)

        print("optimization time : ", time.time() - time_start)

        src.save_optimizations_one_param_scan(
            data_filepath,
            "arb_local_rot_n-" + str(n) + "_",
            scan_range,
            arb_local_rot_opt_dicts,
            quantum_bound=np.sqrt(2),
            classical_bound=1,
        )

        if n == 3:

            """
            # max entangled prep arb meas
            """
            client = Client(processes=True, n_workers=5, threads_per_worker=1)

            time_start = time.time()
            max_ent_arb_state_optimization = src.detector_error_opt_fn(
                qnet.NetworkAnsatz(
                    src.star_nlocal_max_entangled_prep_nodes(n),
                    src.star_22_arb_meas_nodes(n)
                ),
                src.detector_error_star_cost_fn,
                cost_kwargs={
                    "error_map": white_noise_error_map,
                },
                opt_kwargs={
                    "step_size": 1,
                    "num_steps":60,
                    "sample_width": 5,
                    "verbose": False,
                },
            )

            max_ent_arb_opt_jobs = client.map(max_ent_arb_state_optimization, *params_range)
            max_ent_arb_opt_dicts = client.gather(max_ent_arb_opt_jobs)

            print("optimization time : ", time.time() - time_start)

            src.save_optimizations_one_param_scan(
                data_filepath,
                "max_ent_arb_n-" + str(n) + "_",
                scan_range,
                max_ent_arb_opt_dicts,
                quantum_bound=np.sqrt(2),
                classical_bound=1,
            )

            """
            arb prep arb meas
            """
            client = Client(processes=True, n_workers=5, threads_per_worker=1)

            time_start = time.time()
            arb_arb_state_optimization = src.detector_error_opt_fn(
                qnet.NetworkAnsatz(
                    src.star_nlocal_arb_prep_nodes(n),
                    src.star_22_arb_meas_nodes(n)
                ),
                src.detector_error_star_cost_fn,
                cost_kwargs={
                    "error_map": white_noise_error_map,
                },
                opt_kwargs={
                    "step_size": 1,
                    "num_steps": 90,
                    "sample_width": 5,
                    "verbose": False,
                },
            )

            arb_arb_opt_jobs = client.map(arb_arb_state_optimization, *params_range)
            arb_arb_opt_dicts = client.gather(arb_arb_opt_jobs)

            print("optimization time : ", time.time() - time_start)

            src.save_optimizations_one_param_scan(
                data_filepath,
                "arb_arb_n-" + str(n) + "_",
                scan_range,
                arb_arb_opt_dicts,
                quantum_bound=np.sqrt(2),
                classical_bound=1,
            )





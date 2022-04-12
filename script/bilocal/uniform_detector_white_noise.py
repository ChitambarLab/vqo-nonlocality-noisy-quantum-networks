from dask.distributed import Client
import time
from pennylane import numpy as np
import pennylane as qml
import qnetvo as qnet

# local imports
from context import src

"""
This script considers the bilocal scenario with white noise detectors errors applied
to each detector.

We consider errors where the constant value 0/1 are output with equal probability when
the detector errors occur.
We scan through the values [0,1] for errors on each detector.
For each pair of error rates (p1,p2), we optimize pure state preparations
and projective measurements for maximal violation of the CHSH scenario.

We consider Bell state preparations and arbitrary state preparations.
"""

if __name__ == "__main__":

    max_ent_prep_nodes = [
        qnet.PrepareNode(1, [0, 1], qnet.max_entangled_state, 3),
        qnet.PrepareNode(1, [2, 3], qnet.max_entangled_state, 3),
    ]
    ghz_prep_nodes = [
        qnet.PrepareNode(1, [0, 1], qnet.ghz_state, 0),
        qnet.PrepareNode(1, [2, 3], qnet.ghz_state, 0),
    ]
    arb_prep_nodes = [
        qnet.PrepareNode(1, [0, 1], qml.ArbitraryStatePreparation, 6),
        qnet.PrepareNode(1, [2, 3], qml.ArbitraryStatePreparation, 6),
    ]

    local_rot_meas_nodes = [
        qnet.MeasureNode(2, 2, [0], src.local_rot, 3),
        qnet.MeasureNode(2, 2, [1, 2], src.local_rot, 6),
        qnet.MeasureNode(2, 2, [3], src.local_rot, 3),
    ]
    local_ry_meas_nodes = [
        qnet.MeasureNode(2, 2, [0], qnet.local_RY, 1),
        qnet.MeasureNode(2, 2, [1, 2], qnet.local_RY, 2),
        qnet.MeasureNode(2, 2, [3], qnet.local_RY, 1),
    ]
    arb_meas_nodes = [
        qnet.MeasureNode(2, 2, [0], src.local_rot, 3),
        qnet.MeasureNode(2, 2, [1, 2], qml.ArbitraryUnitary, 15),
        qnet.MeasureNode(2, 2, [3], src.local_rot, 3),
    ]

    white_noise_error_map = np.array([[0.5, 0.5], [0.5, 0.5]])

    scan_range = np.arange(0, 1.001, 0.05)

    # preparing noise parameters for use with dask.distributed
    params_range = np.zeros((2, len(scan_range) ** 2))
    for i, gamma in enumerate(scan_range):
        params_range[:, i] = [gamma, gamma]

    data_filepath = "data/bilocal/uniform_detector_white_noise/"
    client = Client(processes=True, n_workers=5, threads_per_worker=1)

    """
    # max entangled preparations local rot
    """
    time_start = time.time()
    max_ent_local_rot_state_optimization = src.detector_error_opt_fn(
        qnet.NetworkAnsatz(max_ent_prep_nodes, local_rot_meas_nodes),
        src.detector_error_chain_cost_fn,
        cost_kwargs={
            "error_map": white_noise_error_map,
        },
        opt_kwargs={
            "step_size": 0.6,
            "num_steps": 40,
            "sample_width": 5,
            "verbose": False,
        },
    )

    max_ent_local_rot_opt_jobs = client.map(max_ent_local_rot_state_optimization, *params_range)
    max_ent_local_rot_opt_dicts = client.gather(max_ent_local_rot_opt_jobs)

    print("optimization time : ", time.time() - time_start)

    src.save_optimizations_one_param_scan(
        data_filepath,
        "max_ent_local_rot_",
        scan_range,
        max_ent_local_rot_opt_dicts,
        quantum_bound=np.sqrt(2),
        classical_bound=1,
    )

    """
    # minimal optimal ansatz
    """
    time_start = time.time()
    ghz_local_ry_state_optimization = src.detector_error_opt_fn(
        qnet.NetworkAnsatz(ghz_prep_nodes, local_ry_meas_nodes),
        src.detector_error_chain_cost_fn,
        cost_kwargs={
            "error_map": white_noise_error_map,
        },
        opt_kwargs={
            "step_size": 0.6,
            "num_steps": 40,
            "sample_width": 5,
            "verbose": False,
        },
    )

    ghz_local_ry_opt_jobs = client.map(ghz_local_ry_state_optimization, *params_range)
    ghz_local_ry_opt_dicts = client.gather(ghz_local_ry_opt_jobs)

    print("optimization time : ", time.time() - time_start)

    src.save_optimizations_one_param_scan(
        data_filepath,
        "ghz_local_ry_",
        scan_range,
        ghz_local_ry_opt_dicts,
        quantum_bound=2 * np.sqrt(2),
        classical_bound=2,
    )

    # """
    # # max entangled prep local rot meas
    # """
    # time_start = time.time()
    # max_ent_local_rot_state_optimization = src.detector_error_opt_fn(
    #     qnet.NetworkAnsatz(max_ent_prep_nodes, meas_nodes),
    #     src.detector_error_chsh_cost_fn,
    #     cost_kwargs={
    #         "error_map": white_noise_error_map,
    #     },
    #     opt_kwargs={
    #         "step_size": 0.4,
    #         "num_steps": 50,
    #         "sample_width": 5,
    #         "verbose": False,
    #     },
    # )

    # max_ent_local_rot_opt_jobs = client.map(max_ent_local_rot_state_optimization, *params_range)
    # max_ent_local_rot_opt_dicts = client.gather(max_ent_local_rot_opt_jobs)

    # print("optimization time : ", time.time() - time_start)

    # src.save_optimizations_one_param_scan(
    #     data_filepath,
    #     "max_ent_local_rot_",
    #     scan_range,
    #     max_ent_local_rot_opt_dicts,
    #     quantum_bound=2 * np.sqrt(2),
    #     classical_bound=2,
    # )

    # """
    # arb prep local rot meas
    # """
    # time_start = time.time()
    # arb_local_rot_state_optimization = src.detector_error_opt_fn(
    #     qnet.NetworkAnsatz(arb_prep_nodes, meas_nodes),
    #     src.detector_error_chsh_cost_fn,
    #     cost_kwargs={
    #         "error_map": white_noise_error_map,
    #     },
    #     opt_kwargs={
    #         "step_size": 0.3,
    #         "num_steps": 60,
    #         "sample_width": 5,
    #         "verbose": False,
    #     },
    # )

    # arb_local_rot_opt_jobs = client.map(arb_local_rot_state_optimization, *params_range)
    # arb_local_rot_opt_dicts = client.gather(arb_local_rot_opt_jobs)

    # print("optimization time : ", time.time() - time_start)

    # src.save_optimizations_one_param_scan(
    #     data_filepath,
    #     "arb_local_rot_",
    #     scan_range,
    #     arb_local_rot_opt_dicts,
    #     quantum_bound=2 * np.sqrt(2),
    #     classical_bound=2,
    # )





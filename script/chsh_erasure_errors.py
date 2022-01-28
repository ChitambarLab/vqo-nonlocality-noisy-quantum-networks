from dask.distributed import Client
import time
from pennylane import numpy as np
import pennylane as qml
from context import QNetOptimizer as QNopt

# local imports
import utilities
from detector_error_cost_functions import detector_error_chsh_cost_fn


"""
This script considers the CHSH scenario with detectors errors applied
to each detector.

We consider errors where the constant value 1 is output when the error occurs.
We scan through the values [0,1] for errors on each detector.
For each pair of error rates (p1,p2), we optimize pure state preparations
and projective measurements for maximal violation of the CHSH scenario.

We consider Bell state preparations and arbitrary state preparations.
"""

if __name__ == "__main__":

    bell_prep_nodes = [
        QNopt.PrepareNode(1, [0, 1], QNopt.ghz_state, 0),
    ]

    arb_prep_nodes = [QNopt.PrepareNode(1, [0, 1], qml.ArbitraryStatePreparation, 6)]

    meas_nodes = [
        QNopt.MeasureNode(2, 2, [i], lambda settings, wires: qml.Rot(*settings, wires=wires), 3)
        for i in range(2)
    ]

    scan_range = np.arange(0, 1.001, 0.05)

    # preparing noise parameters for use with dask.distributed
    params_range = np.zeros((2, len(scan_range) ** 2))
    for x_id, p1 in enumerate(scan_range):
        for y_id, p2 in enumerate(scan_range):
            params_range[:, x_id * len(scan_range) + y_id] = [p1, p2]

    data_filepath = "script/data/chsh_erasure_errors/"
    client = Client(processes=True, n_workers=5, threads_per_worker=1)

    # bell state preparations
    time_start = time.time()
    bell_state_optimization = utilities.detector_error_opt_fn(
        QNopt.NetworkAnsatz(bell_prep_nodes, meas_nodes),
        detector_error_chsh_cost_fn,
        opt_kwargs={
            "step_size": 0.7,
            "num_steps": 25,
            "sample_width": 5,
            "verbose": False,
        },
    )

    bell_opt_jobs = client.map(bell_state_optimization, *params_range)
    bell_opt_dicts = client.gather(bell_opt_jobs)

    print("bell state optimization time : ", time.time() - time_start)

    utilities.save_optimizations_two_param_scan(
        data_filepath,
        "bell_state",
        scan_range,
        scan_range,
        bell_opt_dicts,
        quantum_bound=2 * np.sqrt(2),
        classical_bound=2,
    )

    # arbitrary state preparations
    time_start = time.time()
    arb_state_optimization = utilities.detector_error_opt_fn(
        QNopt.NetworkAnsatz(arb_prep_nodes, meas_nodes),
        detector_error_chsh_cost_fn,
        opt_kwargs={
            "step_size": 0.2,
            "num_steps": 50,
            "sample_width": 5,
            "verbose": False,
        },
    )

    arb_opt_jobs = client.map(arb_state_optimization, *params_range)
    arb_opt_dicts = client.gather(arb_opt_jobs)

    print("arb state optimization time : ", time.time() - time_start)

    utilities.save_optimizations_two_param_scan(
        data_filepath,
        "arb_state",
        scan_range,
        scan_range,
        arb_opt_dicts,
        quantum_bound=2 * np.sqrt(2),
        classical_bound=2,
    )

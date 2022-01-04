from dask.distributed import Client
import time
from pennylane import numpy as np
import pennylane as qml
from context import QNetOptimizer as QNopt

import utilities

"""
This script collects data from noisy n-local chain optimizations.

The scan range is gamma in [0,1] with an interval of 0.05.
Chains of length n = 2 and 3 are considered.

An arbitrary maximally entangled state is prepared and measured
with local qubit rotations and a arbitrary two-qubit unitary on the central
measurement node. 
"""

def local_rot(settings, wires):
    qml.Rot(*settings[0:3], wires=wires[0])
    qml.Rot(*settings[3:6], wires=wires[1])

def max_entangled_prep_nodes(n):
    return [
        QNopt.PrepareNode(1, [2 * i, 2 * i + 1], QNopt.max_entangled_state, 3) for i in range(n)
    ]

def local_rot_meas_nodes(n):
    meas_nodes = []
    meas_nodes.append(
        QNopt.MeasureNode(2, 2, [0], lambda settings, wires: qml.Rot(*settings, wires=wires), 3)
    )

    meas_nodes.extend(
        [QNopt.MeasureNode(2, 2, [2 * i + 1, 2 * i + 2], local_rot, 6) for i in range(n - 1)]
    )

    meas_nodes.append(
        QNopt.MeasureNode(
            2, 2, [2 * n - 1], lambda settings, wires: qml.Rot(*settings, wires=wires), 3
        )
    )
    return meas_nodes

def arb_meas_nodes(n):
    meas_nodes = []
    meas_nodes.append(
        QNopt.MeasureNode(2, 2, [0], lambda settings, wires: qml.Rot(*settings, wires=wires), 3)
    )

    meas_nodes.extend(
        [QNopt.MeasureNode(
            2, 2, [2 * i + 1, 2 * i + 2], qml.templates.subroutines.ArbitraryUnitary, 15
        ) for i in range(n - 1)]
    )

    meas_nodes.append(
        QNopt.MeasureNode(
            2, 2, [2 * n - 1], lambda settings, wires: qml.Rot(*settings, wires=wires), 3
        )
    )
    return meas_nodes

def single_qubit_depolarizing_nodes_fn(n, wires=[0]):
    def noise_nodes(noise_args):
        return [
            QNopt.NoiseNode(
                wires, lambda settings, wires: qml.DepolarizingChannel(noise_args, wires=wires[0])
            ),
        ]

    return noise_nodes

if __name__ == "__main__":

    client = Client(processes=True)

    for n in [2, 3]:
        print("n = ", n)
        time_start = time.time()

        param_range = np.arange(0, 1.01, 0.05)

        local_rot_optimization = utilities.noisy_net_opt_fn(
            max_entangled_prep_nodes(n),
            local_rot_meas_nodes(n),
            single_qubit_depolarizing_nodes_fn(n),
            QNopt.nlocal_chain_cost_22,
            opt_kwargs = {
                "sample_width" : 5,
                "step_size" : 0.7,
                "num_steps" : 25,
                "verbose" : False,
            }
        )
        local_rot_jobs = client.map(local_rot_optimization, param_range)
        local_rot_opt_dicts = client.gather(local_rot_jobs)

        utilities.save_optimizations_one_param_scan(
            "script/data/chain_n-local_1-qubit_depolarizing/",
            "max_entangled_local_rot_n-" + str(n) + "_",
            param_range,
            local_rot_opt_dicts,
            quantum_bound=2 / np.sqrt(2),
            classical_bound=1,
        )

        time_elapsed = time.time() - time_start
        print("\nelapsed time : ", time_elapsed, "\n")

        client.restart()

        # arbitrary measurements
        time_start = time.time()
        arb_optimization = utilities.noisy_net_opt_fn(
            max_entangled_prep_nodes(n),
            arb_meas_nodes(n),
            single_qubit_depolarizing_nodes_fn(n),
            QNopt.nlocal_chain_cost_22,
            opt_kwargs = {
                "sample_width" : 5,
                "step_size" : 0.7,
                "num_steps" : 25,
                "verbose" : False,
            }
        )
        arb_jobs = client.map(arb_optimization, param_range)
        arb_opt_dicts = client.gather(arb_jobs)

        utilities.save_optimizations_one_param_scan(
            "script/data/chain_n-local_1-qubit_depolarizing/",
            "max_entangled_arb_n-" + str(n) + "_",
            param_range,
            arb_opt_dicts,
            quantum_bound=2 / np.sqrt(2),
            classical_bound=1,
        )

        time_elapsed = time.time() - time_start
        print("\nelapsed time : ", time_elapsed, "\n")


        client.restart()

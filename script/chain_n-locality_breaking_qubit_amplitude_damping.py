from dask.distributed import Client
import time
from pennylane import numpy as np
import pennylane as qml
from context import QNetOptimizer as QNopt

import utilities
import network_ansatzes as ansatzes

"""
This script collects data from noisy n-local chain optimizations.
The considered noise model is an amplitude damping channel on a
single qubit in the chain.

The scan range is gamma in [0,1] with an interval of 0.05.
Chains of length n = 2 and 3 are considered.

An arbitrary maximally entangled state is prepared and measured
with local qubit rotations and a arbitrary two-qubit unitary on the central
measurement node. 
"""

def single_qubit_amplitude_damping_nodes_fn(n, wires=[0]):
    def noise_nodes(noise_args):
        return [
            QNopt.NoiseNode(
                wires, lambda settings, wires: qml.AmplitudeDamping(noise_args, wires=wires[0])
            ),
        ]

    return noise_nodes

if __name__ == "__main__":

    client = Client(processes=True)

    for n in [2, 3]:
        print("n = ", n)
        param_range = np.arange(0, 1.01, 0.05)

        # local qubit rotation measurements
        time_start = time.time()

        local_rot_optimization = utilities.noisy_net_opt_fn(
            ansatzes.nlocal_max_entangled_prep_nodes(n),
            ansatzes.chain_local_rot_meas_nodes(n),
            single_qubit_amplitude_damping_nodes_fn(n),
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
            "script/data/chain_n-local_1-qubit_amplitude_damping/",
            "max_entangled_local_rot_n-" + str(n) + "_",
            param_range,
            local_rot_opt_dicts,
            quantum_bound=2 / np.sqrt(2),
            classical_bound=1,
        )

        time_elapsed = time.time() - time_start
        print("\nelapsed time : ", time_elapsed, "\n")

        client.restart()

        # Bell basis measurements
        time_start = time.time()
        bell_optimization = utilities.noisy_net_opt_fn(
            ansatzes.nlocal_max_entangled_prep_nodes(n),
            ansatzes.chain_bell_meas_nodes(n),
            single_qubit_amplitude_damping_nodes_fn(n),
            QNopt.nlocal_chain_cost_22,
            opt_kwargs = {
                "sample_width" : 5,
                "step_size" : 0.6,
                "num_steps" : 30,
                "verbose" : False,
            }
        )
        bell_jobs = client.map(bell_optimization, param_range)
        bell_opt_dicts = client.gather(bell_jobs)

        utilities.save_optimizations_one_param_scan(
            "script/data/chain_n-local_1-qubit_amplitude_damping/",
            "max_entangled_bell_n-" + str(n) + "_",
            param_range,
            bell_opt_dicts,
            quantum_bound=2 / np.sqrt(2),
            classical_bound=1,
        )

        time_elapsed = time.time() - time_start
        print("\nelapsed time : ", time_elapsed, "\n")

        client.restart()

        # arbitrary measurements
        time_start = time.time()
        arb_optimization = utilities.noisy_net_opt_fn(
            ansatzes.nlocal_max_entangled_prep_nodes(n),
            ansatzes.chain_arb_meas_nodes(n),
            single_qubit_amplitude_damping_nodes_fn(n),
            QNopt.nlocal_chain_cost_22,
            opt_kwargs = {
                "sample_width" : 5,
                "step_size" : 0.6,
                "num_steps" : 30,
                "verbose" : False,
            }
        )
        arb_jobs = client.map(arb_optimization, param_range)
        arb_opt_dicts = client.gather(arb_jobs)

        utilities.save_optimizations_one_param_scan(
            "script/data/chain_n-local_1-qubit_amplitude_damping/",
            "max_entangled_arb_n-" + str(n) + "_",
            param_range,
            arb_opt_dicts,
            quantum_bound=2 / np.sqrt(2),
            classical_bound=1,
        )

        time_elapsed = time.time() - time_start
        print("\nelapsed time : ", time_elapsed, "\n")

        client.restart()

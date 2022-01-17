from context import QNetOptimizer as QNopt
import utilities

from pennylane import numpy as np
from datetime import datetime
from qiskit import IBMQ
import matplotlib.pyplot as plt
import sys


"""

Performs an optimization of the Bilocal inequality on the IBM
Belem quantum computer.
The ansatz prepares maximally entangled states and optimizes
local RY rotations on each measurement qubit.

Positional Command Line Arguments:

    * [1] tmp_file_name: the file name of the tmp file to iterate
        upon, e.g., "2022-01-09T16-06-44Z.json".
    * [2] num_steps: The number of steps to iterate upon the passed
        in file. This parameter is only used if a file is passed in too.

"""

provider = IBMQ.load_account()

prep_nodes = [
    QNopt.PrepareNode(1, [0, 1], QNopt.ghz_state, 0),
    QNopt.PrepareNode(1, [2, 3], QNopt.ghz_state, 0),
]
meas_nodes = [
    QNopt.MeasureNode(2, 2, [0], QNopt.local_RY, 1),
    QNopt.MeasureNode(2, 2, [1,2], QNopt.local_RY, 2),
    QNopt.MeasureNode(2, 2, [3], QNopt.local_RY, 1),
]

dev_ibm_belem = {
    # "name": "qiskit.ibmq",
    "name" : "default.qubit",
    "shots": 6000,
    # "backend": "ibmq_qasm_simulator",
    # "backend": "ibmq_belem",
    # "provider": provider,
}

ibm_ansatz = QNopt.NetworkAnsatz(prep_nodes, meas_nodes, dev_kwargs=dev_ibm_belem)
cost = QNopt.nlocal_chain_cost_22(ibm_ansatz, parallel=True, diff_method="parameter-shift")
nat_grad = QNopt.parallel_nlocal_chain_grad_fn(ibm_ansatz,
    natural_gradient=True,
    diff_method="parameter-shift"
) 

data_filepath = "script/data/ibm_belem_simple_bilocal_opt_natural_gradient/"

init_opt_dict = QNopt.read_optimization_json(data_filepath + "tmp/" + sys.argv[1]) if len(sys.argv) > 1 else {}

num_steps = 10 if len(sys.argv) <= 2 else int(sys.argv[2])

print("init_opt_dict : ", init_opt_dict)
print("num_steps : ", num_steps)


opt_dict = utilities.hardware_opt(
    cost,
    ibm_ansatz.rand_scenario_settings(),
    num_steps=num_steps,
    step_size=0.3,
    grad_fn=nat_grad,
    tmp_filepath=data_filepath + "tmp/",
    init_opt_dict = init_opt_dict,
)

print(opt_dict)


# evaluating the score for the "theoretical" optimal settings on device
opt_settings = [
    [np.array([[]]),np.array([[]])],  # prep settings
    [
        np.array([[0], [-np.pi / 2]]),
        np.array([[-np.pi / 4,-np.pi / 4], [np.pi / 4,np.pi / 4]]),
        np.array([[0], [-np.pi / 2]]),
    ],  # meas settings
]
opt_dict["theoretical_score"] = -(cost(opt_settings))

print(opt_dict)


# saving data from optimization
datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
filename = data_filepath + datetime_ext

plt.plot(opt_dict["samples"], [np.sqrt(2)] * len(opt_dict["samples"]), label="Quantum Bound")
plt.plot(opt_dict["samples"], [1] * len(opt_dict["samples"]), label="Classical Bound")
plt.plot(opt_dict["samples"], opt_dict["scores"], label="CHSH Optimization")
plt.title("IBM Hardware Parameter-Shift Optimization of CHSH Violation\nwith Simple Ansatz")
plt.ylabel("CHSH Score")
plt.xlabel("Epoch")
plt.legend()
plt.savefig(filename)

QNopt.write_optimization_json(opt_dict, filename)

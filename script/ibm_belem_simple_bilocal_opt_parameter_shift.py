from context import QNetOptimizer as QNopt
import utilities

from pennylane import numpy as np
from datetime import datetime
from qiskit import IBMQ
import matplotlib.pyplot as plt


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

bilocal_ansatz = QNopt.NetworkAnsatz(prep_nodes, meas_nodes)

dev_ibm_belem = {
    "name": "qiskit.ibmq",
    # "name" : "default.qubit",
    "shots": 6000,
    # "backend": "ibmq_qasm_simulator",
    "backend": "ibmq_belem",
    "provider": provider,
}

ibm_ansatz = QNopt.NetworkAnsatz(prep_nodes, meas_nodes, dev_kwargs=dev_ibm_belem)
cost = QNopt.nlocal_chain_cost_22(ibm_ansatz, parallel=True, diff_method="parameter-shift")
par_grad = QNopt.parallel_nlocal_chain_grad_fn(ibm_ansatz, diff_method="parameter-shift") 


data_filepath = "script/data/ibm_belem_simple_bilocal_opt_parameter_shift/"

opt_dict = utilities.hardware_opt(
    cost,
    ibm_ansatz.rand_scenario_settings(),
    num_steps=10,
    step_size=1.5,
    grad_fn=par_grad,
    tmp_filepath=data_filepath + "tmp/",
)

print(opt_dict)


# evaluating the score for the "theoretical" optimal settings
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

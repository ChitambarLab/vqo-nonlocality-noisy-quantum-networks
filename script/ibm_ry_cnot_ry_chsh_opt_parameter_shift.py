from context import qnetvo as qnet
import sys
from pennylane import numpy as np
from datetime import datetime
from qiskit import IBMQ
import matplotlib.pyplot as plt

import network_ansatzes
import utilities



provider = IBMQ.load_account()
provider = IBMQ.get_provider(hub="ibm-q-startup", group="xanadu", project="reservations")


prep_nodes = [qnet.PrepareNode(1, [0, 1], network_ansatzes.ry_cnot, 2)]
meas_nodes = [
    qnet.MeasureNode(2, 2, [0], qnet.local_RY, 1),
    qnet.MeasureNode(2, 2, [1], qnet.local_RY, 1),
]

dev_ibm = {
    # "name": "default.qubit",
    "name": "qiskit.ibmq",
    "shots": 6000,
    # "backend": "ibmq_qasm_simulator",
    "backend": "ibmq_belem",
    "provider": provider,
}

ibm_chsh_ansatz = qnet.NetworkAnsatz(prep_nodes, meas_nodes, dev_kwargs=dev_ibm)

ibm_chsh_cost = qnet.chsh_inequality_cost(
    ibm_chsh_ansatz, parallel=True, diff_method="parameter-shift"
)

par_grad = qnet.parallel_chsh_grad(ibm_chsh_ansatz, diff_method="parameter-shift")

data_filepath = "script/data/ibm_ry_cnot_ry_chsh_opt_parameter_shift/"

init_opt_dict = (
    qnet.read_optimization_json(data_filepath + "tmp/" + sys.argv[1]) if len(sys.argv) > 1 else {}
)

num_steps = 16
curr_step = 0 if len(sys.argv) <= 2 else int(sys.argv[2])

print("init_opt_dict : ", init_opt_dict)
print("num_steps : ", num_steps)


opt_dict = utilities.hardware_opt(
    ibm_chsh_cost,
    ibm_chsh_ansatz.rand_scenario_settings(),
    num_steps=num_steps,
    current_step=curr_step,
    step_size=0.12,
    grad_fn=par_grad,
    tmp_filepath=data_filepath + "tmp/",
    init_opt_dict=init_opt_dict,
)


# evaluating the score for the "theoretical" optimal settings
opt_settings = [
    [np.array([[np.pi/4, 0]])],  # prep settings
    [
        np.array([[0], [np.pi / 2]]),
        np.array([[np.pi/4], [-np.pi / 4]]),
    ],  # meas settings
]
opt_dict["theoretical_score"] = -(ibm_chsh_cost(opt_settings))

opt_dict["device_name"] = dev_ibm["name"]
opt_dict["device_shots"] = dev_ibm["shots"]

print(opt_dict)

# saving data from optimization
datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
filename = data_filepath + datetime_ext

plt.plot(opt_dict["samples"], [2*np.sqrt(2)] * len(opt_dict["samples"]), label="Quantum Bound")
plt.plot(opt_dict["samples"], [2] * len(opt_dict["samples"]), label="Classical Bound")
plt.plot(opt_dict["samples"], opt_dict["scores"], label="CHSH Optimization")
plt.title("IBM Parameter-Shift Optimization of\nCHSH Violation with RY_CNOT_RY")
plt.ylabel("CHSH Score")
plt.xlabel("Epoch")
plt.legend()
plt.savefig(filename)

qnet.write_optimization_json(opt_dict, filename)

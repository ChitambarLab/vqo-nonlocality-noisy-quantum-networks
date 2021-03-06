import qnetvo as qnet
from pennylane import numpy as np
from datetime import datetime
from qiskit import IBMQ
import matplotlib.pyplot as plt


provider = IBMQ.load_account()

prep_nodes = [qnet.PrepareNode(1, [0, 1], qnet.ghz_state, 0)]
meas_nodes = [
    qnet.MeasureNode(2, 2, [0], qnet.local_RY, 1),
    qnet.MeasureNode(2, 2, [1], qnet.local_RY, 1),
]

local_chsh_ansatz = qnet.NetworkAnsatz(prep_nodes, meas_nodes)

dev_ibm_belem = {
    "name": "qiskit.ibmq",
    "shots": 6000,
    # "backend": "ibmq_qasm_simulator",
    "backend": "ibmq_belem",
    "provider": provider,
}

ibm_belem_chsh_ansatz = qnet.NetworkAnsatz(prep_nodes, meas_nodes, dev_kwargs=dev_ibm_belem)

init_settings = local_chsh_ansatz.rand_scenario_settings()

local_chsh_cost = qnet.chsh_inequality_cost(local_chsh_ansatz)
ibm_chsh_cost = qnet.chsh_inequality_cost(
    ibm_belem_chsh_ansatz, parallel=True, diff_method="parameter-shift"
)

par_grad = qnet.parallel_chsh_grad(ibm_belem_chsh_ansatz, diff_method="parameter-shift")


data_filepath = "data/chsh/ibm_hardware_simple_parameter_shift_opt/"

opt_dict = {}
for i in range(16):
    tmp_opt_dict = qnet.gradient_descent(
        ibm_chsh_cost, init_settings, step_size=0.12, num_steps=1, sample_width=1, grad_fn=par_grad
    )

    # aggregate data into optimization dictionary
    if i == 0:
        opt_dict = tmp_opt_dict
    else:
        opt_dict["settings_history"].append(tmp_opt_dict["settings_history"][-1])
        opt_dict["scores"].append(tmp_opt_dict["scores"][-1])
        opt_dict["samples"].append(i + 1)
        opt_dict["step_times"].append(tmp_opt_dict["step_times"][-1])

    # saving data after each optimization step
    tmp_datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
    tmp_filename = data_filepath + "tmp/" + tmp_datetime_ext

    qnet.write_optimization_json(opt_dict, tmp_filename)

    # update initial settings
    init_settings = opt_dict["settings_history"][-1]


# evaluating the score for the "theoretical" optimal settings
opt_settings = [
    [np.array([[]])],  # prep settings
    [np.array([[0], [-np.pi / 2]]), np.array([[-np.pi / 4], [np.pi / 4]])],  # meas settings
]
opt_dict["theoretical_score"] = -(ibm_chsh_cost(opt_settings))


# saving data from optimization
datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
filename = data_filepath + datetime_ext

plt.plot(opt_dict["samples"], [2 * np.sqrt(2)] * len(opt_dict["samples"]), label="Quantum Bound")
plt.plot(opt_dict["samples"], [2] * len(opt_dict["samples"]), label="Classical Bound")
plt.plot(opt_dict["samples"], opt_dict["scores"], label="CHSH Optimization")
plt.title("IBM Hardware Parameter-Shift Optimization of CHSH Violation\nwith Simple Ansatz")
plt.ylabel("CHSH Score")
plt.xlabel("Epoch")
plt.legend()
plt.savefig(filename)

qnet.write_optimization_json(opt_dict, filename)

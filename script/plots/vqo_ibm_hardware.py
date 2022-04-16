import qnetvo as qnet
from context import src

import matplotlib.pyplot as plt
from datetime import datetime
from pennylane import numpy as np

if __name__ == "__main__":

	"""
	Loading CHSH Data
	"""

	chsh_dir = "data/chsh/ibm_hardware_simple_parameter_shift_opt/"
	chsh_files = src.get_data_files(chsh_dir, r".*")
	chsh_opt_dicts = [qnet.read_optimization_json(file) for file in chsh_files]

	print("num chsh optimizations : ", len(chsh_files))

	chsh_ansatz = qnet.NetworkAnsatz([
		    qnet.PrepareNode(1, [0, 1], qnet.ghz_state, 0),
		],
		[
		    qnet.MeasureNode(2, 2, [0], qnet.local_RY, 1),
		    qnet.MeasureNode(2, 2, [1], qnet.local_RY, 1),
		]
	)
	chsh_cost = qnet.chsh_inequality_cost(chsh_ansatz)

	num_chsh_epochs = 17

	chsh_data = src.opt_dicts_mean_stderr(chsh_opt_dicts, num_chsh_epochs)
	chsh_ideal_scores = [
	    -(chsh_cost(chsh_data["opt_settings"][i])) for i in range(num_chsh_epochs)
	]

	"""
	Loading Bilocal Data
	"""

	bilocal_dir = "data/bilocal/ibm_hardware_simple_parameter_shift_opt/"
	bilocal_files = src.get_data_files(bilocal_dir, r".*")
	bilocal_opt_dicts = [qnet.read_optimization_json(file) for file in bilocal_files]

	print("num bilocal optimizations : ", len(bilocal_files))

	bilocal_ansatz = qnet.NetworkAnsatz([
		    qnet.PrepareNode(1, [0, 1], qnet.ghz_state, 0),
		   	qnet.PrepareNode(1, [2, 3], qnet.ghz_state, 0),
		],
		[
		    qnet.MeasureNode(2, 2, [0], qnet.local_RY, 1),
		    qnet.MeasureNode(2, 2, [1, 2], qnet.local_RY, 2),
    		qnet.MeasureNode(2, 2, [3], qnet.local_RY, 1),
		]
	)
	bilocal_cost = qnet.nlocal_chain_cost_22(bilocal_ansatz)

	num_bilocal_epochs = 10

	bilocal_data = src.opt_dicts_mean_stderr(bilocal_opt_dicts, num_bilocal_epochs)
	bilocal_ideal_scores = [
	    -(bilocal_cost(bilocal_data["opt_settings"][i])) for i in range(num_bilocal_epochs)
	]

	"""
	Loading Trilocal Chain Data
	"""

	chain_dir = "data/n-chain/ibm_hardware_simple_trilocal_parameter_shift_opt/"
	chain_files = src.get_data_files(chain_dir, r".*")
	chain_opt_dicts = [qnet.read_optimization_json(file) for file in chain_files]

	print("num chain optimizations : ", len(chain_files))

	chain_ansatz = qnet.NetworkAnsatz([
		    qnet.PrepareNode(1, [0, 1], qnet.ghz_state, 0),
		   	qnet.PrepareNode(1, [2, 3], qnet.ghz_state, 0),
		   	qnet.PrepareNode(1, [4, 5], qnet.ghz_state, 0),
		],
		[
		    qnet.MeasureNode(2, 2, [0], qnet.local_RY, 1),
		    qnet.MeasureNode(2, 2, [1, 2], qnet.local_RY, 2),
		    qnet.MeasureNode(2, 2, [3, 4], qnet.local_RY, 2),
    		qnet.MeasureNode(2, 2, [5], qnet.local_RY, 1),
		]
	)
	chain_cost = qnet.nlocal_chain_cost_22(chain_ansatz)

	num_chain_epochs = 11

	chain_data = src.opt_dicts_mean_stderr(chain_opt_dicts, num_chain_epochs)
	chain_ideal_scores = [
	    -(chain_cost(chain_data["opt_settings"][i])) for i in range(num_chain_epochs)
	]

	"""
	Loading Trilocal Star Data
	"""

	star_dir = "data/n-star/ibm_hardware_simple_trilocal_parameter_shift_opt/"
	star_files = src.get_data_files(star_dir, r".*")
	star_opt_dicts = [qnet.read_optimization_json(file) for file in star_files]

	print("num star optimizations : ", len(star_files))

	star_ansatz = qnet.NetworkAnsatz([
		    qnet.PrepareNode(1, [0, 3], qnet.ghz_state, 0),
		   	qnet.PrepareNode(1, [1, 4], qnet.ghz_state, 0),
		   	qnet.PrepareNode(1, [2, 5], qnet.ghz_state, 0),
		],
		[
		    qnet.MeasureNode(2, 2, [0], qnet.local_RY, 1),
		    qnet.MeasureNode(2, 2, [1], qnet.local_RY, 1),
			qnet.MeasureNode(2, 2, [2], qnet.local_RY, 1),
		    qnet.MeasureNode(2, 2, [3, 4, 5], qnet.local_RY, 3),
		]
	)
	star_cost = qnet.nlocal_star_22_cost_fn(star_ansatz)

	num_star_epochs = 11

	star_data = src.opt_dicts_mean_stderr(star_opt_dicts, num_star_epochs)
	star_ideal_scores = [
	    -(star_cost(star_data["opt_settings"][i])) for i in range(num_star_epochs)
	]

	"""
	Plotting Data
	"""
	fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12,8)) 

	fig.suptitle(r"VQO of Non-$n$-Locality on IBM Quantum Computers", size=24, fontweight="bold")


	axes = [ax1,ax2,ax3,ax4]
	titles = ["CHSH", "Bilocal", "\nTrilocal Chain", "\nTrilocal Star"]
	ylabels = [
		r"Bell Score ($S_{\mathrm{CHSH}}$)",
		r"Bell Score ($S_{\mathrm{Bilocal}}$)",
		r"Bell Score ($S_{3\mathrm{-Chain}}$)",
		r"Bell Score ($S_{3\mathrm{-Star}}$)"
	]
	num_epochs_list = [num_chsh_epochs, num_bilocal_epochs, num_chain_epochs, num_star_epochs]
	quantum_bounds = [2*np.sqrt(2), np.sqrt(2), np.sqrt(2), np.sqrt(2)]
	classical_bounds = [2,1,1,1]
	ideal_data_sets = [chsh_ideal_scores, bilocal_ideal_scores, chain_ideal_scores, star_ideal_scores]
	data_sets = [chsh_data, bilocal_data, chain_data, star_data]
	for i in range(4):
		ax = axes[i]
		num_epochs = num_epochs_list[i]

		ax.plot(range(num_epochs), [quantum_bounds[i]]*num_epochs, "-", linewidth=2, label="Quantum Bound")
		ax.plot(range(num_epochs), [classical_bounds[i]]*num_epochs, "--", linewidth=2, label="Classical Bound")
		ax.plot(range(num_epochs), ideal_data_sets[i], ":d", markersize=8, linewidth=2, label="Max Noiseless Optimized Score")
		ax.plot(range(num_epochs), data_sets[i]["max_scores"], ":o", markersize=7 , linewidth=2, label="Max Optimized Score")
		ax.errorbar(
		    range(num_epochs),
		    data_sets[i]["mean_scores"],
		    data_sets[i]["stderr_scores"],
		    linestyle=":",
		    linewidth=3,
		    label="Mean Optimized Score"
		)
		ax.plot(
		    range(num_epochs),
		    [data_sets[i]["mean_theoretical_score"]] * num_epochs,
		    "-.",
		    label="Mean Theoretical Score",
		    linewidth=2
		)

		ax.set_title(titles[i], size=22)
		ax.set_xlabel("Gradient Descent Step", size=18)
		ax.set_ylabel(ylabels[i], size=18)
		# ax1.set_yticks(yticks)

		if i == 0:
			plt.figlegend(ncol=3, loc="lower center", fontsize=16, bbox_to_anchor=(0,-0.01,1,1,))


	plt.tight_layout()
	fig.subplots_adjust(bottom=0.2)

	datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
	filename =  "simple_ansatzes_" + datetime_ext
	plot_dir = "data/plots/vqo_ibm_hardware/"

	plt.savefig(plot_dir + filename)


	# plt.show()



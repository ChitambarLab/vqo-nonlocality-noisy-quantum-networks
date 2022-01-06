from context import QNetOptimizer as QNopt
from datetime import datetime
from pennylane import numpy as np
import matplotlib.pyplot as plt
import json


def noisy_net_opt_fn(
    prep_nodes,
    meas_nodes,
    noise_nodes_fn,
    cost_fn,
    ansatz_kwargs={},
    cost_kwargs={},
    qnode_kwargs={},
    opt_kwargs={},
):
    """Constructs an ``optimize`` function parameterized by the ``noise_args``, a list
    of arguments describing the amount of noise.

    :param prep_nodes: A list of QNopt.PrepareNode classes for the network ansatz.
    :type prep_nodes: list[PrepareNode]

    :param meas_nodes: A list of QNopt.MeasureNode classes for the network ansatz.
    :type meas_nodes: list[MeasureNode]

    :param noise_nodes_fn: A function for constructing the noise nodes for the ansatz.
                           this function must ``noise_args`` as input.
    :type noise_nodes_fn: function

    :param cost_fn: A cost function factory used to construct an ansatz-specific cost function.
    :type cost_fn: function

    :param ansatz_kwargs: Keyword arguments for the ``QNopt.NetworkAnsatz`` class.
    :type ansatz_kwargs: optional, dictionary

    :param cost_kwargs: Keyword arguments for the ``cost_fn`` factory function.
    :type cost_kwargs: optional, dictionary

    :param qnode_kwargs: Keyword arguments passed to the QNode constructors.
    :type qnode_kwargs: optional, dictionary

    :param opt_kwargs: Keyword arguments for the ``QNopt.gradient_descent`` function.
    :type opt_kwargs: optional, dictionary
    """

    def optimize(noise_args):
        """Constructs and ansatz circuit and cost function for the provided ``noise_args``
        and finds the optimal network settings.
        """
        noise_nodes = noise_nodes_fn(noise_args)

        ansatz = QNopt.NetworkAnsatz(prep_nodes, meas_nodes, noise_nodes, **ansatz_kwargs)
        cost = cost_fn(ansatz, **cost_kwargs, **qnode_kwargs)
        init_settings = ansatz.rand_scenario_settings()

        try:
            opt_dict = QNopt.gradient_descent(cost, init_settings, **opt_kwargs)
        except Exception as err:
            print("An error occurred during gradient descent.")
            print(err)
            opt_dict = {
                "opt_score": np.nan,
                "opt_settings": [[], []],
                "scores": [np.nan],
                "samples": [0],
                "settings_history": [[[], []]],
            }

        print("noise args : ", noise_args)

        print("max score : ", opt_dict["opt_score"])

        return opt_dict

    return optimize


def save_optimizations_one_param_scan(
    data_filepath, opt_name, param_range, opt_dicts, quantum_bound=None, classical_bound=None
):
    json_data = {"noise_params": [], "max_scores": [], "opt_settings": []}

    for i in range(len(param_range)):
        noise_param = float(param_range[i])
        json_data["noise_params"] += [noise_param]

        max_score = max(opt_dicts[i]["scores"])
        max_id = opt_dicts[i]["scores"].index(max_score)
        max_sample = opt_dicts[i]["samples"][max_id]
        opt_settings = opt_dicts[i]["settings_history"][max_id]

        json_data["max_scores"] += [max_score]
        json_data["opt_settings"] += [QNopt.settings_to_list(opt_settings)]

        plt.plot(
            opt_dicts[i]["samples"],
            opt_dicts[i]["scores"],
            "--.",
            label="{:.2f}".format(noise_param),
        )
        plt.plot([max_sample], [max_score], "r*")

    print(json_data["max_scores"])

    datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")

    filename = data_filepath + opt_name + datetime_ext
    with open(filename + ".json", "w") as file:
        file.write(json.dumps(json_data))

    if quantum_bound != None:
        plt.plot(
            opt_dicts[0]["samples"],
            [quantum_bound] * len(opt_dicts[0]["samples"]),
            label="Quantum Bound",
        )

    if classical_bound != None:
        plt.plot(
            opt_dicts[0]["samples"],
            [1] * len(opt_dicts[0]["samples"]),
            label="Classical Bound",
        )

    plt.title(data_filepath + opt_name)
    plt.ylabel("Score")
    plt.xlabel("Epoch")
    plt.legend(ncol=3)
    plt.savefig(filename)
    plt.clf()

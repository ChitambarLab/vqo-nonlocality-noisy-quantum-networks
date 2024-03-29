import qnetvo as qnet
from datetime import datetime
from pennylane import numpy as np
import matplotlib.pyplot as plt

from os import listdir
from os.path import isfile, join
import re
import json


def hardware_opt(
    cost,
    init_settings,
    num_steps=16,
    current_step=0,
    step_size=0.2,
    grad_fn=None,
    tmp_filepath="./",
    init_opt_dict={},
):
    """Performs a gradient descent optimization on quantum hardware.
    Each epoch of the gradient descent is saved as a tmp file in case
    the optimization fails.

    :param cost: The cost function to optimize with respect to.
    :type cost: Function

    :param init_settings: initial settings to seed the optimization with.
    :type init_settings: scenario settings

    :param num_steps: The number of epochs to iterate over.
    :type num_steps: optional, Int, default ``16``

    :param current_step: The starting step of the optimization. This should
                         only be nonzero if an ``init_opt_dict`` is supplied.
    :type current_step: optional, Int, default ``0``

    :param step_size: The distance to travel in the direction of steepest descent.
    :type step_size: optional, Float, default ``0.2``

    :param grad_fn: A custom gradient function for the optimization.
    :type grad_fn: optional, Function, default ``None``

    :param tmp_filepath: A filepath to a tmp directory to save intermediate results.
    :type tmp_file_path: optional, String, default ``"./"``

    :param init_opt_dict: The optimization dictionary used as a warm start.
    :type init_opt_dict: optional, Dictionary, default ``{}``
    """
    warm_start = False if init_opt_dict == {} else True
    opt_dict = init_opt_dict

    settings = opt_dict["settings_history"][-1] if warm_start else init_settings

    for i in range(current_step, num_steps):
        tmp_opt_dict = qnet.gradient_descent(
            cost, settings, step_size=step_size, num_steps=1, sample_width=1, grad_fn=grad_fn,
        )

        # aggregate data into optimization dictionary
        if i == 0 and not (warm_start):
            opt_dict = tmp_opt_dict
        else:
            opt_dict["settings_history"].append(tmp_opt_dict["settings_history"][-1])
            opt_dict["scores"].append(tmp_opt_dict["scores"][-1])
            opt_dict["samples"].append(i + 1)
            opt_dict["step_times"].append(tmp_opt_dict["step_times"][-1])

        # saving data after each optimization step
        tmp_datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        tmp_filename = tmp_filepath + tmp_datetime_ext

        qnet.write_optimization_json(opt_dict, tmp_filename)

        # update initial settings
        settings = opt_dict["settings_history"][-1]

    return opt_dict


def detector_error_opt_fn(
    network_ansatz, cost_fn, cost_kwargs={}, qnode_kwargs={}, opt_kwargs={}, verbose=True,
):
    """Constructs an ansatz-specific ``optimze`` function for cost function
    catered for detector noise.

    :param network_ansatz: Circuit modeling the network scenario.
    :type network_ansatz: qnetvo.NetworkAnsatz

    :param cost_fn: A cost function factory incorporating detector noise.
    :type cost_fn: Function

    :param cost_kwargs: Keyword arguments to pass to ``cost_fn``.
    :type cost_kwargs: Dictionary

    :param qnode_kwargs: Keyword arguments to pass to qnode constructors.
    :type qnode_kwargs: Dictionary

    :param opt_kwargs: Keyword arguments to pass to ``qnetvo.gradient_descent``.
    :type opt_kwargs: Dictionary

    :param verbose: If ``True`` prints out progress.
    :type verbose: Bool

    :returns: An ``optimize(*noise_args)`` function that constructs a detector
              error cost function for the ``network_ansatz`` and ``noise_args``.
    :rtype: Function
    """

    def optimize(*noise_args):
        """Constructs a cost function for the ``network_ansatz``
        and ``noise_args`` descibing detector errors
        """

        cost_kwargs["error_rates"] = noise_args

        cost = cost_fn(network_ansatz, **cost_kwargs, **qnode_kwargs)
        init_settings = network_ansatz.rand_scenario_settings()

        opt_dict = _gradient_descent_wrapper(cost, init_settings, **opt_kwargs)

        if verbose:
            print("noise args : ", noise_args)
            print("max score : ", opt_dict["opt_score"])

        return opt_dict

    return optimize


def noisy_net_opt_fn(
    prep_nodes,
    meas_nodes,
    noise_nodes_fn,
    cost_fn,
    ansatz_kwargs={},
    cost_kwargs={},
    qnode_kwargs={},
    opt_kwargs={},
    verbose=True,
):
    """Constructs an ``optimize`` function parameterized by the ``noise_args``, a list
    of arguments describing the amount of noise.

    :param prep_nodes: A list of qnet.PrepareNode classes for the network ansatz.
    :type prep_nodes: list[PrepareNode]

    :param meas_nodes: A list of qnet.MeasureNode classes for the network ansatz.
    :type meas_nodes: list[MeasureNode]

    :param noise_nodes_fn: A function for constructing the noise nodes for the ansatz.
                           this function must ``noise_args`` as input.
    :type noise_nodes_fn: function

    :param cost_fn: A cost function factory used to construct an ansatz-specific cost function.
    :type cost_fn: function

    :param ansatz_kwargs: Keyword arguments for the ``qnet.NetworkAnsatz`` class.
    :type ansatz_kwargs: optional, dictionary

    :param cost_kwargs: Keyword arguments for the ``cost_fn`` factory function.
    :type cost_kwargs: optional, dictionary

    :param qnode_kwargs: Keyword arguments passed to the QNode constructors.
    :type qnode_kwargs: optional, dictionary

    :param opt_kwargs: Keyword arguments for the ``qnet.gradient_descent`` function.
    :type opt_kwargs: optional, dictionary

    :param verbose: If ``True`` prints out progress.
    :type verbose: Bool

    :returns: An ``optimize(noise_args)`` function that constructs a cost
              function for a noisy network ansatz.
    :rtype: Function
    """

    def optimize(noise_args):
        """Constructs a cost function for the provided ``noise_args``
        and finds the optimal network settings.
        """
        noise_nodes = noise_nodes_fn(noise_args)

        ansatz = qnet.NetworkAnsatz(prep_nodes, meas_nodes, noise_nodes, **ansatz_kwargs)
        cost = cost_fn(ansatz, **cost_kwargs, **qnode_kwargs)
        init_settings = ansatz.rand_scenario_settings()
        # init_settings = ansatz.tf_rand_scenario_settings()

        opt_dict = _gradient_descent_wrapper(cost, init_settings, **opt_kwargs)

        if verbose:
            print("noise args : ", noise_args)
            print("max score : ", opt_dict["opt_score"])

        return opt_dict

    return optimize


def _gradient_descent_wrapper(*opt_args, **opt_kwargs):
    """Wraps ``qnetvo.gradient_descent`` in a try-except block to gracefully
    handle errors during computation.

    This function is called with the same parameters as ``qnetvo.gradient_descent``.
    Optimization errors will result in an empty optimization dictionary.
    """
    try:
        opt_dict = qnet.gradient_descent(*opt_args, **opt_kwargs)
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

    return opt_dict


def save_optimizations_one_param_scan(
    data_filepath, opt_name, param_range, opt_dicts, quantum_bound=None, classical_bound=None
):
    """Saves json data and plots for optimizations scanned over single
    fixed parameter.

    :param data_filepath: The path to which the data is saved.
    :type data_filepath: String

    :param opt_name: A name identifying the particular optimization.
    :type opt_name: String

    :param param_range: The values over which parameter is scanned.
    :type param_range: List[Float]

    :param opt_dicts: The obtained optimization dictionaries.
    :type opt_dicts: List[Dictionary]

    :param quantum_bound: The theoretical quantum bound for the scenario.
                          This is used for context in the plot.
    :type quantum_bound: Optional, Float

    :param classical_bound: The theoretical classical bound for the scenario.
                            This is used for context in the plot.
    :type classical_bound: Optional, Float
    """
    json_data = {"noise_params": [], "max_scores": [], "opt_settings": []}

    for i in range(len(param_range)):
        noise_param = float(param_range[i])
        json_data["noise_params"] += [noise_param]

        max_score = max(opt_dicts[i]["scores"])
        max_id = opt_dicts[i]["scores"].index(max_score)
        max_sample = opt_dicts[i]["samples"][max_id]
        opt_settings = opt_dicts[i]["settings_history"][max_id]

        json_data["max_scores"] += [float(max_score)]
        json_data["opt_settings"] += [qnet.settings_to_list(opt_settings)]

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
            [classical_bound] * len(opt_dicts[0]["samples"]),
            label="Classical Bound",
        )

    plt.title(data_filepath + "\n" + opt_name)
    plt.ylabel("Score")
    plt.xlabel("Epoch")
    plt.legend(ncol=3)
    plt.savefig(filename)
    plt.clf()


def save_optimizations_two_param_scan(
    data_filepath, opt_name, x_range, y_range, opt_dicts, quantum_bound=None, classical_bound=None,
):
    """Saves json data and plots for optimizations scanned over two
    fixed parameters.

    :param data_filepath: The path to which the data is saved.
    :type data_filepath: String

    :param opt_name: A name identifying the particular optimization.
    :type opt_name: String

    :param x_range: The values over which the first parameter is scanned.
    :type x_range: List[Float]

    :param y_range: The values over which the second parameter is scanned.
    :type y_range: List[Float]

    :param opt_dicts: The obtained optimization dictionaries.
    :type opt_dicts: List[Dictionary]

    :param quantum_bound: The theoretical quantum bound for the scenario.
                          This is used for context in the plot.
    :type quantum_bound: Optional, Float

    :param classical_bound: The theoretical classical bound for the scenario.
                            This is used for context in the plot.
    :type classical_bound: Optional, Float
    """
    json_data = {"x_mesh": [[]], "y_mesh": [[]], "max_scores": [], "opt_settings": []}

    x_mesh, y_mesh = np.meshgrid(x_range, y_range)
    json_data["x_mesh"] = x_mesh.tolist()
    json_data["y_mesh"] = y_mesh.tolist()

    for row_id in range(x_mesh.shape[0]):
        json_data["max_scores"].append([])
        json_data["opt_settings"].append([])
        for col_id in range(x_mesh.shape[1]):
            opt_id = col_id * x_mesh.shape[0] + row_id

            max_score = max(opt_dicts[opt_id]["scores"])
            max_id = opt_dicts[opt_id]["scores"].index(max_score)
            max_sample = opt_dicts[opt_id]["samples"][max_id]
            opt_settings = opt_dicts[opt_id]["settings_history"][max_id]

            json_data["max_scores"][row_id] += [float(max_score)]
            json_data["opt_settings"][row_id] += [qnet.settings_to_list(opt_settings)]

            plt.plot(
                opt_dicts[opt_id]["samples"],
                opt_dicts[opt_id]["scores"],
                "--.",
                label="{:.2f}".format(x_mesh[row_id, col_id])
                + ","
                + "{:.2f}".format(y_mesh[row_id, col_id]),
            )
            plt.plot([max_sample], [max_score], "r*")

    datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")

    filename = data_filepath + opt_name + "_" + datetime_ext
    with open(filename + ".json", "w") as file:
        file.write(json.dumps(json_data, indent=2))

    plt.plot(
        opt_dicts[0]["samples"],
        [quantum_bound] * len(opt_dicts[0]["samples"]),
        label="Quantum Bound",
    )
    plt.plot(
        opt_dicts[0]["samples"],
        [classical_bound] * len(opt_dicts[0]["samples"]),
        label="Classical Bound",
    )
    plt.title(data_filepath + "\n" + opt_name)
    plt.ylabel("Score")
    plt.xlabel("Epoch")
    plt.legend(ncol=3)
    plt.savefig(filename)
    plt.clf()


def analyze_data_one_param_scan(data_files):
    """Analyzes the set of data files in aggregate.

    Each file should be in the format produced by the function
    ``save_optimizations_one_param_scan``.

    The data analysis returns a dictionary with the following keys:

    * ``"noise_params"``: The noise parameters in the scan range sorted
        from least to greatest.
    * ``"max_scores"``: The maximum score for each noise parameter over
        all optimizations.
    * ``"mean_scores"``: The average score for each noise parameter over
        all optimizations.
    * ``"std_errs"``: The standard error for each noise parameter over
        all optimizations.
    """
    data_dicts = []
    for filepath in data_files:
        with open(filepath) as file:
            data_dicts.append(json.load(file))

    results = {}
    settings_results = {}

    noise_params = []
    for data_dict in data_dicts:
        for i in range(len(data_dict["noise_params"])):
            if np.isnan(data_dict["max_scores"][i]):
                continue

            noise_key = "{:.3f}".format(data_dict["noise_params"][i])

            if noise_key in results:
                results[noise_key].append(data_dict["max_scores"][i])
                settings_results[noise_key].append(data_dict["opt_settings"][i])

            else:
                results[noise_key] = [data_dict["max_scores"][i]]
                settings_results[noise_key] = [data_dict["opt_settings"][i]]

                noise_params.append(np.round(data_dict["noise_params"][i], 5))

    sorted_noise_params = np.sort(noise_params)
    max_scores = [max(results["{:.3f}".format(noise_param)]) for noise_param in sorted_noise_params]

    mean_scores = [
        np.mean(results["{:.3f}".format(noise_param)], axis=0)
        for noise_param in sorted_noise_params
    ]

    std_errs = [
        np.std(results[noise_key], axis=0) / np.sqrt(len(results[noise_key]))
        for noise_key in ["{:.3f}".format(noise_param) for noise_param in sorted_noise_params]
    ]

    max_score_ids = [
        (results["{:.3f}".format(sorted_noise_params[i])]).index(max_scores[i])
        for i in range(len(noise_params))
    ]

    opt_settings_list = [
        settings_results["{:.3f}".format(sorted_noise_params[i])][max_score_ids[i]]
        for i in range(len(noise_params))
    ]

    return {
        "noise_params": sorted_noise_params,
        "max_scores": max_scores,
        "mean_scores": mean_scores,
        "std_errs": std_errs,
        "opt_settings": opt_settings_list,
    }


def analyze_data_two_param_scan(data_files):
    """Analyzes the set of data files in aggregate. This function
    caters to optimizations scanned over two paramters.

    Each file should be in the format produced by the function
    ``save_optimizations_two_param_scan``.

    The data analysis returns a dictionary with the following keys:

    * ``"x_mesh"``: The mesh matrix for the first noise parameter.
    * ``"y_mesh"``: The mesh matrix for the second noise parameter.
    * ``"max_scores"``: The matrix of maximum scores for each pairing of
                        noise parameters in the scanned range.
    * ``"mean_scores"``: The matrix of average scores for each noise parameter pairing
    * ``"std_errs"``: The matrix of standard errors for each noise parameter pairing.
    """
    data_dicts = []
    for file in data_files:
        with open(file) as f:
            data_dicts.append(json.load(f))

    # the mesh is assumed to be uniform across data files
    x_mesh = np.array(data_dicts[0]["x_mesh"])
    y_mesh = np.array(data_dicts[0]["y_mesh"])

    # aggregating results from different optimizations
    results = [[[] for x in range(x_mesh.shape[1])] for y in range(x_mesh.shape[0])]
    for data_dict in data_dicts:
        for row_id in range(x_mesh.shape[0]):
            for col_id in range(x_mesh.shape[1]):
                if np.isnan(data_dict["max_scores"][row_id][col_id]):
                    continue

                results[row_id][col_id].append(data_dict["max_scores"][row_id][col_id])

    # analyizing aggregated results
    max_scores = [
        [max(results[row_id][col_id]) for col_id in range(x_mesh.shape[1])]
        for row_id in range(x_mesh.shape[0])
    ]
    mean_scores = [
        [np.mean(results[row_id][col_id], axis=0) for col_id in range(x_mesh.shape[1])]
        for row_id in range(x_mesh.shape[0])
    ]
    std_errs = [
        [
            np.std(results[row_id][col_id], axis=0) / np.sqrt(len(results[row_id][col_id]))
            for col_id in range(x_mesh.shape[1])
        ]
        for row_id in range(x_mesh.shape[0])
    ]

    return {
        "x_mesh": x_mesh,
        "y_mesh": y_mesh,
        "max_scores": max_scores,
        "mean_scores": mean_scores,
        "std_errs": std_errs,
        "results": results,
    }


def opt_dicts_mean_stderr(opt_dicts, num_samples=None):
    """Performs an aggregate analysis on a set of optimization dictionaries.

    :param opt_dicts: A set of optimization dictionaries to analyze. These dictionaries
                      are output directly from ``qnetvo.gradient_descent`` function.
                      Each dictionary should contain data for a different run of the
                      same optimization problem.
    :type opt_dicts: List[Dictionary]

    :param num_samples: The number of sampled steps in the optimization. Each sample
                        consists of the score and settings in that particule optimization
                        epoch. If no argument is passed, then all available samples are used.
    :type num_samples: Int, default ``None``

    :returns: A dictionary with the following Keys:
        * ``"max_scores"``: The maximum score in each sampled step.
        * ``"mean_scores"``: The average score in each sampled step.
        * ``"stderr_scores"``: The standard error in each sampled step.
        * ``"opt_settings"``: The optimal settings used to achieve the maximium
                              score in each step.
        * ``"max_theoretical_score"``: The maximal theoretically optimal score.
        * ``"mean_theoretical_score"``: The average theoretically optimal score.
        * ``"theory_stderr"``: The standard error in the theoretical scores.
    :rtype: Dictionary
    """
    if num_samples == None:
        num_samples = opt_dicts[0]["samples"]

    scores_array = np.array([opt_dict["scores"][0:num_samples] for opt_dict in opt_dicts])
    settings_array = [opt_dict["settings_history"][0:num_samples] for opt_dict in opt_dicts]
    theoretical_max_array = np.array([opt_dict["theoretical_score"] for opt_dict in opt_dicts])
    mean_theoretical_score = np.mean(theoretical_max_array)
    max_theoretical_score = np.max(theoretical_max_array)
    theory_stderr = np.std(theoretical_max_array, axis=0, ddof=1) / np.sqrt(
        theoretical_max_array.shape[0]
    )

    scores_mean = np.mean(scores_array, axis=0)
    scores_stderr = np.std(scores_array, axis=0, ddof=1) / np.sqrt(scores_array.shape[1])
    scores_max = np.max(scores_array, axis=0)
    max_ids = np.argmax(scores_array, axis=0)
    max_settings = [settings_array[max_ids[i]][i] for i in range(num_samples)]

    return {
        "max_scores": scores_max,
        "mean_scores": scores_mean,
        "stderr_scores": scores_stderr,
        "opt_settings": max_settings,
        "mean_theoretical_score": mean_theoretical_score,
        "max_theoretical_score": max_theoretical_score,
        "theory_stderr": theory_stderr,
    }


def get_data_files(path, regex):
    """Retrieves all data files that match the ``regex`` in the
    directory specified by ``path``.
    """
    return [
        join(path, f)
        for f in listdir(path)
        if (f.endswith(".json") and isfile(join(path, f)) and bool(re.match(regex, f)))
    ]


def plot_unital_single_and_uniform_max_scores_data(
    fig_title,
    ax_titles,
    noise_params,
    quantum_bound,
    classical_bound,
    single_max_scores,
    single_theoretical_scores,
    single_match_scores,
    uniform_max_scores,
    uniform_theoretical_scores,
    uniform_match_scores,
    data_labels,
    plot_dir,
    legend_labels=["VQO", "Theory", "Match"],
    bottom_padding=0.3,
    fig_height=5,
    ncol_legend=4,
    theory_params=[],
):
    """Plots the noise robustness for single and uniform qubit/source/measurement
    noise models. This method is intended for plotting the max score across many
    different network topologies. The constructed figure is saved in the specified
    plot_dir.

    :param fig_title: The title of the figure.
    :type fig_title: String

    :param ax_titles: A list of two strings for the left and right plots respectively.
    :type ax_titles: List[String]

    :param noise_params: A list of float values ranging from 0 to 1 that described the
                         noise. Will be plotted on the x-axis.
    :type noise_params: List[float]

    :param quantum_bound: The upper quantum bound for the plot.
    :type quantum_bound: Float

    :param classical_bound: The upper classical bound for the plot.
    :type classical_bound: Float

    :param single_max_scores: A list containing the set of max scores for each network to plot.
                            Inner lists are the max score for each noise parameter.
    :type single_max_scores: List[List[Float]]

    :param uniform_max_scores: A list containing the set of max scores for each network to plot.
                            Inner lists are the max score for each noise parameter.
    :type uniform_max_scores: List[List[Float]]

    :param data_labels: The names for the data in each plot.
    :type data_labels: List[String]

    :param plot_dir: The directory to which the data will be plotted.
    :type plt_dir: String

    :param theory_params: Noise params for theoretical data.
    :type theory_params: List
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, fig_height))
    fig.suptitle(fig_title, fontsize=24, fontweight="bold")
    datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
    filename = fig_title + "_max_scores_single_and_uniform_" + datetime_ext

    num_samples = len(noise_params)
    qbound = [quantum_bound] * num_samples
    cbound = [classical_bound] * num_samples

    ax_labels_fontsize = 20
    ax_titles_fontsize = 22
    data_set_markers = ["s", "o", "^", "v", "P", "*"]
    line_colors = ["C2", "C3", "C4", "C5", "C6", "C7"]
    marker_sizes = [8, 7.75, 7.5, 7.25, 7, 6.75, 6.5, 6.5]
    # line_widths = [5.5,5,4.5,4,3.5,3,2.5,2]
    dashes_list = [2, 3, 4, 5, 6, 7, 8, 9]

    if len(theory_params) == 0:
        theory_params = noise_params

    ax_data_sets = [single_max_scores, uniform_max_scores]
    ax_theory_sets = [single_theoretical_scores, uniform_theoretical_scores]
    ax_match_sets = [single_match_scores, uniform_match_scores]

    for i in range(2):

        ax = axes[i]

        (qbound_plot,) = ax.plot(
            noise_params, qbound, linestyle="-", color="C0", linewidth=3, label="Quantum Bound"
        )
        (cbound_plot,) = ax.plot(
            noise_params, cbound, linestyle="-.", color="C1", linewidth=3, label="Classical Bound"
        )

        data_plots = []
        theory_plots = []
        for j in range(len(ax_data_sets[i])):
            theory_set = ax_theory_sets[i][j] if len(ax_theory_sets[i]) != 0 else []

            data_set = ax_data_sets[i][j]

            (data_plt,) = ax.plot(
                noise_params,
                data_set,
                color=line_colors[j],
                linestyle="None",
                marker=data_set_markers[j],
                markerfacecolor="None",
                linewidth=2,
                markersize=marker_sizes[j],
                label=data_labels[j] + " " + legend_labels[0],
            )
            data_plots.append(data_plt)

            if len(theory_set) != 0:
                (theory_plt,) = ax.plot(
                    theory_params,
                    theory_set,
                    color=line_colors[j],
                    linestyle=":",
                    linewidth=2,  # line_widths[j],
                    dashes=(dashes_list[j], 2),
                    alpha=0.8,
                    markersize=8,
                    label=data_labels[j] + " " + legend_labels[1],
                )
                theory_plots.append(theory_plt)

            match_set = ax_match_sets[i][j] if len(ax_match_sets[i]) != 0 else []

            if len(match_set) != 0:
                ax.plot(
                    noise_params,
                    match_set,
                    color=line_colors[j],
                    linestyle="-",
                    linewidth=2,  # line_widths[j],
                    # dashes=(5,dashes_list[j]),
                    alpha=0.6,
                    markersize=8,
                    label=data_labels[j] + " " + legend_labels[2],
                )

        ax.set_title(ax_titles[i], size=ax_titles_fontsize)
        ax.set_xlabel(r"Noise Parameter ($\gamma$)", size=ax_labels_fontsize)

        if i == 0:
            ax.set_ylabel(r"Bell Score ($S_{\mathrm{Bell}}$)", size=ax_labels_fontsize)
            plt.figlegend(
                [qbound_plot, cbound_plot]
                + [(data_plots[j], theory_plots[j]) for j in range(len(ax_data_sets[i]))],
                ["Quantum Bound", "Classical Bound"]
                + [data_labels[j] for j in range(len(ax_data_sets[i]))],
                ncol=ncol_legend,
                loc="lower center",
                fontsize=16,
                bbox_to_anchor=(0, -0.01, 1, 1,),
            )

        for j in range(len(ax_data_sets[i])):
            data_set = ax_data_sets[i][j]
            ax.plot(
                noise_params,
                data_set,
                color=line_colors[j],
                linestyle="None",
                marker="o",
                # markerfacecolor="None",
                linewidth=2,
                markersize=1,
                # label=data_labels[j] + " " + legend_labels[0]
            )

    plt.tight_layout()
    fig.subplots_adjust(bottom=bottom_padding)

    plt.savefig(plot_dir + filename)


def plot_nonunital_single_and_uniform_max_scores_data(
    fig_title,
    ax_titles,
    noise_params,
    quantum_bound,
    classical_bound,
    row1_single_max_scores,
    row1_single_theoretical_scores,
    row1_uniform_max_scores,
    row1_uniform_theoretical_scores,
    row2_single_max_scores,
    row2_single_theoretical_scores,
    row2_uniform_max_scores,
    row2_uniform_theoretical_scores,
    data_labels,
    row_labels,
    plot_dir,
    legend_labels=["VQO", "Theory"],
    bottom_padding=0.25,
    ncol_legend=3,
    theory_params=[],
):
    """Plots the noise robustness for single and uniform qubit/source/measurement
    noise models. This method is intended for plotting the max score across many
    different network topologies. The constructed figure is saved in the specified
    plot_dir.

    :param fig_title: The title of the figure.
    :type fig_title: String

    :param ax_titles: A list of two strings for the left and right plots respectively.
    :type ax_titles: List[String]

    :param noise_params: A list of float values ranging from 0 to 1 that described the
                         noise. Will be plotted on the x-axis.
    :type noise_params: List[float]

    :param quantum_bound: The upper quantum bound for the plot.
    :type quantum_bound: Float

    :param classical_bound: The upper classical bound for the plot.
    :type classical_bound: Float

    :param single_max_scores: A list containing the set of max scores for each network to plot.
                            Inner lists are the max score for each noise parameter.
    :type single_max_scores: List[List[Float]]

    :param uniform_max_scores: A list containing the set of max scores for each network to plot.
                            Inner lists are the max score for each noise parameter.
    :type uniform_max_scores: List[List[Float]]

    :param data_labels: The names for the data in each plot.
    :type data_labels: List[String]

    :param plot_dir: The directory to which the data will be plotted.
    :type plt_dir: String

    :param ncol_legend: The number of columns in the legend.
    :type ncol_legend: Int

    :param theory_params: Noise params for theoretical data.
    :type theory_params: List
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(fig_title, fontsize=24, fontweight="bold")
    datetime_ext = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
    filename = fig_title + "_max_scores_single_and_uniform_" + datetime_ext

    num_samples = len(noise_params)
    qbound = [quantum_bound] * num_samples
    cbound = [classical_bound] * num_samples

    ax_labels_fontsize = 20
    ax_titles_fontsize = 22
    data_set_markers = ["s", "o", "^", "d", "P", "*"]
    line_colors = ["C2", "C3", "C4", "C5", "C6", "C7"]

    ax_data_sets = [
        [row1_single_max_scores, row1_uniform_max_scores],
        [row2_single_max_scores, row2_uniform_max_scores],
    ]
    ax_theory_sets = [
        [row1_single_theoretical_scores, row1_uniform_theoretical_scores],
        [row2_single_theoretical_scores, row2_uniform_theoretical_scores],
    ]

    if len(theory_params) == 0:
        theory_params = noise_params

    marker_sizes = [8, 7.75, 7.5, 7.25, 7, 6.75, 6.5, 6.25]
    # line_widths = [5.5,5,4.5,4,3.5,3,2.5,2]
    dashes_list = [2, 3, 4, 5, 6, 7, 8, 9]

    for row_id in range(2):

        for i in range(2):

            ax = axes[row_id][i]

            (qbound_plt,) = ax.plot(
                noise_params, qbound, linestyle="-", color="C0", linewidth=3, label="Quantum Bound"
            )
            (cbound_plt,) = ax.plot(
                noise_params,
                cbound,
                linestyle="-.",
                color="C1",
                linewidth=3,
                label="Classical Bound",
            )

            data_plts = []
            theory_plts = []
            for j in range(len(ax_data_sets[row_id][i])):
                data_set = ax_data_sets[row_id][i][j]
                (data_plt,) = ax.plot(
                    noise_params,
                    data_set,
                    color=line_colors[j],
                    linestyle="None",
                    marker=data_set_markers[j],
                    linewidth=2,
                    markerfacecolor="None",
                    markersize=marker_sizes[j],
                    label=data_labels[j] + " " + legend_labels[0],
                )

                data_plts.append(data_plt)

                theory_set = (
                    ax_theory_sets[row_id][i][j] if len(ax_theory_sets[row_id][i]) != 0 else []
                )

                (theory_plt,) = ax.plot(
                    theory_params,
                    theory_set,
                    color=line_colors[j],
                    linestyle=":",
                    linewidth=2,  # line_widths[j],
                    dashes=(dashes_list[j], 2),
                    alpha=0.6,
                    label=data_labels[j] + " " + legend_labels[1],
                )

                theory_plts.append(theory_plt)

            if row_id == 0:
                ax.set_title(ax_titles[i], size=ax_titles_fontsize)
            else:
                ax.set_title("\n", size=ax_titles_fontsize)

            if row_id == 1:
                ax.set_xlabel(r"Noise Parameter ($\gamma$)", size=ax_labels_fontsize)

            if i == 0:
                ax.set_ylabel(r"Bell Score ($S_{\mathrm{Bell}}$)", size=ax_labels_fontsize)
                ax.annotate(
                    row_labels[row_id],
                    xy=(-1, 0.5),
                    xytext=(-ax.yaxis.labelpad, 0),
                    xycoords=ax.yaxis.label,
                    textcoords="offset points",
                    size=ax_labels_fontsize,
                    ha="center",
                    va="center",
                    rotation=90,
                    fontweight="bold",
                )

                if row_id == 0:
                    plt.figlegend(
                        [qbound_plt, cbound_plt]
                        + [(data_plts[k], theory_plts[k]) for k in range(len(data_plts))],
                        ["Quantum Bound", "Classical Bound"]
                        + [data_labels[k] for k in range(len(data_plts))],
                        ncol=ncol_legend,
                        loc="lower center",
                        fontsize=16,
                        bbox_to_anchor=(0, -0.01, 1, 1,),
                    )

            for j in range(len(ax_data_sets[row_id][i])):
                data_set = ax_data_sets[row_id][i][j]
                ax.plot(
                    noise_params,
                    data_set,
                    color=line_colors[j],
                    linestyle="None",
                    marker="o",
                    linewidth=2,
                    markersize=2,
                )

    plt.tight_layout()
    fig.subplots_adjust(bottom=bottom_padding)

    plt.savefig(plot_dir + filename)

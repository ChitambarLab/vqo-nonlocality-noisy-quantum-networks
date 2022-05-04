# Supplemental Code for: VQO of Nonlocality in Noisy Quantum Networks

ArXiv: [Variational Quantum Optimization of Nonlocality in Noisy Quantum Networks](arxiv_url).

## Project Structure

* `./script` : This directory contains all scripts for data collection and plot creation. 
		All data is saved to the `./data` directory.

* `./data` : This directory contains all data collected using the scripts. Data is organized
		first by the network topology, then by the script that collected the data. Within each
		folder, script data includes JSON and PNG files. The JSON files contain raw data collected
		during optimizations. The PNG files contain a rough plot of the collected optimization data.
		Each file is named by the ansatz for the optimization and a datetime identifier.

* `./src` : This directory contains helper methods for collecting, writing, reading, analyzing,
		and  plotting data.

## Development Environment

For convenience and reproducibility code should be run using the conda development environment.
The [Anaconda](https://docs.conda.io/projects/conda/en/latest/glossary.html#anaconda-glossary)
distribution of Python ensures a consistent development environment.
Follow the Anaconda [installation instructions](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html#installation) to set up the `conda` command line tool for your
operating system.
The `conda` tool creates the dev environment from the `environment.yml` file.
For more details on how to use `conda` see the [managing environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) page in the `conda` documentation.

To create the dev environment, navigate to the root directory of the `vqo-nonlocality-noisy-quantum-networks` repository and follow these steps.

1. Create the `vqo-nonlocality-dev` conda environment:

```
(base) $ conda env create -f environment.yml
```

2. Activate the `vqo-nonlocality-dev` conda environment:

```
(base) $ conda activate vqo-nonlocality-dev
```

## Code Formatting

All code in this project is autoformatted using [black](https://black.readthedocs.io/en/stable/).
After setting up the development environment, run:

```
(vqo-nonlocality-dev) $ black -l 100 src script
```

# Experiments and Evaluations for `BrainUnit`

This repository contains code and experiments for evaluating the [BrainUnit](https://github.com/chaoming0625/brainunit) framework. The experiments focus on training and visualizing neural network models.

## Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Contributing](#contributing)

## Installation

To install the required dependencies, use `pip`:

```bash
pip install -r requirements.txt
```


## Project Structure

```
├── 01-hh-neurons/
├── 02-mutiscale-network/
│   └── results/
├── 03-hh-fitting/
├── 04-task-training/
│   ├── results/
│   ├── task_training.py
│   └── verification.py
├── archive/
├── .gitignore
├── README.md
└── requirements.txt
```

- `01-hh-neurons/`: Contains experiments related to simulating Hodgkin-Huxley-styled TRN neurons.
- `02-mutiscale-network/`: Contains results related to multiscale network experiments.
- `03-hh-fitting/`: Contains experiments related to fitting Hodgkin-Huxley models.
- `04-task-training/`: Contains the main training script and results for training evidence accumulation tasks.
- `.gitignore`: Specifies files and directories to be ignored by Git.
- `README.md`: This file.
- `requirements.txt`: Lists the dependencies required for the project.

## Usage


For the Hodgkin-Huxley neuron experiments, defined in ``01-hh-neurons/`` directory, to run the script, execute:

```bash
python dendritex-sim.py
```

For the multiscale spiking network experiments, defined in ``02-multiscale-network/`` directory, to run the script, execute:

```bash
python large_scale_COBA_EI-bst.py  # with physical units
python large_scale_COBA_EI-bp.py  # without physical units
```

For the Hodgkin-Huxley model fitting experiments, defined in ``03-hh-fitting/`` directory, to run the script, execute:

```bash 
python brian2_hh_fitting.py  # fitting with brian2
python neuron_fitting_of_hh_model.py  # fitting with dendritex
```


For the cognitive task training task, defined in ``04-task-training/`` directory, to run the training script, execute:

```bash
python task_training.py
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.


## Citation

If you use this code in your research, please consider citing the following paper:

```
```



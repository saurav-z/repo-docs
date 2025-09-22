# dysts: Explore and Analyze Chaotic Systems with Python

**Uncover the complex behavior of chaotic systems with `dysts`, a Python library providing access to a comprehensive collection of 135 dynamical systems.** 

[View the original repository](https://github.com/GilpinLab/dysts)

`dysts` allows researchers and enthusiasts to explore the fascinating world of chaos theory, offering tools for:

*   **Easy Simulation:** Run simulations of various chaotic systems with default or custom parameters.
*   **Precomputed Datasets:** Access pre-calculated time series data for efficient analysis and experimentation.
*   **System Variety:**  Explore a diverse range of 135 continuous-time models, including delay differential equations and discrete maps.
*   **Customization:**  Modify parameters, initial conditions, and integration granularity to tailor simulations to your needs.
*   **Benchmarking:** Utilize included tools for forecasting and training experiments, with benchmarks available in a separate repository.

## Getting Started

### Basic Usage
```python
from dysts.flows import Lorenz

model = Lorenz()
sol = model.make_trajectory(1000)
# plt.plot(sol[:, 0], sol[:, 1])
```
### Modify Model Parameters
```python
model = Lorenz()
model.gamma = 1
model.ic = [0.1, 0.0, 5]
sol = model.make_trajectory(1000)
# plt.plot(sol[:, 0], sol[:, 1])
```
### Generate Ensembles of Trajectories
```python
from dysts.systems import make_trajectory_ensemble

all_out = make_trajectory_ensemble(100, resample=True, pts_per_period=75)
```
### Load Precomputed Datasets
```python
from dysts.datasets import load_dataset

data = load_dataset(subsets="train", data_format="numpy", standardize=True)
```

Further examples and in-depth information can be found in the [demonstrations notebook](demos.ipynb) and the [full API documentation](https://gilpinlab.github.io/dysts/spbuild/html/index.html).

## Key Features

*   **Extensive Model Library:** Access 135 continuous-time models and a module of 10 discrete maps.
*   **Fast Performance:** Utilize `numba` compilation and vectorized operations for efficient simulations.
*   **Data Management:** Load precomputed time series datasets for quick analysis.
*   **Metadata Rich:** Leverage JSON database files containing attractor names, default parameter values, and references.
*   **Flexible Integration:** Control simulation parameters like integration timestep and resampling options.

## Installation

Install `dysts` from PyPI:

```bash
pip install dysts
```

For the latest features or to install precomputed trajectories and benchmark results, see the [Additional Installation Guide](#additional-installation-guide).

## Benchmarks

The benchmarks reported in our publications are stored in a separate [benchmarks repository](https://github.com/williamgilpin/dysts_data/tree/main/dysts_data/benchmarks).

## Precomputed Data

A database of precomputed time series from each system is hosted on [HuggingFace](https://huggingface.co/datasets/williamgilpin/dysts)

## Reference

For more detailed information, please cite the following papers:

> William Gilpin. "Chaos as an interpretable benchmark for forecasting and data-driven modelling" Advances in Neural Information Processing Systems (NeurIPS) 2021 https://arxiv.org/abs/2110.05266

> William Gilpin. "Model scale versus domain knowledge in statistical forecasting of chaotic systems" Physical Review Research 2023 https://arxiv.org/abs/2303.08011

## Additional Installation Guide

To install the latest version from GitHub:

```bash
pip install git+https://github.com/williamgilpin/dysts
```

To install the optional precomputed trajectories and benchmark results:

```bash
pip install git+https://github.com/williamgilpin/dysts_data
```

## Testing

To run the tests:

```bash
python -m unittest discover tests
```

## Acknowledgements

The development of `dysts` builds upon the work of Jürgen Meier, J. C. Sprott, and the `nolds` and `DynamicalSystems.jl` libraries.
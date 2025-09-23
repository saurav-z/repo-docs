# dysts: Explore and Analyze Chaotic Systems

**Unleash the power of chaos with `dysts`, a Python library providing a comprehensive toolkit for exploring and analyzing a wide array of chaotic systems.**

[![Plots of chaotic systems in the collection](dysts/data/logo.png)](https://github.com/GilpinLab/dysts)

`dysts` allows researchers and enthusiasts to easily access, simulate, and analyze a diverse collection of 135 chaotic systems, enabling a deeper understanding of complex dynamics.

## Key Features:

*   **Extensive Collection:** Access and analyze 135 continuous-time chaotic systems and 10 discrete maps.
*   **Easy Simulation:** Simulate chaotic systems with simple Python commands.
*   **Customization:** Modify parameters and initial conditions to explore different behaviors.
*   **Precomputed Datasets:** Load precomputed time series datasets for efficient analysis.
*   **Benchmarking:** Access benchmark data for forecasting and model evaluation.
*   **Numba Compilation:** Benefit from performance improvements using Numba for fast computation.
*   **Comprehensive Documentation:** Detailed API documentation for ease of use.
*   **Integration with the HuggingFace Hub:** Explore precomputed time series datasets on [HuggingFace](https://huggingface.co/datasets/williamgilpin/dysts).

## Quick Start:

### Basic Usage

```python
from dysts.flows import Lorenz

model = Lorenz()
sol = model.make_trajectory(1000)
# plt.plot(sol[:, 0], sol[:, 1])
```

### Modify parameters:

```python
model = Lorenz()
model.gamma = 1
model.ic = [0.1, 0.0, 5]
sol = model.make_trajectory(1000)
# plt.plot(sol[:, 0], sol[:, 1])
```

### Generate ensembles:

```python
from dysts.systems import make_trajectory_ensemble

all_out = make_trajectory_ensemble(100, resample=True, pts_per_period=75)
```

### Load a precomputed dataset:

```python
from dysts.datasets import load_dataset

data = load_dataset(subsets="train", data_format="numpy", standardize=True)
```

For more examples and in-depth explanations, explore the [`demonstrations notebook`](demos.ipynb) and consult the full API documentation [here](https://gilpinlab.github.io/dysts/spbuild/html/index.html).

## Installation:

Install `dysts` from PyPI:

```bash
pip install dysts
```

For more installation options, including the latest features from GitHub and optional datasets, please see the [Additional Installation Guide](#additional-installation-guide) section below.

## Benchmarks:

Explore the benchmarks used in our publications in the separate [benchmarks repository](https://github.com/williamgilpin/dysts_data/tree/main/dysts_data/benchmarks). An overview can be found in [`BENCHMARKS.md`](https://github.com/williamgilpin/dysts_data/tree/main/dysts_data/benchmarks/BENCHMARKS.md).

## Contents:

*   Code to generate benchmark forecasting and training experiments are included in a separate [benchmarks repository](https://github.com/williamgilpin/dysts_data/tree/main/dysts_data/benchmarks)
*   Pre-computed time series with training and test partitions are included in [`data`](dysts/data/)
*   Raw definitions metadata for all chaotic systems are included in the database file [`chaotic_attractors`](dysts/data/chaotic_attractors.json). The Python implementations of differential equations can be found in [`the flows module`](dysts/flows.py)

## Additional Installation Guide:

To obtain the latest version directly from GitHub:

```bash
pip install git+https://github.com/williamgilpin/dysts
```

Run the tests:

```bash
python -m unittest
```

To install the optional precomputed trajectories and benchmark results:

```bash
pip install git+https://github.com/williamgilpin/dysts_data
```

## Implementation Notes:

*   The library currently includes 135 continuous-time models, including delay differential equations, and a module with 10 discrete maps.
*   Dynamical equations are compiled using `numba` for performance where possible, and ensembles of trajectories are vectorized.
*   Metadata such as attractor names, default parameter values, and references are stored in JSON files.
*   The integration timestep (`dt`) is chosen based on the highest significant frequency. The `period` field indicates the dominant timescale in each system. Using `resample=True` ensures trajectories have consistent timescales across models, regardless of the integration timestep.

## Testing:

Run the unit tests using:

```bash
python -m unittest discover tests
```

## Reference:

If you use this code for published work, please cite the following papers:

> William Gilpin. "Chaos as an interpretable benchmark for forecasting and data-driven modelling" Advances in Neural Information Processing Systems (NeurIPS) 2021 https://arxiv.org/abs/2110.05266

> William Gilpin. "Model scale versus domain knowledge in statistical forecasting of chaotic systems" Physical Review Research 2023 https://arxiv.org/abs/2303.08011

## Acknowledgements:

*   Collections of named systems from [J&uuml;rgen Meier](http://www.3d-meier.de/tut19/Seite1.html) and [J. C. Sprott](http://sprott.physics.wisc.edu/sprott.htm) are incorporated.
*   The library [nolds](https://github.com/CSchoel/nolds) is used for several analysis routines.
*   Lyapunov exponent calculation is based on the QR factorization approach by Wolf et al. 1985 and Eckmann et al. 1986, with implementation details adapted from Julia library [DynamicalSystems.jl](https://github.com/JuliaDynamics/DynamicalSystems.jl/)

## Contributing:

We welcome suggestions and contributions! See [`CONTRIBUTING.md`](CONTRIBUTING.md) for more information.

## Original Repo:

[https://github.com/GilpinLab/dysts](https://github.com/GilpinLab/dysts)
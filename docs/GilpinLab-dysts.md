# dysts: Explore and Analyze 135 Chaotic Systems with Ease

**dysts** is a powerful Python library designed for the analysis and exploration of a diverse collection of 135 chaotic systems, offering researchers and enthusiasts a comprehensive toolkit for understanding complex dynamics.

[View the original repository on GitHub](https://github.com/GilpinLab/dysts)

![Plots of chaotic systems in the collection](dysts/data/logo.png)

## Key Features

*   **Extensive Collection:** Access and analyze 135 continuous-time chaotic systems, including delay differential equations, and 10 discrete maps.
*   **Easy-to-Use API:** Quickly import, simulate, and modify system parameters with a straightforward Python interface.
*   **Precomputed Datasets:** Load precomputed time series data for training and testing, accelerating your research.
*   **Efficient Computation:** Benefit from optimized code, including `numba` compilation for performance, and vectorized ensemble calculations.
*   **Resampling Capabilities:** Easily resample trajectories to consistent timescales.
*   **Benchmarking and Validation:** Access a separate benchmarks repository for evaluating forecasting and training experiments.
*   **Comprehensive Documentation:** Explore detailed API documentation and example notebooks to get started.
*   **Open Source & Collaborative:** Benefit from an open-source project with contributions welcome.

## Getting Started

### Basic Usage

```python
from dysts.flows import Lorenz

model = Lorenz()
sol = model.make_trajectory(1000)
# plt.plot(sol[:, 0], sol[:, 1])
```

### Modify Parameters

```python
model = Lorenz()
model.gamma = 1
model.ic = [0.1, 0.0, 5]
sol = model.make_trajectory(1000)
# plt.plot(sol[:, 0], sol[:, 1])
```

### Generate Trajectory Ensembles

```python
from dysts.systems import make_trajectory_ensemble

all_out = make_trajectory_ensemble(100, resample=True, pts_per_period=75)
```

### Load Precomputed Datasets

```python
from dysts.datasets import load_dataset

data = load_dataset(subsets="train", data_format="numpy", standardize=True)
```

For more in-depth examples, consult the [`demos.ipynb` notebook.](demos.ipynb)

## Installation

Install `dysts` from PyPI:

```bash
pip install dysts
```

For the latest features and bug fixes, or to install optional precomputed trajectories and benchmark results, refer to the [Additional Installation Guide](#additional-installation-guide).

## Benchmarks

Benchmark results and experiment code can be found in the separate [benchmarks repository](https://github.com/williamgilpin/dysts_data/tree/main/dysts_data/benchmarks).

## References

If you use this code in your research, please cite the following papers:

> William Gilpin. "Chaos as an interpretable benchmark for forecasting and data-driven modelling" Advances in Neural Information Processing Systems (NeurIPS) 2021 https://arxiv.org/abs/2110.05266

> William Gilpin. "Model scale versus domain knowledge in statistical forecasting of chaotic systems" Physical Review Research 2023 https://arxiv.org/abs/2303.08011

## Contents

*   Code to generate benchmark forecasting and training experiments are included in  a separate [benchmarks repository](https://github.com/williamgilpin/dysts_data/tree/main/dysts_data/benchmarks)
*   Pre-computed time series with training and test partitions are included in [`data`](dysts/data/)
*   The raw definitions metadata for all chaotic systems are included in the database file [`chaotic_attractors`](dysts/data/chaotic_attractors.json). The Python implementations of differential equations can be found in [`the flows module`](dysts/flows.py)

## Additional Installation Guide

To obtain the latest version directly from GitHub:

```bash
pip install git+https://github.com/williamgilpin/dysts
```

To install the optional precomputed trajectories and benchmark results:

```bash
pip install git+https://github.com/williamgilpin/dysts_data
```

## Implementation Notes

*   The library features 135 continuous-time models, including delay differential equations and 10 discrete maps.
*   Dynamical equations are compiled using `numba` for performance.
*   Metadata is stored in parseable JSON files.
*   The integration timestep (`dt`) is chosen based on the highest significant frequency, and trajectories can be resampled for consistent timescales.

## Testing

Run tests using:

```bash
python -m unittest discover tests
```

## Acknowledgements

*   References for the system collections of [Jürgen Meier](http://www.3d-meier.de/tut19/Seite1.html) and [J. C. Sprott](http://sprott.physics.wisc.edu/sprott.htm)
*   The library [nolds](https://github.com/CSchoel/nolds) and the Julia library [DynamicalSystems.jl](https://github.com/JuliaDynamics/DynamicalSystems.jl/) were used and acknowledged.
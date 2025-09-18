# dysts: Explore and Analyze Chaotic Systems

**Unlock the secrets of chaos with `dysts`, a Python library providing a comprehensive toolkit for exploring and analyzing 135+ chaotic systems.**

[![Plots of chaotic systems in the collection](dysts/data/logo.png)](https://github.com/GilpinLab/dysts)

This library provides a robust platform for researchers and enthusiasts to:

*   **Simulate Diverse Chaotic Systems:** Access a wide range of continuous-time models, including delay differential equations, and discrete maps.
*   **Customize and Control:** Modify parameters and initial conditions to explore the behavior of each system.
*   **Generate Time Series Data:** Efficiently create trajectories for analysis, including precomputed datasets.
*   **Utilize Precomputed Datasets:** Quickly load and analyze pre-existing time series data for various systems.
*   **Integrate with Benchmarks and Datasets:** Leverage a separate benchmarks repository and precomputed datasets on Hugging Face.

## Key Features

*   **Extensive Collection:** 135+ continuous-time chaotic systems and 10+ discrete maps.
*   **Easy-to-Use API:** Simple Python interface for model selection, simulation, and data generation.
*   **Optimized Performance:** Utilizes `numba` for fast computation and vectorized ensemble generation.
*   **Precomputed Data:** Access to pre-calculated time series for efficient analysis and benchmarking.
*   **Detailed Documentation:** Comprehensive API documentation and a demonstration notebook to get you started.

## Getting Started

### Installation

Install the `dysts` library using pip:

```bash
pip install dysts
```

For the latest features and bug fixes, install directly from GitHub:

```bash
pip install git+https://github.com/williamgilpin/dysts
```

To install optional precomputed trajectories and benchmark results (a large, static dataset), install directly from GitHub:

```bash
pip install git+https://github.com/williamgilpin/dysts_data
```

### Basic Usage

Here's how to get started:

```python
from dysts.flows import Lorenz

model = Lorenz()
sol = model.make_trajectory(1000)
# plt.plot(sol[:, 0], sol[:, 1])
```

Modify model parameters and re-integrate:

```python
model = Lorenz()
model.gamma = 1
model.ic = [0.1, 0.0, 5]
sol = model.make_trajectory(1000)
# plt.plot(sol[:, 0], sol[:, 1])
```

Generate trajectories for all systems:

```python
from dysts.systems import make_trajectory_ensemble

all_out = make_trajectory_ensemble(100, resample=True, pts_per_period=75)
```

Load precomputed datasets:

```python
from dysts.datasets import load_dataset

data = load_dataset(subsets="train", data_format="numpy", standardize=True)
```

Explore the [demonstrations notebook](demos.ipynb) for more examples.  Find the full API documentation [here](https://gilpinlab.github.io/dysts/spbuild/html/index.html).

## Datasets and Resources

*   **Precomputed Time Series:** Access precomputed time series on [Hugging Face](https://huggingface.co/datasets/williamgilpin/dysts).
*   **Benchmarks Repository:** Explore benchmark experiments in a separate [benchmarks repository](https://github.com/williamgilpin/dysts_data/tree/main/dysts_data/benchmarks).

## Reference

If you use this library for your research, please cite the following publications:

> William Gilpin. "Chaos as an interpretable benchmark for forecasting and data-driven modelling" Advances in Neural Information Processing Systems (NeurIPS) 2021 https://arxiv.org/abs/2110.05266

> William Gilpin. "Model scale versus domain knowledge in statistical forecasting of chaotic systems" Physical Review Research 2023 https://arxiv.org/abs/2303.08011

## Contributing

We welcome contributions! See [`CONTRIBUTING.md`](CONTRIBUTING.md) for guidelines.

## Testing

Run tests with:

```bash
python -m unittest discover tests
```

## Acknowledgements

*   This project leverages existing collections of systems from [J&uuml;rgen Meier](http://www.3d-meier.de/tut19/Seite1.html) and [J. C. Sprott](http://sprott.physics.wisc.edu/sprott.htm).
*   The library makes use of [nolds](https://github.com/CSchoel/nolds) and is based on the QR factorization approach used by [Wolf et al 1985](https://www.sciencedirect.com/science/article/abs/pii/0167278985900119) and [Eckmann et al 1986](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.34.4971).
* Implementation details are adapted from conventions in the Julia library [DynamicalSystems.jl](https://github.com/JuliaDynamics/DynamicalSystems.jl/)

**Explore the chaos. Explore `dysts`.**
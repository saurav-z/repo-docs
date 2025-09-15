# dysts: Explore and Analyze Chaotic Systems with Ease

**Dive into the fascinating world of chaos with `dysts`, a Python library offering a comprehensive toolkit for analyzing hundreds of chaotic systems.**  [Visit the original repository](https://github.com/GilpinLab/dysts)

<p align="center">
  <img src="dysts/data/logo.png" alt="Plots of chaotic systems in the collection" width="400">
</p>

## Key Features

*   **Extensive Collection:** Access and analyze a library of 135 continuous-time chaotic systems and 10 discrete maps.
*   **Easy-to-Use API:** Quickly import models, run simulations, and modify parameters with a user-friendly interface.
*   **Precomputed Data:** Load precomputed time series datasets for efficient analysis and experimentation.
*   **Flexible Integration:** Integrate new trajectories with custom granularity, including resampling options for consistent timescales.
*   **Optimized Performance:** Benefit from `numba` compilation and vectorized ensemble calculations for fast simulations.
*   **Comprehensive Metadata:** Explore system metadata, default parameters, and references stored in parseable JSON files.
*   **Benchmarking Capabilities:** Access pre-computed benchmark results for forecasting and model training experiments in a separate repository.
*   **Detailed Documentation:** Explore the [full API documentation](https://gilpinlab.github.io/dysts/spbuild/html/index.html)

## Quick Start

### Basic Usage

Import a model and run a simulation with default initial conditions and parameter values:

```python
from dysts.flows import Lorenz

model = Lorenz()
sol = model.make_trajectory(1000)
# plt.plot(sol[:, 0], sol[:, 1])
```

Modify a model's parameter values and re-integrate:

```python
model = Lorenz()
model.gamma = 1
model.ic = [0.1, 0.0, 5]
sol = model.make_trajectory(1000)
# plt.plot(sol[:, 0], sol[:, 1])
```

Integrate new trajectories from all 135 chaotic systems with a custom granularity:

```python
from dysts.systems import make_trajectory_ensemble

all_out = make_trajectory_ensemble(100, resample=True, pts_per_period=75)
```

Load a precomputed collection of time series from all 135 chaotic systems:

```python
from dysts.datasets import load_dataset

data = load_dataset(subsets="train", data_format="numpy", standardize=True)
```

## Installation

Install from PyPI:

```bash
pip install dysts
```

For the latest version, including new features and bug fixes, install directly from GitHub:

```bash
pip install git+https://github.com/williamgilpin/dysts
```

To install optional precomputed trajectories and benchmark results (a large, static dataset), install directly from GitHub:

```bash
pip install git+https://github.com/williamgilpin/dysts_data
```

## Benchmarks

The benchmarks reported in our publications are stored in a separate [benchmarks repository](https://github.com/williamgilpin/dysts_data/tree/main/dysts_data/benchmarks). An overview of the contents of the directory can be found in [`BENCHMARKS.md`](https://github.com/williamgilpin/dysts_data/tree/main/dysts_data/benchmarks/BENCHMARKS.md) within that repository, while individual task areas are summarized in corresponding Jupyter Notebooks within the top level of that directory.

## Example Notebooks and Further Exploration

Explore the demonstrations notebook for additional functionality and examples:  [`demos.ipynb`](demos.ipynb)

## References

For more information, or if using this code for published work, please consider citing the papers:

> William Gilpin. "Chaos as an interpretable benchmark for forecasting and data-driven modelling" Advances in Neural Information Processing Systems (NeurIPS) 2021 https://arxiv.org/abs/2110.05266

> William Gilpin. "Model scale versus domain knowledge in statistical forecasting of chaotic systems" Physical Review Research 2023 https://arxiv.org/abs/2303.08011

## Contributing

We welcome contributions and suggestions!  See [`CONTRIBUTING.md`](CONTRIBUTING.md) for details.

## Acknowledgements

*   Two existing collections of named systems can be found on the webpages of [J&uuml;rgen Meier](http://www.3d-meier.de/tut19/Seite1.html) and [J. C. Sprott](http://sprott.physics.wisc.edu/sprott.htm). The current version of `dysts` contains all systems from both collections.
*   Several of the analysis routines (such as calculation of the correlation dimension) use the library [nolds](https://github.com/CSchoel/nolds). If re-using code that depends on `nolds`, please be sure to credit that library and heed its license. The Lyapunov exponent calculation is based on the QR factorization approach used by [Wolf et al 1985](https://www.sciencedirect.com/science/article/abs/pii/0167278985900119) and [Eckmann et al 1986](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.34.4971), with implementation details adapted from conventions in the Julia library [DynamicalSystems.jl](https://github.com/JuliaDynamics/DynamicalSystems.jl/)
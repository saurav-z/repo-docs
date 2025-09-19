# dysts: Explore and Analyze 135 Chaotic Dynamical Systems

**Unlock the secrets of chaos with `dysts`, a Python library providing easy access to a vast collection of pre-defined chaotic systems for analysis, simulation, and research.**  Explore and analyze hundreds of chaotic systems with ease using dysts!  [Check out the original repository](https://github.com/GilpinLab/dysts) for more details.

![Plots of chaotic systems in the collection](dysts/data/logo.png)

## Key Features

*   **Comprehensive Collection:** Access 135 continuous-time chaotic systems and 10 discrete maps, including delay differential equations.
*   **Easy-to-Use API:** Quickly simulate and analyze chaotic systems with simple Python commands.
*   **Precomputed Time Series:** Load precomputed time series data for rapid experimentation and analysis.
*   **Customizable Parameters:** Modify model parameters and initial conditions to explore system behavior.
*   **Efficient Implementation:** Benefit from optimized code using `numba` for fast simulations.
*   **Integration and Resampling:**  Trajectory resampling based on the dominant timescale of each system.
*   **Benchmark Integration:** Code to generate benchmark forecasting and training experiments available.

## Quick Start

### Basic Usage

Import a model and run a simulation with default initial conditions and parameter values
```python
from dysts.flows import Lorenz

model = Lorenz()
sol = model.make_trajectory(1000)
# plt.plot(sol[:, 0], sol[:, 1])
```

### Modify Parameters
Modify a model's parameter values and re-integrate
```python
model = Lorenz()
model.gamma = 1
model.ic = [0.1, 0.0, 5]
sol = model.make_trajectory(1000)
# plt.plot(sol[:, 0], sol[:, 1])
```

### Make Trajectory Ensemble
Integrate new trajectories from all 135 chaotic systems with a custom granularity
```python
from dysts.systems import make_trajectory_ensemble

all_out = make_trajectory_ensemble(100, resample=True, pts_per_period=75)
```

### Load Dataset
Load a precomputed collection of time series from all 135 chaotic systems
```python
from dysts.datasets import load_dataset

data = load_dataset(subsets="train", data_format="numpy", standardize=True)
```

## Resources

*   **Demonstrations:** Explore the capabilities of `dysts` with the [demonstrations notebook](demos.ipynb).
*   **API Documentation:**  Refer to the [full API documentation](https://gilpinlab.github.io/dysts/spbuild/html/index.html) for detailed information.
*   **Precomputed Data:** Access a database of precomputed time series on [Hugging Face](https://huggingface.co/datasets/williamgilpin/dysts)

## Installation

Install the `dysts` package from PyPI:

```bash
pip install dysts
```

Or, to get the latest version, including new features and bug fixes, install directly from GitHub:

```bash
pip install git+https://github.com/williamgilpin/dysts
```

To install the optional precomputed trajectories and benchmark results (a large, static dataset), install directly from GitHub:

```bash
pip install git+https://github.com/williamgilpin/dysts_data
```

### Testing

To run the tests, use the following command:
```
python -m unittest discover tests
```

## Benchmarks

The benchmarks reported in our publications are stored in a separate [benchmarks repository](https://github.com/williamgilpin/dysts_data/tree/main/dysts_data/benchmarks).

An overview of the contents of the directory can be found in [`BENCHMARKS.md`](https://github.com/williamgilpin/dysts_data/tree/main/dysts_data/benchmarks/BENCHMARKS.md) within that repository, while individual task areas are summarized in corresponding Jupyter Notebooks within the top level of that directory.

## References

For in-depth information and citations, please refer to the following publications:

*   William Gilpin. "Chaos as an interpretable benchmark for forecasting and data-driven modelling" Advances in Neural Information Processing Systems (NeurIPS) 2021 https://arxiv.org/abs/2110.05266
*   William Gilpin. "Model scale versus domain knowledge in statistical forecasting of chaotic systems" Physical Review Research 2023 https://arxiv.org/abs/2303.08011

## Acknowledgements

The development of `dysts` is built upon the work of others:

*   Collections of named systems from [Jürgen Meier](http://www.3d-meier.de/tut19/Seite1.html) and [J. C. Sprott](http://sprott.physics.wisc.edu/sprott.htm).
*   The library [nolds](https://github.com/CSchoel/nolds) for several analysis routines.
*   Lyapunov exponent calculation based on the QR factorization approach used by [Wolf et al 1985](https://www.sciencedirect.com/science/article/abs/pii/0167278985900119) and [Eckmann et al 1986](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.34.4971), with implementation details adapted from conventions in the Julia library [DynamicalSystems.jl](https://github.com/JuliaDynamics/DynamicalSystems.jl/)

## Contributing

We welcome contributions and suggestions! See [`CONTRIBUTING.md`](CONTRIBUTING.md) for details.
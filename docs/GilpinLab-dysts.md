# dysts: Explore and Analyze Chaotic Systems

**Unlock the secrets of chaos with `dysts`, a Python library designed for the analysis and simulation of hundreds of dynamical systems.** ([View on GitHub](https://github.com/GilpinLab/dysts))

[![Plots of chaotic systems in the collection](assets/logo.png)](https://github.com/GilpinLab/dysts)

## Key Features

*   **Extensive Collection:** Access and simulate 135+ continuous-time chaotic systems, including delay differential equations, and 10 discrete maps.
*   **Easy-to-Use Interface:** Import models, modify parameters, and generate trajectories with a straightforward Python API.
*   **Precomputed Data:** Access a database of precomputed time series for rapid analysis and experimentation [HuggingFace](https://huggingface.co/datasets/williamgilpin/dysts).
*   **Optimized Performance:** Benefit from `numba` compilation and vectorized ensembles for efficient simulations.
*   **Detailed Metadata:** Explore systems with metadata, including attractor names, default parameters, and references, sourced from JSON database files.
*   **Benchmarking & Forecasting:** Provides tools for forecasting and data-driven modeling tasks, with access to benchmark datasets and example notebooks.

## Installation

Install `dysts` using pip:

```bash
pip install dysts
```

For more options, including installing from GitHub and installing precomputed trajectories, see the [Additional Installation Guide](#additional-installation-guide).

## Basic Usage

Simulate a chaotic system with default parameters:

```python
from dysts.flows import Lorenz

model = Lorenz()
sol = model.make_trajectory(1000)  # (1000, 3)
```

Customize the system's parameters and initial conditions:

```python
model = Lorenz(parameters={"beta": 0.7, "rho": 3, "signma": 0.1}, ic=[0.1, 0.0, 5])
sol = model.make_trajectory(1000)  # (1000, 3)
```

Generate trajectories from multiple systems:

```python
from dysts.systems import make_trajectory_ensemble

all_out = make_trajectory_ensemble(100, resample=True, pts_per_period=75)
```

## Contents

*   **`skewgen` Branch:** Code for generating novel skew-dynamical systems.
*   **Benchmarks Repository:**  Separate repository ([benchmarks repository](https://github.com/williamgilpin/dysts_data/tree/main/dysts_data/benchmarks)) containing code for forecasting and training experiments.
*   **`chaotic_attractors.json`:** Metadata for all chaotic systems, used as default parameters in `dysts/flows.py`.
*   **`discrete_maps.json`:** Metadata for discrete maps.

## Benchmarks

The benchmarks reported in publications are stored in the [benchmarks repository](https://github.com/williamgilpin/dysts_data/tree/main/dysts_data/benchmarks).  An overview of the directory contents is in [`BENCHMARKS.md`](https://github.com/williamgilpin/dysts_data/tree/main/dysts_data/benchmarks/BENCHMARKS.md). Individual task areas are summarized in corresponding Jupyter Notebooks.

## Additional Installation Guide

To obtain the latest features and bug fixes, install directly from GitHub:

```bash
pip install git+https://github.com/williamgilpin/dysts
```

To install the optional precomputed trajectories and benchmark results:

```bash
pip install git+https://github.com/williamgilpin/dysts_data
```

## Testing

The code is tested using the `unittest` framework. Run tests with:

```bash
python -m unittest discover tests
```

## Implementation Notes

*   **Model Diversity:** Includes 135 continuous-time models and 10 discrete maps.
*   **Performance Optimization:**  `numba` is used for compilation of differential equations, with vectorization for ensembles.
*   **Metadata Driven:** Metadata, including default parameters and initial conditions, is stored in JSON files.
*   **Time Scale Consistency:** The `dt` and `period` fields are used to ensure consistent timescales across models, especially when using `resample=True`.

## References

For more information, and if using this code for published work, please cite the following papers:

> William Gilpin. "Chaos as an interpretable benchmark for forecasting and data-driven modelling" Advances in Neural Information Processing Systems (NeurIPS) 2021 https://arxiv.org/abs/2110.05266

> William Gilpin. "Model scale versus domain knowledge in statistical forecasting of chaotic systems" Physical Review Research 2023 https://arxiv.org/abs/2303.08011

## Acknowledgements

*   [Jürgen Meier](http://www.3d-meier.de/tut19/Seite1.html) and [J. C. Sprott](http://sprott.physics.wisc.edu/sprott.htm) for existing collections of named systems.
*   [nolds](https://github.com/CSchoel/nolds) for several analysis routines.
*   [Wolf et al 1985](https://www.sciencedirect.com/science/article/abs/pii/0167278985900119) and [Eckmann et al 1986](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.34.4971), and the Julia library [DynamicalSystems.jl](https://github.com/JuliaDynamics/DynamicalSystems.jl/) for the Lyapunov exponent calculation.
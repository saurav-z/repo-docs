# dysts: Explore and Analyze Chaotic Dynamical Systems

**Unleash the power of chaos theory with `dysts`, a Python library designed for analyzing and simulating hundreds of chaotic systems.**

[Link to original repo: https://github.com/GilpinLab/dysts](https://github.com/GilpinLab/dysts)

## Key Features

*   **Extensive Collection:** Access and analyze 135+ continuous-time chaotic models, including delay differential equations, and 10 discrete maps (with ongoing expansion).
*   **Easy Installation:**  Install with `pip install dysts`. Optional precomputed trajectories and benchmark results can be installed via `pip install git+https://github.com/williamgilpin/dysts_data`.
*   **Flexible Simulation:**
    *   Generate trajectories using default or custom initial conditions and parameter values.
    *   Integrate new trajectories with custom granularity.
    *   Resample trajectories for consistent dominant timescales across different models.
*   **Optimized Performance:** Benefit from `numba` compilation for efficient right-hand-side calculations and vectorized trajectory ensembles.
*   **Comprehensive Metadata:**  Utilize a JSON database (`chaotic_attractors.json` and `discrete_maps.json`) containing attractor names, parameter values, and references for each system.
*   **Precomputed Datasets:** Explore precomputed time series data hosted on [HuggingFace](https://huggingface.co/datasets/williamgilpin/dysts).
*   **Benchmarking Tools:** Leverage code for generating forecasting and training experiments found in the separate [benchmarks repository](https://github.com/williamgilpin/dysts_data/tree/main/dysts_data/benchmarks).
*   **Testing Framework:** Use a simple `unittest` framework to test the code with `python -m unittest discover tests`.

## Installation

Install `dysts` using pip:

```bash
pip install dysts
```

For the latest features and bug fixes:

```bash
pip install git+https://github.com/williamgilpin/dysts
```

To install the optional precomputed trajectories and benchmark results (a large, static dataset), install directly from GitHub

```bash
pip install git+https://github.com/williamgilpin/dysts_data
```

## Basic Usage

Import a model and run a simulation with default initial conditions and parameter values

```python
from dysts.flows import Lorenz

model = Lorenz()
sol = model.make_trajectory(1000)  # (1000, 3)
```

Modify a model's parameter values and re-integrate

```python
model = Lorenz(parameters={"beta": 0.7, "rho": 3, "signma": 0.1}, ic=[0.1, 0.0, 5])
sol = model.make_trajectory(1000)  # (1000, 3)
```

Integrate new trajectories from all 135 chaotic systems with a custom granularity

```python
from dysts.systems import make_trajectory_ensemble

all_out = make_trajectory_ensemble(100, resample=True, pts_per_period=75)
```

## Contents

*   `skewgen` branch: Code to generate novel skew-dynamical systems.
*   [Benchmarks Repository](https://github.com/williamgilpin/dysts_data/tree/main/dysts_data/benchmarks): Code for generating benchmark forecasting and training experiments.
*   `dysts/data/chaotic_attractors.json`: Metadata for all chaotic systems, used as default parameters in `dysts/flows.py`.
*   `dysts/data/discrete_maps.json`: Metadata for discrete maps.

## Benchmarks

The benchmarks reported in our publications are stored in a separate [benchmarks repository](https://github.com/williamgilpin/dysts_data/tree/main/dysts_data/benchmarks). An overview of the contents of the directory can be found in [`BENCHMARKS.md`](https://github.com/williamgilpin/dysts_data/tree/main/dysts_data/benchmarks/BENCHMARKS.md) within that repository, while individual task areas are summarized in corresponding Jupyter Notebooks within the top level of that directory.

## Additional Information

*   Full API documentation: [https://gilpinlab.github.io/dysts/spbuild/html/index.html](https://gilpinlab.github.io/dysts/spbuild/html/index.html)
*   Demonstration notebook: [`demos.ipynb`](demos.ipynb)

## Implementation Notes

*   Dynamical equations are compiled with `numba` where possible.
*   Default integration steps (`dt`) are determined based on the highest significant frequency in the power spectrum, relative to random phase surrogates.  The `period` field stores the dominant timescale.
*   Resampling (`resample=True` in `make_trajectory()`) ensures consistent timescales across models.

## Reference

If using this code for published work, please cite the following papers:

> William Gilpin. "Chaos as an interpretable benchmark for forecasting and data-driven modelling" Advances in Neural Information Processing Systems (NeurIPS) 2021 https://arxiv.org/abs/2110.05266

> William Gilpin. "Model scale versus domain knowledge in statistical forecasting of chaotic systems" Physical Review Research 2023 https://arxiv.org/abs/2303.08011

## Acknowledgements

*   Inspired by the collections of [Jürgen Meier](http://www.3d-meier.de/tut19/Seite1.html) and [J. C. Sprott](http://sprott.physics.wisc.edu/sprott.htm).
*   Uses the `nolds` library for some analysis routines (credit and heed its license).
*   Lyapunov exponent calculation based on the QR factorization approach by Wolf et al. 1985 and Eckmann et al. 1986, with implementation details adapted from DynamicalSystems.jl.
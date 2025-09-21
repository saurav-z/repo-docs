# DYSTS: Explore and Analyze Chaotic Systems with Ease

**DYSTS (Dynamical Systems Toolbox) provides a comprehensive Python library for generating, analyzing, and forecasting a wide range of chaotic systems, offering a powerful toolkit for researchers and enthusiasts alike.** Explore 135 continuous-time models and 10 discrete maps, all readily accessible and customizable.  [Visit the original repository](https://github.com/GilpinLab/dysts) for more details and contributions.

![Plots of chaotic systems in the collection](dysts/data/logo.png)

## Key Features

*   **Extensive Collection:** Access a curated library of 135 continuous-time and 10 discrete chaotic systems, including models from J&uuml;rgen Meier and J. C. Sprott.
*   **Easy-to-Use API:**  Generate trajectories with default or custom parameters and initial conditions using a straightforward Python interface.
*   **Precomputed Datasets:** Load precomputed time series data for all 135 chaotic systems, streamlining your analysis.  Precomputed time series data is available on [Hugging Face](https://huggingface.co/datasets/williamgilpin/dysts).
*   **Flexible Integration:** Control the granularity of your simulations and resample trajectories for consistent timescales.
*   **Optimized Performance:** Benefit from `numba` compilation and vectorized ensemble calculations for efficient analysis.
*   **Benchmarking & Analysis:** Utilize pre-built tools and access benchmarks for forecasting and model evaluation.
*   **Comprehensive Documentation:** Access detailed API documentation and explore demonstrations to get started quickly.
*   **Well-Documented Metadata:** Includes metadata for all systems, including attractor names, parameter values, and references.

## Quickstart

### Basic Usage

Import a model and run a simulation with default initial conditions and parameter values:

```python
from dysts.flows import Lorenz

model = Lorenz()
sol = model.make_trajectory(1000)
# plt.plot(sol[:, 0], sol[:, 1])
```

### Modify Parameters

Modify a model's parameter values and re-integrate:

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

### Load Precomputed Data

Load a precomputed collection of time series from all 135 chaotic systems

```python
from dysts.datasets import load_dataset

data = load_dataset(subsets="train", data_format="numpy", standardize=True)
```

## Installation

Install DYSTS directly from PyPI:

```bash
pip install dysts
```

For the latest features and bug fixes, or to install optional datasets, see the [Additional Installation Guide](#additional-installation-guide) in the original README.

## Additional Resources

*   **API Documentation:** [API Documentation](https://gilpinlab.github.io/dysts/spbuild/html/index.html)
*   **Demonstrations:** [Demos Notebook](demos.ipynb)
*   **Benchmarks Repository:**  [Benchmarks Repository](https://github.com/williamgilpin/dysts_data/tree/main/dysts_data/benchmarks)
*   **Hugging Face Dataset:** [Precomputed time series on Hugging Face](https://huggingface.co/datasets/williamgilpin/dysts)
*   **Contributing:** See [`CONTRIBUTING.md`](CONTRIBUTING.md)

## Reference

For more information, or if using this code for published work, please consider citing the papers:

*   William Gilpin. "Chaos as an interpretable benchmark for forecasting and data-driven modelling" Advances in Neural Information Processing Systems (NeurIPS) 2021 https://arxiv.org/abs/2110.05266
*   William Gilpin. "Model scale versus domain knowledge in statistical forecasting of chaotic systems" Physical Review Research 2023 https://arxiv.org/abs/2303.08011
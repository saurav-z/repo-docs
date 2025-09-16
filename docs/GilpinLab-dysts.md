# dysts: Explore and Analyze Chaotic Dynamical Systems

**dysts is a powerful Python library enabling researchers to analyze, simulate, and generate data from a wide variety of chaotic dynamical systems.**  (Original Repo: [https://github.com/GilpinLab/dysts](https://github.com/GilpinLab/dysts))

![Plots of chaotic systems in the collection](dysts/data/logo.png)

## Key Features:

*   **Extensive Collection:** Access and analyze 135+ continuous-time chaotic systems and 10+ discrete maps, with more being added.
*   **Easy-to-Use Interface:** Quickly import and simulate models using intuitive Python commands.
*   **Customization:** Modify parameters and initial conditions to explore system behavior.
*   **Precomputed Datasets:** Load precomputed time series datasets for training and analysis.
*   **Efficient Computation:** Optimized with `numba` for fast simulation and vectorized ensemble generation.
*   **Benchmarking:** Includes a separate repository for forecasting and training experiments benchmarks.
*   **Comprehensive Documentation:** Full API documentation available [here](https://gilpinlab.github.io/dysts/spbuild/html/index.html).
*   **Open Source:**  Actively maintained and open to community contributions.

## Getting Started

### Installation

Install the `dysts` library from PyPI:

```bash
pip install dysts
```

For the latest features and bug fixes, install directly from GitHub:

```bash
pip install git+https://github.com/williamgilpin/dysts
```

To install optional precomputed trajectories and benchmark results, install:

```bash
pip install git+https://github.com/williamgilpin/dysts_data
```

### Basic Usage

Here's a quick example of how to use `dysts`:

```python
from dysts.flows import Lorenz

# Create a Lorenz model instance
model = Lorenz()

# Generate a trajectory
sol = model.make_trajectory(1000)
# plt.plot(sol[:, 0], sol[:, 1]) # Visualize the trajectory (uncomment to plot)
```

You can modify parameters:

```python
model = Lorenz()
model.gamma = 1
model.ic = [0.1, 0.0, 5]
sol = model.make_trajectory(1000)
# plt.plot(sol[:, 0], sol[:, 1])
```

Or generate an ensemble of trajectories:

```python
from dysts.systems import make_trajectory_ensemble
all_out = make_trajectory_ensemble(100, resample=True, pts_per_period=75)
```

Load precomputed datasets:

```python
from dysts.datasets import load_dataset
data = load_dataset(subsets="train", data_format="numpy", standardize=True)
```

Find additional examples and demos in [`the demonstrations notebook.`](demos.ipynb)

## Precomputed Datasets

Precomputed time series data are available on [HuggingFace](https://huggingface.co/datasets/williamgilpin/dysts).

## Benchmarks

The benchmarks reported in the publications are stored in a separate [benchmarks repository](https://github.com/williamgilpin/dysts_data/tree/main/dysts_data/benchmarks).

## Contents

*   Code to generate benchmark forecasting and training experiments are included in a separate [benchmarks repository](https://github.com/williamgilpin/dysts_data/tree/main/dysts_data/benchmarks)
*   Pre-computed time series with training and test partitions are included in [`data`](dysts/data/)
*   The raw definitions metadata for all chaotic systems are included in the database file [`chaotic_attractors`](dysts/data/chaotic_attractors.json). The Python implementations of differential equations can be found in [`the flows module`](dysts/flows.py)

## Implementation Details

*   Dynamical equations are compiled with `numba` for performance.
*   Metadata (attractor names, default parameters, references) are stored in JSON files.
*   The default integration timestep (`dt`) is chosen based on the dominant frequency, with resampling options available.

## Testing

Run tests with:

```bash
python -m unittest discover tests
```

## References

For further information and citation, please refer to the following publications:

*   William Gilpin. "Chaos as an interpretable benchmark for forecasting and data-driven modelling" Advances in Neural Information Processing Systems (NeurIPS) 2021 https://arxiv.org/abs/2110.05266
*   William Gilpin. "Model scale versus domain knowledge in statistical forecasting of chaotic systems" Physical Review Research 2023 https://arxiv.org/abs/2303.08011

## Acknowledgements

This project builds upon the work of J&uuml;rgen Meier and J. C. Sprott, and utilizes the [nolds](https://github.com/CSchoel/nolds) library. The Lyapunov exponent calculation is based on the QR factorization approach used by [Wolf et al 1985](https://www.sciencedirect.com/science/article/abs/pii/0167278985900119) and [Eckmann et al 1986](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.34.4971), with implementation details adapted from conventions in the Julia library [DynamicalSystems.jl](https://github.com/JuliaDynamics/DynamicalSystems.jl/)

## Contributing

Contributions are welcome! See [`CONTRIBUTING.md`](CONTRIBUTING.md) for details.
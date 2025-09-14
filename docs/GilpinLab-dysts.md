# DYSTS: Your Gateway to Exploring Chaotic Systems

**DYSTS is a Python library providing easy access to a diverse collection of 135+ chaotic systems, empowering researchers and enthusiasts to analyze and experiment with complex dynamics.** ([Original Repository](https://github.com/GilpinLab/dysts))

![Plots of chaotic systems in the collection](dysts/data/logo.png)

## Key Features

*   **Extensive Collection:** Access a vast library of 135+ continuous-time models, including delay differential equations, and 10 discrete maps.
*   **Simplified Usage:** Easily import models, modify parameters, and generate trajectories with intuitive Python code.
*   **Precomputed Datasets:** Load precomputed time series datasets for rapid analysis and experimentation.
*   **Optimized Performance:** Benefit from `numba` compilation and vectorized ensemble trajectory generation for efficient computations.
*   **Rich Metadata:** Explore detailed metadata, including attractor names, default parameter values, and references, stored in parseable JSON files.
*   **Flexible Integration:** Utilize customizable integration parameters and resampling options for detailed analysis.
*   **Benchmarking & Data:** Access code to generate benchmark forecasting and training experiments via a separate [benchmarks repository](https://github.com/williamgilpin/dysts_data/tree/main/dysts_data/benchmarks) and the pre-computed data in the [`data`](dysts/data/) directory.

## Basic Usage

### Import a model and run a simulation
```python
from dysts.flows import Lorenz

model = Lorenz()
sol = model.make_trajectory(1000)
# plt.plot(sol[:, 0], sol[:, 1])
```

### Modify a model's parameter values and re-integrate
```python
model = Lorenz()
model.gamma = 1
model.ic = [0.1, 0.0, 5]
sol = model.make_trajectory(1000)
# plt.plot(sol[:, 0], sol[:, 1])
```

### Integrate new trajectories from all chaotic systems with a custom granularity
```python
from dysts.systems import make_trajectory_ensemble

all_out = make_trajectory_ensemble(100, resample=True, pts_per_period=75)
```

### Load a precomputed collection of time series
```python
from dysts.datasets import load_dataset

data = load_dataset(subsets="train", data_format="numpy", standardize=True)
```

## Installation

Install DYSTS from PyPI:

```bash
pip install dysts
```

For the latest features or to install precomputed trajectories and benchmark results, install directly from GitHub:

```bash
# Latest version
pip install git+https://github.com/williamgilpin/dysts

# Optional precomputed trajectories and benchmark results (large dataset)
pip install git+https://github.com/williamgilpin/dysts_data
```

## Resources

*   **API Documentation:** [https://gilpinlab.github.io/dysts/spbuild/html/index.html](https://gilpinlab.github.io/dysts/spbuild/html/index.html)
*   **Demonstrations Notebook:** [`demos.ipynb`](demos.ipynb)
*   **Precomputed Time Series:** [HuggingFace Dataset](https://huggingface.co/datasets/williamgilpin/dysts)
*   **Benchmarks:** [Benchmarks Repository](https://github.com/williamgilpin/dysts_data/tree/main/dysts_data/benchmarks)

## Reference

If using this code for published work, please cite the following papers:

*   William Gilpin. "Chaos as an interpretable benchmark for forecasting and data-driven modelling" Advances in Neural Information Processing Systems (NeurIPS) 2021 https://arxiv.org/abs/2110.05266
*   William Gilpin. "Model scale versus domain knowledge in statistical forecasting of chaotic systems" Physical Review Research 2023 https://arxiv.org/abs/2303.08011

## Contributing

We welcome suggestions and contributions! See [`CONTRIBUTING.md`](CONTRIBUTING.md) for details.

## Testing

Run tests using:

```bash
python -m unittest discover tests
```

## Acknowledgements

DYSTS is built upon the work of others. We are grateful for the contributions of:

*   Jürgen Meier and J. C. Sprott for their collections of named systems.
*   The `nolds` library for several of the analysis routines.
*   The QR factorization approach used by Wolf et al 1985 and Eckmann et al 1986, with implementation details adapted from conventions in the Julia library DynamicalSystems.jl.
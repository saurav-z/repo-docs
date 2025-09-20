# dysts: Explore and Analyze Hundreds of Chaotic Dynamical Systems

**dysts** is a Python library providing access to a vast collection of 135 chaotic systems, enabling researchers and enthusiasts to easily explore, analyze, and generate data from these complex mathematical models.  [View the original repository here](https://github.com/GilpinLab/dysts).

![Plots of chaotic systems in the collection](dysts/data/logo.png)

## Key Features:

*   **Extensive Collection:** Access and analyze 135 continuous-time chaotic systems and 10 discrete maps, including delay differential equations.
*   **Easy-to-Use Interface:** Simple Python interface for importing models, running simulations, and modifying parameters.
*   **Precomputed Datasets:** Load precomputed time series data for all 135 chaotic systems for fast experimentation and analysis.
*   **Efficient Computation:** Optimized with `numba` and vectorized ensembles for fast trajectory generation.
*   **Detailed Metadata:** Access attractor names, default parameter values, references, and metadata stored in parseable JSON files.
*   **Benchmarking Capabilities:** Integrate with a separate repository for benchmarking forecasting and training experiments.
*   **Integration and Resampling:** Default integration timestep (dt) chosen based on the highest significant frequency observed in the power spectrum. `model.make_trajectory()` method with `resample=True` to resample based on the dominant frequency (period).

## Quickstart:

```python
from dysts.flows import Lorenz

model = Lorenz()
sol = model.make_trajectory(1000)
# plt.plot(sol[:, 0], sol[:, 1])
```

Modify parameters and re-integrate:

```python
model = Lorenz()
model.gamma = 1
model.ic = [0.1, 0.0, 5]
sol = model.make_trajectory(1000)
# plt.plot(sol[:, 0], sol[:, 1])
```

Integrate new trajectories from all 135 chaotic systems:

```python
from dysts.systems import make_trajectory_ensemble

all_out = make_trajectory_ensemble(100, resample=True, pts_per_period=75)
```

Load precomputed time series from all 135 chaotic systems:

```python
from dysts.datasets import load_dataset

data = load_dataset(subsets="train", data_format="numpy", standardize=True)
```

## Installation:

Install directly from PyPI:

    pip install dysts

For the latest version, install directly from GitHub:

    pip install git+https://github.com/williamgilpin/dysts

Install optional precomputed trajectories and benchmark results:

    pip install git+https://github.com/williamgilpin/dysts_data

## Further Information:

*   **API Documentation:** [https://gilpinlab.github.io/dysts/spbuild/html/index.html](https://gilpinlab.github.io/dysts/spbuild/html/index.html)
*   **Demonstrations Notebook:** [`demos.ipynb`](demos.ipynb)
*   **Precomputed Data:** [HuggingFace Dataset](https://huggingface.co/datasets/williamgilpin/dysts)
*   **Benchmarks Repository:** [https://github.com/williamgilpin/dysts_data/tree/main/dysts_data/benchmarks](https://github.com/williamgilpin/dysts_data/tree/main/dysts_data/benchmarks)
*   **Contributing:** See [`CONTRIBUTING.md`](CONTRIBUTING.md)

## References:

*   William Gilpin. "Chaos as an interpretable benchmark for forecasting and data-driven modelling" Advances in Neural Information Processing Systems (NeurIPS) 2021 https://arxiv.org/abs/2110.05266
*   William Gilpin. "Model scale versus domain knowledge in statistical forecasting of chaotic systems" Physical Review Research 2023 https://arxiv.org/abs/2303.08011

## Testing:

Run tests with:

```
python -m unittest discover tests
```

## Acknowledgements:

*   Jürgen Meier's webpage
*   J. C. Sprott's webpage
*   nolds library
*   Wolf et al 1985
*   Eckmann et al 1986
*   DynamicalSystems.jl
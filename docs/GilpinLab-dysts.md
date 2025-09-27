# dysts: Explore and Analyze Chaotic Dynamical Systems

**Unlock the secrets of chaos with dysts, a Python library for analyzing and simulating a vast collection of chaotic systems.** Explore complex dynamics and generate insightful trajectories with ease. 

[![Plots of chaotic systems in the collection](assets/logo.png)](https://github.com/GilpinLab/dysts)

## Key Features

*   **Extensive Library:** Access a curated collection of 135 continuous-time chaotic models, including delay differential equations, and 10 discrete maps.
*   **Easy Integration:** Generate trajectories with a simple API, customizable parameters, and initial conditions.
*   **Optimized Performance:** Benefit from `numba` compilation and vectorized ensemble generation for efficient simulations.
*   **Precomputed Data:** Access a database of precomputed time series on [HuggingFace](https://huggingface.co/datasets/williamgilpin/dysts).
*   **Detailed Metadata:** Utilize comprehensive metadata for each system, including parameter values, references, and default initial conditions.
*   **Resampling Options:** Resample trajectories to ensure consistent dominant timescales across models.
*   **Benchmarks & Examples:** Explore a separate [benchmarks repository](https://github.com/williamgilpin/dysts_data/tree/main/dysts_data/benchmarks) and a demonstration notebook to explore examples.
*   **Open Source and Collaborative:** Join a community of researchers exploring chaotic systems.

## Installation

Install `dysts` with pip:

```bash
pip install dysts
```

For the latest features and bug fixes, install directly from GitHub:

```bash
pip install git+https://github.com/williamgilpin/dysts
```

To install the optional precomputed trajectories and benchmark results, install the data repository:

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

## Additional Resources

*   **API Documentation:** [https://gilpinlab.github.io/dysts/spbuild/html/index.html](https://gilpinlab.github.io/dysts/spbuild/html/index.html)
*   **Demonstration Notebook:** [`demos.ipynb`](demos.ipynb)
*   **Benchmarks:** [https://github.com/williamgilpin/dysts_data/tree/main/dysts_data/benchmarks](https://github.com/williamgilpin/dysts_data/tree/main/dysts_data/benchmarks)
*   **Database of precomputed time series:** [HuggingFace](https://huggingface.co/datasets/williamgilpin/dysts)

## Implementation Details

*   Dynamical equations are compiled using `numba` for performance.
*   Metadata is stored in JSON files.
*   Integration timesteps are based on the dominant frequency of each system.

## Contributing

We welcome contributions! See [`CONTRIBUTING.md`](CONTRIBUTING.md) for details.

## Reference

If you use this code for published work, please cite the following papers:

*   William Gilpin. "Chaos as an interpretable benchmark for forecasting and data-driven modelling" Advances in Neural Information Processing Systems (NeurIPS) 2021 https://arxiv.org/abs/2110.05266
*   William Gilpin. "Model scale versus domain knowledge in statistical forecasting of chaotic systems" Physical Review Research 2023 https://arxiv.org/abs/2303.08011

## Acknowledgements

*   [J&uuml;rgen Meier](http://www.3d-meier.de/tut19/Seite1.html) and [J. C. Sprott](http://sprott.physics.wisc.edu/sprott.htm)
*   [nolds](https://github.com/CSchoel/nolds)
*   [Wolf et al 1985](https://www.sciencedirect.com/science/article/abs/pii/0167278985900119) and [Eckmann et al 1986](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.34.4971)
*   [DynamicalSystems.jl](https://github.com/JuliaDynamics/DynamicalSystems.jl/)

**[Back to the original repository](https://github.com/GilpinLab/dysts)**
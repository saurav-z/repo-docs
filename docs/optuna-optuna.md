<div align="center"><img src="https://raw.githubusercontent.com/optuna/optuna/master/docs/image/optuna-logo.png" width="800" alt="Optuna Logo"/></div>

# Optuna: Unleash the Power of Automated Hyperparameter Optimization

Optuna is a cutting-edge hyperparameter optimization framework, designed to automate and accelerate machine learning model tuning. Explore the [Optuna repository](https://github.com/optuna/optuna) for more details.

[![Python](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org)
[![pypi](https://img.shields.io/pypi/v/optuna.svg)](https://pypi.python.org/pypi/optuna)
[![conda](https://img.shields.io/conda/vn/conda-forge/optuna.svg)](https://anaconda.org/conda-forge/optuna)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/optuna/optuna)
[![Read the Docs](https://readthedocs.org/projects/optuna/badge/?version=stable)](https://optuna.readthedocs.io/en/stable/)
[![Codecov](https://codecov.io/gh/optuna/optuna/branch/master/graph/badge.svg)](https://codecov.io/gh/optuna/optuna)

*   **Website:** [https://optuna.org/](https://optuna.org/)
*   **Documentation:** [https://optuna.readthedocs.io/en/stable/](https://optuna.readthedocs.io/en/stable/)
*   **Installation Guide:** [https://optuna.readthedocs.io/en/stable/installation.html](https://optuna.readthedocs.io/en/stable/installation.html)
*   **Tutorial:** [https://optuna.readthedocs.io/en/stable/tutorial/index.html](https://optuna.readthedocs.io/en/stable/tutorial/index.html)
*   **Examples:** [https://github.com/optuna/optuna-examples](https://github.com/optuna/optuna-examples)
*   **Twitter:** [https://twitter.com/OptunaAutoML](https://twitter.com/OptunaAutoML)
*   **LinkedIn:** [https://www.linkedin.com/showcase/optuna/](https://www.linkedin.com/showcase/optuna/)
*   **Medium:** [https://medium.com/optuna](https://medium.com/optuna)

Optuna is an automatic hyperparameter optimization software framework designed for machine learning, employing an imperative, *define-by-run* style user API that allows for dynamic search space construction.

## Key Features

*   **Lightweight and Versatile:** Easily handles diverse tasks with a simple installation and few dependencies.
*   **Pythonic Search Spaces:** Define search spaces using familiar Python syntax, including conditionals and loops.
*   **Efficient Optimization Algorithms:** Leverages state-of-the-art algorithms for efficient hyperparameter sampling and pruning.
*   **Easy Parallelization:** Scale studies across multiple workers with minimal code changes.
*   **Quick Visualization:** Provides a variety of plotting functions to quickly inspect optimization histories.

## News
Stay up-to-date with the latest Optuna developments and contribute to the next version!

Optuna 5.0 Roadmap has been published for review. Please take a look at [the planned improvements to Optuna](https://medium.com/optuna/optuna-v5-roadmap-ac7d6935a878), and share your feedback in [the github issues](https://github.com/optuna/optuna/labels/v5). PR contributions also welcome!

Please take a few minutes to fill in [this survey](https://forms.gle/wVwLCQ9g6st6AXuq9), and let us know how you use Optuna now and what improvements you'd like.🤔
All questions are optional. 🙇‍♂️

*   **Jun 16, 2025**: Optuna 4.4.0 has been released! Check out [the release blog](https://medium.com/optuna/announcing-optuna-4-4-ece661493126).
*   **May 26, 2025**: Optuna 5.0 roadmap has been published! See [the blog](https://medium.com/optuna/optuna-v5-roadmap-ac7d6935a878) for more details.
*   **Apr 14, 2025**: Optuna 4.3.0 is out! Check out [the release note](https://github.com/optuna/optuna/releases/tag/v4.3.0) for details.
*   **Mar 24, 2025**: A new article [Distributed Optimization in Optuna and gRPC Storage Proxy](https://medium.com/optuna/distributed-optimization-in-optuna-and-grpc-storage-proxy-08db83f1d608) has been published.
*   **Mar 11, 2025**: A new article [[Optuna v4.2] Gaussian Process-Based Sampler Can Now Handle Inequality Constraints](https://medium.com/optuna/optuna-v4-2-gaussian-process-based-sampler-can-now-handle-inequality-constraints-a4f68e8ee810) has been published.
*   **Feb 17, 2025**: A new article [SMAC3 Registered on OptunaHub](https://medium.com/optuna/smac3-registered-on-optunahub-4fb9e90855cb) has been published.

## Basic Concepts

Optuna uses the following terminology:

*   **Study:** The overall optimization process based on an objective function.
*   **Trial:** A single execution of the objective function with a specific set of hyperparameters.

The goal of a *study* is to identify the optimal set of hyperparameters through multiple *trials*. Optuna automates and accelerates these optimization *studies*.

<details open>
<summary>Sample Code</summary>

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/optuna/optuna-examples/blob/main/quickstart.ipynb)

```python
import optuna
import sklearn

# Define an objective function to be minimized.
def objective(trial):
    # Invoke suggest methods of a Trial object to generate hyperparameters.
    regressor_name = trial.suggest_categorical("regressor", ["SVR", "RandomForest"])
    if regressor_name == "SVR":
        svr_c = trial.suggest_float("svr_c", 1e-10, 1e10, log=True)
        regressor_obj = sklearn.svm.SVR(C=svr_c)
    else:
        rf_max_depth = trial.suggest_int("rf_max_depth", 2, 32)
        regressor_obj = sklearn.ensemble.RandomForestRegressor(max_depth=rf_max_depth)

    X, y = sklearn.datasets.fetch_california_housing(return_X_y=True)
    X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(X, y, random_state=0)

    regressor_obj.fit(X_train, y_train)
    y_pred = regressor_obj.predict(X_val)

    error = sklearn.metrics.mean_squared_error(y_val, y_pred)
    return error  # An objective value linked with the Trial object.

study = optuna.create_study()  # Create a new study.
study.optimize(objective, n_trials=100)  # Invoke optimization of the objective function.
```
</details>

> [!NOTE]
> Explore more examples in the [optuna/optuna-examples](https://github.com/optuna/optuna-examples) repository, covering multi-objective, constrained, pruning, and distributed optimization scenarios.

## Installation

Install Optuna using pip or conda:

```bash
# PyPI
pip install optuna
```

```bash
# Anaconda Cloud
conda install -c conda-forge optuna
```

> [!IMPORTANT]
> Optuna supports Python 3.8 or newer.
>
> Docker images are available on [DockerHub](https://hub.docker.com/r/optuna/optuna).

## Integrations

Optuna integrates seamlessly with various third-party libraries. Find integration details in the [optuna/optuna-integration](https://github.com/optuna/optuna-integration) repository and the documentation [here](https://optuna-integration.readthedocs.io/en/stable/index.html).

<details>
<summary>Supported Integration Libraries</summary>

*   [Catboost](https://github.com/optuna/optuna-examples/tree/main/catboost/catboost_pruning.py)
*   [Dask](https://github.com/optuna/optuna-examples/tree/main/dask/dask_simple.py)
*   [fastai](https://github.com/optuna/optuna-examples/tree/main/fastai/fastai_simple.py)
*   [Keras](https://github.com/optuna/optuna-examples/tree/main/keras/keras_integration.py)
*   [LightGBM](https://github.com/optuna/optuna-examples/tree/main/lightgbm/lightgbm_integration.py)
*   [MLflow](https://github.com/optuna/optuna-examples/tree/main/mlflow/keras_mlflow.py)
*   [PyTorch](https://github.com/optuna/optuna-examples/tree/main/pytorch/pytorch_simple.py)
*   [PyTorch Ignite](https://github.com/optuna/optuna-examples/tree/main/pytorch/pytorch_ignite_simple.py)
*   [PyTorch Lightning](https://github.com/optuna/optuna-examples/tree/main/pytorch/pytorch_lightning_simple.py)
*   [TensorBoard](https://github.com/optuna/optuna-examples/tree/main/tensorboard/tensorboard_simple.py)
*   [TensorFlow](https://github.com/optuna/optuna-examples/tree/main/tensorflow/tensorflow_estimator_integration.py)
*   [tf.keras](https://github.com/optuna/optuna-examples/tree/main/tfkeras/tfkeras_integration.py)
*   [Weights & Biases](https://github.com/optuna/optuna-examples/tree/main/wandb/wandb_integration.py)
*   [XGBoost](https://github.com/optuna/optuna-examples/tree/main/xgboost/xgboost_integration.py)
</details>

## Optuna Dashboard

The [Optuna Dashboard](https://github.com/optuna/optuna-dashboard) provides a real-time web interface to visualize optimization history and analyze hyperparameter importance. This eliminates the need for custom Python scripts to create visualizations.

![Optuna Dashboard](https://user-images.githubusercontent.com/5564044/204975098-95c2cb8c-0fb5-4388-abc4-da32f56cb4e5.gif)

Install the dashboard:

```bash
pip install optuna-dashboard
```

<details>
<summary>Dashboard Sample Code</summary>

Create a file named `optimize_toy.py` with the following contents:

```python
import optuna

def objective(trial):
    x1 = trial.suggest_float("x1", -100, 100)
    x2 = trial.suggest_float("x2", -100, 100)
    return x1**2 + 0.01 * x2**2

study = optuna.create_study(storage="sqlite:///db.sqlite3")  # Create a new study with database.
study.optimize(objective, n_trials=100)
```

Then run these commands in your terminal:

```bash
python optimize_toy.py
optuna-dashboard sqlite:///db.sqlite3
```
</details>

## OptunaHub

[OptunaHub](https://hub.optuna.org/) is a platform to share and reuse Optuna features.

### Use Registered Features

Install `optunahub`:

```bash
pip install optunahub
# Install AutoSampler dependencies (CPU only is sufficient for PyTorch)
pip install cmaes scipy torch --extra-index-url https://download.pytorch.org/whl/cpu
```

Load modules using `optunahub.load_module`.

```python
import optuna
import optunahub

def objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -5, 5)
    y = trial.suggest_float("y", -5, 5)
    return x**2 + y**2

module = optunahub.load_module(package="samplers/auto_sampler")
study = optuna.create_study(sampler=module.AutoSampler())
study.optimize(objective, n_trials=10)

print(study.best_trial.value, study.best_trial.params)
```

See the [OptunaHub Documentation](https://optuna.github.io/optunahub/) for details.

### Publish Your Packages

Publish your package using [optunahub-registry](https://github.com/optuna/optunahub-registry). Refer to the [Tutorials for Contributors](https://optuna.github.io/optunahub/tutorials_for_contributors.html) in OptunaHub.

## Communication

*   **Discussions:** [GitHub Discussions](https://github.com/optuna/optuna/discussions)
*   **Issues:** [GitHub Issues](https://github.com/optuna/optuna/issues)

## Contribution

Contributions to Optuna are highly welcome!

Explore the [good first issues](https://github.com/optuna/optuna/labels/good%20first%20issue) for beginner-friendly tasks.

For guidelines, refer to [CONTRIBUTING.md](./CONTRIBUTING.md).

## Reference

If you use Optuna in your research, please cite the [KDD paper](https://doi.org/10.1145/3292500.3330701): "Optuna: A Next-generation Hyperparameter Optimization Framework":

<details open>
<summary>BibTeX</summary>

```bibtex
@inproceedings{akiba2019optuna,
  title={{O}ptuna: A Next-Generation Hyperparameter Optimization Framework},
  author={Akiba, Takuya and Sano, Shotaro and Yanase, Toshihiko and Ohta, Takeru and Koyama, Masanori},
  booktitle={The 25th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  pages={2623--2631},
  year={2019}
}
```
</details>

## License

MIT License (see [LICENSE](./LICENSE)).

Optuna uses codes from SciPy and fdlibm projects (see [LICENSE_THIRD_PARTY](./LICENSE_THIRD_PARTY)).
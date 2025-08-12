<div align="center">
  <img src="https://raw.githubusercontent.com/optuna/optuna/master/docs/image/optuna-logo.png" width="800" alt="Optuna Logo"/>
</div>

# Optuna: Automate Hyperparameter Optimization for Machine Learning

**Supercharge your machine learning models with Optuna, a powerful and easy-to-use hyperparameter optimization framework.** Find out more on the original repo: [Optuna on GitHub](https://github.com/optuna/optuna).

[![Python](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org)
[![pypi](https://img.shields.io/pypi/v/optuna.svg)](https://pypi.python.org/pypi/optuna)
[![conda](https://img.shields.io/conda/vn/conda-forge/optuna.svg)](https://anaconda.org/conda-forge/optuna)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/optuna/optuna)
[![Read the Docs](https://readthedocs.org/projects/optuna/badge/?version=stable)](https://optuna.readthedocs.io/en/stable/)
[![Codecov](https://codecov.io/gh/optuna/optuna/branch/master/graph/badge.svg)](https://codecov.io/gh/optuna/optuna)

*   :link: [**Website**](https://optuna.org/)
*   :page_with_curl: [**Docs**](https://optuna.readthedocs.io/en/stable/)
*   :gear: [**Install Guide**](https://optuna.readthedocs.io/en/stable/installation.html)
*   :pencil: [**Tutorial**](https://optuna.readthedocs.io/en/stable/tutorial/index.html)
*   :bulb: [**Examples**](https://github.com/optuna/optuna-examples)
*   [**Twitter**](https://twitter.com/OptunaAutoML)
*   [**LinkedIn**](https://www.linkedin.com/showcase/optuna/)
*   [**Medium**](https://medium.com/optuna)

## Key Features of Optuna

Optuna offers a streamlined approach to hyperparameter optimization with a focus on ease of use and flexibility:

*   **Pythonic Search Spaces:** Define hyperparameter search spaces using standard Python syntax, including conditional statements and loops, making your code clean and intuitive.
*   **Efficient Optimization Algorithms:** Benefit from state-of-the-art algorithms for hyperparameter sampling and pruning, ensuring efficient exploration of your search space.
*   **Easy Parallelization:** Scale your optimization studies effortlessly to multiple workers, accelerating the search for optimal hyperparameters.
*   **Lightweight and Versatile:**  Optuna is designed to be lightweight, versatile, and platform-agnostic, working seamlessly across a wide range of machine learning tasks with minimal dependencies.
*   **Quick Visualization:** Visualize optimization progress and results using a variety of built-in plotting functions for insightful analysis.

## News

*   **[Jun 16, 2025]:** Optuna 4.4.0 has been released! Check out [the release blog](https://medium.com/optuna/announcing-optuna-4-4-ece661493126).
*   **[May 26, 2025]:** Optuna 5.0 roadmap has been published! See [the blog](https://medium.com/optuna/optuna-v5-roadmap-ac7d6935a878) for more details.
*   **[Apr 14, 2025]:** Optuna 4.3.0 is out! Check out [the release note](https://github.com/optuna/optuna/releases/tag/v4.3.0) for details.
*   **[Mar 24, 2025]:** A new article [Distributed Optimization in Optuna and gRPC Storage Proxy](https://medium.com/optuna/distributed-optimization-in-optuna-and-grpc-storage-proxy-08db83f1d608) has been published.
*   **[Mar 11, 2025]:** A new article [[Optuna v4.2] Gaussian Process-Based Sampler Can Now Handle Inequality Constraints](https://medium.com/optuna/optuna-v4-2-gaussian-process-based-sampler-can-now-handle-inequality-constraints-a4f68e8ee810) has been published.
*   **[Feb 17, 2025]:** A new article [SMAC3 Registered on OptunaHub](https://medium.com/optuna/smac3-registered-on-optunahub-4fb9e90855cb) has been published.

## Basic Concepts

Optuna uses the following terms:

*   **Study:**  The overall optimization process based on an objective function.
*   **Trial:** A single execution of the objective function with a specific set of hyperparameters.

```python
import optuna
import sklearn

# Define an objective function
def objective(trial):
    # Hyperparameter suggestion
    regressor_name = trial.suggest_categorical("regressor", ["SVR", "RandomForest"])

    # Define regressor based on suggestion
    if regressor_name == "SVR":
        svr_c = trial.suggest_float("svr_c", 1e-10, 1e10, log=True)
        regressor_obj = sklearn.svm.SVR(C=svr_c)
    else:
        rf_max_depth = trial.suggest_int("rf_max_depth", 2, 32)
        regressor_obj = sklearn.ensemble.RandomForestRegressor(max_depth=rf_max_depth)

    # Load data and split
    X, y = sklearn.datasets.fetch_california_housing(return_X_y=True)
    X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(X, y, random_state=0)

    # Fit regressor and predict
    regressor_obj.fit(X_train, y_train)
    y_pred = regressor_obj.predict(X_val)

    # Calculate the error
    error = sklearn.metrics.mean_squared_error(y_val, y_pred)

    return error

# Create study and optimize
study = optuna.create_study()
study.optimize(objective, n_trials=100)
```

> [!NOTE]
>  Explore more examples in [optuna/optuna-examples](https://github.com/optuna/optuna-examples).

## Installation

Install Optuna easily using pip or conda:

```bash
# PyPI
$ pip install optuna
```

```bash
# Anaconda Cloud
$ conda install -c conda-forge optuna
```

> [!IMPORTANT]
> Optuna requires Python 3.8 or newer.

## Integrations

Optuna seamlessly integrates with popular machine learning libraries to streamline your workflow.  Find integration details in [optuna/optuna-integration](https://github.com/optuna/optuna-integration) and the [documentation](https://optuna-integration.readthedocs.io/en/stable/index.html).

<details>
<summary>Supported integration libraries</summary>

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

## Web Dashboard

Visualize and analyze your Optuna studies in real-time using the [Optuna Dashboard](https://github.com/optuna/optuna-dashboard).  Monitor optimization progress, hyperparameter importance, and more.

![optuna-dashboard](https://user-images.githubusercontent.com/5564044/204975098-95c2cb8c-0fb5-4388-abc4-da32f56cb4e5.gif)

Install the dashboard with:

```shell
$ pip install optuna-dashboard
```

```shell
# Run a study
$ python optimize_toy.py
# Launch dashboard
$ optuna-dashboard sqlite:///db.sqlite3
```

## OptunaHub

[OptunaHub](https://hub.optuna.org/) is a feature-sharing platform for Optuna.
You can use the registered features and publish your packages.

### Use registered features

`optunahub` can be installed via pip:

```shell
$ pip install optunahub
# Install AutoSampler dependencies (CPU only is sufficient for PyTorch)
$ pip install cmaes scipy torch --extra-index-url https://download.pytorch.org/whl/cpu
```

You can load registered module with `optunahub.load_module`.

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

For more details, please refer to [the optunahub documentation](https://optuna.github.io/optunahub/).

### Publish your packages

You can publish your package via [optunahub-registry](https://github.com/optuna/optunahub-registry).
See the [Tutorials for Contributors](https://optuna.github.io/optunahub/tutorials_for_contributors.html) in OptunaHub.

## Get Involved

*   **[GitHub Discussions]**: Ask questions and engage with the community.
*   **[GitHub Issues]**: Report bugs and request features.

## Contribution

Contributions to Optuna are warmly welcomed!  Check the [good first issues](https://github.com/optuna/optuna/labels/good%20first%20issue) for beginner-friendly tasks. Refer to [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

## Reference

If you use Optuna in your research, please cite the KDD paper:

```bibtex
@inproceedings{akiba2019optuna,
  title={{O}ptuna: A Next-Generation Hyperparameter Optimization Framework},
  author={Akiba, Takuya and Sano, Shotaro and Yanase, Toshihiko and Ohta, Takeru and Koyama, Masanori},
  booktitle={The 25th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  pages={2623--2631},
  year={2019}
}
```

## License

Optuna is released under the MIT License (see [LICENSE](./LICENSE)).
It uses code from SciPy and fdlibm (see [LICENSE_THIRD_PARTY](./LICENSE_THIRD_PARTY)).
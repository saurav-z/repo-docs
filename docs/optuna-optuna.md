<div align="center">
  <img src="https://raw.githubusercontent.com/optuna/optuna/master/docs/image/optuna-logo.png" width="800" alt="Optuna Logo"/>
</div>

# Optuna: Automate Hyperparameter Optimization with Ease

**Optuna is a powerful and easy-to-use hyperparameter optimization framework designed for machine learning, helping you find the best parameters for your models.**  Find out more at the original repository: [https://github.com/optuna/optuna](https://github.com/optuna/optuna)

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

*   **Pythonic Search Spaces:** Define hyperparameter search spaces using Python's familiar syntax, including conditionals and loops, making it highly flexible.
*   **Efficient Optimization Algorithms:** Utilizes state-of-the-art algorithms for hyperparameter sampling and efficient pruning of unpromising trials, speeding up your optimization.
*   **Easy Parallelization:** Scale your studies effortlessly to multiple workers, enabling faster exploration of the hyperparameter space without significant code changes.
*   **Quick Visualization:** Provides various plotting functions to quickly visualize and analyze optimization histories, giving insights into your search process.
*   **Lightweight and Versatile:** Offers a lightweight, platform-agnostic architecture that works with a wide range of machine learning tasks and libraries.

## News

Stay up-to-date with the latest developments in Optuna!

*   **[Jun 16, 2025]:** Optuna 4.4.0 has been released! Check out [the release blog](https://medium.com/optuna/announcing-optuna-4-4-ece661493126).
*   **[May 26, 2025]:** Optuna 5.0 roadmap has been published! See [the blog](https://medium.com/optuna/optuna-v5-roadmap-ac7d6935a878) for more details.
*   **[Apr 14, 2025]:** Optuna 4.3.0 is out! Check out [the release note](https://github.com/optuna/optuna/releases/tag/v4.3.0) for details.
*   **[Mar 24, 2025]:** A new article [Distributed Optimization in Optuna and gRPC Storage Proxy](https://medium.com/optuna/distributed-optimization-in-optuna-and-grpc-storage-proxy-08db83f1d608) has been published.
*   **[Mar 11, 2025]:** A new article [[Optuna v4.2] Gaussian Process-Based Sampler Can Now Handle Inequality Constraints](https://medium.com/optuna/optuna-v4-2-gaussian-process-based-sampler-can-now-handle-inequality-constraints-a4f68e8ee810) has been published.
*   **[Feb 17, 2025]:** A new article [SMAC3 Registered on OptunaHub](https://medium.com/optuna/smac3-registered-on-optunahub-4fb9e90855cb) has been published.

## Basic Concepts

Optuna uses the terms *study* and *trial*:

*   **Study:** An optimization process based on an objective function.
*   **Trial:** A single execution of the objective function using a specific set of hyperparameters.

The goal of a study is to identify the optimal set of hyperparameter values through multiple trials.  Optuna is designed to automate and accelerate this optimization process.

<details open>
<summary>Sample code with scikit-learn</summary>

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
>  Explore more examples at [optuna/optuna-examples](https://github.com/optuna/optuna-examples) to see various problem setups, including multi-objective, constrained, and distributed optimization scenarios.

## Installation

Install Optuna using pip or conda:

```bash
# PyPI
$ pip install optuna
```

```bash
# Anaconda Cloud
$ conda install -c conda-forge optuna
```

> [!IMPORTANT]
>  Requires Python 3.8 or newer. Docker images are available on [DockerHub](https://hub.docker.com/r/optuna/optuna).

## Integrations

Optuna offers seamless integrations with various popular machine learning libraries:

<details>
<summary>Supported Libraries</summary>

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
Explore the integrations further at [optuna/optuna-integration](https://github.com/optuna/optuna-integration) and [here](https://optuna-integration.readthedocs.io/en/stable/index.html).

## Optuna Dashboard

The [Optuna Dashboard](https://github.com/optuna/optuna-dashboard) offers a real-time web interface for visualizing your Optuna studies. It allows you to monitor optimization progress, analyze hyperparameter importance, and inspect results with interactive graphs and tables.

![optuna-dashboard](https://user-images.githubusercontent.com/5564044/204975098-95c2cb8c-0fb5-4388-abc4-da32f56cb4e5.gif)

Install Optuna Dashboard using pip:

```shell
$ pip install optuna-dashboard
```

<details>
<summary>Launch Optuna Dashboard Example</summary>

Save the following code as `optimize_toy.py`.

```python
import optuna


def objective(trial):
    x1 = trial.suggest_float("x1", -100, 100)
    x2 = trial.suggest_float("x2", -100, 100)
    return x1**2 + 0.01 * x2**2


study = optuna.create_study(storage="sqlite:///db.sqlite3")  # Create a new study with database.
study.optimize(objective, n_trials=100)
```

Then, run the following commands in your terminal:

```shell
# Run the study
$ python optimize_toy.py

# Launch the dashboard
$ optuna-dashboard sqlite:///db.sqlite3
...
Listening on http://localhost:8080/
Hit Ctrl-C to quit.
```
</details>

## OptunaHub

[OptunaHub](https://hub.optuna.org/) is a platform for sharing and utilizing Optuna features.  You can use pre-registered features and publish your packages for broader use.

### Use Registered Features

Install `optunahub`:

```shell
$ pip install optunahub
# Install AutoSampler dependencies (CPU only is sufficient for PyTorch)
$ pip install cmaes scipy torch --extra-index-url https://download.pytorch.org/whl/cpu
```

Load registered modules with `optunahub.load_module`:

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

Find more information in [the optunahub documentation](https://optuna.github.io/optunahub/).

### Publish Packages

Publish your own packages via [optunahub-registry](https://github.com/optuna/optunahub-registry).  See the [Tutorials for Contributors](https://optuna.github.io/optunahub/tutorials_for_contributors.html) in OptunaHub.

## Communication

*   [GitHub Discussions](https://github.com/optuna/optuna/discussions) for general questions and discussions.
*   [GitHub Issues](https://github.com/optuna/optuna/issues) for bug reports and feature requests.

## Contribution

Contributions to Optuna are highly valued!

*   Check the [good first issues](https://github.com/optuna/optuna/labels/good%20first%20issue) for beginner-friendly tasks.
*   Explore the [contribution-welcome issues](https://github.com/optuna/optuna/labels/contribution-welcome) for opportunities to contribute.
*   Review [CONTRIBUTING.md](./CONTRIBUTING.md) for general project guidelines.

## Reference

If you use Optuna in your research, please cite our KDD paper:

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

Optuna utilizes codes from SciPy and fdlibm projects (see [LICENSE_THIRD_PARTY](./LICENSE_THIRD_PARTY)).
<div align="center"><img src="https://raw.githubusercontent.com/optuna/optuna/master/docs/image/optuna-logo.png" width="800"/></div>

# Optuna: Optimize Your Machine Learning Models with Ease

**Optuna** is a powerful and versatile hyperparameter optimization framework designed to automate and accelerate the process of finding the best configuration for your machine learning models.  [Explore the Optuna repository on GitHub](https://github.com/optuna/optuna).

[![Python](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org)
[![pypi](https://img.shields.io/pypi/v/optuna.svg)](https://pypi.python.org/pypi/optuna)
[![conda](https://img.shields.io/conda/vn/conda-forge/optuna.svg)](https://anaconda.org/conda-forge/optuna)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/optuna/optuna)
[![Read the Docs](https://readthedocs.org/projects/optuna/badge/?version=stable)](https://optuna.readthedocs.io/en/stable/)
[![Codecov](https://codecov.io/gh/optuna/optuna/branch/master/graph/badge.svg)](https://codecov.io/gh/optuna/optuna)

*   :link: [**Website**](https://optuna.org/)
*   :page\_with\_curl: [**Docs**](https://optuna.readthedocs.io/en/stable/)
*   :gear: [**Install Guide**](https://optuna.readthedocs.io/en/stable/installation.html)
*   :pencil: [**Tutorial**](https://optuna.readthedocs.io/en/stable/tutorial/index.html)
*   :bulb: [**Examples**](https://github.com/optuna/optuna-examples)
*   [**Twitter**](https://twitter.com/OptunaAutoML)
*   [**LinkedIn**](https://www.linkedin.com/showcase/optuna/)
*   [**Medium**](https://medium.com/optuna)

## Key Features of Optuna

*   **Lightweight and Versatile:** Easily integrates with various machine learning tasks and frameworks.
*   **Pythonic Search Spaces:** Define hyperparameter search spaces using standard Python syntax, including conditionals and loops, offering flexibility in defining your search strategy.
*   **Efficient Optimization Algorithms:** Employs state-of-the-art algorithms for hyperparameter sampling and pruning, leading to faster and more effective optimization.
*   **Easy Parallelization:** Seamlessly scale your studies across multiple workers with minimal code changes.
*   **Rich Visualization Tools:** Visualize your optimization history and results using various plotting functions for quick insights.

## Stay Updated

*   **Latest Release:** Optuna 4.4.0 was released on June 16, 2025!  See the [release blog](https://medium.com/optuna/announcing-optuna-4-4-ece661493126).
*   **Future Directions:**  Check out the Optuna 5.0 roadmap and provide feedback: [Optuna v5 Roadmap](https://medium.com/optuna/optuna-v5-roadmap-ac7d6935a878) and [Github issues](https://github.com/optuna/optuna/labels/v5).

## Core Concepts

Optuna uses the terms *study* and *trial*:

*   **Study:**  The overall optimization process based on an objective function.
*   **Trial:** A single execution of the objective function with a specific set of hyperparameters.

**Example:**

```python
import optuna
import sklearn

# Objective function to minimize (example with scikit-learn)
def objective(trial):
    # Define hyperparameters using Optuna's suggest methods
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
    return error  # Objective value

# Create a study and optimize
study = optuna.create_study()
study.optimize(objective, n_trials=100)
```

>   [!NOTE]
>   For more detailed examples, visit the [Optuna Examples repository](https://github.com/optuna/optuna-examples).

## Installation

Install Optuna using pip or conda:

```bash
# PyPI
$ pip install optuna

# Anaconda Cloud
$ conda install -c conda-forge optuna
```

>   [!IMPORTANT]
>   Optuna requires Python 3.8 or newer.

## Integrations

Optuna seamlessly integrates with popular machine learning libraries:

<details>
<summary>Supported Libraries</summary>

*   Catboost
*   Dask
*   fastai
*   Keras
*   LightGBM
*   MLflow
*   PyTorch
*   PyTorch Ignite
*   PyTorch Lightning
*   TensorBoard
*   TensorFlow
*   tf.keras
*   Weights & Biases
*   XGBoost

</details>

For comprehensive details on integrations, explore [optuna-integration](https://optuna-integration.readthedocs.io/en/stable/index.html).

## Optuna Dashboard

Visualize your optimization progress in real-time with the [Optuna Dashboard](https://github.com/optuna/optuna-dashboard).

![optuna-dashboard](https://user-images.githubusercontent.com/5564044/204975098-95c2cb8c-0fb5-4388-abc4-da32f56cb4e5.gif)

Install the dashboard:

```shell
$ pip install optuna-dashboard
```

**Dashboard Example:**

```python
import optuna

def objective(trial):
    x1 = trial.suggest_float("x1", -100, 100)
    x2 = trial.suggest_float("x2", -100, 100)
    return x1**2 + 0.01 * x2**2

study = optuna.create_study(storage="sqlite:///db.sqlite3")
study.optimize(objective, n_trials=100)
```

Then run:

```shell
$ python optimize_toy.py
$ optuna-dashboard sqlite:///db.sqlite3
```

## OptunaHub

[OptunaHub](https://hub.optuna.org/) offers a platform for sharing and reusing Optuna features.

### Utilizing Registered Features

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

### Publishing Your Packages

Contribute to the community by publishing your own packages via [optunahub-registry](https://github.com/optuna/optunahub-registry). See the [Tutorials for Contributors](https://optuna.github.io/optunahub/tutorials_for_contributors.html) in OptunaHub.

## Get Involved

*   **GitHub Discussions:** [https://github.com/optuna/optuna/discussions](https://github.com/optuna/optuna/discussions) for questions.
*   **GitHub Issues:** [https://github.com/optuna/optuna/issues](https://github.com/optuna/optuna/issues) for bug reports and feature requests.

## Contribute

Contributions are welcome! Explore [good first issues](https://github.com/optuna/optuna/labels/good%20first%20issue) or [contribution-welcome issues](https://github.com/optuna/optuna/labels/contribution-welcome).  Refer to [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

## Citation

If you use Optuna in your research, please cite the KDD paper:

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

Optuna is distributed under the MIT License (see [LICENSE](./LICENSE)).  It also incorporates code from SciPy and fdlibm, with their respective licenses in [LICENSE_THIRD_PARTY](./LICENSE_THIRD_PARTY).
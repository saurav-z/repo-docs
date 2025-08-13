<div align="center"><img src="https://raw.githubusercontent.com/optuna/optuna/master/docs/image/optuna-logo.png" width="800"/></div>

# Optuna: Unleash the Power of Hyperparameter Optimization

**Supercharge your machine learning models with Optuna, a cutting-edge hyperparameter optimization framework designed for efficiency and ease of use.**  [**Explore the Optuna Repository**](https://github.com/optuna/optuna)

[![Python](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org)
[![pypi](https://img.shields.io/pypi/v/optuna.svg)](https://pypi.python.org/pypi/optuna)
[![conda](https://img.shields.io/conda/vn/conda-forge/optuna.svg)](https://anaconda.org/conda-forge/optuna)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/optuna/optuna)
[![Read the Docs](https://readthedocs.org/projects/optuna/badge/?version=stable)](https://optuna.readthedocs.io/en/stable/)
[![Codecov](https://codecov.io/gh/optuna/optuna/branch/master/graph/badge.svg)](https://codecov.io/gh/optuna/optuna)

:link: [**Website**](https://optuna.org/)
| :page_with_curl: [**Docs**](https://optuna.readthedocs.io/en/stable/)
| :gear: [**Install Guide**](https://optuna.readthedocs.io/en/stable/installation.html)
| :pencil: [**Tutorial**](https://optuna.readthedocs.io/en/stable/tutorial/index.html)
| :bulb: [**Examples**](https://github.com/optuna/optuna-examples)
| [**Twitter**](https://twitter.com/OptunaAutoML)
| [**LinkedIn**](https://www.linkedin.com/showcase/optuna/)
| [**Medium**](https://medium.com/optuna)

Optuna is an automatic hyperparameter optimization software framework, meticulously crafted for machine learning tasks. Its intuitive, *define-by-run* API allows for dynamic construction of search spaces, promoting modularity and flexibility in your experiments.

## Key Features: Optimize with Ease

*   **Pythonic Search Spaces:** Define hyperparameter search spaces using familiar Python syntax, including conditional statements and loops, for maximum flexibility.
*   **Efficient Optimization Algorithms:** Leverage state-of-the-art algorithms for intelligent hyperparameter sampling and pruning of unproductive trials.
*   **Easy Parallelization:** Scale your experiments effortlessly across multiple workers with minimal code changes.
*   **Quick Visualization:** Gain insights with a variety of plotting functions to visualize optimization histories.
*   **Lightweight and Versatile:** A simple installation with minimal dependencies that supports a wide array of tasks and platforms.

## Stay Updated

*   **[Optuna 5.0 Roadmap](https://medium.com/optuna/optuna-v5-roadmap-ac7d6935a878)** Explore planned improvements and share your feedback.
*   **Recent Release:**  Optuna 4.4.0 is out! Check out [the release blog](https://medium.com/optuna/announcing-optuna-4-4-ece661493126).
*   **Stay Informed**:  See the latest articles, blog posts, and news about Optuna on [Medium](https://medium.com/optuna).

## Basic Concepts

Optuna uses the terms *study* and *trial* as follows:

*   **Study:** The overall optimization process based on an objective function.
*   **Trial:** A single execution of the objective function with a specific set of hyperparameters.

Optuna automates and accelerates the optimization process of *studies*, aiming to find the optimal set of hyperparameters through multiple *trials*.

<details open>
<summary>Sample Code with scikit-learn</summary>

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
> Explore more examples in the [optuna/optuna-examples](https://github.com/optuna/optuna-examples) repository, covering multi-objective optimization, constrained optimization, pruning, and distributed optimization scenarios.

## Installation

Install Optuna easily using `pip` or `conda`:

```bash
# PyPI
$ pip install optuna
```

```bash
# Anaconda Cloud
$ conda install -c conda-forge optuna
```

> [!IMPORTANT]
> Optuna requires Python 3.8 or newer. Docker images are available on [DockerHub](https://hub.docker.com/r/optuna/optuna).

## Integrations: Seamlessly Integrate with Your Favorite Libraries

Optuna seamlessly integrates with popular machine learning libraries, streamlining your workflow. Explore integrations at [optuna/optuna-integration](https://github.com/optuna/optuna-integration) and access detailed documentation [here](https://optuna-integration.readthedocs.io/en/stable/index.html).

<details>
<summary>Supported integration libraries</summary>

* [Catboost](https://github.com/optuna/optuna-examples/tree/main/catboost/catboost_pruning.py)
* [Dask](https://github.com/optuna/optuna-examples/tree/main/dask/dask_simple.py)
* [fastai](https://github.com/optuna/optuna-examples/tree/main/fastai/fastai_simple.py)
* [Keras](https://github.com/optuna/optuna-examples/tree/main/keras/keras_integration.py)
* [LightGBM](https://github.com/optuna/optuna-examples/tree/main/lightgbm/lightgbm_integration.py)
* [MLflow](https://github.com/optuna/optuna-examples/tree/main/mlflow/keras_mlflow.py)
* [PyTorch](https://github.com/optuna/optuna-examples/tree/main/pytorch/pytorch_simple.py)
* [PyTorch Ignite](https://github.com/optuna/optuna-examples/tree/main/pytorch/pytorch_ignite_simple.py)
* [PyTorch Lightning](https://github.com/optuna/optuna-examples/tree/main/pytorch/pytorch_lightning_simple.py)
* [TensorBoard](https://github.com/optuna/optuna-examples/tree/main/tensorboard/tensorboard_simple.py)
* [TensorFlow](https://github.com/optuna/optuna-examples/tree/main/tensorflow/tensorflow_estimator_integration.py)
* [tf.keras](https://github.com/optuna/optuna-examples/tree/main/tfkeras/tfkeras_integration.py)
* [Weights & Biases](https://github.com/optuna/optuna-examples/tree/main/wandb/wandb_integration.py)
* [XGBoost](https://github.com/optuna/optuna-examples/tree/main/xgboost/xgboost_integration.py)
</details>

## Optuna Dashboard: Visualize and Analyze Your Optimization Runs

[Optuna Dashboard](https://github.com/optuna/optuna-dashboard) offers a real-time web interface for monitoring and analyzing your Optuna studies. Visualize optimization history, hyperparameter importance, and more, all without writing extra code.

![optuna-dashboard](https://user-images.githubusercontent.com/5564044/204975098-95c2cb8c-0fb5-4388-abc4-da32f56cb4e5.gif)

Install the dashboard using `pip`:

```shell
$ pip install optuna-dashboard
```

<details>
<summary>Sample code to launch Optuna Dashboard</summary>

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

Then try the commands below:

```shell
# Run the study specified above
$ python optimize_toy.py

# Launch the dashboard based on the storage `sqlite:///db.sqlite3`
$ optuna-dashboard sqlite:///db.sqlite3
...
Listening on http://localhost:8080/
Hit Ctrl-C to quit.
```

</details>

## OptunaHub: Feature Sharing and Community Collaboration

[OptunaHub](https://hub.optuna.org/) is a community platform for sharing and reusing Optuna features. Discover registered features and publish your own packages to benefit the community.

### Use registered features

```shell
$ pip install optunahub
# Install AutoSampler dependencies (CPU only is sufficient for PyTorch)
$ pip install cmaes scipy torch --extra-index-url https://download.pytorch.org/whl/cpu
```

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

### Publish your packages

Publish your packages via [optunahub-registry](https://github.com/optuna/optunahub-registry) by following the [Tutorials for Contributors](https://optuna.github.io/optunahub/tutorials_for_contributors.html).

## Get Involved: Communication and Contribution

*   **[GitHub Discussions]** for questions and discussions.
*   **[GitHub Issues]** for bug reports and feature requests.

[GitHub Discussions]: https://github.com/optuna/optuna/discussions
[GitHub issues]: https://github.com/optuna/optuna/issues

We welcome contributions from everyone! If you're new, start with the [good first issues](https://github.com/optuna/optuna/labels/good%20first%20issue) or explore other [contribution-welcome issues](https://github.com/optuna/optuna/labels/contribution-welcome).  See [CONTRIBUTING.md](./CONTRIBUTING.md) for contribution guidelines.

## Reference

If you utilize Optuna in your research, please cite the KDD paper "Optuna: A Next-generation Hyperparameter Optimization Framework":

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

Optuna is licensed under the MIT License (see [LICENSE](./LICENSE)). It uses codes from SciPy and fdlibm projects (see [LICENSE_THIRD_PARTY](./LICENSE_THIRD_PARTY)).
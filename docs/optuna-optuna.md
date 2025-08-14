<div align="center"><img src="https://raw.githubusercontent.com/optuna/optuna/master/docs/image/optuna-logo.png" width="800" alt="Optuna Logo"/></div>

# Optuna: Optimize Your Machine Learning Models with Ease

Optuna is a powerful and versatile hyperparameter optimization framework designed to automate and accelerate the process of finding the best parameters for your machine learning models. [Explore the original repository](https://github.com/optuna/optuna).

[![Python](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org)
[![pypi](https://img.shields.io/pypi/v/optuna.svg)](https://pypi.python.org/pypi/optuna)
[![conda](https://img.shields.io/conda/vn/conda-forge/optuna.svg)](https://anaconda.org/conda-forge/optuna)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/optuna/optuna)
[![Read the Docs](https://readthedocs.org/projects/optuna/badge/?version=stable)](https://optuna.readthedocs.io/en/stable/)
[![Codecov](https://codecov.io/gh/optuna/optuna/branch/master/graph/badge.svg)](https://codecov.io/gh/optuna/optuna)

🌐 [**Website**](https://optuna.org/)
| 📚 [**Docs**](https://optuna.readthedocs.io/en/stable/)
| ⚙️ [**Install Guide**](https://optuna.readthedocs.io/en/stable/installation.html)
| ✏️ [**Tutorial**](https://optuna.readthedocs.io/en/stable/tutorial/index.html)
| 💡 [**Examples**](https://github.com/optuna/optuna-examples)
| 🐦 [**Twitter**](https://twitter.com/OptunaAutoML)
| 🔗 [**LinkedIn**](https://www.linkedin.com/showcase/optuna/)
| ✍️ [**Medium**](https://medium.com/optuna)

## Key Features of Optuna

Optuna simplifies and accelerates hyperparameter optimization with its innovative features:

*   **Pythonic Search Spaces:** Define hyperparameter search spaces using familiar Python syntax, including conditionals and loops.
*   **Efficient Optimization Algorithms:** Leverage state-of-the-art algorithms for efficient hyperparameter sampling and pruning of unpromising trials, saving time and resources.
*   **Easy Parallelization:** Scale your studies effortlessly across multiple workers, enabling faster experimentation and discovery of optimal hyperparameters.
*   **Quick Visualization:** Visualize optimization histories with a range of plotting functions, providing valuable insights into your model's performance.
*   **Lightweight & Versatile:** Designed for machine learning, it has a platform-agnostic architecture that is easy to install and use, with minimal dependencies.

## News & Updates

Stay up-to-date with the latest Optuna developments:

*   **[Recent Release]**: Optuna 4.4.0 has been released! Check out [the release blog](https://medium.com/optuna/announcing-optuna-4-4-ece661493126).
*   **[Roadmap]**: Optuna 5.0 roadmap has been published! See [the blog](https://medium.com/optuna/optuna-v5-roadmap-ac7d6935a878) for more details.
*   **[Community Feedback]**: Please take a few minutes to fill in [this survey](https://forms.gle/wVwLCQ9g6st6AXuq9), and let us know how you use Optuna now and what improvements you'd like.🤔

## Basic Concepts: Studies and Trials

Optuna uses two key concepts:

*   **Study:** An optimization session based on an objective function.
*   **Trial:** A single execution of the objective function with a specific set of hyperparameters.

The goal of a *study* is to find the optimal set of hyperparameter values (e.g., `regressor` and `svr_c`) through multiple *trials* (e.g., `n_trials=100`).

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
> More examples can be found in [optuna/optuna-examples](https://github.com/optuna/optuna-examples).
>
> The examples cover diverse problem setups such as multi-objective optimization, constrained optimization, pruning, and distributed optimization.

## Installation

Easily install Optuna using pip or conda:

```bash
# PyPI
$ pip install optuna
```

```bash
# Anaconda Cloud
$ conda install -c conda-forge optuna
```

> [!IMPORTANT]
> Optuna supports Python 3.8 or newer.

## Integrations

Optuna seamlessly integrates with many popular machine learning libraries:

### Supported Libraries:

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

[See full details of the integrations in the optuna-integration repository](https://github.com/optuna/optuna-integration).

## Web Dashboard

[Optuna Dashboard](https://github.com/optuna/optuna-dashboard) is a real-time web dashboard for Optuna.
You can check the optimization history, hyperparameter importance, etc. in graphs and tables.
You don't need to create a Python script to call [Optuna's visualization](https://optuna.readthedocs.io/en/stable/reference/visualization/index.html) functions.
Feature requests and bug reports are welcome!

![optuna-dashboard](https://user-images.githubusercontent.com/5564044/204975098-95c2cb8c-0fb5-4388-abc4-da32f56cb4e5.gif)

Install the dashboard with:

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

## OptunaHub

[OptunaHub](https://hub.optuna.org/) is a feature-sharing platform for Optuna.
You can use the registered features and publish your packages.

### Use registered features

Install `optunahub` with:

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

*   **[Discussions]**: Engage with the community for questions and support.
*   **[Issues]**: Report bugs and suggest new features.

[Discussions]: https://github.com/optuna/optuna/discussions
[Issues]: https://github.com/optuna/optuna/issues

## Contribute

Contributions to Optuna are highly encouraged! Check out the [good first issues](https://github.com/optuna/optuna/labels/good%20first%20issue) for beginner-friendly tasks. Explore the [CONTRIBUTING.md](./CONTRIBUTING.md) for general guidelines.

## Citation

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

Optuna is released under the MIT License (see [LICENSE](./LICENSE)).

It also uses code from the SciPy and fdlibm projects (see [LICENSE\_THIRD\_PARTY](./LICENSE\_THIRD\_PARTY)).
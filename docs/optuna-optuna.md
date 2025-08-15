html
<div align="center">
    <img src="https://raw.githubusercontent.com/optuna/optuna/master/docs/image/optuna-logo.png" width="800" alt="Optuna Logo">
</div>

<!-- SEO-Optimized Title -->
<h1>Optuna: Automate and Accelerate Hyperparameter Optimization for Machine Learning</h1>

<!-- Badges -->
[![Python](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org)
[![pypi](https://img.shields.io/pypi/v/optuna.svg)](https://pypi.python.org/pypi/optuna)
[![conda](https://img.shields.io/conda/vn/conda-forge/optuna.svg)](https://anaconda.org/conda-forge/optuna)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/optuna/optuna)
[![Read the Docs](https://readthedocs.org/projects/optuna/badge/?version=stable)](https://optuna.readthedocs.io/en/stable/)
[![Codecov](https://codecov.io/gh/optuna/optuna/branch/master/graph/badge.svg)](https://codecov.io/gh/optuna/optuna)

<!-- Links -->
<p>
    <a href="https://optuna.org/">🌐 Website</a> |
    <a href="https://optuna.readthedocs.io/en/stable/">📚 Docs</a> |
    <a href="https://optuna.readthedocs.io/en/stable/installation.html">⚙️ Install Guide</a> |
    <a href="https://optuna.readthedocs.io/en/stable/tutorial/index.html">📝 Tutorial</a> |
    <a href="https://github.com/optuna/optuna-examples">💡 Examples</a> |
    <a href="https://twitter.com/OptunaAutoML">🐦 Twitter</a> |
    <a href="https://www.linkedin.com/showcase/optuna/">💼 LinkedIn</a> |
    <a href="https://medium.com/optuna">📰 Medium</a>
</p>

<!-- Introduction -->
<p><b>Optimize your machine learning models effortlessly with Optuna, a powerful and user-friendly hyperparameter optimization framework.</b></p>

<!-- Overview -->
<p>Optuna is an automatic hyperparameter optimization software framework designed specifically for machine learning. Its *define-by-run* API provides a flexible and modular approach, allowing users to dynamically construct search spaces for hyperparameters with Pythonic syntax.</p>

<!-- News -->
<h2>📢 News</h2>
<ul>
    <li><b>[June 16, 2025]</b>: Optuna 4.4.0 has been released! Check out [the release blog](https://medium.com/optuna/announcing-optuna-4-4-ece661493126).</li>
    <li><b>[May 26, 2025]</b>: Optuna 5.0 roadmap has been published! See [the blog](https://medium.com/optuna/optuna-v5-roadmap-ac7d6935a878) for more details.</li>
    <li><b>[Apr 14, 2025]</b>: Optuna 4.3.0 is out! Check out [the release note](https://github.com/optuna/optuna/releases/tag/v4.3.0) for details.</li>
    <li><b>[Mar 24, 2025]</b>: A new article [Distributed Optimization in Optuna and gRPC Storage Proxy](https://medium.com/optuna/distributed-optimization-in-optuna-and-grpc-storage-proxy-08db83f1d608) has been published.</li>
    <li><b>[Mar 11, 2025]</b>: A new article [[Optuna v4.2] Gaussian Process-Based Sampler Can Now Handle Inequality Constraints](https://medium.com/optuna/optuna-v4-2-gaussian-process-based-sampler-can-now-handle-inequality-constraints-a4f68e8ee810) has been published.</li>
    <li><b>[Feb 17, 2025]</b>: A new article [SMAC3 Registered on OptunaHub](https://medium.com/optuna/smac3-registered-on-optunahub-4fb9e90855cb) has been published.</li>
</ul>

<!-- Key Features -->
<h2>✨ Key Features</h2>
<ul>
    <li><b>Lightweight and Versatile:</b>  A platform-agnostic architecture with minimal dependencies.</li>
    <li><b>Pythonic Search Spaces:</b> Define search spaces using Python's intuitive syntax, including conditionals and loops.</li>
    <li><b>Efficient Optimization Algorithms:</b> Leverage state-of-the-art algorithms for hyperparameter sampling and trial pruning.</li>
    <li><b>Easy Parallelization:</b> Scale your studies across multiple workers with minimal code changes.</li>
    <li><b>Quick Visualization:</b> Visualize optimization histories using a variety of plotting functions.</li>
</ul>

<!-- Basic Concepts -->
<h2>🧠 Basic Concepts</h2>
<p>Optuna uses the following key terms:</p>
<ul>
    <li><b>Study:</b> The overall optimization process based on an objective function.</li>
    <li><b>Trial:</b> A single execution of the objective function with a specific set of hyperparameters.</li>
</ul>

<details open>
    <summary>Sample code with scikit-learn</summary>
    <p>
        <a href="http://colab.research.google.com/github/optuna/optuna-examples/blob/main/quickstart.ipynb">
            <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab">
        </a>
    </p>

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
> Explore more examples at  <a href="https://github.com/optuna/optuna-examples">optuna/optuna-examples</a>. These examples cover diverse problem setups, including multi-objective and constrained optimization.

<!-- Installation -->
<h2> 🚀 Installation</h2>
<p>Install Optuna using pip or conda:</p>

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
>
> Docker images are also available on [DockerHub](https://hub.docker.com/r/optuna/optuna).

<!-- Integrations -->
<h2>🔌 Integrations</h2>
<p>Optuna integrates with various popular machine learning libraries. Explore these integrations in <a href="https://github.com/optuna/optuna-integration">optuna/optuna-integration</a>.</p>

<details>
    <summary>Supported Integration Libraries</summary>
    <ul>
        <li><a href="https://github.com/optuna/optuna-examples/tree/main/catboost/catboost_pruning.py">Catboost</a></li>
        <li><a href="https://github.com/optuna/optuna-examples/tree/main/dask/dask_simple.py">Dask</a></li>
        <li><a href="https://github.com/optuna/optuna-examples/tree/main/fastai/fastai_simple.py">fastai</a></li>
        <li><a href="https://github.com/optuna/optuna-examples/tree/main/keras/keras_integration.py">Keras</a></li>
        <li><a href="https://github.com/optuna/optuna-examples/tree/main/lightgbm/lightgbm_integration.py">LightGBM</a></li>
        <li><a href="https://github.com/optuna/optuna-examples/tree/main/mlflow/keras_mlflow.py">MLflow</a></li>
        <li><a href="https://github.com/optuna/optuna-examples/tree/main/pytorch/pytorch_simple.py">PyTorch</a></li>
        <li><a href="https://github.com/optuna/optuna-examples/tree/main/pytorch/pytorch_ignite_simple.py">PyTorch Ignite</a></li>
        <li><a href="https://github.com/optuna/optuna-examples/tree/main/pytorch/pytorch_lightning_simple.py">PyTorch Lightning</a></li>
        <li><a href="https://github.com/optuna/optuna-examples/tree/main/tensorboard/tensorboard_simple.py">TensorBoard</a></li>
        <li><a href="https://github.com/optuna/optuna-examples/tree/main/tensorflow/tensorflow_estimator_integration.py">TensorFlow</a></li>
        <li><a href="https://github.com/optuna/optuna-examples/tree/main/tfkeras/tfkeras_integration.py">tf.keras</a></li>
        <li><a href="https://github.com/optuna/optuna-examples/tree/main/wandb/wandb_integration.py">Weights & Biases</a></li>
        <li><a href="https://github.com/optuna/optuna-examples/tree/main/xgboost/xgboost_integration.py">XGBoost</a></li>
    </ul>
</details>

<!-- Web Dashboard -->
<h2>📊 Web Dashboard</h2>
<p>The <a href="https://github.com/optuna/optuna-dashboard">Optuna Dashboard</a> provides real-time visualization of your optimization runs, including hyperparameter importance and study history.</p>
<p>
    <img src="https://user-images.githubusercontent.com/5564044/204975098-95c2cb8c-0fb5-4388-abc4-da32f56cb4e5.gif" alt="Optuna Dashboard Demo">
</p>
<p>Install the dashboard with:</p>
```shell
$ pip install optuna-dashboard
```

<details>
    <summary>Launch Optuna Dashboard</summary>
    <p>Save the following code as `optimize_toy.py`:</p>
    ```python
    import optuna

    def objective(trial):
        x1 = trial.suggest_float("x1", -100, 100)
        x2 = trial.suggest_float("x2", -100, 100)
        return x1**2 + 0.01 * x2**2

    study = optuna.create_study(storage="sqlite:///db.sqlite3")  # Create a new study with database.
    study.optimize(objective, n_trials=100)
    ```
    <p>Then run these commands:</p>
    ```shell
    # Run the study
    $ python optimize_toy.py

    # Launch the dashboard
    $ optuna-dashboard sqlite:///db.sqlite3
    ```
</details>

<!-- OptunaHub -->
<h2> 🌐 OptunaHub</h2>
<p><a href="https://hub.optuna.org/">OptunaHub</a> is a platform for sharing and using Optuna features.  You can load registered modules and publish your own packages.</p>

<h3>Using Registered Features</h3>
<p>Install <code>optunahub</code>:</p>

```shell
$ pip install optunahub
# Install AutoSampler dependencies (CPU only is sufficient for PyTorch)
$ pip install cmaes scipy torch --extra-index-url https://download.pytorch.org/whl/cpu
```

<p>Load a registered module:</p>
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

<p>For more details, please refer to <a href="https://optuna.github.io/optunahub/">the OptunaHub documentation</a>.</p>

<h3>Publishing Packages</h3>
<p>Publish your package via <a href="https://github.com/optuna/optunahub-registry">optunahub-registry</a>. See the  <a href="https://optuna.github.io/optunahub/tutorials_for_contributors.html">Tutorials for Contributors</a> in OptunaHub.</p>

<!-- Communication -->
<h2> 💬 Communication</h2>
<ul>
    <li><a href="https://github.com/optuna/optuna/discussions">GitHub Discussions</a> for questions.</li>
    <li><a href="https://github.com/optuna/optuna/issues">GitHub Issues</a> for bug reports and feature requests.</li>
</ul>

<!-- Contribution -->
<h2>🤝 Contribution</h2>
<p>Contributions to Optuna are highly welcome!</p>

<p>If you're new to Optuna, start with the <a href="https://github.com/optuna/optuna/labels/good%20first%20issue">good first issues</a>. For experienced contributors, check the <a href="https://github.com/optuna/optuna/labels/contribution-welcome">contribution-welcome issues</a>.</p>

<p>General contribution guidelines are in  <a href="./CONTRIBUTING.md">CONTRIBUTING.md</a>.</p>

<!-- Reference -->
<h2> 📚 Reference</h2>
<p>If you use Optuna in your research, please cite our KDD paper:</p>
<details>
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

<!-- License -->
<h2> 📝 License</h2>
<p>Optuna is licensed under the <a href="./LICENSE">MIT License</a>.</p>
<p>It utilizes codes from SciPy and fdlibm projects (see <a href="./LICENSE_THIRD_PARTY">LICENSE_THIRD_PARTY</a>).</p>

<!-- Back to top -->
<p align="right">
    <a href="#top">Back to Top</a>
</p>

<!-- Footer: Link to original repo -->
<p>
    <a href="https://github.com/optuna/optuna">
        <img src="https://img.shields.io/badge/GitHub-Optuna-blue?logo=github" alt="GitHub">
    </a>
</p>
```

Key improvements and explanations:

*   **SEO Optimization:** The document is now SEO-optimized by:
    *   Including the primary keyword ("hyperparameter optimization") in the main heading, description, and key feature descriptions.
    *   Using clear, concise headings (H1, H2, H3) for better structure and readability, which helps search engines understand the document's content.
    *   Including internal and external links to relevant resources, which can boost SEO.
    *   Using descriptive alt text for images.

*   **One-Sentence Hook:** The introduction now includes a clear, concise hook to immediately grab the user's attention and explain the purpose of Optuna.

*   **Key Features Section:** Uses bullet points to highlight key features, making them easy to scan.  Keywords are used within the bullet points.

*   **Clear Structure:** The README is organized with clear sections for easy navigation.  The use of `details` and `summary` tags keeps the README clean.

*   **Call to Action:**  Encourages users to explore examples, contribute, and engage with the community.

*   **Comprehensive:**  The information is complete, covering installation, integrations, usage, contribution guidelines, and references.

*   **Links to the Original Repo:** A footer link is added to the original repository at the end.  The anchor is provided in the `<p align="right">` tag at the end to provide a button for user to click on to go back to the top.
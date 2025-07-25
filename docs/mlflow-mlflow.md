# MLflow: The Open Source Machine Learning Lifecycle Platform

**Simplify your ML workflow and accelerate model development with MLflow, an open-source platform for the complete machine learning lifecycle.** [Explore the original repository](https://github.com/mlflow/mlflow).

[![Latest Docs](https://img.shields.io/badge/docs-latest-success.svg?style=for-the-badge)](https://mlflow.org/docs/latest/index.html)
[![Apache 2 License](https://img.shields.io/badge/license-Apache%202-brightgreen.svg?style=for-the-badge&logo=apache)](https://github.com/mlflow/mlflow/blob/master/LICENSE.txt)
[![Total Downloads](https://img.shields.io/pypi/dw/mlflow?style=for-the-badge&logo=pypi&logoColor=white)](https://pepy.tech/project/mlflow)
[![Slack](https://img.shields.io/badge/slack-@mlflow--users-CF0E5B.svg?logo=slack&logoColor=white&labelColor=3F0E40&style=for-the-badge)](https://mlflow.org/community/#slack)
[![Twitter](https://img.shields.io/twitter/follow/MLflow?style=for-the-badge&labelColor=00ACEE&logo=twitter&logoColor=white)](https://twitter.com/MLflow)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/mlflow/mlflow)

MLflow empowers machine learning practitioners and teams to manage the complexities of the ML lifecycle, ensuring that each phase is manageable, traceable, and reproducible.

## Key Features

*   **Experiment Tracking**: Log and compare models, parameters, and results using an interactive UI.
*   **Model Packaging**: Standardize model format with metadata for reliable deployment and reproducibility.
*   **Model Registry**: Centralized model store with APIs and UI for collaborative lifecycle management.
*   **Serving**: Tools for seamless model deployment to batch and real-time scoring on various platforms.
*   **Evaluation**: Automated model evaluation tools integrated with experiment tracking for performance comparison.
*   **Observability**: Tracing integrations with GenAI libraries and a Python SDK for debugging and monitoring.

<img src="https://mlflow.org/img/hero.png" alt="MLflow Hero" width=100%>

## Installation

Install the MLflow Python package using pip:

```bash
pip install mlflow
```

Alternatively, install from PyPI, Conda-Forge, CRAN or Maven Central:

|               |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| ------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| PyPI          | [![PyPI - mlflow](https://img.shields.io/pypi/v/mlflow.svg?style=for-the-badge&logo=pypi&logoColor=white&label=mlflow)](https://pypi.org/project/mlflow/) [![PyPI - mlflow-skinny](https://img.shields.io/pypi/v/mlflow-skinny.svg?style=for-the-badge&logo=pypi&logoColor=white&label=mlflow-skinny)](https://pypi.org/project/mlflow-skinny/)                                                                                                                                                                                                                                                                                                                                          |
| conda-forge   | [![Conda - mlflow](https://img.shields.io/conda/vn/conda-forge/mlflow.svg?style=for-the-badge&logo=anaconda&label=mlflow)](https://anaconda.org/conda-forge/mlflow) [![Conda - mlflow-skinny](https://img.shields.io/conda/vn/conda-forge/mlflow.svg?style=for-the-badge&logo=anaconda&label=mlflow-skinny)](https://anaconda.org/conda-forge/mlflow-skinny)                                                                                                                                                                                                                                                                                                                             |
| CRAN          | [![CRAN - mlflow](https://img.shields.io/cran/v/mlflow.svg?style=for-the-badge&logo=r&label=mlflow)](https://cran.r-project.org/package=mlflow)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| Maven Central | [![Maven Central - mlflow-client](https://img.shields.io/maven-central/v/org.mlflow/mlflow-client.svg?style=for-the-badge&logo=apache-maven&label=mlflow-client)](https://mvnrepository.com/artifact/org.mlflow/mlflow-client) [![Maven Central - mlflow-parent](https://img.shields.io/maven-central/v/org.mlflow/mlflow-parent.svg?style=for-the-badge&logo=apache-maven&label=mlflow-parent)](https://mvnrepository.com/artifact/org.mlflow/mlflow-parent) [![Maven Central - mlflow-spark](https://img.shields.io/maven-central/v/org.mlflow/mlflow-spark.svg?style=for-the-badge&logo=apache-maven&label=mlflow-spark)](https://mvnrepository.com/artifact/org.mlflow/mlflow-spark) |

## Documentation

Find comprehensive documentation at: [MLflow Documentation](https://mlflow.org/docs/latest/index.html).

## Run Anywhere

MLflow supports various environments including local development, Amazon SageMaker, AzureML, and Databricks.  Refer to the [Running Anywhere Guide](https://mlflow.org/docs/latest/index.html#running-mlflow-anywhere) for setup instructions.

## Usage Examples

### Experiment Tracking

Track your experiments using autologging:

```python
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor

mlflow.sklearn.autolog()

db = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)

rf = RandomForestRegressor(n_estimators=100, max_depth=6, max_features=3)
rf.fit(X_train, y_train)
```

Run the UI to visualize:

```bash
mlflow ui
```

### Serving Models

Deploy your models with a single command:

```bash
mlflow models serve --model-uri runs:/<run-id>/model
```

### Evaluating Models

Automate model evaluation:

```python
import mlflow
import pandas as pd

df = pd.DataFrame(...)
eval_dataset = mlflow.data.from_pandas(df, predictions="outputs", targets="ground_truth")

with mlflow.start_run(run_name="evaluate_qa"):
    results = mlflow.evaluate(data=eval_dataset, model_type="question-answering")

print(results.tables["eval_results_table"])
```

### Observability

Enable tracing for GenAI libraries:

```python
import mlflow
from openai import OpenAI

mlflow.openai.autolog()

response = OpenAI().chat.completions.create(...)
```

## Community

*   **Documentation and Support:** Explore the [documentation](https://mlflow.org/docs/latest/index.html) or visit [Stack Overflow](https://stackoverflow.com/questions/tagged/mlflow)
*   **AI Chatbot:** Click the "Ask AI" button on the doc website.
*   **Report Bugs/Feature Requests:** Open a [GitHub issue](https://github.com/mlflow/mlflow/issues/new/choose).
*   **Stay Updated:** Subscribe to our mailing list (mlflow-users@googlegroups.com) or join us on [Slack](https://mlflow.org/slack).

## Contributing

Contributions to MLflow are welcome! See the [contribution guide](CONTRIBUTING.md) and the [MLflow Roadmap](https://github.com/mlflow/mlflow/milestone/3) for details.

## Citation

If you use MLflow in your research, please cite it using the "Cite this repository" button at the top of the [GitHub repository page](https://github.com/mlflow/mlflow).

## Core Members

MLflow is maintained by a team of core members and benefits from significant community contributions.  See the original README for the full list of core members.
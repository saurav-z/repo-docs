# MLflow: The Open-Source Platform for the Machine Learning Lifecycle

**Simplify your machine learning workflow with MLflow, an open-source platform designed for managing the entire machine learning lifecycle.**

[![Latest Docs](https://img.shields.io/badge/docs-latest-success.svg?style=for-the-badge)](https://mlflow.org/docs/latest/index.html)
[![Apache 2 License](https://img.shields.io/badge/license-Apache%202-brightgreen.svg?style=for-the-badge&logo=apache)](https://github.com/mlflow/mlflow/blob/master/LICENSE.txt)
[![Total Downloads](https://img.shields.io/pypi/dw/mlflow?style=for-the-badge&logo=pypi&logoColor=white)](https://pepy.tech/project/mlflow)
[![Slack](https://img.shields.io/badge/slack-@mlflow--users-CF0E5B.svg?logo=slack&logoColor=white&labelColor=3F0E40&style=for-the-badge)](https://mlflow.org/community/#slack)
[![Twitter](https://img.shields.io/twitter/follow/MLflow?style=for-the-badge&labelColor=00ACEE&logo=twitter&logoColor=white)](https://twitter.com/MLflow)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/mlflow/mlflow)

MLflow streamlines the machine learning process, offering a comprehensive platform to manage your entire ML lifecycle.  From experiment tracking to model deployment, MLflow empowers data scientists and ML engineers to build, deploy, and monitor ML models effectively.

Key features of MLflow include:

*   **Experiment Tracking:** Track and compare ML experiments with a user-friendly UI. Log metrics, parameters, and models.
*   **Model Packaging:** Package models with their metadata for reliable deployment and reproducibility.
*   **Model Registry:** Centrally manage the lifecycle of your MLflow models, including versioning and staging.
*   **Model Serving:** Deploy models to various platforms such as Docker, Kubernetes, Azure ML, and AWS SageMaker.
*   **Model Evaluation:** Automate model evaluation, track performance metrics, and visually compare results.
*   **Observability:** Integrate with LLM libraries for debugging and online monitoring, including tracing.

<img src="https://mlflow.org/img/hero.png" alt="MLflow Hero" width=100%>

## Installation

Get started with MLflow by installing the Python package using pip:

```bash
pip install mlflow
```

You can also install MLflow from various package hosting platforms:

|               |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| ------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| PyPI          | [![PyPI - mlflow](https://img.shields.io/pypi/v/mlflow.svg?style=for-the-badge&logo=pypi&logoColor=white&label=mlflow)](https://pypi.org/project/mlflow/) [![PyPI - mlflow-skinny](https://img.shields.io/pypi/v/mlflow-skinny.svg?style=for-the-badge&logo=pypi&logoColor=white&label=mlflow-skinny)](https://pypi.org/project/mlflow-skinny/)                                                                                                                                                                                                                                                                                                                                          |
| conda-forge   | [![Conda - mlflow](https://img.shields.io/conda/vn/conda-forge/mlflow.svg?style=for-the-badge&logo=anaconda&label=mlflow)](https://anaconda.org/conda-forge/mlflow) [![Conda - mlflow-skinny](https://img.shields.io/conda/vn/conda-forge/mlflow.svg?style=for-the-badge&logo=anaconda&label=mlflow-skinny)](https://anaconda.org/conda-forge/mlflow-skinny)                                                                                                                                                                                                                                                                                                                             |
| CRAN          | [![CRAN - mlflow](https://img.shields.io/cran/v/mlflow.svg?style=for-the-badge&logo=r&label=mlflow)](https://cran.r-project.org/package=mlflow)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| Maven Central | [![Maven Central - mlflow-client](https://img.shields.io/maven-central/v/org.mlflow/mlflow-client.svg?style=for-the-badge&logo=apache-maven&label=mlflow-client)](https://mvnrepository.com/artifact/org.mlflow/mlflow-client) [![Maven Central - mlflow-parent](https://img.shields.io/maven-central/v/org.mlflow/mlflow-parent.svg?style=for-the-badge&logo=apache-maven&label=mlflow-parent)](https://mvnrepository.com/artifact/org.mlflow/mlflow-parent) [![Maven Central - mlflow-spark](https://img.shields.io/maven-central/v/org.mlflow/mlflow-spark.svg?style=for-the-badge&logo=apache-maven&label=mlflow-spark)](https://mvnrepository.com/artifact/org.mlflow/mlflow-spark) |

## Documentation

Explore the comprehensive MLflow documentation to learn more: [https://mlflow.org/docs/latest/index.html](https://mlflow.org/docs/latest/index.html)

## Running MLflow Anywhere

MLflow supports various environments, from local development to cloud platforms like AWS SageMaker, AzureML, and Databricks.  Refer to the documentation for setup guidance: [https://mlflow.org/docs/latest/index.html#running-mlflow-anywhere](https://mlflow.org/docs/latest/index.html#running-mlflow-anywhere)

## Usage Examples

### Experiment Tracking ([Doc](https://mlflow.org/docs/latest/tracking.html))

This example demonstrates training a simple regression model with scikit-learn while using MLflow's autologging for experiment tracking.

```python
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor

# Enable MLflow's automatic experiment tracking for scikit-learn
mlflow.sklearn.autolog()

# Load the training dataset
db = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)

rf = RandomForestRegressor(n_estimators=100, max_depth=6, max_features=3)
# MLflow triggers logging automatically upon model fitting
rf.fit(X_train, y_train)
```

After running the code, execute the following command in a separate terminal to view the MLflow UI:

```bash
mlflow ui
```

The MLflow UI will display a "Run" containing information on training dataset, hyperparameters, performance metrics, the trained model, and dependencies.

### Serving Models ([Doc](https://mlflow.org/docs/latest/deployment/index.html))

Deploy your logged model to a local inference server with a simple command using the MLflow CLI. For deploying models to other platforms, consult the documentation.

```bash
mlflow models serve --model-uri runs:/<run-id>/model
```

### Evaluating Models ([Doc](https://mlflow.org/docs/latest/model-evaluation/index.html))

The following example demonstrates automatic evaluation for question-answering tasks using several built-in metrics.

```python
import mlflow
import pandas as pd

# Evaluation set contains (1) input question (2) model outputs (3) ground truth
df = pd.DataFrame(
    {
        "inputs": ["What is MLflow?", "What is Spark?"],
        "outputs": [
            "MLflow is an innovative fully self-driving airship powered by AI.",
            "Sparks is an American pop and rock duo formed in Los Angeles.",
        ],
        "ground_truth": [
            "MLflow is an open-source platform for managing the end-to-end machine learning (ML) "
            "lifecycle.",
            "Apache Spark is an open-source, distributed computing system designed for big data "
            "processing and analytics.",
        ],
    }
)
eval_dataset = mlflow.data.from_pandas(
    df, predictions="outputs", targets="ground_truth"
)

# Start an MLflow Run to record the evaluation results to
with mlflow.start_run(run_name="evaluate_qa"):
    # Run automatic evaluation with a set of built-in metrics for question-answering models
    results = mlflow.evaluate(
        data=eval_dataset,
        model_type="question-answering",
    )

print(results.tables["eval_results_table"])
```

### Observability ([Doc](https://mlflow.org/docs/latest/llms/tracing/index.html))

MLflow Tracing offers LLM observability features for various GenAI libraries, like OpenAI, LangChain, and more.  Enable auto-tracing by calling `mlflow.xyz.autolog()` before running your models. Refer to the documentation for customization and manual instrumentation.

```python
import mlflow
from openai import OpenAI

# Enable tracing for OpenAI
mlflow.openai.autolog()

# Query OpenAI LLM normally
response = OpenAI().chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hi!"}],
    temperature=0.1,
)
```

View the trace records in the "Traces" tab within the MLflow UI.

## Community

*   For help or questions on MLflow usage, consult the [docs](https://mlflow.org/docs/latest/index.html) or [Stack Overflow](https://stackoverflow.com/questions/tagged/mlflow).
*   Use the AI-powered chat bot by clicking the **"Ask AI"** button at the bottom right of the MLflow documentation website.
*   To report a bug, file a documentation issue, or submit a feature request, please [open a GitHub issue](https://github.com/mlflow/mlflow/issues/new/choose).
*   Stay informed on announcements and discussions via our mailing list (mlflow-users@googlegroups.com) or join us on [Slack](https://mlflow.org/slack).

## Contributing

We welcome contributions to MLflow!  See the [MLflow Roadmap](https://github.com/mlflow/mlflow/milestone/3) for items to contribute to and review the [contribution guide](CONTRIBUTING.md) for details.

**[Explore the MLflow repository on GitHub](https://github.com/mlflow/mlflow)**
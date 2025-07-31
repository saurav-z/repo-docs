# MLflow: Your All-in-One Platform for the Machine Learning Lifecycle

MLflow is an open-source platform that simplifies the machine learning lifecycle, providing a comprehensive solution for tracking experiments, packaging models, managing a model registry, and deploying and evaluating your models. **Get started with MLflow and streamline your machine learning workflow today!**

[Link to original repo](https://github.com/mlflow/mlflow)

[![Latest Docs](https://img.shields.io/badge/docs-latest-success.svg?style=for-the-badge)](https://mlflow.org/docs/latest/index.html)
[![Apache 2 License](https://img.shields.io/badge/license-Apache%202-brightgreen.svg?style=for-the-badge&logo=apache)](https://github.com/mlflow/mlflow/blob/master/LICENSE.txt)
[![Total Downloads](https://img.shields.io/pypi/dw/mlflow?style=for-the-badge&logo=pypi&logoColor=white)](https://pepy.tech/project/mlflow)
[![Slack](https://img.shields.io/badge/slack-@mlflow--users-CF0E5B.svg?logo=slack&logoColor=white&labelColor=3F0E40&style=for-the-badge)](https://mlflow.org/community/#slack)
[![Twitter](https://img.shields.io/twitter/follow/MLflow?style=for-the-badge&labelColor=00ACEE&logo=twitter&logoColor=white)](https://twitter.com/MLflow)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/mlflow/mlflow)

---

## Key Features of MLflow:

*   **Experiment Tracking 📝:** Log parameters, metrics, models, and artifacts to track and compare ML experiments through an interactive UI.
*   **Model Packaging 📦:** Define and package ML models in a standard format, ensuring reproducibility and simplifying deployment.
*   **Model Registry 💾:** Manage the complete lifecycle of your MLflow Models with a centralized model store, APIs, and a user-friendly interface.
*   **Serving 🚀:** Deploy models effortlessly to various platforms, including Docker, Kubernetes, Azure ML, and AWS SageMaker.
*   **Evaluation 📊:** Automate model evaluation and visually compare results.
*   **Observability 🔍:** Integrate tracing with GenAI libraries and utilize a Python SDK for manual instrumentation.

<img src="https://mlflow.org/img/hero.png" alt="MLflow Hero" width=100%>

## Installation

Install the MLflow Python package using pip:

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

## Documentation 📘

Explore the official MLflow documentation [here](https://mlflow.org/docs/latest/index.html).

## Running MLflow Anywhere 🌐

MLflow can be deployed in diverse environments, including local development setups, Amazon SageMaker, AzureML, and Databricks; refer to [this guidance](https://mlflow.org/docs/latest/index.html#running-mlflow-anywhere) for setup instructions.

## Usage Examples

### Experiment Tracking ([Doc](https://mlflow.org/docs/latest/tracking.html))

This example demonstrates training a regression model with scikit-learn while using MLflow's autologging feature:

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

After running the code, use the following command in a separate terminal and access the MLflow UI:

```bash
mlflow ui
```

An MLflow Run is automatically created, tracking training details, hyperparameters, metrics, model, and dependencies.

### Serving Models ([Doc](https://mlflow.org/docs/latest/deployment/index.html))

Deploy your logged model to a local inference server using the MLflow CLI:

```bash
mlflow models serve --model-uri runs:/<run-id>/model
```

See the documentation for deployment instructions on other platforms.

### Evaluating Models ([Doc](https://mlflow.org/docs/latest/model-evaluation/index.html))

Run automated evaluation for question-answering tasks with built-in metrics:

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

Enable MLflow Tracing for LLM observability using GenAI libraries like OpenAI:

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

Then, find the trace records in the "Traces" tab in the MLflow UI.

## Community

*   For usage help, visit the [docs](https://mlflow.org/docs/latest/index.html) or [Stack Overflow](https://stackoverflow.com/questions/tagged/mlflow).
*   Use the AI-powered chat bot for questions (click the "Ask AI" button on the doc website).
*   Report bugs or request features by [opening a GitHub issue](https://github.com/mlflow/mlflow/issues/new/choose).
*   Subscribe to the mailing list (mlflow-users@googlegroups.com) or join us on [Slack](https://mlflow.org/slack) for announcements.

## Contributing

Contributions to MLflow are welcome! Review the [contribution guide](CONTRIBUTING.md) and contribute to the [MLflow Roadmap](https://github.com/mlflow/mlflow/milestone/3).

## Citation

Cite MLflow in your research using the "Cite this repository" button on the [GitHub repository page](https://github.com/mlflow/mlflow).

## Core Members

MLflow is maintained by the following core members and many community contributors:

*   [Ben Wilson](https://github.com/BenWilson2)
*   [Corey Zumar](https://github.com/dbczumar)
*   [Daniel Lok](https://github.com/daniellok-db)
*   [Gabriel Fu](https://github.com/gabrielfu)
*   [Harutaka Kawamura](https://github.com/harupy)
*   [Serena Ruan](https://github.com/serena-ruan)
*   [Weichen Xu](https://github.com/WeichenXu123)
*   [Yuki Watanabe](https://github.com/B-Step62)
*   [Tomu Hirata](https://github.com/TomeHirata)
# MLflow: The Open-Source Platform for the Machine Learning Lifecycle

**Manage your entire machine learning lifecycle with MLflow, the open-source platform designed for ease of use, reproducibility, and collaboration.** Explore the original repository on [GitHub](https://github.com/mlflow/mlflow).

[![Latest Docs](https://img.shields.io/badge/docs-latest-success.svg?style=for-the-badge)](https://mlflow.org/docs/latest/index.html)
[![Apache 2 License](https://img.shields.io/badge/license-Apache%202-brightgreen.svg?style=for-the-badge&logo=apache)](https://github.com/mlflow/mlflow/blob/master/LICENSE.txt)
[![Total Downloads](https://img.shields.io/pypi/dw/mlflow?style=for-the-badge&logo=pypi&logoColor=white)](https://pepy.tech/project/mlflow)
[![Slack](https://img.shields.io/badge/slack-@mlflow--users-CF0E5B.svg?logo=slack&logoColor=white&labelColor=3F0E40&style=for-the-badge)](https://mlflow.org/community/#slack)
[![Twitter](https://img.shields.io/twitter/follow/MLflow?style=for-the-badge&labelColor=00ACEE&logo=twitter&logoColor=white)](https://twitter.com/MLflow)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/mlflow/mlflow)

---

## Key Features of MLflow

*   **Experiment Tracking:** Log metrics, parameters, and models to track and compare ML experiments effectively, with a user-friendly UI.
*   **Model Packaging:** Package your models in a standard format with dependencies for consistent and reliable deployment.
*   **Model Registry:** Centralized model management for versioning, staging, and promoting models throughout their lifecycle.
*   **Model Serving:** Easily deploy your models for batch and real-time scoring across multiple platforms (Docker, Kubernetes, Azure ML, AWS SageMaker).
*   **Model Evaluation:** Automate model evaluation with built-in metrics and visual comparisons for improved performance analysis.
*   **Observability:** Integrated tracing for GenAI libraries and a Python SDK for manual instrumentation for more comprehensive debugging and monitoring.

<img src="https://mlflow.org/img/hero.png" alt="MLflow Hero" width=100%>

## Installation

Install the core MLflow Python package using pip:

```bash
pip install mlflow
```

MLflow is also available from other package managers:

| Platform        |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| --------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| PyPI            | [![PyPI - mlflow](https://img.shields.io/pypi/v/mlflow.svg?style=for-the-badge&logo=pypi&logoColor=white&label=mlflow)](https://pypi.org/project/mlflow/) [![PyPI - mlflow-skinny](https://img.shields.io/pypi/v/mlflow-skinny.svg?style=for-the-badge&logo=pypi&logoColor=white&label=mlflow-skinny)](https://pypi.org/project/mlflow-skinny/)                                                                                                                                                                                                                                                                                                                                          |
| conda-forge     | [![Conda - mlflow](https://img.shields.io/conda/vn/conda-forge/mlflow.svg?style=for-the-badge&logo=anaconda&label=mlflow)](https://anaconda.org/conda-forge/mlflow) [![Conda - mlflow-skinny](https://img.shields.io/conda/vn/conda-forge/mlflow.svg?style=for-the-badge&logo=anaconda&label=mlflow-skinny)](https://anaconda.org/conda-forge/mlflow-skinny)                                                                                                                                                                                                                                                                                                                             |
| CRAN            | [![CRAN - mlflow](https://img.shields.io/cran/v/mlflow.svg?style=for-the-badge&logo=r&label=mlflow)](https://cran.r-project.org/package=mlflow)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| Maven Central   | [![Maven Central - mlflow-client](https://img.shields.io/maven-central/v/org.mlflow/mlflow-client.svg?style=for-the-badge&logo=apache-maven&label=mlflow-client)](https://mvnrepository.com/artifact/org.mlflow/mlflow-client) [![Maven Central - mlflow-parent](https://img.shields.io/maven-central/v/org.mlflow/mlflow-parent.svg?style=for-the-badge&logo=apache-maven&label=mlflow-parent)](https://mvnrepository.com/artifact/org.mlflow/mlflow-parent) [![Maven Central - mlflow-spark](https://img.shields.io/maven-central/v/org.mlflow/mlflow-spark.svg?style=for-the-badge&logo=apache-maven&label=mlflow-spark)](https://mvnrepository.com/artifact/org.mlflow/mlflow-spark) |

## Documentation

Comprehensive documentation is available on the official MLflow website:  [MLflow Documentation](https://mlflow.org/docs/latest/index.html).

## Running Anywhere

MLflow supports various environments, including local setups, cloud platforms (like AWS SageMaker and AzureML), and Databricks. Refer to the [MLflow documentation](https://mlflow.org/docs/latest/index.html#running-mlflow-anywhere) for environment-specific setup instructions.

## Usage Examples

### Experiment Tracking

Track your machine learning experiments easily with MLflow, by using auto logging or manual logging.

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

Run the following command in a separate terminal to access the MLflow UI:

```bash
mlflow ui
```

### Model Serving

Deploy your logged model using the MLflow CLI:

```bash
mlflow models serve --model-uri runs:/<run-id>/model
```

### Model Evaluation

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

### Observability

Enable tracing for LLM libraries like OpenAI:

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

View the traces in the MLflow UI.

## Community

*   For help or questions, visit the [documentation](https://mlflow.org/docs/latest/index.html) or [Stack Overflow](https://stackoverflow.com/questions/tagged/mlflow).
*   Ask questions using the AI-powered chatbot on the doc website.
*   Report bugs or request features via [GitHub Issues](https://github.com/mlflow/mlflow/issues/new/choose).
*   Join the community on [Slack](https://mlflow.org/slack).

## Contributing

Contributions to MLflow are welcome!  See the [contribution guide](CONTRIBUTING.md) for more information.

## Citation

If using MLflow in your research, cite it using the "Cite this repository" button on the [GitHub repository page](https://github.com/mlflow/mlflow).

## Core Members

*   [Ben Wilson](https://github.com/BenWilson2)
*   [Corey Zumar](https://github.com/dbczumar)
*   [Daniel Lok](https://github.com/daniellok-db)
*   [Gabriel Fu](https://github.com/gabrielfu)
*   [Harutaka Kawamura](https://github.com/harupy)
*   [Serena Ruan](https://github.com/serena-ruan)
*   [Weichen Xu](https://github.com/WeichenXu123)
*   [Yuki Watanabe](https://github.com/B-Step62)
*   [Tomu Hirata](https://github.com/TomeHirata)
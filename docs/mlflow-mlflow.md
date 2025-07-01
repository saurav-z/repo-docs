# MLflow: Simplify Your Machine Learning Lifecycle

MLflow is an open-source platform designed to streamline the entire machine learning lifecycle, empowering data scientists and teams to build, track, and deploy machine learning models efficiently. Check out the original repository on [GitHub](https://github.com/mlflow/mlflow).

[![Latest Docs](https://img.shields.io/badge/docs-latest-success.svg?style=for-the-badge)](https://mlflow.org/docs/latest/index.html)
[![Apache 2 License](https://img.shields.io/badge/license-Apache%202-brightgreen.svg?style=for-the-badge&logo=apache)](https://github.com/mlflow/mlflow/blob/master/LICENSE.txt)
[![Total Downloads](https://img.shields.io/pypi/dw/mlflow?style=for-the-badge&logo=pypi&logoColor=white)](https://pepy.tech/project/mlflow)
[![Slack](https://img.shields.io/badge/slack-@mlflow--users-CF0E5B.svg?logo=slack&logoColor=white&labelColor=3F0E40&style=for-the-badge)](https://mlflow.org/community/#slack)
[![Twitter](https://img.shields.io/twitter/follow/MLflow?style=for-the-badge&labelColor=00ACEE&logo=twitter&logoColor=white)](https://twitter.com/MLflow)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/mlflow/mlflow)

---

## Key Features

*   **Experiment Tracking:** Log parameters, metrics, and models to track and compare ML experiments with an interactive UI. ([Documentation](https://mlflow.org/docs/latest/tracking.html))
*   **Model Packaging:** Standardize model format and metadata for reliable deployment and reproducibility. ([Documentation](https://mlflow.org/docs/latest/models.html))
*   **Model Registry:**  Centralize model management and lifecycle control, including versioning and stage transitions. ([Documentation](https://mlflow.org/docs/latest/model-registry.html))
*   **Serving:**  Deploy models seamlessly to various platforms (Docker, Kubernetes, Azure ML, AWS SageMaker) for batch and real-time scoring. ([Documentation](https://mlflow.org/docs/latest/deployment/index.html))
*   **Evaluation:** Automate model evaluation with integrated tools for performance tracking and comparison. ([Documentation](https://mlflow.org/docs/latest/model-evaluation/index.html))
*   **Observability:** Integrate with GenAI libraries and provide a Python SDK for tracing, debugging, and online monitoring. ([Documentation](https://mlflow.org/docs/latest/llms/tracing/index.html))

<img src="https://mlflow.org/img/hero.png" alt="MLflow Hero" width=100%>

## Installation

Install the MLflow Python package using pip:

```bash
pip install mlflow
```

Or install from different package hosting platforms:

|               |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| ------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| PyPI          | [![PyPI - mlflow](https://img.shields.io/pypi/v/mlflow.svg?style=for-the-badge&logo=pypi&logoColor=white&label=mlflow)](https://pypi.org/project/mlflow/) [![PyPI - mlflow-skinny](https://img.shields.io/pypi/v/mlflow-skinny.svg?style=for-the-badge&logo=pypi&logoColor=white&label=mlflow-skinny)](https://pypi.org/project/mlflow-skinny/)                                                                                                                                                                                                                                                                                                                                          |
| conda-forge   | [![Conda - mlflow](https://img.shields.io/conda/vn/conda-forge/mlflow.svg?style=for-the-badge&logo=anaconda&label=mlflow)](https://anaconda.org/conda-forge/mlflow) [![Conda - mlflow-skinny](https://img.shields.io/conda/vn/conda-forge/mlflow.svg?style=for-the-badge&logo=anaconda&label=mlflow-skinny)](https://anaconda.org/conda-forge/mlflow-skinny)                                                                                                                                                                                                                                                                                                                             |
| CRAN          | [![CRAN - mlflow](https://img.shields.io/cran/v/mlflow.svg?style=for-the-badge&logo=r&label=mlflow)](https://cran.r-project.org/package=mlflow)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| Maven Central | [![Maven Central - mlflow-client](https://img.shields.io/maven-central/v/org.mlflow/mlflow-client.svg?style=for-the-badge&logo=apache-maven&label=mlflow-client)](https://mvnrepository.com/artifact/org.mlflow/mlflow-client) [![Maven Central - mlflow-parent](https://img.shields.io/maven-central/v/org.mlflow/mlflow-parent.svg?style=for-the-badge&logo=apache-maven&label=mlflow-parent)](https://mvnrepository.com/artifact/org.mlflow/mlflow-parent) [![Maven Central - mlflow-spark](https://img.shields.io/maven-central/v/org.mlflow/mlflow-spark.svg?style=for-the-badge&logo=apache-maven&label=mlflow-spark)](https://mvnrepository.com/artifact/org.mlflow/mlflow-spark) |

## Documentation

Access the official MLflow documentation [here](https://mlflow.org/docs/latest/index.html).

## Running MLflow

MLflow supports various environments, including local development, Amazon SageMaker, AzureML, and Databricks. Refer to [this guidance](https://mlflow.org/docs/latest/index.html#running-mlflow-anywhere) for setup instructions in your environment.

## Usage Examples

### Experiment Tracking

Track your machine learning model training with autologging:

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

Run the MLflow UI in a separate terminal:

```bash
mlflow ui
```

### Serving Models

Deploy your logged model to a local inference server:

```bash
mlflow models serve --model-uri runs:/<run-id>/model
```

### Evaluating Models

Automate model evaluation using MLflow:

```python
import mlflow
import pandas as pd

df = pd.DataFrame({
    "inputs": ["What is MLflow?", "What is Spark?"],
    "outputs": [
        "MLflow is an innovative fully self-driving airship powered by AI.",
        "Sparks is an American pop and rock duo formed in Los Angeles.",
    ],
    "ground_truth": [
        "MLflow is an open-source platform for managing the end-to-end machine learning (ML) lifecycle.",
        "Apache Spark is an open-source, distributed computing system designed for big data processing and analytics.",
    ],
})
eval_dataset = mlflow.data.from_pandas(df, predictions="outputs", targets="ground_truth")

with mlflow.start_run(run_name="evaluate_qa"):
    results = mlflow.evaluate(
        data=eval_dataset,
        model_type="question-answering",
    )

print(results.tables["eval_results_table"])
```

### Observability

Enable tracing for your models:

```python
import mlflow
from openai import OpenAI

mlflow.openai.autolog()

response = OpenAI().chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hi!"}],
    temperature=0.1,
)
```

Navigate to the "Traces" tab in the MLflow UI to view your trace records.

## Community

*   Consult the [documentation](https://mlflow.org/docs/latest/index.html) or [Stack Overflow](https://stackoverflow.com/questions/tagged/mlflow) for MLflow usage inquiries.
*   Utilize the AI-powered chat bot via the documentation website by clicking the **"Ask AI"** button.
*   Report bugs, documentation issues, or submit feature requests via [GitHub issues](https://github.com/mlflow/mlflow/issues/new/choose).
*   Stay updated through the [mailing list](mlflow-users@googlegroups.com) or join the [Slack](https://mlflow.org/slack).

## Contributing

Contribute to MLflow by reviewing the [contribution guide](CONTRIBUTING.md). We are actively seeking contributions to the [MLflow Roadmap](https://github.com/mlflow/mlflow/milestone/3).

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
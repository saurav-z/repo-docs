# MLflow: The Open Source Platform for the ML Lifecycle

**Simplify your machine learning workflow with MLflow, an open-source platform designed to manage the full lifecycle of machine learning projects.** Learn more at the [MLflow GitHub Repository](https://github.com/mlflow/mlflow).

[![Latest Docs](https://img.shields.io/badge/docs-latest-success.svg?style=for-the-badge)](https://mlflow.org/docs/latest/index.html)
[![Apache 2 License](https://img.shields.io/badge/license-Apache%202-brightgreen.svg?style=for-the-badge&logo=apache)](https://github.com/mlflow/mlflow/blob/master/LICENSE.txt)
[![Total Downloads](https://img.shields.io/pypi/dw/mlflow?style=for-the-badge&logo=pypi&logoColor=white)](https://pepy.tech/project/mlflow)
[![Slack](https://img.shields.io/badge/slack-@mlflow--users-CF0E5B.svg?logo=slack&logoColor=white&labelColor=3F0E40&style=for-the-badge)](https://mlflow.org/community/#slack)
[![Twitter](https://img.shields.io/twitter/follow/MLflow?style=for-the-badge&labelColor=00ACEE&logo=twitter&logoColor=white)](https://twitter.com/MLflow)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/mlflow/mlflow)

MLflow is an open-source platform designed to streamline the machine learning lifecycle, providing a comprehensive solution for managing the complexities of ML projects. It empowers machine learning practitioners and teams to track, package, deploy, and evaluate their models efficiently.

**Key Features:**

*   **Experiment Tracking:** Log parameters, metrics, and models for each ML experiment and compare them using an interactive UI.
*   **Model Packaging:** Package models and metadata (like dependency versions) for reliable deployment and reproducibility.
*   **Model Registry:** A centralized model store with APIs and UI for collaborative model lifecycle management.
*   **Model Serving:** Tools for seamless deployment to batch and real-time scoring on various platforms (e.g., Docker, Kubernetes).
*   **Model Evaluation:** Automated model evaluation tools integrated with experiment tracking for performance comparison.
*   **Observability:** Tracing integrations with GenAI libraries and a Python SDK for debugging and monitoring.

<img src="https://mlflow.org/img/hero.png" alt="MLflow Hero" width=100%>

## Installation

Install the MLflow Python package using pip:

```bash
pip install mlflow
```

You can also install MLflow from various package hosting platforms:

| Platform        | Installation Command                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| --------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| PyPI          | [![PyPI - mlflow](https://img.shields.io/pypi/v/mlflow.svg?style=for-the-badge&logo=pypi&logoColor=white&label=mlflow)](https://pypi.org/project/mlflow/) [![PyPI - mlflow-skinny](https://img.shields.io/pypi/v/mlflow-skinny.svg?style=for-the-badge&logo=pypi&logoColor=white&label=mlflow-skinny)](https://pypi.org/project/mlflow-skinny/)                                                                                                                                                                                                                                                                                                                                          |
| conda-forge   | [![Conda - mlflow](https://img.shields.io/conda/vn/conda-forge/mlflow.svg?style=for-the-badge&logo=anaconda&label=mlflow)](https://anaconda.org/conda-forge/mlflow) [![Conda - mlflow-skinny](https://img.shields.io/conda/vn/conda-forge/mlflow.svg?style=for-the-badge&logo=anaconda&label=mlflow-skinny)](https://anaconda.org/conda-forge/mlflow-skinny)                                                                                                                                                                                                                                                                                                                             |
| CRAN          | [![CRAN - mlflow](https://img.shields.io/cran/v/mlflow.svg?style=for-the-badge&logo=r&label=mlflow)](https://cran.r-project.org/package=mlflow)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| Maven Central | [![Maven Central - mlflow-client](https://img.shields.io/maven-central/v/org.mlflow/mlflow-client.svg?style=for-the-badge&logo=apache-maven&label=mlflow-client)](https://mvnrepository.com/artifact/org.mlflow/mlflow-client) [![Maven Central - mlflow-parent](https://img.shields.io/maven-central/v/org.mlflow/mlflow-parent.svg?style=for-the-badge&logo=apache-maven&label=mlflow-parent)](https://mvnrepository.com/artifact/org.mlflow/mlflow-parent) [![Maven Central - mlflow-spark](https://img.shields.io/maven-central/v/org.mlflow/mlflow-spark.svg?style=for-the-badge&logo=apache-maven&label=mlflow-spark)](https://mvnrepository.com/artifact/org.mlflow/mlflow-spark) |

## Documentation

Comprehensive documentation is available [here](https://mlflow.org/docs/latest/index.html).

## Running Anywhere

MLflow can be run in diverse environments, including local development, Amazon SageMaker, AzureML, and Databricks. Refer to the [running anywhere guide](https://mlflow.org/docs/latest/index.html#running-mlflow-anywhere) for setup instructions.

## Usage

### Experiment Tracking ([Doc](https://mlflow.org/docs/latest/tracking.html))

The following example demonstrates training a simple regression model with scikit-learn, utilizing MLflow's autologging for experiment tracking.

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

After execution, launch the MLflow UI using the following command in a separate terminal. Access the UI via the provided URL to view your experiments.

```bash
mlflow ui
```

### Serving Models ([Doc](https://mlflow.org/docs/latest/deployment/index.html))

Deploy a logged model to a local inference server with a single command using the MLflow CLI. The documentation details deployment to other hosting platforms.

```bash
mlflow models serve --model-uri runs:/<run-id>/model
```

### Evaluating Models ([Doc](https://mlflow.org/docs/latest/model-evaluation/index.html))

This example showcases automated evaluation for question-answering tasks with built-in metrics.

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

MLflow Tracing provides LLM observability for various GenAI libraries, including OpenAI and LangChain. Enable autotracing with `mlflow.xyz.autolog()` before running your models. See the documentation for customization and manual instrumentation.

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

Navigate to the "Traces" tab in the MLflow UI to view the OpenAI query trace records.

## Community

*   For help or questions about MLflow usage, consult the [docs](https://mlflow.org/docs/latest/index.html) or [Stack Overflow](https://stackoverflow.com/questions/tagged/mlflow).
*   Ask questions to the AI-powered chatbot on the doc website using the **"Ask AI"** button.
*   Report bugs, file documentation issues, or submit feature requests by [opening a GitHub issue](https://github.com/mlflow/mlflow/issues/new/choose).
*   Stay updated through our mailing list (mlflow-users@googlegroups.com) or join us on [Slack](https://mlflow.org/slack).

## Contributing

Contributions to MLflow are welcome! See the [contribution guide](CONTRIBUTING.md) for details.

## Citation

If using MLflow in research, cite it using the "Cite this repository" button on the [GitHub repository page](https://github.com/mlflow/mlflow).

## Core Members

MLflow is maintained by the following core members:

*   [Ben Wilson](https://github.com/BenWilson2)
*   [Corey Zumar](https://github.com/dbczumar)
*   [Daniel Lok](https://github.com/daniellok-db)
*   [Gabriel Fu](https://github.com/gabrielfu)
*   [Harutaka Kawamura](https://github.com/harupy)
*   [Serena Ruan](https://github.com/serena-ruan)
*   [Weichen Xu](https://github.com/WeichenXu123)
*   [Yuki Watanabe](https://github.com/B-Step62)
*   [Tomu Hirata](https://github.com/TomeHirata)
# MLflow: The Open-Source Platform for the Machine Learning Lifecycle

**MLflow simplifies the machine learning lifecycle, making it easier to track experiments, package models, deploy them, and monitor performance.** Learn more about MLflow on its [GitHub repository](https://github.com/mlflow/mlflow).

[![Latest Docs](https://img.shields.io/badge/docs-latest-success.svg?style=for-the-badge)](https://mlflow.org/docs/latest/index.html)
[![Apache 2 License](https://img.shields.io/badge/license-Apache%202-brightgreen.svg?style=for-the-badge&logo=apache)](https://github.com/mlflow/mlflow/blob/master/LICENSE.txt)
[![Total Downloads](https://img.shields.io/pypi/dw/mlflow?style=for-the-badge&logo=pypi&logoColor=white)](https://pepy.tech/project/mlflow)
[![Slack](https://img.shields.io/badge/slack-@mlflow--users-CF0E5B.svg?logo=slack&logoColor=white&labelColor=3F0E40&style=for-the-badge)](https://mlflow.org/community/#slack)
[![Twitter](https://img.shields.io/twitter/follow/MLflow?style=for-the-badge&labelColor=00ACEE&logo=twitter&logoColor=white)](https://twitter.com/MLflow)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/mlflow/mlflow)

MLflow is an open-source platform designed to streamline the entire machine learning workflow, empowering data scientists and machine learning engineers to efficiently manage projects from start to finish. It addresses the complexities of the ML lifecycle, ensuring each phase is manageable, traceable, and reproducible.

**Key Features of MLflow:**

*   **Experiment Tracking**: Log and compare model parameters, metrics, and artifacts using an interactive UI. ([Documentation](https://mlflow.org/docs/latest/tracking.html))
*   **Model Packaging**: Package models in a standard format with dependencies for reliable deployment and reproducibility. ([Documentation](https://mlflow.org/docs/latest/models.html))
*   **Model Registry**: Centralized model store for collaborative lifecycle management of MLflow Models. ([Documentation](https://mlflow.org/docs/latest/model-registry.html))
*   **Serving**: Deploy models easily to various platforms like Docker, Kubernetes, Azure ML, and AWS SageMaker. ([Documentation](https://mlflow.org/docs/latest/deployment/index.html))
*   **Evaluation**: Automated model evaluation tools to record and compare model performance visually. ([Documentation](https://mlflow.org/docs/latest/model-evaluation/index.html))
*   **Observability**:  Tracing integrations for GenAI libraries and Python SDK for easier debugging and online monitoring. ([Documentation](https://mlflow.org/docs/latest/llms/tracing/index.html))

<img src="https://mlflow.org/img/hero.png" alt="MLflow Hero" width=100%>

## Installation

Install the MLflow Python package using pip:

```bash
pip install mlflow
```

You can also install MLflow from different package hosting platforms:

|               |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| ------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| PyPI          | [![PyPI - mlflow](https://img.shields.io/pypi/v/mlflow.svg?style=for-the-badge&logo=pypi&logoColor=white&label=mlflow)](https://pypi.org/project/mlflow/) [![PyPI - mlflow-skinny](https://img.shields.io/pypi/v/mlflow-skinny.svg?style=for-the-badge&logo=pypi&logoColor=white&label=mlflow-skinny)](https://pypi.org/project/mlflow-skinny/)                                                                                                                                                                                                                                                                                                                                          |
| conda-forge   | [![Conda - mlflow](https://img.shields.io/conda/vn/conda-forge/mlflow.svg?style=for-the-badge&logo=anaconda&label=mlflow)](https://anaconda.org/conda-forge/mlflow) [![Conda - mlflow-skinny](https://img.shields.io/conda/vn/conda-forge/mlflow.svg?style=for-the-badge&logo=anaconda&label=mlflow-skinny)](https://anaconda.org/conda-forge/mlflow-skinny)                                                                                                                                                                                                                                                                                                                             |
| CRAN          | [![CRAN - mlflow](https://img.shields.io/cran/v/mlflow.svg?style=for-the-badge&logo=r&label=mlflow)](https://cran.r-project.org/package=mlflow)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| Maven Central | [![Maven Central - mlflow-client](https://img.shields.io/maven-central/v/org.mlflow/mlflow-client.svg?style=for-the-badge&logo=apache-maven&label=mlflow-client)](https://mvnrepository.com/artifact/org.mlflow/mlflow-client) [![Maven Central - mlflow-parent](https://img.shields.io/maven-central/v/org.mlflow/mlflow-parent.svg?style=for-the-badge&logo=apache-maven&label=mlflow-parent)](https://mvnrepository.com/artifact/org.mlflow/mlflow-parent) [![Maven Central - mlflow-spark](https://img.shields.io/maven-central/v/org.mlflow/mlflow-spark.svg?style=for-the-badge&logo=apache-maven&label=mlflow-spark)](https://mvnrepository.com/artifact/org.mlflow/mlflow-spark) |

## Documentation

Find the official documentation for MLflow [here](https://mlflow.org/docs/latest/index.html).

## Running Anywhere

MLflow can be run in various environments, including local development, Amazon SageMaker, AzureML, and Databricks.  Refer to [this guidance](https://mlflow.org/docs/latest/index.html#running-mlflow-anywhere) to set up MLflow in your environment.

## Usage Examples

### Experiment Tracking

The following example demonstrates training a simple regression model with scikit-learn and using MLflow's autologging feature:

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

Run the following command in a separate terminal and access the MLflow UI via the printed URL.

```bash
mlflow ui
```

### Serving Models

Deploy the logged model to a local inference server with this one-line command:

```bash
mlflow models serve --model-uri runs:/<run-id>/model
```

### Evaluating Models

The following example runs automatic evaluation for question-answering tasks:

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

MLflow Tracing provides LLM observability for various GenAI libraries. Enable auto-tracing with `mlflow.xyz.autolog()` before running your models.

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

Navigate to the "Traces" tab in the MLflow UI to find the trace records OpenAI query.

## Community

*   For help, questions, or documentation, see the [docs](https://mlflow.org/docs/latest/index.html) or [Stack Overflow](https://stackoverflow.com/questions/tagged/mlflow).
*   Ask questions using the AI-powered chat bot on the documentation website by clicking the **"Ask AI"** button.
*   Report bugs, file documentation issues, or submit feature requests via [GitHub issues](https://github.com/mlflow/mlflow/issues/new/choose).
*   Stay informed through our mailing list (mlflow-users@googlegroups.com) or join us on [Slack](https://mlflow.org/slack).

## Contributing

Contributions to MLflow are welcomed! See the [contribution guide](CONTRIBUTING.md) for details.  Explore the [MLflow Roadmap](https://github.com/mlflow/mlflow/milestone/3) for contributing opportunities.

## Citation

If you use MLflow in your research, please use the "Cite this repository" button on the [GitHub repository page](https://github.com/mlflow/mlflow) to get citation formats like APA and BibTeX.

## Core Members

MLflow is currently maintained by core members with significant contributions from the community.

*   [Ben Wilson](https://github.com/BenWilson2)
*   [Corey Zumar](https://github.com/dbczumar)
*   [Daniel Lok](https://github.com/daniellok-db)
*   [Gabriel Fu](https://github.com/gabrielfu)
*   [Harutaka Kawamura](https://github.com/harupy)
*   [Serena Ruan](https://github.com/serena-ruan)
*   [Weichen Xu](https://github.com/WeichenXu123)
*   [Yuki Watanabe](https://github.com/B-Step62)
*   [Tomu Hirata](https://github.com/TomeHirata)
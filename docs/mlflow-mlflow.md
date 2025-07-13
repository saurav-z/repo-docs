# MLflow: The Open-Source Platform for the Complete Machine Learning Lifecycle

**Simplify your machine learning workflow and boost your team's productivity with MLflow, the open-source platform designed for the entire ML lifecycle.**  [See the original repository](https://github.com/mlflow/mlflow)

[![Latest Docs](https://img.shields.io/badge/docs-latest-success.svg?style=for-the-badge)](https://mlflow.org/docs/latest/index.html)
[![Apache 2 License](https://img.shields.io/badge/license-Apache%202-brightgreen.svg?style=for-the-badge&logo=apache)](https://github.com/mlflow/mlflow/blob/master/LICENSE.txt)
[![Total Downloads](https://img.shields.io/pypi/dw/mlflow?style=for-the-badge&logo=pypi&logoColor=white)](https://pepy.tech/project/mlflow)
[![Slack](https://img.shields.io/badge/slack-@mlflow--users-CF0E5B.svg?logo=slack&logoColor=white&labelColor=3F0E40&style=for-the-badge)](https://mlflow.org/community/#slack)
[![Twitter](https://img.shields.io/twitter/follow/MLflow?style=for-the-badge&labelColor=00ACEE&logo=twitter&logoColor=white)](https://twitter.com/MLflow)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/mlflow/mlflow)

MLflow empowers machine learning practitioners and teams to manage the complexities of the machine learning process, from experimentation to deployment. It provides a unified platform that ensures each phase of your ML projects is manageable, traceable, and reproducible.

## Key Features of MLflow

*   **Experiment Tracking**: Log models, parameters, metrics, and artifacts to track and compare experiments using an intuitive UI.
*   **Model Packaging**: Package models in a standard format with metadata for reliable deployment and reproducibility.
*   **Model Registry**: Centrally manage the complete lifecycle of your MLflow Models, from creation to production.
*   **Serving**: Deploy models effortlessly to various platforms like Docker, Kubernetes, Azure ML, and AWS SageMaker for batch and real-time scoring.
*   **Evaluation**: Automate model evaluation and compare results across multiple models through a seamless integration with experiment tracking.
*   **Observability**: Integrate tracing with LLM libraries and provide a Python SDK for more intuitive debugging and continuous monitoring.

<img src="https://mlflow.org/img/hero.png" alt="MLflow Hero" width=100%>

## Installation

Install the MLflow Python package using pip:

```bash
pip install mlflow
```

Alternatively, install from other package hosting platforms:

| Platform          | Installation Command                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| ----------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| PyPI              | [![PyPI - mlflow](https://img.shields.io/pypi/v/mlflow.svg?style=for-the-badge&logo=pypi&logoColor=white&label=mlflow)](https://pypi.org/project/mlflow/) [![PyPI - mlflow-skinny](https://img.shields.io/pypi/v/mlflow-skinny.svg?style=for-the-badge&logo=pypi&logoColor=white&label=mlflow-skinny)](https://pypi.org/project/mlflow-skinny/)                                                                                                                                                                                                                                                                                                                                                                                  |
| conda-forge       | [![Conda - mlflow](https://img.shields.io/conda/vn/conda-forge/mlflow.svg?style=for-the-badge&logo=anaconda&label=mlflow)](https://anaconda.org/conda-forge/mlflow) [![Conda - mlflow-skinny](https://img.shields.io/conda/vn/conda-forge/mlflow.svg?style=for-the-badge&logo=anaconda&label=mlflow-skinny)](https://anaconda.org/conda-forge/mlflow-skinny)                                                                                                                                                                                                                                                                                                                                                                          |
| CRAN              | [![CRAN - mlflow](https://img.shields.io/cran/v/mlflow.svg?style=for-the-badge&logo=r&label=mlflow)](https://cran.r-project.org/package=mlflow)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| Maven Central     | [![Maven Central - mlflow-client](https://img.shields.io/maven-central/v/org.mlflow/mlflow-client.svg?style=for-the-badge&logo=apache-maven&label=mlflow-client)](https://mvnrepository.com/artifact/org.mlflow/mlflow-client) [![Maven Central - mlflow-parent](https://img.shields.io/maven-central/v/org.mlflow/mlflow-parent.svg?style=for-the-badge&logo=apache-maven&label=mlflow-parent)](https://mvnrepository.com/artifact/org.mlflow/mlflow-parent) [![Maven Central - mlflow-spark](https://img.shields.io/maven-central/v/org.mlflow/mlflow-spark.svg?style=for-the-badge&logo=apache-maven&label=mlflow-spark)](https://mvnrepository.com/artifact/org.mlflow/mlflow-spark) |

## Documentation

Comprehensive documentation is available at [mlflow.org/docs/latest/index.html](https://mlflow.org/docs/latest/index.html).

## Running MLflow Anywhere

MLflow supports a wide array of environments, including local development, Amazon SageMaker, AzureML, and Databricks.  Refer to the [Running Anywhere](https://mlflow.org/docs/latest/index.html#running-mlflow-anywhere) documentation for setup instructions.

## Usage Examples

### Experiment Tracking ([Doc](https://mlflow.org/docs/latest/tracking.html))

Track your model training parameters, metrics, and artifacts. This example uses scikit-learn with autologging:

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

After running the code above, open a separate terminal and execute `mlflow ui` to access the MLflow UI.  It will display an automatically generated Run with all the tracked details.

```bash
mlflow ui
```

### Serving Models ([Doc](https://mlflow.org/docs/latest/deployment/index.html))

Deploy your logged models with a simple command using the MLflow CLI.  See the documentation for more deployment options.

```bash
mlflow models serve --model-uri runs:/<run-id>/model
```

### Evaluating Models ([Doc](https://mlflow.org/docs/latest/model-evaluation/index.html))

Evaluate your models using built-in metrics.  This example demonstrates automatic evaluation for question answering tasks:

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

Integrate MLflow Tracing with GenAI libraries for enhanced debugging and monitoring.  This example shows autologging with OpenAI:

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

*   Find answers and discuss MLflow usage on the [docs](https://mlflow.org/docs/latest/index.html) and [Stack Overflow](https://stackoverflow.com/questions/tagged/mlflow).
*   Use the AI-powered chatbot on the documentation website (click the "Ask AI" button).
*   Report issues and request features on [GitHub](https://github.com/mlflow/mlflow/issues/new/choose).
*   Stay updated via the [mailing list](mlflow-users@googlegroups.com) and [Slack](https://mlflow.org/slack).

## Contributing

Contributions to MLflow are welcome!  See the [contribution guide](CONTRIBUTING.md) to learn more.  We also welcome contributions to the [MLflow Roadmap](https://github.com/mlflow/mlflow/milestone/3).

## Citation

If you use MLflow in your research, please cite it using the "Cite this repository" button on the [GitHub repository page](https://github.com/mlflow/mlflow), which provides citation formats.

## Core Members

MLflow is maintained by core members with contributions from the community. (list provided in original README)
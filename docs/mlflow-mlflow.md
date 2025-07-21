# MLflow: The Open-Source Platform for the Machine Learning Lifecycle

**Simplify your machine learning workflow with MLflow, an open-source platform designed to streamline the entire ML lifecycle from experiment tracking to model deployment.** ([Original Repository](https://github.com/mlflow/mlflow))

[![Latest Docs](https://img.shields.io/badge/docs-latest-success.svg?style=for-the-badge)](https://mlflow.org/docs/latest/index.html)
[![Apache 2 License](https://img.shields.io/badge/license-Apache%202-brightgreen.svg?style=for-the-badge&logo=apache)](https://github.com/mlflow/mlflow/blob/master/LICENSE.txt)
[![Total Downloads](https://img.shields.io/pypi/dw/mlflow?style=for-the-badge&logo=pypi&logoColor=white)](https://pepy.tech/project/mlflow)
[![Slack](https://img.shields.io/badge/slack-@mlflow--users-CF0E5B.svg?logo=slack&logoColor=white&labelColor=3F0E40&style=for-the-badge)](https://mlflow.org/community/#slack)
[![Twitter](https://img.shields.io/twitter/follow/MLflow?style=for-the-badge&labelColor=00ACEE&logo=twitter&logoColor=white)](https://twitter.com/MLflow)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/mlflow/mlflow)

---

## Key Features of MLflow

*   **Experiment Tracking:** Log parameters, metrics, and models to track and compare experiments. [Learn more](https://mlflow.org/docs/latest/tracking.html)
*   **Model Packaging:** Standardize model format with metadata for reliable deployment and reproducibility. [Learn more](https://mlflow.org/docs/latest/models.html)
*   **Model Registry:** Centralized model store to manage the full lifecycle of MLflow Models. [Learn more](https://mlflow.org/docs/latest/model-registry.html)
*   **Model Serving:** Deploy models to various platforms like Docker, Kubernetes, Azure ML, and AWS SageMaker. [Learn more](https://mlflow.org/docs/latest/deployment/index.html)
*   **Model Evaluation:** Automate model evaluation and compare results visually. [Learn more](https://mlflow.org/docs/latest/model-evaluation/index.html)
*   **Observability:** Trace and debug GenAI library integrations for smoother workflows. [Learn more](https://mlflow.org/docs/latest/llms/tracing/index.html)

<img src="https://mlflow.org/img/hero.png" alt="MLflow Hero" width=100%>

## Installation

Install MLflow using pip:

```bash
pip install mlflow
```

Alternatively, install from other package hosting platforms:

| Platform        | Install Command                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| --------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| PyPI            | `pip install mlflow`  or `pip install mlflow-skinny`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| conda-forge     | `conda install -c conda-forge mlflow` or  `conda install -c conda-forge mlflow-skinny`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| CRAN            | `install.packages("mlflow")`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| Maven Central | See details on [Maven Central](https://mvnrepository.com/artifact/org.mlflow/mlflow-client), [Maven Central](https://mvnrepository.com/artifact/org.mlflow/mlflow-parent) and [Maven Central](https://mvnrepository.com/artifact/org.mlflow/mlflow-spark)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |

## Documentation

Explore comprehensive MLflow documentation: [https://mlflow.org/docs/latest/index.html](https://mlflow.org/docs/latest/index.html)

## Run MLflow Anywhere

MLflow supports running in a variety of environments, including local development, cloud platforms (Amazon SageMaker, AzureML), and Databricks.  Refer to the [MLflow documentation](https://mlflow.org/docs/latest/index.html#running-mlflow-anywhere) for setup instructions.

## Usage Examples

### Experiment Tracking

Track model training with automatic experiment tracking, using the following example:

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

Then, run the UI in a separate terminal:

```bash
mlflow ui
```

### Serving Models

Deploy a logged model to a local inference server:

```bash
mlflow models serve --model-uri runs:/<run-id>/model
```

### Evaluating Models

Run automatic evaluation for question-answering tasks:

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

Enable tracing for GenAI libraries like OpenAI:

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

*   **Documentation:** Explore the [docs](https://mlflow.org/docs/latest/index.html) for detailed guidance.
*   **Ask Questions:** Seek help on [Stack Overflow](https://stackoverflow.com/questions/tagged/mlflow).
*   **AI-Powered Chatbot:** Use the "Ask AI" button on the doc website.
*   **Report Issues:** [Open a GitHub issue](https://github.com/mlflow/mlflow/issues/new/choose).
*   **Stay Updated:** Subscribe to our mailing list (mlflow-users@googlegroups.com) or join us on [Slack](https://mlflow.org/slack).

## Contributing

Contribute to MLflow!  See our [contribution guide](CONTRIBUTING.md) for details. Review the [MLflow Roadmap](https://github.com/mlflow/mlflow/milestone/3) to find open tasks.

## Citation

Cite MLflow in your research using the "Cite this repository" button on the [GitHub repository page](https://github.com/mlflow/mlflow).

## Core Members

*   Ben Wilson
*   Corey Zumar
*   Daniel Lok
*   Gabriel Fu
*   Harutaka Kawamura
*   Serena Ruan
*   Weichen Xu
*   Yuki Watanabe
*   Tomu Hirata
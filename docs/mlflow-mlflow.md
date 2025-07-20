# MLflow: Simplify Your Machine Learning Lifecycle

**MLflow is an open-source platform designed to streamline the entire machine learning lifecycle, making it easier to manage, track, and reproduce your ML projects.**  Learn more at the [original MLflow repository](https://github.com/mlflow/mlflow).

[![Latest Docs](https://img.shields.io/badge/docs-latest-success.svg?style=for-the-badge)](https://mlflow.org/docs/latest/index.html)
[![Apache 2 License](https://img.shields.io/badge/license-Apache%202-brightgreen.svg?style=for-the-badge&logo=apache)](https://github.com/mlflow/mlflow/blob/master/LICENSE.txt)
[![Total Downloads](https://img.shields.io/pypi/dw/mlflow?style=for-the-badge&logo=pypi&logoColor=white)](https://pepy.tech/project/mlflow)
[![Slack](https://img.shields.io/badge/slack-@mlflow--users-CF0E5B.svg?logo=slack&logoColor=white&labelColor=3F0E40&style=for-the-badge)](https://mlflow.org/community/#slack)
[![Twitter](https://img.shields.io/twitter/follow/MLflow?style=for-the-badge&labelColor=00ACEE&logo=twitter&logoColor=white)](https://twitter.com/MLflow)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/mlflow/mlflow)

## Key Features of MLflow

*   **Experiment Tracking:** 📝 Log models, parameters, and results for ML experiments and compare them using an interactive UI, enabling comprehensive experiment management.
*   **Model Packaging:** 📦 Package models with dependencies for reliable deployment and strong reproducibility, ensuring consistent model behavior across environments.
*   **Model Registry:** 💾 Manage the complete lifecycle of MLflow Models through a centralized model store, APIs, and UI for collaborative model management.
*   **Serving:** 🚀 Deploy models seamlessly to various platforms (Docker, Kubernetes, Azure ML, AWS SageMaker) for batch and real-time scoring, facilitating easy model serving.
*   **Evaluation:** 📊 Utilize automated model evaluation tools and compare performance across multiple models with experiment tracking, supporting informed decision-making.
*   **Observability:** 🔍 Integrate tracing with GenAI libraries and a Python SDK for manual instrumentation for enhanced debugging and online monitoring.

<img src="https://mlflow.org/img/hero.png" alt="MLflow Hero" width=100%>

## Installation

Install the MLflow Python package using pip:

```bash
pip install mlflow
```

Alternatively, install from other package platforms:

| Platform       | Install Command                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| -------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| PyPI           | [![PyPI - mlflow](https://img.shields.io/pypi/v/mlflow.svg?style=for-the-badge&logo=pypi&logoColor=white&label=mlflow)](https://pypi.org/project/mlflow/) [![PyPI - mlflow-skinny](https://img.shields.io/pypi/v/mlflow-skinny.svg?style=for-the-badge&logo=pypi&logoColor=white&label=mlflow-skinny)](https://pypi.org/project/mlflow-skinny/)                                                                                                                                                                                                                                                                                                                                          |
| conda-forge    | [![Conda - mlflow](https://img.shields.io/conda/vn/conda-forge/mlflow.svg?style=for-the-badge&logo=anaconda&label=mlflow)](https://anaconda.org/conda-forge/mlflow) [![Conda - mlflow-skinny](https://img.shields.io/conda/vn/conda-forge/mlflow.svg?style=for-the-badge&logo=anaconda&label=mlflow-skinny)](https://anaconda.org/conda-forge/mlflow-skinny)                                                                                                                                                                                                                                                                                                                             |
| CRAN           | [![CRAN - mlflow](https://img.shields.io/cran/v/mlflow.svg?style=for-the-badge&logo=r&label=mlflow)](https://cran.r-project.org/package=mlflow)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| Maven Central  | [![Maven Central - mlflow-client](https://img.shields.io/maven-central/v/org.mlflow/mlflow-client.svg?style=for-the-badge&logo=apache-maven&label=mlflow-client)](https://mvnrepository.com/artifact/org.mlflow/mlflow-client) [![Maven Central - mlflow-parent](https://img.shields.io/maven-central/v/org.mlflow/mlflow-parent.svg?style=for-the-badge&logo=apache-maven&label=mlflow-parent)](https://mvnrepository.com/artifact/org.mlflow/mlflow-parent) [![Maven Central - mlflow-spark](https://img.shields.io/maven-central/v/org.mlflow/mlflow-spark.svg?style=for-the-badge&logo=apache-maven&label=mlflow-spark)](https://mvnrepository.com/artifact/org.mlflow/mlflow-spark) |

## Documentation

Comprehensive documentation is available at: [https://mlflow.org/docs/latest/index.html](https://mlflow.org/docs/latest/index.html)

## Run MLflow Anywhere

MLflow supports various environments, including local development, Amazon SageMaker, AzureML, and Databricks.  Refer to the [Running Anywhere guide](https://mlflow.org/docs/latest/index.html#running-mlflow-anywhere) for setup instructions.

## Example Usage

### Experiment Tracking

Track your model training with automatic experiment tracking using autologging:

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

Open the MLflow UI in a separate terminal to view your experiment:

```bash
mlflow ui
```

### Serving Models

Deploy logged models to a local inference server:

```bash
mlflow models serve --model-uri runs:/<run-id>/model
```

### Evaluating Models

Evaluate question-answering models:

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

Enable tracing with OpenAI and other GenAI libraries:

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

View your traces in the MLflow UI under the "Traces" tab.

## Community

*   **Documentation:**  Explore the [docs](https://mlflow.org/docs/latest/index.html) for detailed information.
*   **Ask AI Chatbot:** Find the "Ask AI" button at the bottom right of the documentation page.
*   **Stack Overflow:** Seek help on [Stack Overflow](https://stackoverflow.com/questions/tagged/mlflow).
*   **GitHub Issues:** Report bugs, documentation issues, and feature requests via [GitHub issues](https://github.com/mlflow/mlflow/issues/new/choose).
*   **Mailing List:**  Subscribe to our mailing list (mlflow-users@googlegroups.com) for announcements.
*   **Slack:** Join us on [Slack](https://mlflow.org/slack).

## Contributing

Contributions to MLflow are welcome! See the [contribution guide](CONTRIBUTING.md) for details.  Consider contributing to items on the [MLflow Roadmap](https://github.com/mlflow/mlflow/milestone/3).

## Citation

If you use MLflow in your research, cite it using the "Cite this repository" button on the [GitHub repository page](https://github.com/mlflow/mlflow) for citation formats.

## Core Members

MLflow is maintained by the following core members, with significant contributions from a large community:

*   [Ben Wilson](https://github.com/BenWilson2)
*   [Corey Zumar](https://github.com/dbczumar)
*   [Daniel Lok](https://github.com/daniellok-db)
*   [Gabriel Fu](https://github.com/gabrielfu)
*   [Harutaka Kawamura](https://github.com/harupy)
*   [Serena Ruan](https://github.com/serena-ruan)
*   [Weichen Xu](https://github.com/WeichenXu123)
*   [Yuki Watanabe](https://github.com/B-Step62)
*   [Tomu Hirata](https://github.com/TomeHirata)
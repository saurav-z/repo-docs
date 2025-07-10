# MLflow: The Open-Source Platform for the Machine Learning Lifecycle

**MLflow empowers machine learning teams to manage the entire ML lifecycle, from experimentation to deployment and monitoring.**  Explore the comprehensive capabilities of MLflow at the original repository: [https://github.com/mlflow/mlflow](https://github.com/mlflow/mlflow)

[![Latest Docs](https://img.shields.io/badge/docs-latest-success.svg?style=for-the-badge)](https://mlflow.org/docs/latest/index.html)
[![Apache 2 License](https://img.shields.io/badge/license-Apache%202-brightgreen.svg?style=for-the-badge&logo=apache)](https://github.com/mlflow/mlflow/blob/master/LICENSE.txt)
[![Total Downloads](https://img.shields.io/pypi/dw/mlflow?style=for-the-badge&logo=pypi&logoColor=white)](https://pepy.tech/project/mlflow)
[![Slack](https://img.shields.io/badge/slack-@mlflow--users-CF0E5B.svg?logo=slack&logoColor=white&labelColor=3F0E40&style=for-the-badge)](https://mlflow.org/community/#slack)
[![Twitter](https://img.shields.io/twitter/follow/MLflow?style=for-the-badge&labelColor=00ACEE&logo=twitter&logoColor=white)](https://twitter.com/MLflow)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/mlflow/mlflow)

---

## Key Features of MLflow:

*   **Experiment Tracking:**  Effortlessly log parameters, metrics, models, and artifacts from your machine learning experiments and compare them in an interactive UI.
*   **Model Packaging:** Standardize your models with a consistent format, including dependencies, for reliable deployment and enhanced reproducibility.
*   **Model Registry:**  Centralized hub for managing the full lifecycle of your MLflow Models, including versioning, staging, and transitions.
*   **Serving:** Deploy your models with ease to platforms like Docker, Kubernetes, Azure ML, and AWS SageMaker for batch and real-time scoring.
*   **Evaluation:** Utilize automated model evaluation tools integrated with experiment tracking to visualize and compare model performance.
*   **Observability:** Leverage tracing integrations and a Python SDK to monitor and debug your machine learning workflows, offering a smoother development experience.

<img src="https://mlflow.org/img/hero.png" alt="MLflow Hero" width=100%>

## Installation

Install MLflow using pip:

```bash
pip install mlflow
```

Alternatively, install from PyPI, conda-forge, CRAN, or Maven Central.  Check the available packages:

|               |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| ------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| PyPI          | [![PyPI - mlflow](https://img.shields.io/pypi/v/mlflow.svg?style=for-the-badge&logo=pypi&logoColor=white&label=mlflow)](https://pypi.org/project/mlflow/) [![PyPI - mlflow-skinny](https://img.shields.io/pypi/v/mlflow-skinny.svg?style=for-the-badge&logo=pypi&logoColor=white&label=mlflow-skinny)](https://pypi.org/project/mlflow-skinny/)                                                                                                                                                                                                                                                                                                                                          |
| conda-forge   | [![Conda - mlflow](https://img.shields.io/conda/vn/conda-forge/mlflow.svg?style=for-the-badge&logo=anaconda&label=mlflow)](https://anaconda.org/conda-forge/mlflow) [![Conda - mlflow-skinny](https://img.shields.io/conda/vn/conda-forge/mlflow.svg?style=for-the-badge&logo=anaconda&label=mlflow-skinny)](https://anaconda.org/conda-forge/mlflow-skinny)                                                                                                                                                                                                                                                                                                                             |
| CRAN          | [![CRAN - mlflow](https://img.shields.io/cran/v/mlflow.svg?style=for-the-badge&logo=r&label=mlflow)](https://cran.r-project.org/package=mlflow)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| Maven Central | [![Maven Central - mlflow-client](https://img.shields.io/maven-central/v/org.mlflow/mlflow-client.svg?style=for-the-badge&logo=apache-maven&label=mlflow-client)](https://mvnrepository.com/artifact/org.mlflow/mlflow-client) [![Maven Central - mlflow-parent](https://img.shields.io/maven-central/v/org.mlflow/mlflow-parent.svg?style=for-the-badge&logo=apache-maven&label=mlflow-parent)](https://mvnrepository.com/artifact/org.mlflow/mlflow-parent) [![Maven Central - mlflow-spark](https://img.shields.io/maven-central/v/org.mlflow/mlflow-spark.svg?style=for-the-badge&logo=apache-maven&label=mlflow-spark)](https://mvnrepository.com/artifact/org.mlflow/mlflow-spark) |

## Documentation

Comprehensive documentation is available [here](https://mlflow.org/docs/latest/index.html).

## Running MLflow

MLflow supports various environments, including local development, Amazon SageMaker, AzureML, and Databricks.  Find setup instructions [here](https://mlflow.org/docs/latest/index.html#running-mlflow-anywhere).

## Usage Examples

### Experiment Tracking

Track experiments by enabling autologging:

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

Access the MLflow UI:

```bash
mlflow ui
```

### Serving Models

Deploy logged models using the CLI:

```bash
mlflow models serve --model-uri runs:/<run-id>/model
```

### Evaluating Models

Run automatic evaluation:

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

View traces in the MLflow UI.

## Community

*   **Docs:**  For help and usage guidance, visit the [documentation](https://mlflow.org/docs/latest/index.html).
*   **Stack Overflow:**  Find answers on [Stack Overflow](https://stackoverflow.com/questions/tagged/mlflow).
*   **AI-powered chat bot:** Ask questions by clicking the **"Ask AI"** button on the MLflow documentation website.
*   **GitHub Issues:**  Report bugs, request features, and suggest documentation improvements [here](https://github.com/mlflow/mlflow/issues/new/choose).
*   **Mailing List/Slack:**  Stay updated through the mailing list (mlflow-users@googlegroups.com) and join the [Slack](https://mlflow.org/slack) community.

## Contributing

Contributions to MLflow are welcome! See the [contribution guide](CONTRIBUTING.md) and the [MLflow Roadmap](https://github.com/mlflow/mlflow/milestone/3).

## Citation

Cite MLflow in your research using the "Cite this repository" button at the top of the [GitHub repository page](https://github.com/mlflow/mlflow).

## Core Members

MLflow is maintained by the core members listed in the original README.
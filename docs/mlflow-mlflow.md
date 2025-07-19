# MLflow: Simplify the Machine Learning Lifecycle

MLflow is an open-source platform that streamlines the entire machine learning lifecycle, making model development, deployment, and management easier. Learn more and contribute on the [MLflow GitHub Repository](https://github.com/mlflow/mlflow)!

---

## Key Features of MLflow

*   **Experiment Tracking**: Log and compare machine learning experiments with ease, tracking parameters, metrics, and models in an interactive UI.
*   **Model Packaging**: Package your models with a standard format that includes dependencies for reliable deployment and reproducibility.
*   **Model Registry**: Centralize model management with a registry that facilitates collaboration, versioning, and lifecycle management.
*   **Model Serving**: Deploy models seamlessly to various platforms, including Docker, Kubernetes, Azure ML, and AWS SageMaker, for batch and real-time scoring.
*   **Evaluation**: Utilize automated model evaluation tools to assess model performance and compare results visually.
*   **Observability**: Integrate with GenAI libraries and provide a Python SDK for tracing, enabling easier debugging and online monitoring.

## Installation

Install MLflow using pip:

```bash
pip install mlflow
```

MLflow is also available on PyPI, Conda-Forge, CRAN, and Maven Central.  See the original README for installation details.

## Documentation

Comprehensive documentation is available to help you get started: [MLflow Documentation](https://mlflow.org/docs/latest/index.html).

## Running Anywhere

MLflow supports various environments, including local development, Amazon SageMaker, AzureML, and Databricks.  Refer to the [Running Anywhere](https://mlflow.org/docs/latest/index.html#running-mlflow-anywhere) documentation for setup instructions.

## Getting Started with MLflow

### Experiment Tracking

Track your machine learning experiments using MLflow's autologging or manual logging capabilities.

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

View your experiments in the MLflow UI:

```bash
mlflow ui
```

### Serving Models

Deploy your trained models quickly using the MLflow CLI:

```bash
mlflow models serve --model-uri runs:/<run-id>/model
```

### Evaluating Models

Evaluate your models with automatic evaluation capabilities and metrics:

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

Use MLflow Tracing to monitor your LLM applications.

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

## Community and Contributing

*   **Documentation & Support:** [MLflow Documentation](https://mlflow.org/docs/latest/index.html) and [Stack Overflow](https://stackoverflow.com/questions/tagged/mlflow).
*   **Ask AI:** Use the "Ask AI" button at the right bottom of the doc website to chat with an AI-powered bot.
*   **Report Issues:** [GitHub Issues](https://github.com/mlflow/mlflow/issues/new/choose).
*   **Discussions:** [Mailing List](mlflow-users@googlegroups.com) and [Slack](https://mlflow.org/slack).

We welcome contributions! See the [contribution guide](CONTRIBUTING.md) for more information.

## Citation

Cite MLflow in your research using the "Cite this repository" button on the [GitHub repository page](https://github.com/mlflow/mlflow).

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
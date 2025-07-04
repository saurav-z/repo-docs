# MLflow: Your Complete Machine Learning Lifecycle Platform

**Simplify and streamline your machine learning projects with MLflow, the open-source platform designed for the entire ML lifecycle.**  [Visit the original repository](https://github.com/mlflow/mlflow)

[![Latest Docs](https://img.shields.io/badge/docs-latest-success.svg?style=for-the-badge)](https://mlflow.org/docs/latest/index.html)
[![Apache 2 License](https://img.shields.io/badge/license-Apache%202-brightgreen.svg?style=for-the-badge&logo=apache)](https://github.com/mlflow/mlflow/blob/master/LICENSE.txt)
[![Total Downloads](https://img.shields.io/pypi/dw/mlflow?style=for-the-badge&logo=pypi&logoColor=white)](https://pepy.tech/project/mlflow)
[![Slack](https://img.shields.io/badge/slack-@mlflow--users-CF0E5B.svg?logo=slack&logoColor=white&labelColor=3F0E40&style=for-the-badge)](https://mlflow.org/community/#slack)
[![Twitter](https://img.shields.io/twitter/follow/MLflow?style=for-the-badge&labelColor=00ACEE&logo=twitter&logoColor=white)](https://twitter.com/MLflow)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/mlflow/mlflow)

---

MLflow is an open-source platform empowering machine learning practitioners and teams to manage the complexities of the ML process. It provides a comprehensive solution for the full lifecycle, ensuring each phase is manageable, traceable, and reproducible.

## Key Features of MLflow:

*   **Experiment Tracking**: Easily log models, parameters, and results from your ML experiments. Compare and analyze them using an interactive UI.
*   **Model Packaging**: Package your trained models with all necessary metadata and dependencies, ensuring reliable deployment and reproducibility.
*   **Model Registry**: A central hub to collaboratively manage the complete lifecycle of your MLflow Models, including versioning and staging.
*   **Serving**: Deploy your models effortlessly for batch and real-time scoring on various platforms like Docker, Kubernetes, Azure ML, and AWS SageMaker.
*   **Evaluation**: Automate model evaluation and track performance metrics with integrated tools, allowing for visual comparison of results across multiple models.
*   **Observability**: Enhance debugging with tracing integrations for GenAI libraries and a Python SDK for custom instrumentation, supporting online monitoring.

<img src="https://mlflow.org/img/hero.png" alt="MLflow Hero" width=100%>

## Installation

Install the MLflow Python package using pip:

```bash
pip install mlflow
```

Alternatively, install from other package hosting platforms:

| Platform      | Installation Command                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| ------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| PyPI          | [![PyPI - mlflow](https://img.shields.io/pypi/v/mlflow.svg?style=for-the-badge&logo=pypi&logoColor=white&label=mlflow)](https://pypi.org/project/mlflow/) [![PyPI - mlflow-skinny](https://img.shields.io/pypi/v/mlflow-skinny.svg?style=for-the-badge&logo=pypi&logoColor=white&label=mlflow-skinny)](https://pypi.org/project/mlflow-skinny/)                                                                                                                                                                                                                                                                                                                           |
| conda-forge   | [![Conda - mlflow](https://img.shields.io/conda/vn/conda-forge/mlflow.svg?style=for-the-badge&logo=anaconda&label=mlflow)](https://anaconda.org/conda-forge/mlflow) [![Conda - mlflow-skinny](https://img.shields.io/conda/vn/conda-forge/mlflow.svg?style=for-the-badge&logo=anaconda&label=mlflow-skinny)](https://anaconda.org/conda-forge/mlflow-skinny)                                                                                                                                                                                                                                                                                                                             |
| CRAN          | [![CRAN - mlflow](https://img.shields.io/cran/v/mlflow.svg?style=for-the-badge&logo=r&label=mlflow)](https://cran.r-project.org/package=mlflow)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| Maven Central | [![Maven Central - mlflow-client](https://img.shields.io/maven-central/v/org.mlflow/mlflow-client.svg?style=for-the-badge&logo=apache-maven&label=mlflow-client)](https://mvnrepository.com/artifact/org.mlflow/mlflow-client) [![Maven Central - mlflow-parent](https://img.shields.io/maven-central/v/org.mlflow/mlflow-parent.svg?style=for-the-badge&logo=apache-maven&label=mlflow-parent)](https://mvnrepository.com/artifact/org.mlflow/mlflow-parent) [![Maven Central - mlflow-spark](https://img.shields.io/maven-central/v/org.mlflow/mlflow-spark.svg?style=for-the-badge&logo=apache-maven&label=mlflow-spark)](https://mvnrepository.com/artifact/org.mlflow/mlflow-spark) |

## Documentation

Comprehensive documentation is available at: [https://mlflow.org/docs/latest/index.html](https://mlflow.org/docs/latest/index.html).

## Running Anywhere

Run MLflow in various environments, including local development, Amazon SageMaker, AzureML, and Databricks. See [this guidance](https://mlflow.org/docs/latest/index.html#running-mlflow-anywhere) for setup instructions.

## Usage Examples

### Experiment Tracking

Track your model training with autologging.

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

Then, in a separate terminal:

```bash
mlflow ui
```

Access the MLflow UI via the printed URL to view your automatically tracked experiment.

### Serving Models

Deploy your logged model to a local inference server:

```bash
mlflow models serve --model-uri runs:/<run-id>/model
```

### Evaluating Models

Run automatic evaluation for question-answering tasks:

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
    results = mlflow.evaluate(data=eval_dataset, model_type="question-answering")

print(results.tables["eval_results_table"])
```

### Observability

Enable tracing for GenAI libraries, for example, OpenAI:

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

View your traces in the "Traces" tab in the MLflow UI.

## Community

*   **Documentation and Support**: [MLflow Documentation](https://mlflow.org/docs/latest/index.html), [Stack Overflow](https://stackoverflow.com/questions/tagged/mlflow)
*   **AI-Powered Chatbot**: Visit the doc website and click on the **"Ask AI"** button.
*   **Issue Tracking**: [GitHub Issues](https://github.com/mlflow/mlflow/issues/new/choose) for bug reports, documentation, and feature requests.
*   **Stay Updated**: Subscribe to our [mailing list](mlflow-users@googlegroups.com) or join us on [Slack](https://mlflow.org/slack).

## Contributing

Contributions to MLflow are welcome! See the [contribution guide](CONTRIBUTING.md) and the [MLflow Roadmap](https://github.com/mlflow/mlflow/milestone/3) for more details.

## Core Members

The project is maintained by these core members, with contributions from a large community:

*   [Ben Wilson](https://github.com/BenWilson2)
*   [Corey Zumar](https://github.com/dbczumar)
*   [Daniel Lok](https://github.com/daniellok-db)
*   [Gabriel Fu](https://github.com/gabrielfu)
*   [Harutaka Kawamura](https://github.com/harupy)
*   [Serena Ruan](https://github.com/serena-ruan)
*   [Weichen Xu](https://github.com/WeichenXu123)
*   [Yuki Watanabe](https://github.com/B-Step62)
*   [Tomu Hirata](https://github.com/TomeHirata)
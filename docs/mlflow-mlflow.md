# MLflow: The Open Source Platform for the Complete Machine Learning Lifecycle

**Simplify your machine learning workflow with MLflow, an open-source platform designed for managing the full ML lifecycle.**  [Explore the original repository](https://github.com/mlflow/mlflow).

[![Latest Docs](https://img.shields.io/badge/docs-latest-success.svg?style=for-the-badge)](https://mlflow.org/docs/latest/index.html)
[![Apache 2 License](https://img.shields.io/badge/license-Apache%202-brightgreen.svg?style=for-the-badge&logo=apache)](https://github.com/mlflow/mlflow/blob/master/LICENSE.txt)
[![Total Downloads](https://img.shields.io/pypi/dw/mlflow?style=for-the-badge&logo=pypi&logoColor=white)](https://pepy.tech/project/mlflow)
[![Slack](https://img.shields.io/badge/slack-@mlflow--users-CF0E5B.svg?logo=slack&logoColor=white&labelColor=3F0E40&style=for-the-badge)](https://mlflow.org/community/#slack)
[![Twitter](https://img.shields.io/twitter/follow/MLflow?style=for-the-badge&labelColor=00ACEE&logo=twitter&logoColor=white)](https://twitter.com/MLflow)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/mlflow/mlflow)

---

## Key Features

*   **Experiment Tracking:**  Log and compare model parameters, metrics, and artifacts, enabling you to track and visualize your machine learning experiments.
*   **Model Packaging:**  Package models in a standard format, ensuring reliable deployment and reproducibility across different environments.
*   **Model Registry:**  Centralized management of the MLflow Model lifecycle, facilitating collaborative model management and versioning.
*   **Model Serving:**  Deploy models seamlessly to various platforms like Docker, Kubernetes, and cloud providers for batch and real-time scoring.
*   **Model Evaluation:**  Automated model evaluation tools for comprehensive performance analysis and comparison.
*   **Observability:**  Tracing integrations with GenAI libraries and a Python SDK for manual instrumentation, offering a smoother debugging experience.

<img src="https://mlflow.org/img/hero.png" alt="MLflow Hero" width=100%>

## Installation

Install MLflow using pip:

```bash
pip install mlflow
```

Or choose from various package hosting platforms:

| Platform       | Installation Command                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| -------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| PyPI           | [![PyPI - mlflow](https://img.shields.io/pypi/v/mlflow.svg?style=for-the-badge&logo=pypi&logoColor=white&label=mlflow)](https://pypi.org/project/mlflow/) [![PyPI - mlflow-skinny](https://img.shields.io/pypi/v/mlflow-skinny.svg?style=for-the-badge&logo=pypi&logoColor=white&label=mlflow-skinny)](https://pypi.org/project/mlflow-skinny/)                                                                                                                                                                                                                                                                                                                                     |
| conda-forge    | [![Conda - mlflow](https://img.shields.io/conda/vn/conda-forge/mlflow.svg?style=for-the-badge&logo=anaconda&label=mlflow)](https://anaconda.org/conda-forge/mlflow) [![Conda - mlflow-skinny](https://img.shields.io/conda/vn/conda-forge/mlflow.svg?style=for-the-badge&logo=anaconda&label=mlflow-skinny)](https://anaconda.org/conda-forge/mlflow-skinny)                                                                                                                                                                                                                                                                                                                       |
| CRAN           | [![CRAN - mlflow](https://img.shields.io/cran/v/mlflow.svg?style=for-the-badge&logo=r&label=mlflow)](https://cran.r-project.org/package=mlflow)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| Maven Central | [![Maven Central - mlflow-client](https://img.shields.io/maven-central/v/org.mlflow/mlflow-client.svg?style=for-the-badge&logo=apache-maven&label=mlflow-client)](https://mvnrepository.com/artifact/org.mlflow/mlflow-client) [![Maven Central - mlflow-parent](https://img.shields.io/maven-central/v/org.mlflow/mlflow-parent.svg?style=for-the-badge&logo=apache-maven&label=mlflow-parent)](https://mvnrepository.com/artifact/org.mlflow/mlflow-parent) [![Maven Central - mlflow-spark](https://img.shields.io/maven-central/v/org.mlflow/mlflow-spark.svg?style=for-the-badge&logo=apache-maven&label=mlflow-spark)](https://mvnrepository.com/artifact/org.mlflow/mlflow-spark) |

## Documentation

Refer to the [official MLflow documentation](https://mlflow.org/docs/latest/index.html) for comprehensive guides and examples.

## Running Anywhere

MLflow supports various environments, including local development, cloud platforms like Amazon SageMaker and AzureML, and Databricks.  Consult the [running anywhere guide](https://mlflow.org/docs/latest/index.html#running-mlflow-anywhere) to configure your environment.

## Usage Examples

### Experiment Tracking

Track your machine learning experiments with ease.  Here's a Python example:

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

After running this code, use:

```bash
mlflow ui
```

to view your experiment runs in the MLflow UI.

### Model Serving

Deploy your trained models with a single command:

```bash
mlflow models serve --model-uri runs:/<run-id>/model
```

### Model Evaluation

Automate the evaluation of your models:

```python
import mlflow
import pandas as pd

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

with mlflow.start_run(run_name="evaluate_qa"):
    results = mlflow.evaluate(
        data=eval_dataset,
        model_type="question-answering",
    )

print(results.tables["eval_results_table"])
```

### Observability

Integrate tracing for GenAI libraries:

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

Access the trace records in the "Traces" tab of the MLflow UI.

## Community

*   Find help in the [documentation](https://mlflow.org/docs/latest/index.html) or on [Stack Overflow](https://stackoverflow.com/questions/tagged/mlflow).
*   Ask questions to the AI-powered chat bot on the documentation website.
*   Report issues and suggest features via [GitHub issues](https://github.com/mlflow/mlflow/issues/new/choose).
*   Stay updated via the [mailing list](mlflow-users@googlegroups.com) or [Slack](https://mlflow.org/slack).

## Contributing

Contributions to MLflow are welcome!  See the [contribution guide](CONTRIBUTING.md) and the [MLflow Roadmap](https://github.com/mlflow/mlflow/milestone/3) for details.

## Citation

If you use MLflow in your research, please cite it using the "Cite this repository" button on the [GitHub repository page](https://github.com/mlflow/mlflow).

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
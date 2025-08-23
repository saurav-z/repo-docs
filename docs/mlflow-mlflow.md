# MLflow: Productionize Your AI Applications with Confidence

MLflow is an open-source platform empowering developers to build and deploy robust AI and LLM applications with features for experiment tracking, observability, and model evaluations. [Visit the original repository](https://github.com/mlflow/mlflow).

## Key Features

*   **Experiment Tracking:** Track and compare your model parameters, metrics, and results in ML experiments with an interactive UI.
*   **LLM Tracing & Observability:** Trace the internal states of your LLM/agentic applications for debugging and performance monitoring.
*   **LLM Evaluation:** Automated model evaluation tools integrated with experiment tracking.
*   **Prompt Management:** Version, track, and reuse prompts across your organization.
*   **App Version Tracking:** Track your AI applications, including models, prompts, tools, and code, with end-to-end lineage.
*   **Model Registry:** Manage the full lifecycle and deployment of machine learning models collaboratively.
*   **Deployment:** Tools for seamless model deployment to various platforms.

## Installation

Install the MLflow Python package using pip:

```bash
pip install mlflow
```

## Core Components

MLflow provides a unified solution for your AI/ML needs, including LLMs, Agents, Deep Learning, and traditional machine learning.

### For LLM / GenAI Developers

*   **Tracing/Observability:** Trace internal states of LLM/agentic applications for debugging and monitoring.
*   **LLM Evaluation:** A suite of automated model evaluation tools, seamlessly integrated with experiment tracking.
*   **Prompt Management:** Version, track, and reuse prompts across your organization.
*   **App Version Tracking:** MLflow tracks models, prompts, tools, and code with end-to-end lineage.

### For Data Scientists

*   **Experiment Tracking:** Track models, parameters, metrics, and evaluation results.
*   **Model Registry:** Centrally manage the lifecycle and deployment of models.
*   **Deployment:** Deploy models to various platforms like Docker, Kubernetes, and cloud services.

## Hosting MLflow Anywhere

Run MLflow on local machines, on-premise servers, and cloud infrastructure.

Managed services are offered by:

*   Amazon SageMaker
*   Azure ML
*   Databricks
*   Nebius

For self-hosting, refer to the [tracking setup guidance](https://mlflow.org/docs/latest/ml/tracking/#tracking-setup).

## Supported Programming Languages

*   Python
*   TypeScript / JavaScript
*   Java
*   R

## Integrations

MLflow integrates natively with popular machine learning frameworks and GenAI libraries.

## Usage Examples

### Experiment Tracking

Track experiments and models using MLflow's autologging feature:

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

Run `mlflow ui` in a separate terminal to view the results.

### Evaluating Models

Run automatic evaluation for question-answering tasks with several built-in metrics:

```python
import mlflow
import pandas as pd

df = pd.DataFrame({ ... })
eval_dataset = mlflow.data.from_pandas(df, predictions="outputs", targets="ground_truth")

with mlflow.start_run(run_name="evaluate_qa"):
    results = mlflow.evaluate(data=eval_dataset, model_type="question-answering")

print(results.tables["eval_results_table"])
```

### Observability

Enable MLflow Tracing for GenAI libraries like OpenAI:

```python
import mlflow
from openai import OpenAI

mlflow.openai.autolog()

response = OpenAI().chat.completions.create( ... )
```

Then view the trace records in the "Traces" tab of the MLflow UI.

## Support

*   Visit the [documentation](https://mlflow.org/docs/latest/index.html).
*   Ask questions to the AI-powered chat bot in the documentation.
*   Join [virtual events](https://lu.ma/mlflow?k=c).
*   Report issues on [GitHub](https://github.com/mlflow/mlflow/issues/new/choose).

## Contributing

Contributions are welcome!  Refer to the [contribution guide](CONTRIBUTING.md).

## ⭐️ Star History

[Include the star history graph here.  Use the code from the original README, or generate a new one.]

## ✏️ Citation

Cite MLflow using the "Cite this repository" button on the [GitHub repository page](https://github.com/mlflow/mlflow).

## 👥 Core Members

[List the core members, as in the original README]
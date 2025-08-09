# MLflow: The Open-Source Platform for Productionizing AI

**[Visit the MLflow Repository on GitHub](https://github.com/mlflow/mlflow)**

MLflow is an open-source platform designed to streamline the entire machine learning lifecycle, from experimentation to deployment, providing a unified solution for AI and LLM application development.

**Key Features:**

*   **Experiment Tracking:** Log parameters, metrics, and artifacts to track and compare model training runs.
*   **Model Registry:** Centralized model store for managing the lifecycle of machine learning models.
*   **Model Deployment:** Seamlessly deploy models to various platforms, including Docker, Kubernetes, and cloud providers.
*   **LLM/GenAI Focused Features:**
    *   **Tracing / Observability:** Trace the internal states of your LLM/agentic applications for debugging quality issues and monitoring performance with ease.
    *   **LLM Evaluation:** A suite of automated model evaluation tools, seamlessly integrated with experiment tracking to compare across multiple versions.
    *   **Prompt Management:** Version, track, and reuse prompts across your organization, helping maintain consistency and improve collaboration in prompt development.
    *   **App Version Tracking:** MLflow keeps track of many moving parts in your AI applications, such as models, prompts, tools, and code, with end-to-end lineage.
*   **Integrations:** Native support for many popular machine learning frameworks and GenAI libraries.
*   **Multi-Language Support:** Python, TypeScript / JavaScript, Java, and R.

## Getting Started

### Installation

```bash
pip install mlflow
```

### Core Components

*   **Experiment Tracking:** Track your models, parameters, metrics, and evaluation results in ML experiments and compare them using an interactive UI.
*   **Model Registry:** A centralized model store designed to collaboratively manage the full lifecycle and deployment of machine learning models.
*   **Deployment:** Tools for seamless model deployment to batch and real-time scoring on platforms like Docker, Kubernetes, Azure ML, and AWS SageMaker.

### Example: Experiment Tracking

Track your models with automatic experiment tracking for scikit-learn:

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

Run `mlflow ui` in a separate terminal to view the UI with your results.

### Example: Evaluating Models

Evaluate models with built-in metrics and evaluation:

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
            "MLflow is an open-source platform for productionizing AI.",
            "Apache Spark is an open-source, distributed computing system.",
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

### Example: Observability

Enable MLflow Tracing for observability for OpenAI, LangChain, LlamaIndex, DSPy, AutoGen, and more.

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

Then navigate to the "Traces" tab in the MLflow UI to find the trace records OpenAI query.

## Hosting MLflow

MLflow can be hosted in various environments, including local machines, on-premise servers, and cloud infrastructure.

Trusted by thousands of organizations, MLflow is now offered as a managed service by most major cloud providers:

-   [Amazon SageMaker](https://aws.amazon.com/sagemaker-ai/experiments/)
-   [Azure ML](https://learn.microsoft.com/en-us/azure/machine-learning/concept-mlflow?view=azureml-api-2)
-   [Databricks](https://www.databricks.com/product/managed-mlflow)
-   [Nebius](https://nebius.com/services/managed-mlflow)

For hosting MLflow on your own infrastructure, please refer to [this guidance](https://mlflow.org/docs/latest/ml/tracking/#tracking-setup).

## Integrations

MLflow is natively integrated with many popular machine learning frameworks and GenAI libraries.

## Support and Community

*   **Documentation:** [https://mlflow.org/docs/latest/index.html](https://mlflow.org/docs/latest/index.html)
*   **Ask AI:** Ask questions in the documentation using the "Ask AI" button.
*   **Events:** [https://lu.ma/mlflow?k=c](https://lu.ma/mlflow?k=c)
*   **GitHub Issues:** [https://github.com/mlflow/mlflow/issues/new/choose](https://github.com/mlflow/mlflow/issues/new/choose)
*   **Mailing List:** mlflow-users@googlegroups.com
*   **Slack:** [https://mlflow.org/slack](https://mlflow.org/slack)

## Contributing

We welcome contributions! Check out the [contribution guide](CONTRIBUTING.md).

## Citation

If you use MLflow in your research, cite it using the "Cite this repository" button at the top of the [GitHub repository page](https://github.com/mlflow/mlflow).

## Core Members

MLflow is currently maintained by the following core members with significant contributions from hundreds of exceptionally talented community members.
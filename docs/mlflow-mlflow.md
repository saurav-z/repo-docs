<h1 align="center" style="border-bottom: none">
    <a href="https://mlflow.org/">
        <img alt="MLflow logo" src="https://raw.githubusercontent.com/mlflow/mlflow/refs/heads/master/assets/logo.svg" width="200" />
    </a>
</h1>

<h2 align="center" style="border-bottom: none">MLflow: Your All-in-One Platform for AI/LLM Application Development and Productionization</h2>

MLflow is the leading open-source platform empowering developers to build, track, deploy, and manage AI/LLM applications with ease and confidence; **transforming the way you build AI applications with a unified, end-to-end solution.**  Get started today and explore the [MLflow repository](https://github.com/mlflow/mlflow).

<div align="center">

[![Python SDK](https://img.shields.io/pypi/v/mlflow)](https://pypi.org/project/mlflow/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/mlflow)](https://pypi.org/project/mlflow/)
[![License](https://img.shields.io/github/license/mlflow/mlflow)](https://github.com/mlflow/mlflow/blob/main/LICENSE)
<a href="https://twitter.com/intent/follow?screen_name=mlflow" target="_blank">
<img src="https://img.shields.io/twitter/follow/mlflow?logo=X&color=%20%23f5f5f5"
      alt="follow on X(Twitter)"></a>
<a href="https://www.linkedin.com/company/mlflow-org/" target="_blank">
<img src="https://custom-icon-badges.demolab.com/badge/LinkedIn-0A66C2?logo=linkedin-white&logoColor=fff"
      alt="follow on LinkedIn"></a>
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/mlflow/mlflow)

</div>

<div align="center">
   <div>
      <a href="https://mlflow.org/"><strong>Website</strong></a> ·
      <a href="https://mlflow.org/docs/latest/index.html"><strong>Docs</strong></a> ·
      <a href="https://github.com/mlflow/mlflow/issues/new/choose"><strong>Feature Request</strong></a> ·
      <a href="https://mlflow.org/blog"><strong>News</strong></a> ·
      <a href="https://www.youtube.com/@mlflowoss"><strong>YouTube</strong></a> ·
      <a href="https://lu.ma/mlflow?k=c"><strong>Events</strong></a>
   </div>
</div>

<br>

## Key Features of MLflow

*   **Experiment Tracking:** Log and compare model parameters, metrics, and artifacts within ML experiments for comprehensive analysis.
*   **LLM Tracing and Observability:**  Gain deep insights into your LLM/agentic applications for debugging and performance monitoring.
*   **LLM Evaluation & Monitoring:** Automatically evaluate and monitor LLM models with integrated experiment tracking.
*   **Prompt Management:** Version, track, and reuse prompts for consistent and collaborative prompt development.
*   **Model Registry:** Centralized model store to collaboratively manage the full lifecycle of ML models.
*   **App Version Tracking:**  Track models, prompts, tools, and code with end-to-end lineage for your AI applications.
*   **Model Deployment:** Deploy models seamlessly to various platforms including Docker, Kubernetes, and cloud providers.
*   **Broad Integrations:** Native integrations with popular machine learning frameworks and GenAI libraries.

## Installation

Install the MLflow Python package with the following command:

```bash
pip install mlflow
```

## Core Components

MLflow provides a unified solution for all your AI/ML needs, covering LLMs, Agents, Deep Learning, and traditional machine learning.

### For LLM / GenAI Developers

| Feature                      | Description                                                                                                       |
| ---------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| **🔍 Tracing / Observability** | Trace the internal states of your LLM/agentic applications for debugging and performance monitoring.            |
| **📊 LLM Evaluation**          | Automate model evaluation with tools seamlessly integrated with experiment tracking for comparing across versions. |
| **🤖 Prompt Management**       | Version, track, and reuse prompts across your organization.                                                    |
| **📦 App Version Tracking**    |  Track the many moving parts in your AI applications with end-to-end lineage.                              |

### For Data Scientists

| Feature              | Description                                                                                                        |
| -------------------- | ------------------------------------------------------------------------------------------------------------------ |
| **📝 Experiment Tracking** | Track your models, parameters, metrics, and evaluation results in ML experiments and compare them using an interactive UI. |
| **💾 Model Registry**  | Centralized model store designed to collaboratively manage the full lifecycle and deployment of machine learning models. |
| **🚀 Deployment**     | Tools for seamless model deployment to batch and real-time scoring.                                               |

## Hosting MLflow

Run MLflow in various environments, including local machines, on-premise servers, and cloud infrastructure.

MLflow is offered as a managed service by major cloud providers:

-   [Amazon SageMaker](https://aws.amazon.com/sagemaker-ai/experiments/)
-   [Azure ML](https://learn.microsoft.com/en-us/azure/machine-learning/concept-mlflow?view=azureml-api-2)
-   [Databricks](https://www.databricks.com/product/managed-mlflow)
-   [Nebius](https://nebius.com/services/managed-mlflow)

For self-hosting, refer to [this guidance](https://mlflow.org/docs/latest/ml/tracking/#tracking-setup).

## Supported Languages

*   [Python](https://pypi.org/project/mlflow/)
*   [TypeScript / JavaScript](https://www.npmjs.com/package/mlflow-tracing)
*   [Java](https://mvnrepository.com/artifact/org.mlflow/mlflow-client)
*   [R](https://cran.r-project.org/web/packages/mlflow/readme/README.html)

## Integrations

MLflow is natively integrated with popular machine learning frameworks and GenAI libraries.  (See the integrations image in original README).

## Usage Examples

### Experiment Tracking

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

Run `mlflow ui` and access the MLflow UI via the printed URL.

### Evaluating Models

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

### Observability

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

Then navigate to the "Traces" tab in the MLflow UI.

## Support

*   Visit the [documentation](https://mlflow.org/docs/latest/index.html) for help.
*   Use the AI-powered chat bot in the documentation.
*   Join the [virtual events](https://lu.ma/mlflow?k=c).
*   [Open a GitHub issue](https://github.com/mlflow/mlflow/issues/new/choose) to report a bug or request a feature.
*   Subscribe to the mailing list or join us on [Slack](https://mlflow.org/slack).

## Contributing

Contributions are welcome!

*   Submit [bug reports](https://github.com/mlflow/mlflow/issues/new?template=bug_report_template.yaml) and [feature requests](https://github.com/mlflow/mlflow/issues/new?template=feature_request_template.yaml)
*   Contribute for [good-first-issues](https://github.com/mlflow/mlflow/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) and [help-wanted](https://github.com/mlflow/mlflow/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22)
*   Share your experience by writing about MLflow

See the [contribution guide](CONTRIBUTING.md) to learn more.

## ⭐️ Star History

(See Star History in original README)

## ✏️ Citation

Cite MLflow using the "Cite this repository" button on the [GitHub repository page](https://github.com/mlflow/mlflow).

## 👥 Core Members

(See list of core members in the original README)
```
Key changes and improvements:

*   **SEO Optimization:** Added a concise, keyword-rich title and description, and used relevant headings.
*   **One-Sentence Hook:**  Provided a compelling opening sentence to immediately grab the reader's attention.
*   **Key Features Section:**  Outlined key features with descriptive bullet points.
*   **Clearer Structure:** Improved the overall structure and readability.
*   **Conciseness:**  Condensed the text while retaining essential information.
*   **Active Voice:** Used active voice for better engagement.
*   **Call to Action:** Included a call to action (Get started today) to encourage engagement.
*   **Links:** Included direct links where appropriate, including back to the original repo.
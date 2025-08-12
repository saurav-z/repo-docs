# MLflow: The Open-Source Platform for Productionizing AI

[![MLflow logo](https://raw.githubusercontent.com/mlflow/mlflow/refs/heads/master/assets/logo.svg)](https://mlflow.org/)

**MLflow empowers data scientists and developers to build, deploy, and manage AI/LLM applications with ease, offering end-to-end capabilities for every stage of the machine learning lifecycle.** [Check out the original repository](https://github.com/mlflow/mlflow).

<div align="center">

[![Python SDK](https://img.shields.io/pypi/v/mlflow)](https://pypi.org/project/mlflow/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/mlflow)](https://pepy.tech/projects/mlflow)
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

## Key Features

*   **Experiment Tracking:** Log and visualize model parameters, metrics, and artifacts to understand and compare experiments.
*   **Model Registry:** Centralized model management for collaborative lifecycle and deployment.
*   **Model Deployment:** Deploy models for batch and real-time scoring on various platforms.
*   **LLM Tracing / Observability:** Trace LLM/Agentic applications for debugging, monitoring and performance analysis.
*   **LLM Evaluation:** Evaluate and compare across multiple LLM versions.
*   **Prompt Management:** Version, track, and reuse prompts to maintain consistency.
*   **App Version Tracking:** Track the evolution of your application, from code to models and prompts.

## 🚀 Installation

```bash
pip install mlflow
```

## 📦 Core Components

MLflow provides a unified solution for all AI/ML needs, including LLMs, Agents, Deep Learning, and traditional machine learning.

### 💡 For LLM / GenAI Developers

*   **Tracing / Observability:** Monitor the internal states of your LLM/agentic applications for debugging, quality issues, and performance.  [Get Started](https://mlflow.org/docs/latest/llms/tracing/index.html)
*   **LLM Evaluation:** Automated model evaluation tools, integrated with experiment tracking. [Get Started](https://mlflow.org/docs/latest/genai/eval-monitor/)
*   **Prompt Management:** Version, track, and reuse prompts across your organization. [Get Started](https://mlflow.org/docs/latest/genai/prompt-registry/create-and-edit-prompts/)
*   **App Version Tracking:** Track models, prompts, tools, and code, with end-to-end lineage. [Get Started](https://mlflow.org/docs/latest/genai/version-tracking/quickstart/)

### 🎓 For Data Scientists

*   **Experiment Tracking:** Track models, parameters, metrics, and results in ML experiments. [Get Started](https://mlflow.org/docs/latest/ml/tracking/quickstart/)
*   **Model Registry:** Manage the full lifecycle and deployment of machine learning models. [Get Started](https://mlflow.org/docs/latest/ml/model-registry/tutorial/)
*   **Deployment:** Seamlessly deploy models to batch and real-time scoring. [Get Started](https://mlflow.org/docs/latest/ml/deployment/)

## 🌐 Hosting MLflow Anywhere

Run MLflow on local machines, on-premise servers, and cloud infrastructure. MLflow is offered as a managed service by major cloud providers:

*   [Amazon SageMaker](https://aws.amazon.com/sagemaker-ai/experiments/)
*   [Azure ML](https://learn.microsoft.com/en-us/azure/machine-learning/concept-mlflow?view=azureml-api-2)
*   [Databricks](https://www.databricks.com/product/managed-mlflow)
*   [Nebius](https://nebius.com/services/managed-mlflow)

For self-hosting, see [this guidance](https://mlflow.org/docs/latest/ml/tracking/#tracking-setup).

## 🗣️ Supported Programming Languages

*   Python
*   TypeScript / JavaScript
*   Java
*   R

## 🔗 Integrations

MLflow integrates with many popular machine learning frameworks and GenAI libraries.

![Integrations](https://raw.githubusercontent.com/mlflow/mlflow/refs/heads/master/assets/readme-integrations.png)

## Usage Examples

### Experiment Tracking

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

Run `mlflow ui` in a separate terminal and access the MLflow UI.

### Evaluating Models

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
            "MLflow is an open-source platform for productionizing AI.",
            "Apache Spark is an open-source, distributed computing system.",
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

Navigate to the "Traces" tab in the MLflow UI.

## 💭 Support

*   [Documentation](https://mlflow.org/docs/latest/index.html)
*   Ask the AI-powered chatbot in the documentation.
*   [Virtual events](https://lu.ma/mlflow?k=c)
*   [GitHub issues](https://github.com/mlflow/mlflow/issues/new/choose)
*   [Mailing list](mlflow-users@googlegroups.com) / [Slack](https://mlflow.org/slack)

## 🤝 Contributing

*   [Bug reports](https://github.com/mlflow/mlflow/issues/new?template=bug_report_template.yaml) and [feature requests](https://github.com/mlflow/mlflow/issues/new?template=feature_request_template.yaml)
*   [Good first issues](https://github.com/mlflow/mlflow/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) and [help wanted](https://github.com/mlflow/mlflow/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22)
*   Share your experience

See our [contribution guide](CONTRIBUTING.md).

## ⭐️ Star History

<a href="https://star-history.com/#mlflow/mlflow&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=mlflow/mlflow&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=mlflow/mlflow&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=mlflow/mlflow&type=Date" />
 </picture>
</a>

## ✏️ Citation

Use the "Cite this repository" button at the top of the [GitHub repository page](https://github.com/mlflow/mlflow).

## 👥 Core Members

*   Ben Wilson
*   Corey Zumar
*   Daniel Lok
*   Gabriel Fu
*   Harutaka Kawamura
*   Serena Ruan
*   Tomu Hirata
*   Weichen Xu
*   Yuki Watanabe
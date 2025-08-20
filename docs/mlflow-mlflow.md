# MLflow: The Open-Source Platform for Productionizing AI

**Unlock the full potential of your AI/LLM projects with MLflow, the leading open-source platform for the entire AI lifecycle.** ([Original Repo](https://github.com/mlflow/mlflow))

<div align="center">
  <img src="https://raw.githubusercontent.com/mlflow/mlflow/refs/heads/master/assets/logo.svg" width="200" alt="MLflow Logo"/>
</div>

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

## Key Features of MLflow:

*   **Experiment Tracking:** Effortlessly track and compare model parameters, metrics, and artifacts.
*   **LLM Tracing / Observability:** Gain insights into the inner workings of your LLM and agentic applications.
*   **LLM Evaluation & Monitoring:** Automate model evaluation with a suite of tools.
*   **Prompt Management:** Version, track, and reuse prompts to improve consistency.
*   **App Version Tracking:** Monitor various AI application elements (models, prompts, tools, and code).
*   **Model Registry:** Centralized model storage to manage the lifecycle and deployment of machine learning models.
*   **Deployment:** Deploy models seamlessly to various platforms like Docker, Kubernetes, Azure ML, and AWS SageMaker.
*   **Integration with Leading Frameworks:**  Works seamlessly with popular machine learning and GenAI libraries.

## 🚀 Installation

Install the MLflow Python package with:

```bash
pip install mlflow
```

## 📦 Core Components

MLflow provides a unified solution for all your AI/ML needs, including LLMs, Agents, Deep Learning, and traditional machine learning.

### 💡 For LLM / GenAI Developers

| Feature                                                                  | Description                                                                                                                                       |
| ------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| **🔍 Tracing / Observability** ([Docs](https://mlflow.org/docs/latest/llms/tracing/index.html))                                | Trace the internal states of your LLM/agentic applications for debugging quality issues and monitoring performance.                      |
| **📊 LLM Evaluation** ([Docs](https://mlflow.org/docs/latest/genai/eval-monitor/))                                            | Evaluate LLM models automatically and compare multiple versions using built-in metrics.                                                   |
| **🤖 Prompt Management** ([Docs](https://mlflow.org/docs/latest/genai/prompt-version-mgmt/prompt-registry/))                  | Version, track, and reuse prompts, ensuring consistency and collaboration.                                                          |
| **📦 App Version Tracking** ([Docs](https://mlflow.org/docs/latest/genai/prompt-version-mgmt/version-tracking/))            | Track key application components such as models, prompts, tools, and code with end-to-end lineage.                                           |

### 🎓 For Data Scientists

| Feature                                                                  | Description                                                                                                                                                              |
| ------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **📝 Experiment Tracking** ([Docs](https://mlflow.org/docs/latest/ml/tracking/))                                              | Log parameters, metrics, models, and artifacts to organize and compare experiments.                                                                 |
| **💾 Model Registry** ([Docs](https://mlflow.org/docs/latest/ml/model-registry/))                                            | Manage the full lifecycle of your ML models with versioning, staging, and deployment tools.                                                                    |
| **🚀 Deployment** ([Docs](https://mlflow.org/docs/latest/ml/deployment/))                                                  | Deploy your models easily to various platforms for batch and real-time scoring.                                                                      |

## 🌐 Hosting MLflow

MLflow can be hosted in various environments, including:

*   Local Machines
*   On-Premise Servers
*   Cloud Infrastructure

Managed services are available from:

*   [Amazon SageMaker](https://aws.amazon.com/sagemaker-ai/experiments/)
*   [Azure ML](https://learn.microsoft.com/en-us/azure/machine-learning/concept-mlflow?view=azureml-api-2)
*   [Databricks](https://www.databricks.com/product/managed-mlflow)
*   [Nebius](https://nebius.com/services/managed-mlflow)

For self-hosting, refer to [this guide](https://mlflow.org/docs/latest/ml/tracking/#tracking-setup).

## 🗣️ Supported Programming Languages

*   [Python](https://pypi.org/project/mlflow/)
*   [TypeScript / JavaScript](https://www.npmjs.com/package/mlflow-tracing)
*   [Java](https://mvnrepository.com/artifact/org.mlflow/mlflow-client)
*   [R](https://cran.r-project.org/web/packages/mlflow/readme/README.html)

## 🔗 Integrations

MLflow natively integrates with numerous popular machine learning frameworks and GenAI libraries.

![Integrations](https://raw.githubusercontent.com/mlflow/mlflow/refs/heads/master/assets/readme-integrations.png)

## Usage Examples

### Experiment Tracking ([Doc](https://mlflow.org/docs/latest/ml/tracking/))

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

Run `mlflow ui` in the terminal to view the MLflow UI.

### Evaluating Models ([Doc](https://mlflow.org/docs/latest/model-evaluation/index.html))

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

### Observability ([Doc](https://mlflow.org/docs/latest/llms/tracing/index.html))

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

Access the "Traces" tab in the MLflow UI.

## 💭 Support

*   [Documentation](https://mlflow.org/docs/latest/index.html)
*   AI-powered chat bot in the documentation
*   [Virtual events](https://lu.ma/mlflow?k=c)
*   [GitHub Issues](https://github.com/mlflow/mlflow/issues/new/choose)
*   [Mailing list](mlflow-users@googlegroups.com)
*   [Slack](https://mlflow.org/slack)

## 🤝 Contributing

Contributions are welcome!  Check out the [contribution guide](CONTRIBUTING.md) and:

*   Submit [bug reports](https://github.com/mlflow/mlflow/issues/new?template=bug_report_template.yaml) and [feature requests](https://github.com/mlflow/mlflow/issues/new?template=feature_request_template.yaml)
*   Work on [good-first-issues](https://github.com/mlflow/mlflow/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) and [help-wanted](https://github.com/mlflow/mlflow/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22)
*   Write about MLflow

## ⭐️ Star History

<a href="https://star-history.com/#mlflow/mlflow&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=mlflow/mlflow&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=mlflow/mlflow&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=mlflow/mlflow&type=Date" />
 </picture>
</a>

## ✏️ Citation

Cite MLflow using the "Cite this repository" button on the [GitHub repository page](https://github.com/mlflow/mlflow).

## 👥 Core Members

(List of Core Members)
# MLflow: Open-Source Platform for Productionizing AI

[![MLflow Logo](https://raw.githubusercontent.com/mlflow/mlflow/refs/heads/master/assets/logo.svg)](https://github.com/mlflow/mlflow)

**MLflow empowers developers to build and deploy AI/LLM applications and models with confidence by providing a unified platform for the entire AI lifecycle.** 

[<img src="https://img.shields.io/pypi/v/mlflow?label=PyPI" alt="PyPI Version">](https://pypi.org/project/mlflow/)
[<img src="https://img.shields.io/pypi/dm/mlflow?label=Downloads" alt="PyPI Downloads">](https://pepy.tech/projects/mlflow)
[<img src="https://img.shields.io/github/license/mlflow/mlflow" alt="License">](https://github.com/mlflow/mlflow/blob/main/LICENSE)
<a href="https://twitter.com/intent/follow?screen_name=mlflow" target="_blank">
<img src="https://img.shields.io/twitter/follow/mlflow?logo=X&color=%20%23f5f5f5"
      alt="follow on X(Twitter)"></a>
<a href="https://www.linkedin.com/company/mlflow-org/" target="_blank">
<img src="https://custom-icon-badges.demolab.com/badge/LinkedIn-0A66C2?logo=linkedin-white&logoColor=fff"
      alt="follow on LinkedIn"></a>
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/mlflow/mlflow)

**Key Features:**

*   **Experiment Tracking:** Easily track and compare your machine learning experiments, including model parameters, metrics, and artifacts.
*   **Model Registry:**  Collaboratively manage the full lifecycle of your machine learning models with a centralized model store.
*   **Model Deployment:** Deploy models to various platforms like Docker, Kubernetes, Azure ML, and AWS SageMaker.
*   **LLM Observability & Tracing:** Gain insights into the internal states of your LLM/agentic applications to debug issues and monitor performance.
*   **LLM Evaluation:** Automate model evaluation with a suite of tools to compare model versions.
*   **Prompt Management:** Version, track, and reuse prompts across your organization.
*   **App Version Tracking:** Track models, prompts, tools, and code with end-to-end lineage.

**Get Started:**

*   [Website](https://mlflow.org/)
*   [Documentation](https://mlflow.org/docs/latest/index.html)
*   [GitHub Repository](https://github.com/mlflow/mlflow)
*   [Feature Requests](https://github.com/mlflow/mlflow/issues/new/choose)
*   [Blog](https://mlflow.org/blog)
*   [YouTube](https://www.youtube.com/@mlflowoss)
*   [Events](https://lu.ma/mlflow?k=c)

## Installation

Install the MLflow Python package using pip:

```bash
pip install mlflow
```

## Core Components

MLflow provides a unified solution for all your AI/ML needs, including LLMs, Agents, Deep Learning, and traditional machine learning.

### For LLM / GenAI Developers

Explore these powerful features for building and managing LLM-powered applications:

| Feature                 | Description                                                                                                                                                                                             | Getting Started                                                                                                                                   |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| **🔍 Tracing / Observability** | Trace the internal states of your LLM/agentic applications for debugging quality issues and monitoring performance with ease.                                                                                                      | [Tracing / Observability Docs](https://mlflow.org/docs/latest/llms/tracing/index.html)   |
| **📊 LLM Evaluation**    | A suite of automated model evaluation tools, seamlessly integrated with experiment tracking to compare across multiple versions.                                                           | [LLM Evaluation Docs](https://mlflow.org/docs/latest/genai/eval-monitor/)    |
| **🤖 Prompt Management**   | Version, track, and reuse prompts across your organization, helping maintain consistency and improve collaboration in prompt development.                                           | [Prompt Management Docs](https://mlflow.org/docs/latest/genai/prompt-version-mgmt/prompt-registry/)        |
| **📦 App Version Tracking**    | MLflow keeps track of many moving parts in your AI applications, such as models, prompts, tools, and code, with end-to-end lineage. | [App Version Tracking Docs](https://mlflow.org/docs/latest/genai/version-tracking/quickstart/)          |

### For Data Scientists

Maximize your machine learning workflow with these core capabilities:

| Feature                 | Description                                                                                                                               | Getting Started                                                                           |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| **📝 Experiment Tracking** | Track your models, parameters, metrics, and evaluation results in ML experiments and compare them using an interactive UI.               | [Experiment Tracking Docs](https://mlflow.org/docs/latest/ml/tracking/quickstart/)        |
| **💾 Model Registry**    | A centralized model store designed to collaboratively manage the full lifecycle and deployment of machine learning models.               | [Model Registry Docs](https://mlflow.org/docs/latest/ml/model-registry/tutorial/)       |
| **🚀 Deployment**        | Tools for seamless model deployment to batch and real-time scoring on platforms like Docker, Kubernetes, Azure ML, and AWS SageMaker.     | [Deployment Docs](https://mlflow.org/docs/latest/ml/deployment/)                         |

## Hosting MLflow Anywhere

MLflow can be run in various environments, including local machines, on-premise servers, and cloud infrastructure. It is offered as a managed service by major cloud providers:

*   [Amazon SageMaker](https://aws.amazon.com/sagemaker-ai/experiments/)
*   [Azure ML](https://learn.microsoft.com/en-us/azure/machine-learning/concept-mlflow?view=azureml-api-2)
*   [Databricks](https://www.databricks.com/product/managed-mlflow)
*   [Nebius](https://nebius.com/services/managed-mlflow)

For hosting MLflow on your own infrastructure, see the [tracking setup guidance](https://mlflow.org/docs/latest/ml/tracking/#tracking-setup).

## Supported Programming Languages

*   [Python](https://pypi.org/project/mlflow/)
*   [TypeScript / JavaScript](https://www.npmjs.com/package/mlflow-tracing)
*   [Java](https://mvnrepository.com/artifact/org.mlflow/mlflow-client)
*   [R](https://cran.r-project.org/web/packages/mlflow/readme/README.html)

## Integrations

MLflow seamlessly integrates with popular machine learning frameworks and GenAI libraries.

![Integrations](https://raw.githubusercontent.com/mlflow/mlflow/refs/heads/master/assets/readme-integrations.png)

## Usage Examples

### Experiment Tracking

Track your model training experiments with MLflow's autologging feature:

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

Run `mlflow ui` in your terminal and access the UI via the printed URL to view the results.

### Evaluating Models

Perform automatic model evaluation for question-answering tasks:

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

Enable LLM observability with MLflow Tracing for insights into your GenAI applications:

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

View the traces in the "Traces" tab in the MLflow UI.

## Support

*   [Documentation](https://mlflow.org/docs/latest/index.html)
*   **Ask AI** - AI-powered chatbot in the documentation.
*   [Virtual Events](https://lu.ma/mlflow?k=c)
*   [GitHub Issues](https://github.com/mlflow/mlflow/issues/new/choose) - for bug reports, documentation issues, and feature requests.
*   [Mailing List](mlflow-users@googlegroups.com) or [Slack](https://mlflow.org/slack) for release announcements and discussions.

## Contributing

Contributions to MLflow are welcome!

*   [Bug Reports](https://github.com/mlflow/mlflow/issues/new?template=bug_report_template.yaml)
*   [Feature Requests](https://github.com/mlflow/mlflow/issues/new?template=feature_request_template.yaml)
*   [Good First Issues](https://github.com/mlflow/mlflow/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
*   [Help Wanted Issues](https://github.com/mlflow/mlflow/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22)
*   Share your experiences by writing about MLflow.

See the [contribution guide](CONTRIBUTING.md) for details.

## Star History

<a href="https://star-history.com/#mlflow/mlflow&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=mlflow/mlflow&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=mlflow/mlflow&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=mlflow/mlflow&type=Date" />
 </picture>
</a>

## Citation

Cite MLflow in your research by using the "Cite this repository" button on the [GitHub repository page](https://github.com/mlflow/mlflow).

## Core Members

MLflow is maintained by the following core members and supported by a large community:

*   [Ben Wilson](https://github.com/BenWilson2)
*   [Corey Zumar](https://github.com/dbczumar)
*   [Daniel Lok](https://github.com/daniellok-db)
*   [Gabriel Fu](https://github.com/gabrielfu)
*   [Harutaka Kawamura](https://github.com/harupy)
*   [Serena Ruan](https://github.com/serena-ruan)
*   [Tomu Hirata](https://github.com/TomeHirata)
*   [Weichen Xu](https://github.com/WeichenXu123)
*   [Yuki Watanabe](https://github.com/B-Step62)
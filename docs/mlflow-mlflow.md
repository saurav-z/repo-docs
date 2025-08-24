# MLflow: The Open-Source Platform for Productionizing AI

**Unlock the power of your AI/LLM projects with MLflow, the comprehensive open-source platform for end-to-end AI development and deployment.**  [Visit the original repository](https://github.com/mlflow/mlflow)

[![PyPI version](https://img.shields.io/pypi/v/mlflow)](https://pypi.org/project/mlflow/)
[![PyPI downloads](https://img.shields.io/pypi/dm/mlflow)](https://pepy.tech/projects/mlflow)
[![License](https://img.shields.io/github/license/mlflow/mlflow)](https://github.com/mlflow/mlflow/blob/main/LICENSE)
[![Twitter Follow](https://img.shields.io/twitter/follow/mlflow?logo=X&color=%20%23f5f5f5)](https://twitter.com/intent/follow?screen_name=mlflow)
[![LinkedIn Follow](https://img.shields.io/badge/LinkedIn-0A66C2?logo=linkedin-white&logoColor=fff)](https://www.linkedin.com/company/mlflow-org/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/mlflow/mlflow)

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

*   **Experiment Tracking:** Effortlessly track and compare your model parameters, metrics, and results for optimal performance.
*   **Model Registry:** Centralized model store for managing the full lifecycle of your machine learning models.
*   **Model Deployment:** Seamlessly deploy your models to various platforms, including Docker, Kubernetes, Azure ML, and AWS SageMaker.
*   **LLM Tracing & Observability:** Gain insights into LLM application behavior for debugging and performance monitoring.
*   **LLM Evaluation:** Automate model evaluation with a suite of tools integrated with experiment tracking.
*   **Prompt Management:** Version, track, and reuse prompts to ensure consistency and collaboration in prompt development.
*   **App Version Tracking:** Track all the moving parts in your AI applications, including models, prompts, tools, and code, with end-to-end lineage.

## Installation

Install the MLflow Python package using pip:

```bash
pip install mlflow
```

## Core Components

MLflow offers a unified solution to meet all your AI/ML needs, including LLMs, Agents, Deep Learning, and traditional machine learning.

### For LLM / GenAI Developers

#### LLM Tracing / Observability

<img src="https://raw.githubusercontent.com/mlflow/mlflow/refs/heads/master/assets/readme-tracing.png" alt="Tracing" width=100%>
*Trace the internal states of your LLM/agentic applications for debugging quality issues and monitoring performance with ease.*
<br><br>
[Tracing Docs](https://mlflow.org/docs/latest/llms/tracing/index.html)

#### LLM Evaluation

<img src="https://raw.githubusercontent.com/mlflow/mlflow/refs/heads/master/assets/readme-llm-eval.png" alt="LLM Evaluation" width=100%>
*A suite of automated model evaluation tools, seamlessly integrated with experiment tracking to compare across multiple versions.*
<br><br>
[Evaluation Docs](https://mlflow.org/docs/latest/genai/eval-monitor/)

#### Prompt Management

<img src="https://raw.githubusercontent.com/mlflow/mlflow/refs/heads/master/assets/readme-prompt.png" alt="Prompt Management">
*Version, track, and reuse prompts across your organization, helping maintain consistency and improve collaboration in prompt development.*
<br><br>
[Prompt Management Docs](https://mlflow.org/docs/latest/genai/prompt-version-mgmt/prompt-registry/)

#### App Version Tracking

<img src="https://raw.githubusercontent.com/mlflow/mlflow/refs/heads/master/assets/readme-logged-model.png" alt="MLflow Hero">
*MLflow keeps track of many moving parts in your AI applications, such as models, prompts, tools, and code, with end-to-end lineage.*
<br><br>
[Version Tracking Docs](https://mlflow.org/docs/latest/genai/version-tracking/quickstart/)

### For Data Scientists

#### Experiment Tracking

<img src="https://raw.githubusercontent.com/mlflow/mlflow/refs/heads/master/assets/readme-experiment.png" alt="Tracking" width=50%>
*Track your models, parameters, metrics, and evaluation results in ML experiments and compare them using an interactive UI.*
<br><br>
[Experiment Tracking Docs](https://mlflow.org/docs/latest/ml/tracking/)

#### Model Registry

<img src="https://raw.githubusercontent.com/mlflow/mlflow/refs/heads/master/assets/readme-model-registry.png" alt="Model Registry" width=100%>
*A centralized model store designed to collaboratively manage the full lifecycle and deployment of machine learning models.*
<br><br>
[Model Registry Docs](https://mlflow.org/docs/latest/ml/model-registry/)

#### Deployment

<img src="https://raw.githubusercontent.com/mlflow/mlflow/refs/heads/master/assets/readme-deployment.png" alt="Deployment" width=100%>
*Tools for seamless model deployment to batch and real-time scoring on platforms like Docker, Kubernetes, Azure ML, and AWS SageMaker.*
<br><br>
[Deployment Docs](https://mlflow.org/docs/latest/ml/deployment/)

## Hosting MLflow Anywhere

MLflow offers flexible deployment options, allowing you to run it on local machines, on-premise servers, and cloud infrastructure.

MLflow is offered as a managed service by major cloud providers:

*   [Amazon SageMaker](https://aws.amazon.com/sagemaker-ai/experiments/)
*   [Azure ML](https://learn.microsoft.com/en-us/azure/machine-learning/concept-mlflow?view=azureml-api-2)
*   [Databricks](https://www.databricks.com/product/managed-mlflow)
*   [Nebius](https://nebius.com/services/managed-mlflow)

For self-hosting, refer to the [tracking setup guidance](https://mlflow.org/docs/latest/ml/tracking/#tracking-setup).

## Supported Programming Languages

*   [Python](https://pypi.org/project/mlflow/)
*   [TypeScript / JavaScript](https://www.npmjs.com/package/mlflow-tracing)
*   [Java](https://mvnrepository.com/artifact/org.mlflow/mlflow-client)
*   [R](https://cran.r-project.org/web/packages/mlflow/readme/README.html)

## Integrations

MLflow seamlessly integrates with many popular machine learning frameworks and GenAI libraries.

![Integrations](https://raw.githubusercontent.com/mlflow/mlflow/refs/heads/master/assets/readme-integrations.png)

## Usage Examples

### Experiment Tracking ([Doc](https://mlflow.org/docs/latest/ml/tracking/))

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

Run `mlflow ui` in a separate terminal and access the MLflow UI via the URL to view the run.

### Evaluating Models ([Doc](https://mlflow.org/docs/latest/model-evaluation/index.html))

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

### Observability ([Doc](https://mlflow.org/docs/latest/llms/tracing/index.html))

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

## Support

*   Find answers in the [documentation](https://mlflow.org/docs/latest/index.html)
*   Ask our AI-powered chatbot in the documentation by clicking the **"Ask AI"** button at the right bottom.
*   Join [virtual events](https://lu.ma/mlflow?k=c)
*   Report bugs/feature requests: [open a GitHub issue](https://github.com/mlflow/mlflow/issues/new/choose).
*   Subscribe to the mailing list (mlflow-users@googlegroups.com) or join us on [Slack](https://mlflow.org/slack).

## Contributing

We welcome contributions!

*   Submit [bug reports](https://github.com/mlflow/mlflow/issues/new?template=bug_report_template.yaml) and [feature requests](https://github.com/mlflow/mlflow/issues/new?template=feature_request_template.yaml)
*   Contribute for [good-first-issues](https://github.com/mlflow/mlflow/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) and [help-wanted](https://github.com/mlflow/mlflow/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22)
*   Share your experience by writing about MLflow

See our [contribution guide](CONTRIBUTING.md).

## Star History

<a href="https://star-history.com/#mlflow/mlflow&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=mlflow/mlflow&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=mlflow/mlflow&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=mlflow/mlflow&type=Date" />
 </picture>
</a>

## Citation

Cite MLflow using the "Cite this repository" button on the [GitHub repository page](https://github.com/mlflow/mlflow).

## Core Members

Maintained by core members and community contributors.
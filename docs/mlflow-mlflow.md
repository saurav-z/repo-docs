<h1 align="center" style="border-bottom: none">
    <a href="https://mlflow.org/">
        <img alt="MLflow logo" src="https://raw.githubusercontent.com/mlflow/mlflow/refs/heads/master/assets/logo.svg" width="200" />
    </a>
</h1>

## MLflow: The Open-Source Platform for Productionizing AI

**MLflow empowers developers to build, deploy, and manage AI/LLM applications with confidence.** This comprehensive platform streamlines the entire machine learning lifecycle, from experiment tracking to model deployment.

[Visit the original repo on GitHub](https://github.com/mlflow/mlflow)

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

## Key Features of MLflow

*   **Experiment Tracking:** Track and compare model parameters, metrics, and results in a centralized location for easy analysis.
*   **LLM Tracing and Observability:** Monitor the internal states of LLM/agentic applications for debugging and performance optimization.
*   **LLM Evaluation:** Automatically evaluate and compare LLM performance using built-in metrics.
*   **Prompt Management:** Version, track, and reuse prompts across your organization to maintain consistency and improve collaboration.
*   **App Version Tracking:**  MLflow tracks models, prompts, tools, and code with end-to-end lineage for comprehensive AI application management.
*   **Model Registry:** A central model store for collaborative lifecycle management and deployment.
*   **Deployment:** Deploy models seamlessly to various platforms, including Docker, Kubernetes, and cloud services.

## Installation

Install the MLflow Python package using pip:

```bash
pip install mlflow
```

## Core Components

MLflow provides a unified solution for all AI/ML needs, including LLMs, Agents, Deep Learning, and traditional machine learning.

### For LLM / GenAI Developers

<details>
  <summary><b>Tracing / Observability</b></summary>
  <img src="https://raw.githubusercontent.com/mlflow/mlflow/refs/heads/master/assets/readme-tracing.png" alt="Tracing" width="100%">
  Trace the internal states of your LLM/agentic applications for debugging quality issues and monitoring performance with ease.
  <br>
  <a href="https://mlflow.org/docs/latest/llms/tracing/index.html"><strong>🔍 Tracing / Observability</strong></a>
  <br>
  <a href="https://mlflow.org/docs/latest/genai/tracing/quickstart/python-openai/">Getting Started →</a>
</details>

<details>
  <summary><b>LLM Evaluation</b></summary>
  <img src="https://raw.githubusercontent.com/mlflow/mlflow/refs/heads/master/assets/readme-llm-eval.png" alt="LLM Evaluation" width="100%">
  A suite of automated model evaluation tools, seamlessly integrated with experiment tracking to compare across multiple versions.
  <br>
  <a href="https://mlflow.org/docs/latest/genai/eval-monitor/"><strong>📊 LLM Evaluation</strong></a>
  <br>
  <a href="https://mlflow.org/docs/latest/genai/eval-monitor/">Getting Started →</a>
</details>

<details>
  <summary><b>Prompt Management</b></summary>
  <img src="https://raw.githubusercontent.com/mlflow/mlflow/refs/heads/master/assets/readme-prompt.png" alt="Prompt Management">
  Version, track, and reuse prompts across your organization, helping maintain consistency and improve collaboration in prompt development.
  <br>
  <a href="https://mlflow.org/docs/latest/genai/prompt-version-mgmt/prompt-registry/"><strong>🤖 Prompt Management</strong></a>
  <br>
  <a href="https://mlflow.org/docs/latest/genai/prompt-registry/create-and-edit-prompts/">Getting Started →</a>
</details>

<details>
  <summary><b>App Version Tracking</b></summary>
  <img src="https://raw.githubusercontent.com/mlflow/mlflow/refs/heads/master/assets/readme-logged-model.png" alt="MLflow Hero">
  MLflow keeps track of many moving parts in your AI applications, such as models, prompts, tools, and code, with end-to-end lineage.
  <br>
  <a href="https://mlflow.org/docs/latest/genai/prompt-version-mgmt/version-tracking/"><strong>📦 App Version Tracking</strong></a>
  <br>
  <a href="https://mlflow.org/docs/latest/genai/version-tracking/quickstart/">Getting Started →</a>
</details>

### For Data Scientists

<details>
  <summary><b>Experiment Tracking</b></summary>
  <img src="https://raw.githubusercontent.com/mlflow/mlflow/refs/heads/master/assets/readme-experiment.png" alt="Tracking" width="50%">
  Track your models, parameters, metrics, and evaluation results in ML experiments and compare them using an interactive UI.
  <br>
  <a href="https://mlflow.org/docs/latest/ml/tracking/"><strong>📝 Experiment Tracking</strong></a>
  <br>
  <a href="https://mlflow.org/docs/latest/ml/tracking/quickstart/">Getting Started →</a>
</details>

<details>
  <summary><b>Model Registry</b></summary>
  <img src="https://raw.githubusercontent.com/mlflow/mlflow/refs/heads/master/assets/readme-model-registry.png" alt="Model Registry" width="100%">
  A centralized model store designed to collaboratively manage the full lifecycle and deployment of machine learning models.
  <br>
  <a href="https://mlflow.org/docs/latest/ml/model-registry/"><strong>💾 Model Registry</strong></a>
  <br>
  <a href="https://mlflow.org/docs/latest/ml/model-registry/tutorial/">Getting Started →</a>
</details>

<details>
  <summary><b>Deployment</b></summary>
  <img src="https://raw.githubusercontent.com/mlflow/mlflow/refs/heads/master/assets/readme-deployment.png" alt="Deployment" width="100%">
  Tools for seamless model deployment to batch and real-time scoring on platforms like Docker, Kubernetes, Azure ML, and AWS SageMaker.
  <br>
  <a href="https://mlflow.org/docs/latest/ml/deployment/"><strong>🚀 Deployment</strong></a>
  <br>
  <a href="https://mlflow.org/docs/latest/ml/deployment/">Getting Started →</a>
</details>

## Hosting MLflow Anywhere

MLflow supports deployment in various environments, including:

*   Local machines
*   On-premise servers
*   Cloud infrastructure

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

MLflow seamlessly integrates with a wide array of machine learning frameworks and GenAI libraries.

![Integrations](https://raw.githubusercontent.com/mlflow/mlflow/refs/heads/master/assets/readme-integrations.png)

## Usage Examples

### Experiment Tracking ([Doc](https://mlflow.org/docs/latest/ml/tracking/))

This example trains a scikit-learn regression model with MLflow's autologging feature.

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

After running the code, run `mlflow ui` in a separate terminal.  Access the MLflow UI via the URL provided to view the automatically created MLflow Run and related tracking information.

```bash
mlflow ui
```

### Evaluating Models ([Doc](https://mlflow.org/docs/latest/model-evaluation/index.html))

The following example runs automatic evaluation for question-answering tasks.

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

MLflow Tracing provides LLM observability for GenAI libraries like OpenAI, LangChain, etc.  Enable auto-tracing with `mlflow.xyz.autolog()`.

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

Navigate to the "Traces" tab in the MLflow UI to view the trace records.

## Support

*   Find help in the [documentation](https://mlflow.org/docs/latest/index.html).  Use the **"Ask AI"** button.
*   Join [virtual events](https://lu.ma/mlflow?k=c).
*   Report bugs and make feature requests via [GitHub issues](https://github.com/mlflow/mlflow/issues/new/choose).
*   Subscribe to the mailing list (mlflow-users@googlegroups.com) or join us on [Slack](https://mlflow.org/slack).

## Contributing

Contribute to MLflow!

*   Submit [bug reports](https://github.com/mlflow/mlflow/issues/new?template=bug_report_template.yaml) and [feature requests](https://github.com/mlflow/mlflow/issues/new?template=feature_request_template.yaml)
*   Work on [good-first-issues](https://github.com/mlflow/mlflow/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) and [help-wanted](https://github.com/mlflow/mlflow/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22)
*   Share your MLflow experience

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

Maintained by the following core members and community contributors.

-   [Ben Wilson](https://github.com/BenWilson2)
-   [Corey Zumar](https://github.com/dbczumar)
-   [Daniel Lok](https://github.com/daniellok-db)
-   [Gabriel Fu](https://github.com/gabrielfu)
-   [Harutaka Kawamura](https://github.com/harupy)
-   [Serena Ruan](https://github.com/serena-ruan)
-   [Tomu Hirata](https://github.com/TomeHirata)
-   [Weichen Xu](https://github.com/WeichenXu123)
-   [Yuki Watanabe](https://github.com/B-Step62)
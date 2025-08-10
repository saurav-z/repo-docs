# MLflow: Productionize AI with Confidence

**MLflow is the open-source platform that empowers developers to build, deploy, and manage AI/LLM applications efficiently.**

[<img src="https://raw.githubusercontent.com/mlflow/mlflow/refs/heads/master/assets/logo.svg" width="200" alt="MLflow logo" />](https://mlflow.org/)

MLflow provides an integrated solution for the entire AI/ML lifecycle, including LLMs, Agents, Deep Learning, and traditional machine learning. With MLflow, you can easily track experiments, manage models, deploy them, and monitor performance.

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

## Key Features of MLflow

*   **Experiment Tracking:** Track and compare model parameters, metrics, and artifacts across multiple experiments.
*   **Model Management & Registry:** Centralized model store to manage the full lifecycle of your machine learning models, including versioning and staging.
*   **Deployment:** Deploy models to various platforms like Docker, Kubernetes, Azure ML, and AWS SageMaker.
*   **LLM/GenAI Tracing & Observability:** Trace the internal states of your LLM/agentic applications for debugging and performance monitoring.
*   **LLM Evaluation & Monitoring:** Automated model evaluation tools to compare different model versions and measure LLM performance.
*   **Prompt Management:** Version, track, and reuse prompts across your organization.
*   **App Version Tracking:** Track models, prompts, tools, and code with end-to-end lineage.
*   **Integration:** Native integration with popular machine learning frameworks and GenAI libraries, simplifying your workflow.

## Installation

Install the MLflow Python package using pip:

```bash
pip install mlflow
```

## Core Components - Your End-to-End AI Platform

MLflow provides a unified solution to handle your entire AI/ML lifecycle for LLMs, Agents, Deep Learning, and traditional machine learning.

### LLM / GenAI Developers

*   **Tracing / Observability:** Easily debug and monitor the performance of your LLM/agentic applications. [Learn More](https://mlflow.org/docs/latest/llms/tracing/index.html)
*   **LLM Evaluation:** Automated model evaluation tools to compare multiple versions. [Learn More](https://mlflow.org/docs/latest/genai/eval-monitor/)
*   **Prompt Management:** Version and track prompts to maintain consistency and collaboration. [Learn More](https://mlflow.org/docs/latest/genai/prompt-version-mgmt/prompt-registry/)
*   **App Version Tracking:** Track and manage models, prompts, tools, and code within your AI applications. [Learn More](https://mlflow.org/docs/latest/genai/version-tracking/quickstart/)

### Data Scientists

*   **Experiment Tracking:** Track models, parameters, metrics, and evaluation results in ML experiments and compare them. [Learn More](https://mlflow.org/docs/latest/ml/tracking/quickstart/)
*   **Model Registry:** Manage the full lifecycle of your machine learning models, including versioning and staging. [Learn More](https://mlflow.org/docs/latest/ml/model-registry/tutorial/)
*   **Deployment:** Deploy models to various platforms for batch and real-time scoring. [Learn More](https://mlflow.org/docs/latest/ml/deployment/)

## Hosting MLflow

MLflow can be hosted in various environments, including:

*   Local Machines
*   On-Premise Servers
*   Cloud Infrastructure

MLflow is offered as a managed service by:

*   [Amazon SageMaker](https://aws.amazon.com/sagemaker-ai/experiments/)
*   [Azure ML](https://learn.microsoft.com/en-us/azure/machine-learning/concept-mlflow?view=azureml-api-2)
*   [Databricks](https://www.databricks.com/product/managed-mlflow)
*   [Nebius](https://nebius.com/services/managed-mlflow)

For self-hosting guidance, refer to [this documentation](https://mlflow.org/docs/latest/ml/tracking/#tracking-setup).

## Supported Languages

*   Python
*   TypeScript / JavaScript
*   Java
*   R

## Integrations

MLflow integrates with popular machine learning frameworks and GenAI libraries.

![Integrations](https://raw.githubusercontent.com/mlflow/mlflow/refs/heads/master/assets/readme-integrations.png)

## Usage Examples

### Experiment Tracking

This example trains a regression model with scikit-learn and uses MLflow's autologging for experiment tracking:

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

Run `mlflow ui` in a separate terminal to access the MLflow UI and view the automatically tracked experiment.

### Evaluating Models

This example runs automated evaluation for question-answering tasks with built-in metrics:

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

MLflow Tracing provides LLM observability. Enable auto-tracing with `mlflow.xyz.autolog()` before running your models.

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

View trace records in the "Traces" tab in the MLflow UI.

## Support

*   **Documentation:** [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
*   **AI-Powered Chat Bot:**  Ask questions using the "Ask AI" button within the documentation.
*   **Events:** [Virtual Events](https://lu.ma/mlflow?k=c)
*   **GitHub Issues:**  Report bugs, file documentation issues, or submit feature requests: [Open a GitHub issue](https://github.com/mlflow/mlflow/issues/new/choose)
*   **Mailing List:** Subscribe for release announcements and discussions (mlflow-users@googlegroups.com)
*   **Slack:** [Join us on Slack](https://mlflow.org/slack)

## Contributing

We welcome contributions!

*   [Bug Reports](https://github.com/mlflow/mlflow/issues/new?template=bug_report_template.yaml)
*   [Feature Requests](https://github.com/mlflow/mlflow/issues/new?template=feature_request_template.yaml)
*   [Good First Issues](https://github.com/mlflow/mlflow/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
*   [Help Wanted](https://github.com/mlflow/mlflow/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22)
*   Writing about MLflow and sharing your experience

See our [contribution guide](CONTRIBUTING.md) for details.

## Star History

<a href="https://star-history.com/#mlflow/mlflow&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=mlflow/mlflow&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=mlflow/mlflow&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=mlflow/mlflow&type=Date" />
 </picture>
</a>

## Citation

If you use MLflow in your research, cite it using the "Cite this repository" button on the [GitHub repository page](https://github.com/mlflow/mlflow).

## Core Members

The project is maintained by the following core members, with contributions from the community.

*   [Ben Wilson](https://github.com/BenWilson2)
*   [Corey Zumar](https://github.com/dbczumar)
*   [Daniel Lok](https://github.com/daniellok-db)
*   [Gabriel Fu](https://github.com/gabrielfu)
*   [Harutaka Kawamura](https://github.com/harupy)
*   [Serena Ruan](https://github.com/serena-ruan)
*   [Tomu Hirata](https://github.com/TomeHirata)
*   [Weichen Xu](https://github.com/WeichenXu123)
*   [Yuki Watanabe](https://github.com/B-Step62)

[Back to top](#mlflow-productionize-ai-with-confidence)
```
Key improvements:

*   **SEO-optimized Headline:**  Uses "MLflow: Productionize AI with Confidence" which includes the keyword "MLflow" and speaks to the key benefit.
*   **Concise Hook:** The introductory sentence immediately highlights the core value proposition.
*   **Clear Headings:** Uses descriptive headings (Key Features, Installation, Core Components, Hosting MLflow, etc.) for better readability and SEO.
*   **Bulleted Key Features:** Makes the most important aspects easy to scan.
*   **Detailed Descriptions:** Expands on the benefits of each section.
*   **Complete Lifecycle Coverage:** Emphasizes the end-to-end nature of the platform.
*   **Links to Documentation & Getting Started:** Include links where users can get more help and start using the tool.
*   **Clear Call to Action:** Directs users to the GitHub repository.
*   **Comprehensive Overview:** Addresses all sections of the original README and includes code examples.
*   **Clean Formatting:** Uses markdown for improved readability.
*   **Back to top anchor:** Added a back to top anchor at the bottom for users to jump back to the beginning.
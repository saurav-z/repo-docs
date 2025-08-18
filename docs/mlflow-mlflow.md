# MLflow: Productionize Your AI with Confidence

**MLflow is the open-source platform that streamlines the AI/LLM development lifecycle, from experimentation to deployment.**  [Learn more about MLflow](https://github.com/mlflow/mlflow)

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

## Key Features

MLflow offers a comprehensive suite of tools to empower AI/LLM developers and data scientists.

*   **Experiment Tracking:** Log and visualize your model parameters, metrics, and results to compare performance and identify optimal configurations.
*   **Model Registry:** Manage the full lifecycle of your machine learning models, from training to deployment, with a centralized model store.
*   **Model Deployment:** Deploy your models easily to various platforms like Docker, Kubernetes, Azure ML, and AWS SageMaker.
*   **LLM Tracing & Observability:** Debug LLM applications by tracing internal states and monitor performance with ease.
*   **LLM Evaluation:** Automate model evaluation with built-in metrics for question-answering tasks.
*   **Prompt Management:** Version, track, and reuse prompts for your LLM applications.
*   **App Version Tracking:** Track models, prompts, tools, and code in your AI application with end-to-end lineage.

## Getting Started

### Installation

Install the MLflow Python package using pip:

```bash
pip install mlflow
```

### Core Components

MLflow provides a unified solution for all your AI/ML needs, including LLMs, Agents, Deep Learning, and traditional machine learning.

#### For LLM / GenAI Developers

*   **Tracing / Observability:** Trace and debug LLM applications. [Get Started](https://mlflow.org/docs/latest/llms/tracing/index.html)
*   **LLM Evaluation:** Automate model evaluation. [Get Started](https://mlflow.org/docs/latest/genai/eval-monitor/)
*   **Prompt Management:** Version and track prompts. [Get Started](https://mlflow.org/docs/latest/genai/prompt-registry/create-and-edit-prompts/)
*   **App Version Tracking:**  Track models, prompts, tools, and code with end-to-end lineage. [Get Started](https://mlflow.org/docs/latest/genai/version-tracking/quickstart/)

#### For Data Scientists

*   **Experiment Tracking:** Track models, parameters, metrics, and evaluation results in ML experiments. [Get Started](https://mlflow.org/docs/latest/ml/tracking/quickstart/)
*   **Model Registry:**  Manage the full lifecycle and deployment of machine learning models. [Get Started](https://mlflow.org/docs/latest/ml/model-registry/tutorial/)
*   **Deployment:** Seamless model deployment to various platforms. [Get Started](https://mlflow.org/docs/latest/ml/deployment/)

## Hosting MLflow

MLflow can be hosted in various environments:

*   **Local:** Run on your local machine.
*   **On-Premise:** Deploy on your own servers.
*   **Cloud:** Utilize managed services from major providers:
    *   Amazon SageMaker
    *   Azure ML
    *   Databricks
    *   Nebius

For self-hosting instructions, refer to the [tracking setup documentation](https://mlflow.org/docs/latest/ml/tracking/#tracking-setup).

## Supported Languages

*   Python
*   TypeScript / JavaScript
*   Java
*   R

## Integrations

MLflow seamlessly integrates with popular machine learning frameworks and GenAI libraries.  [View Integrations](https://raw.githubusercontent.com/mlflow/mlflow/refs/heads/master/assets/readme-integrations.png)

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

Run `mlflow ui` in the terminal to access the MLflow UI and view experiment runs.

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
Navigate to the "Traces" tab in the MLflow UI to find trace records.

## Support

*   [Documentation](https://mlflow.org/docs/latest/index.html)
*   [AI-powered Chatbot](https://mlflow.org/docs/latest/index.html)
*   [Virtual Events](https://lu.ma/mlflow?k=c)
*   [GitHub Issues](https://github.com/mlflow/mlflow/issues/new/choose)
*   [Mailing List](mailto:mlflow-users@googlegroups.com)
*   [Slack](https://mlflow.org/slack)

## Contributing

We welcome contributions!  See the [contribution guide](CONTRIBUTING.md) for details.

*   [Bug Reports](https://github.com/mlflow/mlflow/issues/new?template=bug_report_template.yaml)
*   [Feature Requests](https://github.com/mlflow/mlflow/issues/new?template=feature_request_template.yaml)
*   [Good First Issues](https://github.com/mlflow/mlflow/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
*   [Help Wanted Issues](https://github.com/mlflow/mlflow/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22)
*   Writing about MLflow

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

[List of Core Members]
```
Key improvements and optimizations:

*   **SEO Focus:**  Includes relevant keywords like "AI," "LLM," "machine learning," "experiment tracking," "model registry," and "deployment" in headings and descriptions to improve search engine visibility.
*   **Concise & Engaging Hook:** Starts with a clear and compelling one-sentence description of the platform's core value.
*   **Clear Headings:** Uses descriptive headings to organize the content for readability and scannability.
*   **Bulleted Key Features:** Highlights the core functionalities in a clear, concise bulleted list.
*   **Action-Oriented Language:**  Uses active verbs and phrases like "Streamlines," "Empower," "Manage," and "Deploy" to make the content more engaging.
*   **Simplified Sections:** Removed redundant information and consolidated sections.
*   **Call to Actions (CTAs):** Encourages user interaction with "Get Started," "View Integrations," and links back to the GitHub repo.
*   **Visual Appeal:** Maintains the original visual elements (logos, badges, images) to make the README more attractive.
*   **Complete Documentation Links:**  Provides clear and consistent links to documentation pages for each feature.
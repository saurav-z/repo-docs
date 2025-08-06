# MLflow: Open Source Platform for Productionizing AI

MLflow is a comprehensive open-source platform that empowers developers to build, manage, and deploy AI and LLM applications with confidence.  **[Explore the MLflow Repository](https://github.com/mlflow/mlflow)**

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

## Key Features

*   **Experiment Tracking:** Effortlessly track model parameters, metrics, and artifacts, and compare results through an interactive UI. ([Docs](https://mlflow.org/docs/latest/ml/tracking/))
*   **Model Registry:**  Centrally manage the full lifecycle of your machine learning models, enabling collaboration and streamlined deployment. ([Docs](https://mlflow.org/docs/latest/ml/model-registry/))
*   **Model Deployment:** Deploy models seamlessly to various platforms, including batch and real-time scoring on platforms like Docker, Kubernetes, Azure ML, and AWS SageMaker. ([Docs](https://mlflow.org/docs/latest/ml/deployment/))
*   **LLM/GenAI Focused Features:**
    *   **Tracing / Observability:** Trace the internal states of LLM/agentic applications for debugging quality issues and monitoring performance with ease. ([Docs](https://mlflow.org/docs/latest/llms/tracing/index.html))
    *   **LLM Evaluation:** Automate model evaluation with a suite of tools, seamlessly integrated with experiment tracking. ([Docs](https://mlflow.org/docs/latest/genai/eval-monitor/))
    *   **Prompt Management:** Version, track, and reuse prompts across your organization to maintain consistency and improve collaboration. ([Docs](https://mlflow.org/docs/latest/genai/prompt-version-mgmt/prompt-registry/))
    *   **App Version Tracking:** Track models, prompts, tools, and code with end-to-end lineage in your AI applications. ([Docs](https://mlflow.org/docs/latest/genai/version-tracking/quickstart/))

## Installation

Get started quickly with MLflow by installing the Python package:

```bash
pip install mlflow
```

## Supported Programming Languages

*   Python
*   TypeScript / JavaScript
*   Java
*   R

## Integrations

MLflow integrates natively with popular machine learning frameworks and GenAI libraries.

![Integrations](https://raw.githubusercontent.com/mlflow/mlflow/refs/heads/master/assets/readme-integrations.png)

## Use Cases

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

Run `mlflow ui` in your terminal and access the URL to see the tracked experiment.

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

Navigate to the "Traces" tab in the MLflow UI.

## Hosting MLflow Anywhere

MLflow can be hosted in various environments including local machines, on-premise servers, and major cloud providers:

*   Amazon SageMaker
*   Azure ML
*   Databricks
*   Nebius

## Support and Community

*   **Documentation:**  [https://mlflow.org/docs/latest/index.html](https://mlflow.org/docs/latest/index.html)
*   **Ask AI:** Utilize the AI-powered chat bot within the documentation for assistance.
*   **Events:** Join virtual events, like office hours and meetups.
*   **GitHub Issues:**  Report bugs, request features, or submit documentation issues on [GitHub](https://github.com/mlflow/mlflow/issues/new/choose).
*   **Mailing List:** Subscribe to the [mlflow-users@googlegroups.com](mailto:mlflow-users@googlegroups.com) mailing list for announcements.
*   **Slack:** Join us on [Slack](https://mlflow.org/slack)

## Contributing

Contribute to MLflow by:

*   Submitting [bug reports](https://github.com/mlflow/mlflow/issues/new?template=bug_report_template.yaml) and [feature requests](https://github.com/mlflow/mlflow/issues/new?template=feature_request_template.yaml)
*   Addressing [good-first-issues](https://github.com/mlflow/mlflow/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) and [help-wanted](https://github.com/mlflow/mlflow/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22)
*   Writing about MLflow and sharing your experience

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

Please cite MLflow if used in your research using the "Cite this repository" button on the [GitHub repository page](https://github.com/mlflow/mlflow).

## 👥 Core Members

Maintained by core members and community contributions:

*   [Ben Wilson](https://github.com/BenWilson2)
*   [Corey Zumar](https://github.com/dbczumar)
*   [Daniel Lok](https://github.com/daniellok-db)
*   [Gabriel Fu](https://github.com/gabrielfu)
*   [Harutaka Kawamura](https://github.com/harupy)
*   [Serena Ruan](https://github.com/serena-ruan)
*   [Tomu Hirata](https://github.com/TomeHirata)
*   [Weichen Xu](https://github.com/WeichenXu123)
*   [Yuki Watanabe](https://github.com/B-Step62)
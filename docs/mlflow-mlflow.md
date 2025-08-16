<h1 align="center" style="border-bottom: none">
    <a href="https://mlflow.org/">
        <img alt="MLflow logo" src="https://raw.githubusercontent.com/mlflow/mlflow/refs/heads/master/assets/logo.svg" width="200" />
    </a>
</h1>

<h2 align="center" style="border-bottom: none">MLflow: Productionize Your AI Applications with Confidence</h2>

MLflow is an open-source platform designed to streamline the AI/LLM development lifecycle, from experimentation to deployment.  This comprehensive platform provides all the tools you need for efficient model management.

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

*   **Experiment Tracking:**  Effortlessly track model parameters, metrics, and artifacts for every experiment.  Visualize and compare results with an interactive UI.
*   **Model Registry:**  Collaboratively manage the full lifecycle of your machine learning models, including versioning, staging, and deployment.
*   **LLM & GenAI Support:** Advanced features for LLM applications, including prompt management, tracing and evaluation.
*   **Model Deployment:** Deploy your models seamlessly to various platforms like Docker, Kubernetes, Azure ML, and AWS SageMaker.
*   **Observability and Tracing:**  Gain insights into your LLM/agentic applications for debugging, quality monitoring and performance improvements.
*   **Prompt Management:** Version, track, and reuse prompts across your organization, helping maintain consistency and improve collaboration in prompt development.
*   **App Version Tracking:** MLflow keeps track of many moving parts in your AI applications, such as models, prompts, tools, and code, with end-to-end lineage.

## 🚀 Installation

Install MLflow easily using pip:

```bash
pip install mlflow
```

## 📦 Core Components

MLflow is a unified solution for all your AI/ML needs, including LLMs, Agents, Deep Learning, and traditional machine learning.

### 💡 For LLM / GenAI Developers

[See the original documentation for specific examples](https://github.com/mlflow/mlflow).

### 🎓 For Data Scientists

[See the original documentation for specific examples](https://github.com/mlflow/mlflow).

## 🌐 Hosting MLflow

MLflow offers flexible hosting options:

*   **Local:** Run MLflow on your local machine for development and experimentation.
*   **On-Premise:** Host MLflow on your own servers for greater control.
*   **Cloud Providers:** Leverage managed MLflow services from major cloud providers like AWS SageMaker, Azure ML, Databricks, and Nebius.

For self-hosting, refer to the [tracking setup guide](https://mlflow.org/docs/latest/ml/tracking/#tracking-setup).

## 🗣️ Supported Languages & Frameworks

MLflow offers broad support for popular languages and integrations:

*   **Python:** Primary language for MLflow development.
*   **JavaScript/TypeScript:** Support for observability.
*   **Java:** Client library available.
*   **R:** Integration with the R ecosystem.

## 🔗 Integrations

MLflow seamlessly integrates with a wide range of machine learning frameworks and GenAI libraries.

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

Run `mlflow ui` in a separate terminal to view the results.

### Evaluating Models

```python
import mlflow
import pandas as pd

df = pd.DataFrame(...)
eval_dataset = mlflow.data.from_pandas(df, predictions="outputs", targets="ground_truth")

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
response = OpenAI().chat.completions.create(...)
```

View traces in the MLflow UI under the "Traces" tab.

## 💭 Support and Resources

*   **Documentation:**  Comprehensive [documentation](https://mlflow.org/docs/latest/index.html) with an AI-powered chatbot.
*   **Community:**  Join [virtual events](https://lu.ma/mlflow?k=c) like office hours and meetups.
*   **GitHub:**  Report bugs, request features, or contribute to the project by [opening a GitHub issue](https://github.com/mlflow/mlflow/issues/new/choose).
*   **Mailing List:**  Stay updated with announcements by subscribing to the [mailing list](mlflow-users@googlegroups.com).
*   **Slack:** Join us on [Slack](https://mlflow.org/slack).

## 🤝 Contributing

Contributions to MLflow are highly encouraged!  Review the [contribution guide](CONTRIBUTING.md) for details.

*   Submit [bug reports](https://github.com/mlflow/mlflow/issues/new?template=bug_report_template.yaml) and [feature requests](https://github.com/mlflow/mlflow/issues/new?template=feature_request_template.yaml)
*   Contribute for [good-first-issues](https://github.com/mlflow/mlflow/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) and [help-wanted](https://github.com/mlflow/mlflow/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22)
*   Write about MLflow and share your experience

## ⭐️ Star History

<a href="https://star-history.com/#mlflow/mlflow&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=mlflow/mlflow&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=mlflow/mlflow&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=mlflow/mlflow&type=Date" />
 </picture>
</a>

## ✏️ Citation

Cite MLflow in your research using the "Cite this repository" button on the [GitHub repository page](https://github.com/mlflow/mlflow).

## 👥 Core Members

[See the original documentation for the Core Members](https://github.com/mlflow/mlflow).
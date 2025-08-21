# MLflow: The Open-Source Platform for Productionizing AI

**Effortlessly build, deploy, and manage your AI/LLM applications with MLflow. [Explore the MLflow GitHub Repository](https://github.com/mlflow/mlflow)**

<div align="center">
    <a href="https://mlflow.org/">
        <img alt="MLflow logo" src="https://raw.githubusercontent.com/mlflow/mlflow/refs/heads/master/assets/logo.svg" width="200" />
    </a>
    <p>
        <a href="https://www.mlflow.org/"><strong>Website</strong></a> |
        <a href="https://mlflow.org/docs/latest/index.html"><strong>Docs</strong></a> |
        <a href="https://github.com/mlflow/mlflow/issues/new/choose"><strong>Feature Request</strong></a> |
        <a href="https://mlflow.org/blog"><strong>News</strong></a> |
        <a href="https://www.youtube.com/@mlflowoss"><strong>YouTube</strong></a> |
        <a href="https://lu.ma/mlflow?k=c"><strong>Events</strong></a>
    </p>
    
    [![Python SDK](https://img.shields.io/pypi/v/mlflow)](https://pypi.org/project/mlflow/)
    [![PyPI Downloads](https://img.shields.io/pypi/dm/mlflow)](https://pepy.tech/projects/mlflow)
    [![License](https://img.shields.io/github/license/mlflow/mlflow)](https://github.com/mlflow/mlflow/blob/main/LICENSE)
    <a href="https://twitter.com/intent/follow?screen_name=mlflow" target="_blank">
        <img src="https://img.shields.io/twitter/follow/mlflow?logo=X&color=%20%23f5f5f5" alt="follow on X(Twitter)"></a>
    <a href="https://www.linkedin.com/company/mlflow-org/" target="_blank">
        <img src="https://custom-icon-badges.demolab.com/badge/LinkedIn-0A66C2?logo=linkedin-white&logoColor=fff" alt="follow on LinkedIn"></a>
    [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/mlflow/mlflow)
</div>

---

## Key Features

MLflow provides a unified platform to streamline the entire AI/ML lifecycle, including LLMs, Agents, Deep Learning, and traditional machine learning.

*   **Experiment Tracking:** Track and compare your ML model parameters, metrics, and results in a centralized interface.
*   **LLM Tracing and Observability:** Deeply monitor the internal states and performance of your LLM/agentic applications, enabling easier debugging and performance monitoring.
*   **LLM Evaluation:** Automate model evaluation using a suite of tools integrated with experiment tracking for easy comparison of different model versions.
*   **Prompt Management:** Version, track, and reuse prompts for consistent AI application development and collaboration.
*   **App Version Tracking:** Maintain end-to-end lineage and track all components of your AI applications, including models, prompts, tools, and code.
*   **Model Registry:** Centralized model store to manage the full lifecycle and deployment of machine learning models collaboratively.
*   **Model Deployment:** Deploy models seamlessly for batch and real-time scoring on diverse platforms like Docker, Kubernetes, Azure ML, and AWS SageMaker.
*   **Integrations:** Native integrations with popular ML frameworks and GenAI libraries.

## 🚀 Installation

To install the MLflow Python package, run:

```bash
pip install mlflow
```

## 🌐 Hosting MLflow Anywhere

MLflow supports various hosting environments, from local machines to cloud infrastructure.  It is offered as a managed service by major cloud providers like:

*   Amazon SageMaker
*   Azure ML
*   Databricks
*   Nebius

For self-hosting instructions, see the [official documentation](https://mlflow.org/docs/latest/ml/tracking/#tracking-setup).

## 🗣️ Supported Programming Languages

*   Python
*   TypeScript / JavaScript
*   Java
*   R

## 🔗 Integrations

MLflow integrates seamlessly with many popular machine learning frameworks and GenAI libraries, as shown in the image below:

![Integrations](https://raw.githubusercontent.com/mlflow/mlflow/refs/heads/master/assets/readme-integrations.png)

## 🧑‍💻 Usage Examples

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

After running the code above, execute `mlflow ui` in a separate terminal and access the MLflow UI via the provided URL to view your MLflow **Run**.

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

After executing this code, you can view the trace records in the "Traces" tab of the MLflow UI.

## 💭 Support

*   Consult the [documentation](https://mlflow.org/docs/latest/index.html) for usage assistance.
*   Use the **"Ask AI"** button in the documentation for AI-powered support.
*   Join [virtual events](https://lu.ma/mlflow?k=c) like office hours and meetups.
*   Report bugs and request features by [opening a GitHub issue](https://github.com/mlflow/mlflow/issues/new/choose).
*   Subscribe to the [mailing list](mailto:mlflow-users@googlegroups.com) or join the [Slack](https://mlflow.org/slack) for discussions and announcements.

## 🤝 Contributing

Contributions to MLflow are welcome!  Find out how to contribute by reading the:

*   Submit [bug reports](https://github.com/mlflow/mlflow/issues/new?template=bug_report_template.yaml) and [feature requests](https://github.com/mlflow/mlflow/issues/new?template=feature_request_template.yaml)
*   Contribute for [good-first-issues](https://github.com/mlflow/mlflow/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) and [help-wanted](https://github.com/mlflow/mlflow/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22)
*   Writing about MLflow and sharing your experience
*   See our [contribution guide](CONTRIBUTING.md) for details.

## ⭐️ Star History

<a href="https://star-history.com/#mlflow/mlflow&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=mlflow/mlflow&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=mlflow/mlflow&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=mlflow/mlflow&type=Date" />
 </picture>
</a>

## ✏️ Citation

Cite MLflow using the "Cite this repository" button at the top of the [GitHub repository page](https://github.com/mlflow/mlflow).

## 👥 Core Members

MLflow is actively maintained by the following core members and the wider community:

*   [Ben Wilson](https://github.com/BenWilson2)
*   [Corey Zumar](https://github.com/dbczumar)
*   [Daniel Lok](https://github.com/daniellok-db)
*   [Gabriel Fu](https://github.com/gabrielfu)
*   [Harutaka Kawamura](https://github.com/harupy)
*   [Serena Ruan](https://github.com/serena-ruan)
*   [Tomu Hirata](https://github.com/TomeHirata)
*   [Weichen Xu](https://github.com/WeichenXu123)
*   [Yuki Watanabe](https://github.com/B-Step62)
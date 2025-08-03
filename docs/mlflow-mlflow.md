# MLflow: The Open-Source Platform for Productionizing AI

[![MLflow Logo](https://raw.githubusercontent.com/mlflow/mlflow/refs/heads/master/assets/logo.svg)](https://mlflow.org/)

**MLflow simplifies the AI lifecycle, enabling data scientists and developers to build, deploy, and manage AI applications with confidence.** This comprehensive platform provides robust solutions for experiment tracking, model management, and deployment, all in one place.  Find the original repository [here](https://github.com/mlflow/mlflow).

<div align="center">

[![Python SDK](https://img.shields.io/pypi/v/mlflow)](https://pypi.org/project/mlflow/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/mlflow)](https://pypi.org/project/mlflow/)
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

*   **Experiment Tracking:** Log and compare model parameters, metrics, and artifacts to track and understand your experiments.
*   **Model Registry:** Centralized repository for managing the full lifecycle of your machine learning models.
*   **Model Deployment:** Deploy your models seamlessly to various platforms like Docker, Kubernetes, and cloud providers.
*   **LLM/GenAI Support:** Specialized tools for tracing, evaluation, prompt management, and version tracking in LLM applications.
*   **Observability:**  Trace the internal states of LLM/agentic applications for debugging, quality monitoring, and performance optimization.

## Installation

Install the MLflow Python package using pip:

```bash
pip install mlflow
```

## Core Components

MLflow provides a unified solution for all your AI/ML needs, including LLMs, Agents, Deep Learning, and traditional machine learning.

### For LLM / GenAI Developers

| Feature            | Description                                                                                                                              |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------- |
| **🔍 Tracing / Observability** | Trace the internal states of your LLM/agentic applications for debugging quality issues and monitoring performance with ease. |
| **📊 LLM Evaluation**     | Automated model evaluation tools, seamlessly integrated with experiment tracking.                                                  |
| **🤖 Prompt Management**    | Version, track, and reuse prompts across your organization, helping maintain consistency and improve collaboration.                 |
| **📦 App Version Tracking**  | Track models, prompts, tools, and code with end-to-end lineage.                                                                  |

### For Data Scientists

| Feature             | Description                                                                                                  |
| ------------------- | ------------------------------------------------------------------------------------------------------------ |
| **📝 Experiment Tracking** | Track models, parameters, metrics, and evaluation results in ML experiments using an interactive UI.     |
| **💾 Model Registry**   | Centrally manage the full lifecycle and deployment of machine learning models.                             |
| **🚀 Deployment**    | Seamless model deployment to batch and real-time scoring on platforms like Docker, Kubernetes, and more. |

## Hosting MLflow Anywhere

MLflow can be run in various environments, including local machines, on-premise servers, and cloud infrastructure.  It's a managed service by major cloud providers:

*   [Amazon SageMaker](https://aws.amazon.com/sagemaker-ai/experiments/)
*   [Azure ML](https://learn.microsoft.com/en-us/azure/machine-learning/concept-mlflow?view=azureml-api-2)
*   [Databricks](https://www.databricks.com/product/managed-mlflow)
*   [Nebius](https://nebius.com/services/managed-mlflow)

For hosting on your own infrastructure, refer to [this guidance](https://mlflow.org/docs/latest/ml/tracking/#tracking-setup).

## Supported Programming Languages

*   [Python](https://pypi.org/project/mlflow/)
*   [TypeScript / JavaScript](https://www.npmjs.com/package/mlflow-tracing)
*   [Java](https://mvnrepository.com/artifact/org.mlflow/mlflow-client)
*   [R](https://cran.r-project.org/web/packages/mlflow/readme/README.html)

## Integrations

MLflow integrates seamlessly with popular machine learning frameworks and GenAI libraries.

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

Run `mlflow ui` in a separate terminal and access the MLflow UI via the printed URL.

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

Navigate to the "Traces" tab in the MLflow UI.

## Support

*   **Documentation:** [https://mlflow.org/docs/latest/index.html](https://mlflow.org/docs/latest/index.html)
*   **AI Chatbot:** Ask questions within the documentation.
*   **Events:** [Virtual events](https://lu.ma/mlflow?k=c)
*   **GitHub Issues:** [Open a GitHub issue](https://github.com/mlflow/mlflow/issues/new/choose) to report a bug or request a feature.
*   **Mailing List:** Subscribe to our mailing list (mlflow-users@googlegroups.com)
*   **Slack:** [Slack](https://mlflow.org/slack)

## Contributing

Contributions to MLflow are welcome!  See our [contribution guide](CONTRIBUTING.md).

*   [Bug reports](https://github.com/mlflow/mlflow/issues/new?template=bug_report_template.yaml) and [feature requests](https://github.com/mlflow/mlflow/issues/new?template=feature_request_template.yaml)
*   [Good-first-issues](https://github.com/mlflow/mlflow/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) and [help-wanted](https://github.com/mlflow/mlflow/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22)
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

If you use MLflow in your research, please cite it using the "Cite this repository" button at the top of the [GitHub repository page](https://github.com/mlflow/mlflow).

## 👥 Core Members

Maintained by core members and community contributors:

-   [Ben Wilson](https://github.com/BenWilson2)
-   [Corey Zumar](https://github.com/dbczumar)
-   [Daniel Lok](https://github.com/daniellok-db)
-   [Gabriel Fu](https://github.com/gabrielfu)
-   [Harutaka Kawamura](https://github.com/harupy)
-   [Serena Ruan](https://github.com/serena-ruan)
-   [Tomu Hirata](https://github.com/TomeHirata)
-   [Weichen Xu](https://github.com/WeichenXu123)
-   [Yuki Watanabe](https://github.com/B-Step62)
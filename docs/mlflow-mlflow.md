# MLflow: Productionize Your AI with Confidence

[![MLflow Logo](https://raw.githubusercontent.com/mlflow/mlflow/refs/heads/master/assets/logo.svg)](https://mlflow.org/)

**MLflow is an open-source platform that simplifies the entire machine learning lifecycle, from experiment tracking to model deployment.**  This comprehensive platform empowers data scientists and AI engineers to build, train, and deploy AI/LLM applications efficiently and reliably.  Check out the original repository [here](https://github.com/mlflow/mlflow).

<div align="center">
    <a href="https://www.mlflow.org/">
        <img src="https://img.shields.io/pypi/v/mlflow.svg?color=blue&label=PyPI&logo=pypi" alt="PyPI version"></a>
    <a href="https://pypi.org/project/mlflow/">
        <img src="https://img.shields.io/pypi/dm/mlflow.svg?color=blue&label=Downloads&logo=pypi" alt="PyPI Downloads"></a>
    <a href="https://github.com/mlflow/mlflow/blob/main/LICENSE">
        <img src="https://img.shields.io/github/license/mlflow/mlflow.svg?color=blue" alt="License"></a>
    <a href="https://twitter.com/intent/follow?screen_name=mlflow" target="_blank">
        <img src="https://img.shields.io/twitter/follow/mlflow?logo=X&color=%20%23f5f5f5" alt="Follow on X(Twitter)"></a>
    <a href="https://www.linkedin.com/company/mlflow-org/" target="_blank">
        <img src="https://custom-icon-badges.demolab.com/badge/LinkedIn-0A66C2?logo=linkedin-white&logoColor=fff" alt="Follow on LinkedIn"></a>
    <a href="https://deepwiki.com/mlflow/mlflow">
        <img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki"></a>
</div>

<div align="center">
    <div>
        <a href="https://mlflow.org/"><strong>Website</strong></a> |
        <a href="https://mlflow.org/docs/latest/index.html"><strong>Docs</strong></a> |
        <a href="https://github.com/mlflow/mlflow/issues/new/choose"><strong>Feature Request</strong></a> |
        <a href="https://mlflow.org/blog"><strong>News</strong></a> |
        <a href="https://www.youtube.com/@mlflowoss"><strong>YouTube</strong></a> |
        <a href="https://lu.ma/mlflow?k=c"><strong>Events</strong></a>
    </div>
</div>

<br>

## Key Features

*   **Experiment Tracking:** Log parameters, metrics, code versions, and artifacts for easy comparison and reproducibility.
*   **Model Registry:**  Collaboratively manage the complete lifecycle of your machine learning models, from development to production.
*   **Model Deployment:** Deploy models to various platforms, including Docker, Kubernetes, Azure ML, and AWS SageMaker.
*   **LLM/GenAI Support:**
    *   **Tracing / Observability:** Trace the internal states of your LLM/agentic applications for debugging quality issues and monitoring performance.
    *   **LLM Evaluation:** Automated model evaluation tools integrated with experiment tracking.
    *   **Prompt Management:** Version, track, and reuse prompts across your organization.
    *   **App Version Tracking:** Track models, prompts, tools, and code in your AI applications with end-to-end lineage.
*   **Integrations:** Seamlessly integrates with popular machine learning frameworks and libraries.
*   **Open Source:**  A community-driven platform, fostering collaboration and innovation.

## Installation

Install the MLflow Python package using pip:

```bash
pip install mlflow
```

## Core Components: Your All-in-One AI/ML Solution

MLflow offers a unified solution for all your AI/ML needs, including LLMs, Agents, Deep Learning, and traditional machine learning.

### For LLM / GenAI Developers

| Feature                                                                                   | Description                                                                                                                                                                                                                                                                                                                                 |
| :---------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **🔍 Tracing / Observability** ([Docs](https://mlflow.org/docs/latest/llms/tracing/index.html))    | Trace the internal states of your LLM/agentic applications for debugging quality issues and monitoring performance with ease.  [Getting Started](https://mlflow.org/docs/latest/genai/tracing/quickstart/python-openai/)                                                                                                                                                                |
| **📊 LLM Evaluation** ([Docs](https://mlflow.org/docs/latest/genai/eval-monitor/))                  | A suite of automated model evaluation tools, seamlessly integrated with experiment tracking to compare across multiple versions.  [Getting Started](https://mlflow.org/docs/latest/genai/eval-monitor/)                                                                                                                                                             |
| **🤖 Prompt Management** ([Docs](https://mlflow.org/docs/latest/genai/prompt-version-mgmt/prompt-registry/)) | Version, track, and reuse prompts across your organization, helping maintain consistency and improve collaboration in prompt development.  [Getting Started](https://mlflow.org/docs/latest/genai/prompt-registry/create-and-edit-prompts/)                                                                                                                                 |
| **📦 App Version Tracking** ([Docs](https://mlflow.org/docs/latest/genai/prompt-version-mgmt/version-tracking/))          | MLflow keeps track of many moving parts in your AI applications, such as models, prompts, tools, and code, with end-to-end lineage.  [Getting Started](https://mlflow.org/docs/latest/genai/version-tracking/quickstart/)                                                                                                                                           |

### For Data Scientists

| Feature                                                                             | Description                                                                                                                                                                                                                                                                                                                                 |
| :---------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **📝 Experiment Tracking** ([Docs](https://mlflow.org/docs/latest/ml/tracking/))           | Track your models, parameters, metrics, and evaluation results in ML experiments and compare them using an interactive UI.  [Getting Started](https://mlflow.org/docs/latest/ml/tracking/quickstart/)                                                                                                                                                                |
| **💾 Model Registry** ([Docs](https://mlflow.org/docs/latest/ml/model-registry/))              | A centralized model store designed to collaboratively manage the full lifecycle and deployment of machine learning models.  [Getting Started](https://mlflow.org/docs/latest/ml/model-registry/tutorial/)                                                                                                                                                             |
| **🚀 Deployment** ([Docs](https://mlflow.org/docs/latest/ml/deployment/))                    | Tools for seamless model deployment to batch and real-time scoring on platforms like Docker, Kubernetes, Azure ML, and AWS SageMaker.  [Getting Started](https://mlflow.org/docs/latest/ml/deployment/)                                                                                                                                                                  |

## 🌐 Hosting MLflow

MLflow can be run in various environments:

*   **Locally** on your machine.
*   **On-premise** servers.
*   **Cloud Infrastructure:**

    *   [Amazon SageMaker](https://aws.amazon.com/sagemaker-ai/experiments/)
    *   [Azure ML](https://learn.microsoft.com/en-us/azure/machine-learning/concept-mlflow?view=azureml-api-2)
    *   [Databricks](https://www.databricks.com/product/managed-mlflow)
    *   [Nebius](https://nebius.com/services/managed-mlflow)

For guidance on self-hosting, see [this documentation](https://mlflow.org/docs/latest/ml/tracking/#tracking-setup).

## 🗣️ Supported Languages

*   [Python](https://pypi.org/project/mlflow/)
*   [TypeScript / JavaScript](https://www.npmjs.com/package/mlflow-tracing)
*   [Java](https://mvnrepository.com/artifact/org.mlflow/mlflow-client)
*   [R](https://cran.r-project.org/web/packages/mlflow/readme/README.html)

## 🔗 Integrations

MLflow seamlessly integrates with popular machine learning frameworks and GenAI libraries, expanding its usability and reducing manual effort.

![Integrations](https://raw.githubusercontent.com/mlflow/mlflow/refs/heads/master/assets/readme-integrations.png)

## Usage Examples

### Experiment Tracking ([Doc](https://mlflow.org/docs/latest/ml/tracking/))

This example demonstrates training a regression model with scikit-learn, enabling MLflow's autologging for experiment tracking:

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

Run this command in a separate terminal to launch the MLflow UI, which displays logged training data, hyper-parameters, and model information:

```bash
mlflow ui
```

### Evaluating Models ([Doc](https://mlflow.org/docs/latest/model-evaluation/index.html))

This example performs automatic evaluation of a question-answering task:

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

MLflow Tracing provides LLM observability for various GenAI libraries such as OpenAI, LangChain, LlamaIndex, DSPy, AutoGen, and more. To enable auto-tracing, call `mlflow.xyz.autolog()` before running your models. Refer to the documentation for customization and manual instrumentation.

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

## 💭 Support

*   **Documentation:** Explore detailed documentation at [https://mlflow.org/docs/latest/index.html](https://mlflow.org/docs/latest/index.html)
*   **AI Chatbot:** Get instant answers using the "Ask AI" button within the documentation.
*   **Community:** Join virtual events like office hours and meetups at [https://lu.ma/mlflow?k=c](https://lu.ma/mlflow?k=c).
*   **Issue Tracking:** Report bugs, request features, or suggest documentation improvements by opening a [GitHub issue](https://github.com/mlflow/mlflow/issues/new/choose).
*   **Community Forum:**  Join the mailing list (mlflow-users@googlegroups.com) or Slack channel ([https://mlflow.org/slack](https://mlflow.org/slack)) for announcements and discussions.

## 🤝 Contributing

Contributions to MLflow are welcome!

*   Submit [bug reports](https://github.com/mlflow/mlflow/issues/new?template=bug_report_template.yaml) and [feature requests](https://github.com/mlflow/mlflow/issues/new?template=feature_request_template.yaml).
*   Contribute to [good-first-issues](https://github.com/mlflow/mlflow/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) and [help-wanted](https://github.com/mlflow/mlflow/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22) tasks.
*   Write about MLflow and share your experiences.

See the [contribution guide](CONTRIBUTING.md) for details.

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

MLflow is maintained by the following core members and supported by many community contributors:

*   [Ben Wilson](https://github.com/BenWilson2)
*   [Corey Zumar](https://github.com/dbczumar)
*   [Daniel Lok](https://github.com/daniellok-db)
*   [Gabriel Fu](https://github.com/gabrielfu)
*   [Harutaka Kawamura](https://github.com/harupy)
*   [Serena Ruan](https://github.com/serena-ruan)
*   [Tomu Hirata](https://github.com/TomeHirata)
*   [Weichen Xu](https://github.com/WeichenXu123)
*   [Yuki Watanabe](https://github.com/B-Step62)
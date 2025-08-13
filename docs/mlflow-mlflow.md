<h1 align="center" style="border-bottom: none">
    <a href="https://mlflow.org/">
        <img alt="MLflow logo" src="https://raw.githubusercontent.com/mlflow/mlflow/refs/heads/master/assets/logo.svg" width="200" />
    </a>
</h1>

<h2 align="center" style="border-bottom: none">MLflow: The Open-Source Platform for the AI Lifecycle</h2>

**MLflow is the open-source platform empowering developers to build and deploy AI/LLM applications with confidence.**  This platform streamlines the entire AI lifecycle from experimentation to production.

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

*   **Experiment Tracking:** Log and manage model parameters, metrics, and artifacts to compare results and optimize your models.
*   **LLM Tracing & Observability:** Gain insights into your LLM/Agentic applications for debugging and performance monitoring.
*   **LLM Evaluation:** Evaluate models with automated tools and compare results across different versions seamlessly.
*   **Prompt Management:** Version, track, and reuse prompts to maintain consistency and streamline collaboration.
*   **Model Registry:** Centralized model storage for managing the full lifecycle of machine learning models.
*   **Deployment:** Deploy models to various platforms like Docker, Kubernetes, and cloud services (Azure ML, AWS SageMaker).

## Getting Started: Installation

Install the MLflow Python package:

```bash
pip install mlflow
```

## Core Components & Use Cases

MLflow provides a unified solution for your AI/ML needs, including LLMs, Agents, Deep Learning, and traditional machine learning.

### For LLM / GenAI Developers

<table>
  <tr>
    <td>
    <img src="https://raw.githubusercontent.com/mlflow/mlflow/refs/heads/master/assets/readme-tracing.png" alt="Tracing" width=100%>
    <div align="center">
        <br>
        <a href="https://mlflow.org/docs/latest/llms/tracing/index.html"><strong>🔍 Tracing / Observability</strong></a>
        <br><br>
        <div>Trace the internal states of your LLM/agentic applications for debugging quality issues and monitoring performance with ease.</div><br>
        <a href="https://mlflow.org/docs/latest/genai/tracing/quickstart/python-openai/">Getting Started →</a>
        <br><br>
    </div>
    </td>
    <td>
    <img src="https://raw.githubusercontent.com/mlflow/mlflow/refs/heads/master/assets/readme-llm-eval.png" alt="LLM Evaluation" width=100%>
    <div align="center">
        <br>
        <a href="https://mlflow.org/docs/latest/genai/eval-monitor/"><strong>📊 LLM Evaluation</strong></a>
        <br><br>
        <div>A suite of automated model evaluation tools, seamlessly integrated with experiment tracking to compare across multiple versions.</div><br>
        <a href="https://mlflow.org/docs/latest/genai/eval-monitor/">Getting Started →</a>
        <br><br>
    </div>
    </td>
  </tr>
  <tr>
    <td>
      <img src="https://raw.githubusercontent.com/mlflow/mlflow/refs/heads/master/assets/readme-prompt.png" alt="Prompt Management">
    <div align="center">
        <br>
        <a href="https://mlflow.org/docs/latest/genai/prompt-version-mgmt/prompt-registry/"><strong>🤖 Prompt Management</strong></a>
        <br><br>
        <div>Version, track, and reuse prompts across your organization, helping maintain consistency and improve collaboration in prompt development.</div><br>
        <a href="https://mlflow.org/docs/latest/genai/prompt-registry/create-and-edit-prompts/">Getting Started →</a>
        <br><br>
    </div>
    </td>
    <td>
      <img src="https://raw.githubusercontent.com/mlflow/mlflow/refs/heads/master/assets/readme-logged-model.png" alt="MLflow Hero">
    <div align="center">
        <br>
        <a href="https://mlflow.org/docs/latest/genai/prompt-version-mgmt/version-tracking/"><strong>📦 App Version Tracking</strong></a>
        <br><br>
        <div>MLflow keeps track of many moving parts in your AI applications, such as models, prompts, tools, and code, with end-to-end lineage.</div><br>
        <a href="https://mlflow.org/docs/latest/genai/version-tracking/quickstart/">Getting Started →</a>
        <br><br>
    </div>
    </td>
  </tr>
</table>

### For Data Scientists

<table>
  <tr>
    <td colspan="2" align="center" >
      <img src="https://raw.githubusercontent.com/mlflow/mlflow/refs/heads/master/assets/readme-experiment.png" alt="Tracking" width=50%>
    <div align="center">
        <br>
        <a href="https://mlflow.org/docs/latest/ml/tracking/"><strong>📝 Experiment Tracking</strong></a>
        <br><br>
        <div>Track your models, parameters, metrics, and evaluation results in ML experiments and compare them using an interactive UI.</div><br>
        <a href="https://mlflow.org/docs/latest/ml/tracking/quickstart/">Getting Started →</a>
        <br><br>
    </div>
    </td>
  </tr>
  <tr>
    <td>
      <img src="https://raw.githubusercontent.com/mlflow/mlflow/refs/heads/master/assets/readme-model-registry.png" alt="Model Registry" width=100%>
    <div align="center">
        <br>
        <a href="https://mlflow.org/docs/latest/ml/model-registry/"><strong>💾 Model Registry</strong></a>
        <br><br>
        <div> A centralized model store designed to collaboratively manage the full lifecycle and deployment of machine learning models.</div><br>
        <a href="https://mlflow.org/docs/latest/ml/model-registry/tutorial/">Getting Started →</a>
        <br><br>
    </div>
    </td>
    <td>
      <img src="https://raw.githubusercontent.com/mlflow/mlflow/refs/heads/master/assets/readme-deployment.png" alt="Deployment" width=100%>
    <div align="center">
        <br>
        <a href="https://mlflow.org/docs/latest/ml/deployment/"><strong>🚀 Deployment</strong></a>
        <br><br>
        <div> Tools for seamless model deployment to batch and real-time scoring on platforms like Docker, Kubernetes, Azure ML, and AWS SageMaker.</div><br>
        <a href="https://mlflow.org/docs/latest/ml/deployment/">Getting Started →</a>
        <br><br>
    </div>
    </td>
  </tr>
</table>

## Hosting MLflow

MLflow can be hosted in various environments, including local machines, on-premise servers, and cloud infrastructure. Managed services are offered by major cloud providers such as:

*   [Amazon SageMaker](https://aws.amazon.com/sagemaker-ai/experiments/)
*   [Azure ML](https://learn.microsoft.com/en-us/azure/machine-learning/concept-mlflow?view=azureml-api-2)
*   [Databricks](https://www.databricks.com/product/managed-mlflow)
*   [Nebius](https://nebius.com/services/managed-mlflow)

For self-hosting guidance, please refer to [the documentation](https://mlflow.org/docs/latest/ml/tracking/#tracking-setup).

## Supported Programming Languages

MLflow supports the following languages:

*   [Python](https://pypi.org/project/mlflow/)
*   [TypeScript / JavaScript](https://www.npmjs.com/package/mlflow-tracing)
*   [Java](https://mvnrepository.com/artifact/org.mlflow/mlflow-client)
*   [R](https://cran.r-project.org/web/packages/mlflow/readme/README.html)

## Integrations

MLflow integrates natively with many popular machine learning frameworks and GenAI libraries.

![Integrations](https://raw.githubusercontent.com/mlflow/mlflow/refs/heads/master/assets/readme-integrations.png)

## Usage Examples

### Experiment Tracking

*   **[Documentation](https://mlflow.org/docs/latest/ml/tracking/)**

The following examples trains a simple regression model with scikit-learn, while enabling MLflow's [autologging](https://mlflow.org/docs/latest/tracking/autolog.html) feature for experiment tracking.

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

To view the results, run the following command and access the MLflow UI via the URL:

```bash
mlflow ui
```

### Evaluating Models

*   **[Documentation](https://mlflow.org/docs/latest/model-evaluation/index.html)**

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

*   **[Documentation](https://mlflow.org/docs/latest/llms/tracing/index.html)**

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

Navigate to the "Traces" tab in the MLflow UI to find the trace records OpenAI query.

## Support & Resources

*   **Documentation:** [https://mlflow.org/docs/latest/index.html](https://mlflow.org/docs/latest/index.html)
*   **AI-Powered Chatbot:** Access the chatbot in the documentation by clicking the **"Ask AI"** button.
*   **Virtual Events:** Join [virtual events](https://lu.ma/mlflow?k=c) like office hours and meetups.
*   **Report Issues:** [Open a GitHub issue](https://github.com/mlflow/mlflow/issues/new/choose).
*   **Mailing List:** Subscribe to our mailing list (mlflow-users@googlegroups.com).
*   **Slack:** Join us on [Slack](https://mlflow.org/slack).

## Contributing

Contributions to MLflow are welcome!

*   [Bug reports](https://github.com/mlflow/mlflow/issues/new?template=bug_report_template.yaml)
*   [Feature requests](https://github.com/mlflow/mlflow/issues/new?template=feature_request_template.yaml)
*   [Good-first-issues](https://github.com/mlflow/mlflow/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) and [help-wanted](https://github.com/mlflow/mlflow/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22)
*   Share your experience by writing about MLflow.

See our [contribution guide](CONTRIBUTING.md) for more details.

## Star History

<a href="https://star-history.com/#mlflow/mlflow&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=mlflow/mlflow&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=mlflow/mlflow&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=mlflow/mlflow&type=Date" />
 </picture>
</a>

## Citation

If you use MLflow in your research, please cite it using the "Cite this repository" button at the top of the [GitHub repository page](https://github.com/mlflow/mlflow).

## Core Members

MLflow is maintained by core members with significant contributions from the community.

*   [Ben Wilson](https://github.com/BenWilson2)
*   [Corey Zumar](https://github.com/dbczumar)
*   [Daniel Lok](https://github.com/daniellok-db)
*   [Gabriel Fu](https://github.com/gabrielfu)
*   [Harutaka Kawamura](https://github.com/harupy)
*   [Serena Ruan](https://github.com/serena-ruan)
*   [Tomu Hirata](https://github.com/TomeHirata)
*   [Weichen Xu](https://github.com/WeichenXu123)
*   [Yuki Watanabe](https://github.com/B-Step62)

[Back to the top](#mlflow-the-open-source-platform-for-the-ai-lifecycle)
```
Key improvements and SEO considerations:

*   **Concise Hook:**  Replaced the longer introductory sentences with a clear, one-sentence hook.  Keywords like "AI lifecycle" are included.
*   **Keyword Optimization:**  Strategically used keywords like "AI/LLM applications," "experiment tracking," "model registry," and "deployment" throughout the headings and key features.
*   **Clear Headings:** Used HTML heading tags (<h1>, <h2>, etc.) to organize the content and improve readability for both humans and search engines.
*   **Bulleted Key Features:**  Highlights the core functionalities in a concise, easy-to-scan format.  Each bullet point acts as a concise description, incorporating relevant keywords.
*   **Documentation Links:**  Added [Back to the top](#mlflow-the-open-source-platform-for-the-ai-lifecycle) for enhanced navigation
*   **Installation Section:** This section is separate and early in the document, so it is easier for a new user to quickly find and use.
*   **Call to action and links to the core repo**.
*   **Removed Redundancy:**  Trimmed unnecessary wording to make it more direct and impactful.
*   **Improved Readability:** Better formatting for easier scanning and comprehension.
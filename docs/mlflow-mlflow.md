<h1 align="center" style="border-bottom: none">
    <a href="https://mlflow.org/">
        <img alt="MLflow logo" src="https://raw.githubusercontent.com/mlflow/mlflow/refs/heads/master/assets/logo.svg" width="200" />
    </a>
</h1>

<h2 align="center" style="border-bottom: none">MLflow: The Open Source Platform for the Complete AI Lifecycle</h2>

**MLflow empowers developers to build and deploy AI/LLM applications with confidence, providing end-to-end solutions for every stage of the AI lifecycle.**  Visit the [MLflow GitHub Repository](https://github.com/mlflow/mlflow) for the latest updates.

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

*   ✅ **Experiment Tracking:** Track and compare model parameters, metrics, and results with an interactive UI.
*   ✅ **Model Registry:** Centralized model store for managing the full lifecycle and deployment of machine learning models.
*   ✅ **Deployment:** Seamlessly deploy models to various platforms, including Docker, Kubernetes, and cloud providers.
*   ✅ **LLM/GenAI Tracing & Observability:** Monitor and debug your LLM/agentic applications with ease.
*   ✅ **LLM Evaluation:** Automate model evaluation with a suite of tools integrated with experiment tracking.
*   ✅ **Prompt Management:** Version, track, and reuse prompts for consistency and collaboration.
*   ✅ **App Version Tracking:** Track all the moving parts of your AI applications, including models, prompts, tools, and code.

## Installation

Install the MLflow Python package:

```bash
pip install mlflow
```

## Core Components

MLflow provides a unified platform for all your AI/ML needs, including LLMs, Agents, Deep Learning, and traditional machine learning.

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

MLflow can be hosted in various environments, including:

*   Local Machines
*   On-Premise Servers
*   Cloud Infrastructure

MLflow is offered as a managed service by major cloud providers like:

*   [Amazon SageMaker](https://aws.amazon.com/sagemaker-ai/experiments/)
*   [Azure ML](https://learn.microsoft.com/en-us/azure/machine-learning/concept-mlflow?view=azureml-api-2)
*   [Databricks](https://www.databricks.com/product/managed-mlflow)
*   [Nebius](https://nebius.com/services/managed-mlflow)

For hosting MLflow on your own infrastructure, please refer to [this guidance](https://mlflow.org/docs/latest/ml/tracking/#tracking-setup).

## Supported Languages

*   Python
*   TypeScript / JavaScript
*   Java
*   R

## Integrations

MLflow natively integrates with popular machine learning frameworks and GenAI libraries:

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

Run the following command in a separate terminal:
```bash
mlflow ui
```

### Evaluating Models

```python
import mlflow
import pandas as pd

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

response = OpenAI().chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hi!"}],
    temperature=0.1,
)
```

Navigate to the "Traces" tab in the MLflow UI.

## Support

*   Documentation: [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
*   Ask AI Chatbot: Available within the documentation.
*   Virtual Events: [MLflow Events](https://lu.ma/mlflow?k=c)
*   Report Issues: [GitHub Issues](https://github.com/mlflow/mlflow/issues/new/choose)
*   Mailing List: mlflow-users@googlegroups.com
*   Slack: [MLflow Slack](https://mlflow.org/slack)

## Contributing

Contribute to MLflow by:

*   Submitting bug reports and feature requests.
*   Working on good-first-issues and help-wanted issues.
*   Writing about MLflow and sharing your experiences.

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

*   Ben Wilson
*   Corey Zumar
*   Daniel Lok
*   Gabriel Fu
*   Harutaka Kawamura
*   Serena Ruan
*   Tomu Hirata
*   Weichen Xu
*   Yuki Watanabe
```
Key improvements and SEO optimizations:

*   **Clear Title and Hook:**  Uses "MLflow: The Open Source Platform for the Complete AI Lifecycle" as the primary title and a concise, benefit-driven hook to immediately grab attention.
*   **SEO-Friendly Headings:** Organizes content with clear, descriptive headings and subheadings (e.g., "Key Features," "Installation," "Supported Languages").
*   **Bulleted Key Features:** Uses bullet points to highlight key benefits and features, making the content easy to scan and digest.  Includes checkmarks for visual appeal.
*   **Keyword Optimization:**  Incorporates relevant keywords such as "AI," "LLM," "Machine Learning," "Experiment Tracking," "Model Registry," "Deployment," and "GenAI."
*   **Concise Summaries:** Replaces longer paragraphs with more succinct descriptions.
*   **Clear Call to Action:** Encourages users to visit the documentation, contribute, and report issues.
*   **Link to Original Repo:** Includes a prominent link back to the original GitHub repository.
*   **Simplified Code Blocks:** Keeps code examples short and focused on the core functionality.
*   **Enhanced Visual Appeal:** Uses consistent formatting, image alt text, and visual cues like checkmarks.
*   **Structure and Readability:** Improves overall structure and readability for a better user experience.
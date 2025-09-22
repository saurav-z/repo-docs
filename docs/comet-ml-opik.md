<!--
  SPDX-FileCopyrightText: 2024 Comet ML, Inc.
  SPDX-License-Identifier: Apache-2.0
-->

<div align="center"><b><a href="README.md">English</a> | <a href="readme_CN.md">简体中文</a> | <a href="readme_JP.md">日本語</a> | <a href="readme_KO.md">한국어</a></b></div>

<div align="center">
  <a href="https://github.com/comet-ml/opik">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/comet-ml/opik/refs/heads/main/apps/opik-documentation/documentation/static/img/logo-dark-mode.svg" alt="Opik Logo - Dark Mode">
      <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/comet-ml/opik/refs/heads/main/apps/opik-documentation/documentation/static/img/opik-logo.svg" alt="Opik Logo - Light Mode">
      <img src="https://raw.githubusercontent.com/comet-ml/opik/refs/heads/main/apps/opik-documentation/documentation/static/img/opik-logo.svg" alt="Opik Logo" width="200">
    </picture>
  </a>
  <h1>Opik: Your Open-Source LLM Application Platform</h1>
  <p>
    <i>Supercharge your LLM applications with Opik, the open-source platform for building, evaluating, and optimizing Large Language Model systems.</i>
  </p>
</div>

<div align="center">

[![PyPI Version](https://img.shields.io/pypi/v/opik)](https://pypi.org/project/opik/)
[![GitHub License](https://img.shields.io/github/license/comet-ml/opik)](https://github.com/comet-ml/opik/blob/main/LICENSE)
[![Build Status](https://github.com/comet-ml/opik/actions/workflows/build_apps.yml/badge.svg)](https://github.com/comet-ml/opik/actions/workflows/build_apps.yml)
[![Bounties](https://img.shields.io/endpoint?url=https%3A%2F%2Falgora.io%2Fapi%2Fshields%2Fcomet-ml%2Fbounties%3Fstatus%3Dopen)](https://algora.io/comet-ml/bounties?status=open)

</div>

<div align="center">
  <p>
    <a href="https://www.comet.com/site/products/opik/?from=llm&utm_source=opik&utm_medium=github&utm_content=website_button&utm_campaign=opik"><b>Website</b></a> •
    <a href="https://chat.comet.com"><b>Slack Community</b></a> •
    <a href="https://x.com/Cometml"><b>Twitter</b></a> •
    <a href="https://www.comet.com/docs/opik/changelog"><b>Changelog</b></a> •
    <a href="https://www.comet.com/docs/opik/?from=llm&utm_source=opik&utm_medium=github&utm_content=docs_button&utm_campaign=opik"><b>Documentation</b></a>
  </p>
</div>

<div align="center" style="margin-top: 1em; margin-bottom: 1em;">
  <a href="#-key-features">🚀 Key Features</a> • <a href="#-installation">🛠️ Installation</a> • <a href="#-client-sdk">💻 Client SDK</a> • <a href="#-integrations">📝 Integrations</a><br>
  <a href="#-llm-as-a-judge-metrics">🧑‍⚖️ LLM-as-a-Judge Metrics</a> • <a href="#-evaluating-your-application">🔍 Evaluating Your Application</a> • <a href="#-star-us">⭐ Star Us</a> • <a href="#-contributing">🤝 Contributing</a>
</div>

<br>

<a href="https://www.comet.com/signup?from=llm&utm_source=opik&utm_medium=github&utm_content=readme_banner&utm_campaign=opik">
  <img src="readme-thumbnail-new.png" alt="Opik Platform Screenshot" width="100%">
</a>

## 🚀 Key Features

Opik provides a comprehensive set of tools to build, evaluate, and monitor your LLM applications, offering:

*   **Comprehensive Observability:** Deep tracing of LLM calls, conversation logging, and agent activity.
*   **Advanced Evaluation:** Robust prompt evaluation, LLM-as-a-judge, and experiment management.
*   **Production-Ready:** Scalable monitoring dashboards and online evaluation rules.
*   **Opik Agent Optimizer:** Dedicated SDK and set of optimizers to enhance prompts and agents.
*   **Opik Guardrails:** Features to help you implement safe and responsible AI practices.

Here's a breakdown of core capabilities:

*   **Development & Tracing:**
    *   Track all LLM calls and traces with detailed context ([Quickstart](https://www.comet.com/docs/opik/quickstart/?from=llm&utm_source=opik&utm_medium=github&utm_content=quickstart_link&utm_campaign=opik)).
    *   Extensive 3rd-party integrations.  Seamlessly integrate with frameworks like **Google ADK**, **Autogen**, and **Flowise AI**. ([Integrations](https://www.comet.com/docs/opik/tracing/integrations/overview/?from=llm&utm_source=opik&utm_medium=github&utm_content=integrations_link&utm_campaign=opik))
    *   Annotate traces and spans with feedback scores via the [Python SDK](https://www.comet.com/docs/opik/tracing/annotate_traces/#annotating-traces-and-spans-using-the-sdk?from=llm&utm_source=opik&utm_medium=github&utm_content=sdk_link&utm_campaign=opik) or the [UI](https://www.comet.com/docs/opik/tracing/annotate_traces/#annotating-traces-through-the-ui?from=llm&utm_source=opik&utm_medium=github&utm_content=ui_link&utm_campaign=opik).
    *   Experiment with prompts and models in the [Prompt Playground](https://www.comet.com/docs/opik/prompt_engineering/playground).

*   **Evaluation & Testing:**
    *   Automate your LLM application evaluation with [Datasets](https://www.comet.com/docs/opik/evaluation/manage_datasets/?from=llm&utm_source=opik&utm_medium=github&utm_content=datasets_link&utm_campaign=opik) and [Experiments](https://www.comet.com/docs/opik/evaluation/evaluate_your_llm/?from=llm&utm_source=opik&utm_medium=github&utm_content=eval_link&utm_campaign=opik).
    *   Leverage powerful LLM-as-a-judge metrics, including [hallucination detection](https://www.comet.com/docs/opik/evaluation/metrics/hallucination/?from=llm&utm_source=opik&utm_medium=github&utm_content=hallucination_link&utm_campaign=opik), [moderation](https://www.comet.com/docs/opik/evaluation/metrics/moderation/?from=llm&utm_source=opik&utm_medium=github&utm_content=moderation_link&utm_campaign=opik), and RAG assessment ([Answer Relevance](https://www.comet.com/docs/opik/evaluation/metrics/answer_relevance/?from=llm&utm_source=opik&utm_medium=github&utm_content=alex_link&utm_campaign=opik), [Context Precision](https://www.comet.com/docs/opik/evaluation/metrics/context_precision/?from=llm&utm_source=opik&utm_medium=github&utm_content=context_link&utm_campaign=opik)).
    *   Integrate evaluations into your CI/CD pipeline with our [PyTest integration](https://www.comet.com/docs/opik/testing/pytest_integration/?from=llm&utm_source=opik&utm_medium=github&utm_content=pytest_link&utm_campaign=opik).

*   **Production Monitoring & Optimization:**
    *   Log high volumes of production traces: Opik is designed for scale (40M+ traces/day).
    *   Monitor feedback scores, trace counts, and token usage in the [Opik Dashboard](https://www.comet.com/docs/opik/production/production_monitoring/?from=llm&utm_source=opik&utm_medium=github&utm_content=dashboard_link&utm_campaign=opik).
    *   Utilize [Online Evaluation Rules](https://www.comet.com/docs/opik/production/rules/?from=llm&utm_source=opik&utm_medium=github&utm_content=dashboard_link&utm_campaign=opik) with LLM-as-a-Judge metrics.
    *   Leverage **Opik Agent Optimizer** and **Opik Guardrails** to continuously improve and secure your LLM applications.

> [!TIP]
> If you're looking for features not currently available in Opik, please submit a [Feature request](https://github.com/comet-ml/opik/issues/new/choose) 🚀

<br>

## 🛠️ Installation

Get started with Opik quickly:

### Option 1: Comet.com Cloud (Easiest)

Get instant access to Opik without any setup. Ideal for quick testing and hassle-free management.

👉 [Create your free Comet account](https://www.comet.com/signup?from=llm&utm_source=opik&utm_medium=github&utm_content=install_create_link&utm_campaign=opik)

### Option 2: Self-Host (Full Control)

Deploy Opik in your own environment using Docker or Kubernetes.

#### Self-Hosting with Docker Compose (Local Development)

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/comet-ml/opik.git
    cd opik
    ```

2.  **Run the Installation Script:**

    *   **Linux/Mac:** `./opik.sh`
    *   **Windows:** `powershell -ExecutionPolicy ByPass -c ".\\opik.ps1"`

    Use service profiles:
    ```bash
    ./opik.sh # Full Opik suite
    ./opik.sh --infra # Infrastructure services only
    ./opik.sh --backend # Infrastructure + backend services
    ./opik.sh --guardrails # Enable guardrails with any profile
    ```
    See the [Local Deployment Guide](https://www.comet.com/docs/opik/self-host/local_deployment?from=llm&utm_source=opik&utm_medium=github&utm_content=self_host_link&utm_campaign=opik).

3.  **Access Opik:** Open [http://localhost:5173](http://localhost:5173) in your browser.

#### Self-Hosting with Kubernetes & Helm (Scalable Deployments)

For production deployments, use our Helm chart.  See the [Kubernetes Installation Guide](https://www.comet.com/docs/opik/self-host/kubernetes/#kubernetes-installation?from=llm&utm_source=opik&utm_medium=github&utm_content=kubernetes_link&utm_campaign=opik).

[![Kubernetes](https://img.shields.io/badge/Kubernetes-%23326ce5.svg?&logo=kubernetes&logoColor=white)](https://www.comet.com/docs/opik/self-host/kubernetes/#kubernetes-installation?from=llm&utm_source=opik&utm_medium=github&utm_content=kubernetes_link&utm_campaign=opik)

> [!IMPORTANT]
> **Version 1.7.0 Changes**: Check the [changelog](https://github.com/comet-ml/opik/blob/main/CHANGELOG.md) for important updates.

## 💻 Client SDK

Opik offers client libraries and a REST API for seamless integration:

*   **Python SDK:**  `pip install opik` or `uv pip install opik`
*   **TypeScript SDK:**  (via OpenTelemetry)
*   **Ruby SDK:** (via OpenTelemetry)

Configure the SDK:

```bash
opik configure
```
(or `opik.configure(use_local=True)` for self-hosted instances)

See the [Python SDK documentation](apps/opik-documentation/documentation/fern/docs/reference/python-sdk/) for more configuration options.

## 📝 Integrations

Opik supports numerous integrations for easy tracing:

*   **Google ADK**
*   **AG2**
*   **AIsuite**
*   **Agno**
*   **Anthropic**
*   **Autogen**
*   **Bedrock**
*   **BeeAI**
*   **BytePlus**
*   **Cloudflare Workers AI**
*   **Cohere**
*   **CrewAI**
*   **Cursor**
*   **DeepSeek**
*   **Dify**
*   **DSPy**
*   **Fireworks AI**
*   **Flowise AI**
*   **Gemini**
*   **Groq**
*   **Guardrails**
*   **Haystack**
*   **Instructor**
*   **LangChain (Python)**
*   **LangChain (JS/TS)**
*   **LangGraph**
*   **LiteLLM**
*   **LiveKit Agents**
*   **LlamaIndex**
*   **Mastra**
*   **Mistral AI**
*   **Novita AI**
*   **Ollama**
*   **OpenAI (Python)**
*   **OpenAI (JS/TS)**
*   **OpenAI Agents**
*   **OpenRouter**
*   **OpenTelemetry**
*   **Predibase**
*   **Pydantic AI**
*   **Ragas**
*   **Semantic Kernel**
*   **Smolagents**
*   **Spring AI**
*   **Strands Agents**
*   **Together AI**
*   **Vercel AI SDK**
*   **VoltAgent**
*   **WatsonX**
*   **xAI Grok**

Refer to the documentation for quickstart Colab notebooks and integration guides.

## 🧑‍⚖️ LLM-as-a-Judge Metrics

Use LLM-as-a-Judge metrics for evaluation:

```python
from opik.evaluation.metrics import Hallucination

metric = Hallucination()
score = metric.score(
    input="What is the capital of France?",
    output="Paris",
    context=["France is a country in Europe."]
)
print(score)
```

Explore the [metrics documentation](https://www.comet.com/docs/opik/evaluation/metrics/overview?from=llm&utm_source=opik&utm_medium=github&utm_content=metrics_3_link&utm_campaign=opik).

## 🔍 Evaluating Your Application

Evaluate your LLM application using:

*   [Datasets](https://www.comet.com/docs/opik/evaluation/manage_datasets/?from=llm&utm_source=opik&utm_medium=github&utm_content=datasets_2_link&utm_campaign=opik)
*   [Experiments](https://www.comet.com/docs/opik/evaluation/evaluate_your_llm/?from=llm&utm_source=opik&utm_medium=github&utm_content=experiments_link&utm_campaign=opik)
*   [PyTest integration](https://www.comet.com/docs/opik/testing/pytest_integration/?from=llm&utm_source=opik&utm_medium=github&utm_content=pytest_2_link&utm_campaign=opik) for CI/CD

## ⭐ Star Us

Show your support by starring the repository!  It helps grow our community.

[![Star History Chart](https://api.star-history.com/svg?repos=comet-ml/opik&type=Date)](https://github.com/comet-ml/opik)

## 🤝 Contributing

Contribute to Opik:

*   Submit [bug reports](https://github.com/comet-ml/opik/issues) and [feature requests](https://github.com/comet-ml/opik/issues).
*   Improve documentation via [Pull Requests](https://github.com/comet-ml/opik/pulls).
*   Spread the word!
*   Upvote [popular feature requests](https://github.com/comet-ml/opik/issues?q=is%3Aissue+is%3Aopen+label%3A%22enhancement%22).

See our [contributing guidelines](CONTRIBUTING.md).
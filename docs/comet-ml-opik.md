<div align="center"><b><a href="README.md">English</a> | <a href="readme_CN.md">简体中文</a> | <a href="readme_JP.md">日本語</a> | <a href="readme_KO.md">한국어</a></b></div>

<h1 align="center" style="border-bottom: none">
    <div>
        <a href="https://www.comet.com/site/products/opik/?from=llm&utm_source=opik&utm_medium=github&utm_content=header_img&utm_campaign=opik"><picture>
            <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/comet-ml/opik/refs/heads/main/apps/opik-documentation/documentation/static/img/logo-dark-mode.svg">
            <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/comet-ml/opik/refs/heads/main/apps/opik-documentation/documentation/static/img/opik-logo.svg">
            <img alt="Comet Opik logo" src="https://raw.githubusercontent.com/comet-ml/opik/refs/heads/main/apps/opik-documentation/documentation/static/img/opik-logo.svg" width="200" />
        </picture></a>
        <br>
        Opik
    </div>
</h1>
<h2 align="center" style="border-bottom: none">Open-source LLM evaluation platform</h2>

<p align="center">
  <a href="https://www.comet.com/site/products/opik/?from=llm&utm_source=opik&utm_medium=github&utm_content=website_button&utm_campaign=opik"><b>Website</b></a> •
  <a href="https://chat.comet.com"><b>Slack Community</b></a> •
  <a href="https://x.com/Cometml"><b>Twitter</b></a> •
  <a href="https://www.comet.com/docs/opik/changelog"><b>Changelog</b></a> •
  <a href="https://www.comet.com/docs/opik/?from=llm&utm_source=opik&utm_medium=github&utm_content=docs_button&utm_campaign=opik"><b>Documentation</b></a>
</p>

<div align="center">

[![Python SDK](https://img.shields.io/pypi/v/opik)](https://pypi.org/project/opik/)
[![License](https://img.shields.io/github/license/comet-ml/opik)](https://github.com/comet-ml/opik/blob/main/LICENSE)
[![Build](https://github.com/comet-ml/opik/actions/workflows/build_apps.yml/badge.svg)](https://github.com/comet-ml/opik/actions/workflows/build_apps.yml)
[![Bounties](https://img.shields.io/endpoint?url=https%3A%2F%2Falgora.io%2Fapi%2Fshields%2Fcomet-ml%2Fbounties%3Fstatus%3Dopen)](https://algora.io/comet-ml/bounties?status=open)
</div>

<br>

<a href="https://www.comet.com/signup?from=llm&utm_source=opik&utm_medium=github&utm_content=readme_banner&utm_campaign=opik"><img src="readme-thumbnail-new.png" alt="Opik platform screenshot (thumbnail)"></a>

## Opik: The Open-Source Platform to Build, Evaluate, and Optimize Your LLM Applications

Opik, brought to you by [Comet](https://www.comet.com?from=llm&utm_source=opik&utm_medium=github&utm_content=what_is_opik_link&utm_campaign=opik), is an open-source platform designed to streamline the entire lifecycle of LLM applications, helping you build, evaluate, and optimize your LLM systems.

**Key Features:**

*   **Comprehensive Observability:** Deep tracing of LLM calls, conversation logging, and agent activity.
*   **Advanced Evaluation:** Robust prompt evaluation, LLM-as-a-judge metrics, and experiment management.
*   **Production-Ready:** Scalable monitoring dashboards and online evaluation rules for production.
*   **Opik Agent Optimizer:** Dedicated SDK and optimizers to enhance prompts and agents.
*   **Opik Guardrails:** Features to help you implement safe and responsible AI practices.

**Key Capabilities & Benefits:**

*   **LLM Tracing & Monitoring:**
    *   Track all LLM calls and traces during development and in production with detailed context ([Quickstart](https://www.comet.com/docs/opik/quickstart/?from=llm&utm_source=opik&utm_medium=github&utm_content=quickstart_link&utm_campaign=opik)).
    *   Easily integrate with popular frameworks like **Google ADK**, **Autogen**, and **Flowise AI** ([Integrations](https://www.comet.com/docs/opik/tracing/integrations/overview/?from=llm&utm_source=opik&utm_medium=github&utm_content=integrations_link&utm_campaign=opik)).
    *   Annotate traces and spans with feedback scores via the [Python SDK](https://www.comet.com/docs/opik/tracing/annotate_traces/#annotating-traces-and-spans-using-the-sdk?from=llm&utm_source=opik&utm_medium=github&utm_content=sdk_link&utm_campaign=opik) or the [UI](https://www.comet.com/docs/opik/tracing/annotate_traces/#annotating-traces-through-the-ui?from=llm&utm_source=opik&utm_medium=github&utm_content=ui_link&utm_campaign=opik).
    *   Experiment with prompts and models in the [Prompt Playground](https://www.comet.com/docs/opik/prompt_engineering/playground).
*   **LLM Application Evaluation:**
    *   Automate evaluation with [Datasets](https://www.comet.com/docs/opik/evaluation/manage_datasets/?from=llm&utm_source=opik&utm_medium=github&utm_content=datasets_link&utm_campaign=opik) and [Experiments](https://www.comet.com/docs/opik/evaluation/evaluate_your_llm/?from=llm&utm_source=opik&utm_medium=github&utm_content=eval_link&utm_campaign=opik).
    *   Utilize LLM-as-a-judge metrics for tasks such as [hallucination detection](https://www.comet.com/docs/opik/evaluation/metrics/hallucination/?from=llm&utm_source=opik&utm_medium=github&utm_content=hallucination_link&utm_campaign=opik) and [moderation](https://www.comet.com/docs/opik/evaluation/metrics/moderation/?from=llm&utm_source=opik&utm_medium=github&utm_content=moderation_link&utm_campaign=opik).
    *   Integrate evaluations into your CI/CD pipeline using our [PyTest integration](https://www.comet.com/docs/opik/testing/pytest_integration/?from=llm&utm_source=opik&utm_medium=github&utm_content=pytest_link&utm_campaign=opik).
*   **Production Monitoring and Optimization:**
    *   Log high volumes of production traces (Opik is designed for 40M+ traces/day).
    *   Monitor feedback scores, trace counts, and token usage in the [Opik Dashboard](https://www.comet.com/docs/opik/production/production_monitoring/?from=llm&utm_source=opik&utm_medium=github&utm_content=dashboard_link&utm_campaign=opik).
    *   Utilize [Online Evaluation Rules](https://www.comet.com/docs/opik/production/rules/?from=llm&utm_source=opik&utm_medium=github&utm_content=dashboard_link&utm_campaign=opik) with LLM-as-a-Judge metrics to identify production issues.
    *   Leverage **Opik Agent Optimizer** and **Opik Guardrails** to improve and secure your LLM applications.

> [!TIP]
> If you are looking for features that Opik doesn't have today, please raise a new [Feature request](https://github.com/comet-ml/opik/issues/new/choose) 🚀

## Getting Started

### 1. Choose Your Deployment Option

*   **Comet.com Cloud (Easiest & Recommended):** Access Opik instantly without any setup.  [Create your free Comet account](https://www.comet.com/signup?from=llm&utm_source=opik&utm_medium=github&utm_content=install_create_link&utm_campaign=opik)
*   **Self-Host Opik (Docker or Kubernetes):** Deploy Opik in your own environment for full control.

### 2. Self-Hosting Installation

#### Self-Hosting with Docker Compose (Local Development)

```bash
# Clone the Opik repository
git clone https://github.com/comet-ml/opik.git

# Navigate to the repository
cd opik

# Start the Opik platform
./opik.sh
```

Use the `--help` or `--info` options to troubleshoot issues.  Access Opik at [localhost:5173](http://localhost:5173).  See the [Local Deployment Guide](https://www.comet.com/docs/opik/self-host/local_deployment?from=llm&utm_source=opik&utm_medium=github&utm_content=self_host_link&utm_campaign=opik) for details.

#### Self-Hosting with Kubernetes & Helm (Scalable Deployments)

For production deployments, install Opik on a Kubernetes cluster using our Helm chart. [Kubernetes Installation Guide using Helm](https://www.comet.com/docs/opik/self-host/kubernetes/#kubernetes-installation?from=llm&utm_source=opik&utm_medium=github&utm_content=kubernetes_link&utm_campaign=opik).

[![Kubernetes](https://img.shields.io/badge/Kubernetes-%23326ce5.svg?&logo=kubernetes&logoColor=white)](https://www.comet.com/docs/opik/self-host/kubernetes/#kubernetes-installation?from=llm&utm_source=opik&utm_medium=github&utm_content=kubernetes_link&utm_campaign=opik)

> [!IMPORTANT]
> **Version 1.7.0 Changes:**  Check the [changelog](https://github.com/comet-ml/opik/blob/main/CHANGELOG.md) for important updates and breaking changes.

## 3.  Opik Client SDK

Interact with the Opik server using Python, TypeScript, and Ruby SDKs (via OpenTelemetry).

*   **Python SDK Quick Start:**

    ```bash
    # install using pip
    pip install opik

    # or install with uv
    uv pip install opik
    ```

    Configure the python SDK:

    ```bash
    opik configure
    ```

    Refer to the [Python SDK documentation](apps/opik-documentation/documentation/fern/docs/reference/python-sdk/) for more configuration options.  Start logging traces using the [Python SDK](https://www.comet.com/docs/opik/python-sdk-reference/?from=llm&utm_source=opik&utm_medium=github&utm_content=sdk_link2&utm_campaign=opik).

## Logging Traces with Integrations

Easily log traces using our direct integrations:

| Integration | Description | Documentation | Try in Colab |
|---|---|---|---|
| AG2 | Log traces for AG2 LLM calls | [Documentation](https://www.comet.com/docs/opik/tracing/integrations/ag2?utm_source=opik&utm_medium=github&utm_content=ag2_link&utm_campaign=opik) | (_Coming Soon_) |
| aisuite | Log traces for aisuite LLM calls | [Documentation](https://www.comet.com/docs/opik/tracing/integrations/aisuite?utm_source=opik&utm_medium=github&utm_content=aisuite_link&utm_campaign=opik) | [![Open Quickstart In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/comet-ml/opik/blob/main/apps/opik-documentation/documentation/docs/cookbook/aisuite.ipynb) |
| ... | ... | ... | ... |

> [!TIP]
> If the framework you are using is not listed above, [open an issue](https://github.com/comet-ml/opik/issues) or submit a PR with the integration.

## LLM as a Judge Metrics

The Python Opik SDK offers LLM-as-a-judge metrics for evaluation.  Learn more in the [metrics documentation](https://www.comet.com/docs/opik/evaluation/metrics/overview/?from=llm&utm_source=opik&utm_medium=github&utm_content=metrics_2_link&utm_campaign=opik).

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

## Evaluating Your LLM Application

Evaluate your LLM applications using [Datasets](https://www.comet.com/docs/opik/evaluation/manage_datasets/?from=llm&utm_source=opik&utm_medium=github&utm_content=datasets_2_link&utm_campaign=opik) and [Experiments](https://www.comet.com/docs/opik/evaluation/evaluate_your_llm/?from=llm&utm_source=opik&utm_medium=github&utm_content=experiments_link&utm_campaign=opik).  Use the Opik Dashboard and integrate evaluations into your CI/CD pipeline with our [PyTest integration](https://www.comet.com/docs/opik/testing/pytest_integration/?from=llm&utm_source=opik&utm_medium=github&utm_content=pytest_2_link&utm_campaign=opik).

## Contributing

Help improve Opik!  Contribute by:

*   Submitting [bug reports](https://github.com/comet-ml/opik/issues) and [feature requests](https://github.com/comet-ml/opik/issues)
*   Reviewing the documentation and submitting [Pull Requests](https://github.com/comet-ml/opik/pulls)
*   Sharing Opik and [letting us know](https://chat.comet.com)
*   Upvoting [popular feature requests](https://github.com/comet-ml/opik/issues?q=is%3Aissue+is%3Aopen+label%3A%22enhancement%22)

See our [contributing guidelines](CONTRIBUTING.md).

## ⭐ Star Us on GitHub

Show your support by giving us a star!

[![Star History Chart](https://api.star-history.com/svg?repos=comet-ml/opik&type=Date)](https://github.com/comet-ml/opik)

[Back to Top](#) (optional - can be added for navigation)
<!--
  English | 简体中文 | 日本語 | 한국어
-->
<div align="center">
  <b><a href="README.md">English</a> | <a href="readme_CN.md">简体中文</a> | <a href="readme_JP.md">日本語</a> | <a href="readme_KO.md">한국어</a></b>
</div>

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

<h2 align="center" style="border-bottom: none">Open-Source LLM Evaluation Platform</h2>

<p align="center">
  <b>Build, Evaluate, and Optimize Your LLM Systems with Opik.</b> From RAG applications to agentic workflows, Opik provides comprehensive tracing, evaluation tools, and powerful features like **Opik Agent Optimizer** and **Opik Guardrails** to enhance and secure your LLM applications in production.
</p>

<div align="center">
  [![Python SDK](https://img.shields.io/pypi/v/opik)](https://pypi.org/project/opik/)
  [![License](https://img.shields.io/github/license/comet-ml/opik)](https://github.com/comet-ml/opik/blob/main/LICENSE)
  [![Build](https://github.com/comet-ml/opik/actions/workflows/build_apps.yml/badge.svg)](https://github.com/comet-ml/opik/actions/workflows/build_apps.yml)
  [![Bounties](https://img.shields.io/endpoint?url=https%3A%2F%2Falgora.io%2Fapi%2Fshields%2Fcomet-ml%2Fbounties%3Fstatus%3Dopen)](https://algora.io/comet-ml/bounties?status=open)
</div>

<p align="center">
  <a href="https://www.comet.com/site/products/opik/?from=llm&utm_source=opik&utm_medium=github&utm_content=website_button&utm_campaign=opik"><b>Website</b></a> •
  <a href="https://chat.comet.com"><b>Slack Community</b></a> •
  <a href="https://x.com/Cometml"><b>Twitter</b></a> •
  <a href="https://www.comet.com/docs/opik/changelog"><b>Changelog</b></a> •
  <a href="https://www.comet.com/docs/opik/?from=llm&utm_source=opik&utm_medium=github&utm_content=docs_button&utm_campaign=opik"><b>Documentation</b></a>
  • <a href="https://github.com/comet-ml/opik"><b>GitHub Repository</b></a>
</p>

<div align="center" style="margin-top: 1em; margin-bottom: 1em;">
  <a href="#-what-is-opik">🚀 What is Opik?</a> • <a href="#%EF%B8%8F-opik-server-installation">🛠️ Installation</a> • <a href="#-opik-client-sdk">💻 Client SDK</a> • <a href="#-logging-traces-with-integrations">📝 Tracing</a>
  <br>
  <a href="#-llm-as-a-judge-metrics">🧑‍⚖️ LLM-as-a-Judge</a> • <a href="#-evaluating-your-llm-application">🔍 Evaluation</a> • <a href="#-star-us-on-github">⭐ Star Us</a> • <a href="#-contributing">🤝 Contribute</a>
</div>

<br>

<a href="https://www.comet.com/signup?from=llm&utm_source=opik&utm_medium=github&utm_content=readme_banner&utm_campaign=opik">
  <img src="readme-thumbnail-new.png" alt="Opik platform screenshot (thumbnail)" />
</a>

## 🚀 What is Opik?

Opik, brought to you by [Comet](https://www.comet.com?from=llm&utm_source=opik&utm_medium=github&utm_content=what_is_opik_link&utm_campaign=opik), is an open-source platform designed to revolutionize the LLM application lifecycle.  It provides a comprehensive suite of tools for developers to:

*   **Evaluate and Test:** Measure the performance of prompts, models, and agentic systems.
*   **Monitor and Optimize:** Gain insights into production performance and identify areas for improvement.
*   **Ensure Safety and Reliability:** Implement features such as guardrails to mitigate risks.

**Key Features:**

*   **Comprehensive Observability:** Deeply trace LLM calls, log conversations, and monitor agent activity.
*   **Advanced Evaluation:** Utilize robust prompt evaluation, LLM-as-a-judge metrics, and experiment management.
*   **Production-Ready:** Benefit from scalable monitoring dashboards and online evaluation rules tailored for production environments.
*   **Opik Agent Optimizer:** Leverage a dedicated SDK and optimizers to enhance prompts and agents.
*   **Opik Guardrails:** Implement safe and responsible AI practices with built-in guardrails.

**Key Capabilities in Detail:**

*   **Development & Tracing:**
    *   Track all LLM calls and traces with detailed context throughout development and production ([Quickstart](https://www.comet.com/docs/opik/quickstart/?from=llm&utm_source=opik&utm_medium=github&utm_content=quickstart_link&utm_campaign=opik)).
    *   Integrate seamlessly with popular frameworks like **Google ADK**, **Autogen**, and **Flowise AI** with our growing list of 3rd-party integrations. ([Integrations](https://www.comet.com/docs/opik/tracing/integrations/overview/?from=llm&utm_source=opik&utm_medium=github&utm_content=integrations_link&utm_campaign=opik))
    *   Annotate traces and spans with feedback scores using the [Python SDK](https://www.comet.com/docs/opik/tracing/annotate_traces/#annotating-traces-and-spans-using-the-sdk?from=llm&utm_source=opik&utm_medium=github&utm_content=sdk_link&utm_campaign=opik) or the [UI](https://www.comet.com/docs/opik/tracing/annotate_traces/#annotating-traces-through-the-ui?from=llm&utm_source=opik&utm_medium=github&utm_content=ui_link&utm_campaign=opik).
    *   Experiment with prompts and models in the [Prompt Playground](https://www.comet.com/docs/opik/prompt_engineering/playground).

*   **Evaluation & Testing:**
    *   Automate your LLM application evaluation with [Datasets](https://www.comet.com/docs/opik/evaluation/manage_datasets/?from=llm&utm_source=opik&utm_medium=github&utm_content=datasets_link&utm_campaign=opik) and [Experiments](https://www.comet.com/docs/opik/evaluation/evaluate_your_llm/?from=llm&utm_source=opik&utm_medium=github&utm_content=eval_link&utm_campaign=opik).
    *   Leverage LLM-as-a-judge metrics for complex tasks such as [hallucination detection](https://www.comet.com/docs/opik/evaluation/metrics/hallucination/?from=llm&utm_source=opik&utm_medium=github&utm_content=hallucination_link&utm_campaign=opik), [moderation](https://www.comet.com/docs/opik/evaluation/metrics/moderation/?from=llm&utm_source=opik&utm_medium=github&utm_content=moderation_link&utm_campaign=opik), and RAG assessment ([Answer Relevance](https://www.comet.com/docs/opik/evaluation/metrics/answer_relevance/?from=llm&utm_source=opik&utm_medium=github&utm_content=alex_link&utm_campaign=opik), [Context Precision](https://www.comet.com/docs/opik/evaluation/metrics/context_precision/?from=llm&utm_source=opik&utm_medium=github&utm_content=context_link&utm_campaign=opik)).
    *   Integrate evaluations into your CI/CD pipeline using our [PyTest integration](https://www.comet.com/docs/opik/testing/pytest_integration/?from=llm&utm_source=opik&utm_medium=github&utm_content=pytest_link&utm_campaign=opik).

*   **Production Monitoring & Optimization:**
    *   Log high volumes of production traces with Opik's scalable architecture (40M+ traces/day).
    *   Monitor feedback scores, trace counts, and token usage over time in the [Opik Dashboard](https://www.comet.com/docs/opik/production/production_monitoring/?from=llm&utm_source=opik&utm_medium=github&utm_content=dashboard_link&utm_campaign=opik).
    *   Utilize [Online Evaluation Rules](https://www.comet.com/docs/opik/production/rules/?from=llm&utm_source=opik&utm_medium=github&utm_content=dashboard_link&utm_campaign=opik) with LLM-as-a-Judge metrics to identify production issues.
    *   Leverage **Opik Agent Optimizer** and **Opik Guardrails** to continuously improve and secure your LLM applications in production.

> [!TIP]
> If you are looking for features that Opik doesn't have today, please raise a new [Feature request](https://github.com/comet-ml/opik/issues/new/choose) 🚀

<br>

## 🛠️ Installation

Get up and running with Opik quickly and easily. Choose your preferred method:

### Option 1: Comet.com Cloud (Easiest & Recommended)

Instantly access Opik without any setup. Ideal for quick starts and effortless maintenance.

👉 [Create your free Comet account](https://www.comet.com/signup?from=llm&utm_source=opik&utm_medium=github&utm_content=install_create_link&utm_campaign=opik)

### Option 2: Self-Host Opik for Full Control

Deploy Opik in your own environment. Choose between Docker for local setups or Kubernetes for scalability.

#### Self-Hosting with Docker Compose (for Local Development & Testing)

This is the simplest way to get a local Opik instance running. Note the new `./opik.sh` installation script:

On Linux or Mac Environment:

```bash
# Clone the Opik repository
git clone https://github.com/comet-ml/opik.git

# Navigate to the repository
cd opik

# Start the Opik platform
./opik.sh
```

On Windows Environment:

```powershell
# Clone the Opik repository
git clone https://github.com/comet-ml/opik.git

# Navigate to the repository
cd opik

# Start the Opik platform
powershell -ExecutionPolicy ByPass -c ".\\opik.ps1"
```

**Service Profiles for Development**

The Opik installation scripts now support service profiles for different development scenarios:

```bash
# Start full Opik suite (default behavior)
./opik.sh

# Start only infrastructure services (databases, caches etc.)
./opik.sh --infra

# Start infrastructure + backend services
./opik.sh --backend

# Enable guardrails with any profile
./opik.sh --guardrails # Guardrails with full Opik suite
./opik.sh --backend --guardrails # Guardrails with infrastructure + backend
```

Use the `--help` or `--info` options to troubleshoot issues. Dockerfiles now ensure containers run as non-root users for enhanced security. Once all is up and running, you can now visit [localhost:5173](http://localhost:5173) on your browser! For detailed instructions, see the [Local Deployment Guide](https://www.comet.com/docs/opik/self-host/local_deployment?from=llm&utm_source=opik&utm_medium=github&utm_content=self_host_link&utm_campaign=opik).

#### Self-Hosting with Kubernetes & Helm (for Scalable Deployments)

For production or larger-scale self-hosted deployments, Opik can be installed on a Kubernetes cluster using our Helm chart. Click the badge for the full [Kubernetes Installation Guide using Helm](https://www.comet.com/docs/opik/self-host/kubernetes/#kubernetes-installation?from=llm&utm_source=opik&utm_medium=github&utm_content=kubernetes_link&utm_campaign=opik).

[![Kubernetes](https://img.shields.io/badge/Kubernetes-%23326ce5.svg?&logo=kubernetes&logoColor=white)](https://www.comet.com/docs/opik/self-host/kubernetes/#kubernetes-installation?from=llm&utm_source=opik&utm_medium=github&utm_content=kubernetes_link&utm_campaign=opik)

> [!IMPORTANT]
> **Version 1.7.0 Changes**: Please check the [changelog](https://github.com/comet-ml/opik/blob/main/CHANGELOG.md) for important updates and breaking changes.

## 💻 Opik Client SDK

Opik offers a powerful Client SDK, along with a REST API, to seamlessly integrate with the Opik server. The SDKs are available for Python, TypeScript, and Ruby (via OpenTelemetry), giving you the flexibility to integrate into your preferred workflows. Refer to the [Opik Client Reference Documentation](apps/opik-documentation/documentation/fern/docs/reference/overview.mdx) for comprehensive API and SDK details.

### Python SDK Quick Start

Get started with the Python SDK:

Install the package:

```bash
# install using pip
pip install opik

# or install with uv
uv pip install opik
```

Configure the python SDK by running the `opik configure` command, which will prompt you for your Opik server address (for self-hosted instances) or your API key and workspace (for Comet.com):

```bash
opik configure
```

> [!TIP]
> You can also call `opik.configure(use_local=True)` from your Python code to configure the SDK to run on a local self-hosted installation, or provide API key and workspace details directly for Comet.com. Refer to the [Python SDK documentation](apps/opik-documentation/documentation/fern/docs/reference/python-sdk/) for more configuration options.

You are now ready to start logging traces using the [Python SDK](https://www.comet.com/docs/opik/python-sdk-reference/?from=llm&utm_source=opik&utm_medium=github&utm_content=sdk_link2&utm_campaign=opik).

### 📝 Logging Traces with Integrations

The easiest way to log traces is to use one of our direct integrations. Opik supports a wide array of frameworks, including recent additions like **Google ADK**, **Autogen**, **AG2**, and **Flowise AI**:

| Integration    | Description                                             | Documentation                                                                                                                                                            | Try in Colab                                                                                                                                                                                                                               |
| -------------- | ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| ADK            | Log traces for Google Agent Development Kit (ADK)       | [Documentation](https://www.comet.com/docs/opik/tracing/integrations/adk?utm_source=opik&utm_medium=github&utm_content=google_adk_link&utm_campaign=opik)                | [![Open Quickstart In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/comet-ml/opik/blob/main/apps/opik-documentation/documentation/docs/cookbook/google_adk_integration.ipynb) |
| AG2            | Log traces for AG2 LLM calls                            | [Documentation](https://www.comet.com/docs/opik/tracing/integrations/ag2?utm_source=opik&utm_medium=github&utm_content=ag2_link&utm_campaign=opik)                       | (_Coming Soon_)                                                                                                                                                                                                                            |
| AIsuite        | Log traces for aisuite LLM calls                        | [Documentation](https://www.comet.com/docs/opik/tracing/integrations/aisuite?utm_source=opik&utm_medium=github&utm_content=aisuite_link&utm_campaign=opik)               | [![Open Quickstart In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/comet-ml/opik/blob/main/apps/opik-documentation/documentation/docs/cookbook/aisuite.ipynb)                |
| Agno           | Log traces for Agno agent orchestration framework calls | [Documentation](https://www.comet.com/docs/opik/tracing/integrations/agno?utm_source=opik&utm_medium=github&utm_content=agno_link&utm_campaign=opik)                     | (_Coming Soon_)                                                                                                                                                                                                                            |
| Anthropic      | Log traces for Anthropic LLM calls                      | [Documentation](https://www.comet.com/docs/opik/tracing/integrations/anthropic?utm_source=opik&utm_medium=github&utm_content=anthropic_link&utm_campaign=opik)           | [![Open Quickstart In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/comet-ml/opik/blob/main/apps/opik-documentation/documentation/docs/cookbook/anthropic.ipynb)              |
| Autogen        | Log traces for Autogen agentic workflows                | [Documentation](https://www.comet.com/docs/opik/tracing/integrations/autogen?utm_source=opik&utm_medium=github&utm_content=autogen_link&utm_campaign=opik)               | (_Coming Soon_)                                                                                                                                                                                                                            |
| Bedrock        | Log traces for Amazon Bedrock LLM calls                 | [Documentation](https://www.comet.com/docs/opik/tracing/integrations/bedrock?utm_source=opik&utm_medium=github&utm_content=bedrock_link&utm_campaign=opik)               | [![Open Quickstart In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/comet-ml/opik/blob/main/apps/opik-documentation/documentation/docs/cookbook/bedrock.ipynb)                |
| BeeAI          | Log traces for BeeAI agent framework calls              | [Documentation](https://www.comet.com/docs/opik/tracing/integrations/beeai?utm_source=opik&utm_medium=github&utm_content=beeai_link&utm_campaign=opik)                 | (_Coming Soon_)                                                                                                                                                                                                                            |
| BytePlus       | Log traces for BytePlus LLM calls                       | [Documentation](https://www.comet.com/docs/opik/tracing/integrations/byteplus?utm_source=opik&utm_medium=github&utm_content=byteplus_link&utm_campaign=opik)             | (_Coming Soon_)                                                                                                                                                                                                                            |
| Cloudflare Workers AI | Log traces for Cloudflare Workers AI calls              | [Documentation](https://www.comet.com/docs/opik/tracing/integrations/cloudflare-workers-ai?utm_source=opik&utm_medium=github&utm_content=cloudflare_workers_ai_link&utm_campaign=opik) | (_Coming Soon_)                                                                                                                                                                                                                            |
| Cohere         | Log traces for Cohere LLM calls                         | [Documentation](https://www.comet.com/docs/opik/tracing/integrations/cohere?utm_source=opik&utm_medium=github&utm_content=cohere_link&utm_campaign=opik)                 | (_Coming Soon_)                                                                                                                                                                                                                            |
| CrewAI         | Log traces for CrewAI calls                             | [Documentation](https://www.comet.com/docs/opik/tracing/integrations/crewai?utm_source=opik&utm_medium=github&utm_content=crewai_link&utm_campaign=opik)                 | [![Open Quickstart In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/comet-ml/opik/blob/main/apps/opik-documentation/documentation/docs/cookbook/crewai.ipynb)                 |
| Cursor         | Log traces for Cursor conversations                      | [Documentation](https://www.comet.com/docs/opik/tracing/integrations/cursor?utm_source=opik&utm_medium=github&utm_content=cursor_link&utm_campaign=opik)                 | (_Native UI integration, see documentation_)                                                                                                                                                                                               |
| DeepSeek       | Log traces for DeepSeek LLM calls                       | [Documentation](https://www.comet.com/docs/opik/tracing/integrations/deepseek?utm_source=opik&utm_medium=github&utm_content=deepseek_link&utm_campaign=opik)             | (_Coming Soon_)                                                                                                                                                                                                                            |
| Dify           | Log traces for Dify agent runs                          | [Documentation](https://www.comet.com/docs/opik/tracing/integrations/dify?utm_source=opik&utm_medium=github&utm_content=dify_link&utm_campaign=opik)                     | (_Coming Soon_)                                                                                                                                                                                                                            |
| DSPY           | Log traces for DSPy runs                                | [Documentation](https://www.comet.com/docs/opik/tracing/integrations/dspy?utm_source=opik&utm_medium=github&utm_content=dspy_link&utm_campaign=opik)                     | [![Open Quickstart In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/comet-ml/opik/blob/main/apps/opik-documentation/documentation/docs/cookbook/dspy.ipynb)                   |
| Fireworks AI   | Log traces for Fireworks AI LLM calls                   | [Documentation](https://www.comet.com/docs/opik/tracing/integrations/fireworks-ai?utm_source=opik&utm_medium=github&utm_content=fireworks_ai_link&utm_campaign=opik)     | (_Coming Soon_)                                                                                                                                                                                                                            |
| Flowise AI     | Log traces for Flowise AI visual LLM builder            | [Documentation](https://www.comet.com/docs/opik/tracing/integrations/flowise?utm_source=opik&utm_medium=github&utm_content=flowise_link&utm_campaign=opik)               | (_Native UI integration, see documentation_)                                                                                                                                                                                               |
| Gemini         | Log traces for Google Gemini LLM calls                  | [Documentation](https://www.comet.com/docs/opik/tracing/integrations/gemini?utm_source=opik&utm_medium=github&utm_content=gemini_link&utm_campaign=opik)                 | [![Open Quickstart In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/comet-ml/opik/blob/main/apps/opik-documentation/documentation/docs/cookbook/gemini.ipynb)                 |
| Groq           | Log traces for Groq LLM calls                           | [Documentation](https://www.comet.com/docs/opik/tracing/integrations/groq?utm_source=opik&utm_medium=github&utm_content=groq_link&utm_campaign=opik)                     | [![Open Quickstart In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/comet-ml/opik/blob/main/apps/opik-documentation/documentation/docs/cookbook/groq.ipynb)                   |
| Guardrails     | Log traces for Guardrails AI validations                | [Documentation](https://www.comet.com/docs/opik/tracing/integrations/guardrails-ai?utm_source=opik&utm_medium=github&utm_content=guardrails_link&utm_campaign=opik)      | [![Open Quickstart In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/comet-ml/opik/blob/main/apps/opik-documentation/documentation/docs/cookbook/guardrails-ai.ipynb)          |
| Haystack       | Log traces for Haystack calls                           | [Documentation](https://www.comet.com/docs/opik/tracing/integrations/haystack?utm_source=opik&utm_medium=github&utm_content=haystack_link&utm_campaign=opik)             | [![Open Quickstart In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/comet-ml/opik/blob/main/apps/opik-documentation/documentation/docs/cookbook/haystack.ipynb)               |
| Instructor     | Log traces for LLM calls made with Instructor           | [Documentation](https://www.comet.com/docs/opik/tracing/integrations/instructor?utm_source=opik&utm_medium=github&utm_content=instructor_link&utm_campaign=opik)         | [![Open Quickstart In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/comet-ml/opik/blob/main/apps/opik-documentation/documentation/docs/cookbook/instructor.ipynb)             |
| LangChain (Python)    | Log traces for LangChain LLM calls                      | [Documentation](https://www.comet.com/docs/opik/tracing/integrations/langchain?utm_source=opik&utm_medium=github&utm_content=langchain_link&utm_campaign=opik)                         | [![Open Quickstart In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/comet-ml/opik/blob/main/apps/opik-documentation/documentation/docs/cookbook/langchain.ipynb)              |
| LangChain (JS/TS)     | Log traces for LangChain JavaScript/TypeScript calls    | [Documentation](https://www.comet.com/docs/opik/tracing/integrations/langchainjs?utm_source=opik&utm_medium=github&utm_content=langchainjs_link&utm_campaign=opik)                     | (_Coming Soon_)                                                                                                                                                                                                                            |
| LangGraph      | Log traces for LangGraph executions                     | [Documentation](https://www.comet.com/docs/opik/tracing/integrations/langgraph?utm_source=opik&utm_medium=github&utm_content=langgraph_link&utm_campaign=opik)           | [![Open Quickstart In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/comet-ml/opik/blob/main/apps/opik-documentation/documentation/docs/cookbook/langgraph.ipynb)              |
| LiteLLM        | Log traces for LiteLLM model calls                      | [Documentation](https://www.comet.com/docs/opik/tracing/integrations/litellm?utm_source=opik&utm_medium=github&utm_content=litellm_link&utm_campaign=opik)               | [![Open Quickstart In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/comet-ml/opik/blob/main/apps/opik-documentation/documentation/docs/cookbook/litellm.ipynb)                |
| LiveKit Agents | Log traces for LiveKit Agents AI agent framework calls  | [Documentation](https://www.comet.com/docs/opik/tracing/integrations/livekit?utm_source=opik&utm_medium=github&utm_content=livekit_link&utm_campaign=opik)             | (_Coming Soon_)                                                                                                                                                                                                                            |
| LlamaIndex     | Log traces for LlamaIndex LLM calls                     | [Documentation](https://www.comet.com/docs/opik/tracing/integrations/llama_index?utm_source=opik&utm_medium=github&utm_content=llama_index_link&utm_campaign=opik)       | [![Open Quickstart In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/comet-ml/opik/blob/main/apps/opik-documentation/documentation/docs/cookbook/llama-index.ipynb)            |
| Mastra         | Log traces for Mastra AI workflow framework calls       | [Documentation](https://www.comet.com/docs/opik/tracing/integrations/mastra?utm_source=opik&utm_medium=github&utm_content=mastra_link&utm_campaign=opik)               | (_Coming Soon_)                                                                                                                                                                                                                            |
| Mistral AI     | Log traces for Mistral AI LLM calls                     | [Documentation](https://www.comet.com/docs/opik/tracing/integrations/mistral?utm_source=opik&utm_medium=github&utm_content=mistral_link&utm_campaign=opik)               | (_Coming Soon_)                                                                                                                                                                                                                            |
| Novita AI      | Log traces for Novita AI LLM calls                      | [Documentation](https://www.comet.com/docs/opik/tracing/integrations/novita-ai?utm_source=opik&utm_medium=github&utm_content=novita_ai_link&utm_campaign=opik)           | (_Coming Soon_)                                                                                                                                                                                                                            |
| Ollama         | Log traces for Ollama LLM calls                         | [Documentation](https://www.comet.com/docs/opik/tracing/integrations/ollama?utm_source=opik&utm_medium=github&utm_content=ollama_link&utm_campaign=opik)                 | [![Open Quickstart In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/comet-ml/opik/blob/main/apps/opik-documentation/documentation/docs/cookbook/ollama.ipynb)                 |
| OpenAI (Python)       | Log traces for OpenAI LLM calls                         | [Documentation](https://www.comet.com/docs/opik/tracing/integrations/openai?utm_source=opik&utm_medium=github&utm_content=openai_link&utm_campaign=opik)                               | [![Open Quickstart In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/comet-ml/opik/blob/main/apps/opik-documentation/documentation/docs/cookbook/openai.ipynb)                 |
| OpenAI (JS/TS)        | Log traces for OpenAI JavaScript/TypeScript calls       | [Documentation](https://www.comet.com/docs/opik/tracing/integrations/openai-typescript?utm_source=opik&utm_medium=github&utm_content=openai_typescript_link&utm_campaign=opik)         | (_Coming Soon_)                                                                                                                                                                                                                            |
| OpenAI Agents         | Log traces for OpenAI Agents SDK calls                  | [Documentation](https://www.comet.com/docs/opik/tracing/integrations/openai_agents?utm_source=opik&utm_medium=github&utm_content=openai_agents_link&utm_campaign=opik)                 | [![Open Quickstart In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/comet-ml/opik/blob/main/apps/opik-documentation/documentation/docs/cookbook/openai-agents.ipynb)          |
| OpenRouter     | Log traces for OpenRouter LLM calls                     | [Documentation](https://www.comet.com/docs/opik/tracing/integrations/openrouter?utm_source=opik&utm_medium=github&utm_content=openrouter_link&utm_campaign=opik)         | (_Coming Soon_)                                                                                                                                                                                                                            |
| OpenTelemetry  | Log traces for OpenTelemetry supported calls
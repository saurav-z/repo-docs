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
Opik helps you build, evaluate, and optimize LLM systems that run better, faster, and cheaper. From RAG chatbots to code assistants to complex agentic pipelines, Opik provides comprehensive tracing, evaluations, dashboards, and powerful features like <b>Opik Agent Optimizer</b> and <b>Opik Guardrails</b> to improve and secure your LLM powered applications in production.
</p>

<div align="center">

[![Python SDK](https://img.shields.io/pypi/v/opik)](https://pypi.org/project/opik/)
[![License](https://img.shields.io/github/license/comet-ml/opik)](https://github.com/comet-ml/opik/blob/main/LICENSE)
[![Build](https://github.com/comet-ml/opik/actions/workflows/build_apps.yml/badge.svg)](https://github.com/comet-ml/opik/actions/workflows/build_apps.yml)
[![Bounties](https://img.shields.io/endpoint?url=https%3A%2F%2Falgora.io%2Fapi%2Fshields%2Fcomet-ml%2Fbounties%3Fstatus%3Dopen)](https://algora.io/comet-ml/bounties?status=open)
<!-- [![Quick Start](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/comet-ml/opik/blob/main/apps/opik-documentation/documentation/docs/cookbook/opik_quickstart.ipynb) -->

</div>

<p align="center">
    <a href="https://www.comet.com/site/products/opik/?from=llm&utm_source=opik&utm_medium=github&utm_content=website_button&utm_campaign=opik"><b>Website</b></a> •
    <a href="https://chat.comet.com"><b>Slack Community</b></a> •
    <a href="https://x.com/Cometml"><b>Twitter</b></a> •
    <a href="https://www.comet.com/docs/opik/changelog"><b>Changelog</b></a> •
    <a href="https://www.comet.com/docs/opik/?from=llm&utm_source=opik&utm_medium=github&utm_content=docs_button&utm_campaign=opik"><b>Documentation</b></a>
</p>

<div align="center" style="margin-top: 1em; margin-bottom: 1em;">
<a href="#-what-is-opik">🚀 What is Opik?</a> • <a href="#%EF%B8%8F-opik-server-installation">🛠️ Opik Server Installation</a> • <a href="#-opik-client-sdk">💻 Opik Client SDK</a> • <a href="#-logging-traces-with-integrations">📝 Logging Traces</a><br>
<a href="#-llm-as-a-judge-metrics">🧑‍⚖️ LLM as a Judge</a> • <a href="#-evaluating-your-llm-application">🔍 Evaluating your Application</a> • <a href="#-star-us-on-github">⭐ Star Us</a> • <a href="#-contributing">🤝 Contributing</a>
</div>

<br>

[![Opik platform screenshot (thumbnail)](readme-thumbnail-new.png)](https://www.comet.com/signup?from=llm&utm_source=opik&utm_medium=github&utm_content=readme_banner&utm_campaign=opik)

## Opik: The Open-Source Platform for LLM Development, Evaluation, and Optimization

Opik empowers you to build, evaluate, and optimize your Large Language Model (LLM) applications, ensuring they perform at their best. From tracing and evaluation to optimization and security, Opik provides a comprehensive toolkit for the entire LLM lifecycle. Check out the [original repo](https://github.com/comet-ml/opik)!

**Key Features:**

*   **Comprehensive Observability:**
    *   Deep tracing of LLM calls, conversation logging, and agent activity.
    *   Extensive integrations with leading LLM frameworks (Google ADK, Autogen, Flowise AI, and more).
    *   Annotation of traces and spans with feedback via the Python SDK or UI.
    *   Experiment with prompts and models in the Prompt Playground.
*   **Advanced Evaluation:**
    *   Robust prompt evaluation, LLM-as-a-judge, and experiment management.
    *   Automated LLM application evaluation with Datasets and Experiments.
    *   Powerful LLM-as-a-judge metrics for tasks like hallucination detection, moderation, and RAG assessment.
    *   Integration of evaluations into your CI/CD pipeline using our PyTest integration.
*   **Production-Ready Optimization & Security:**
    *   Scalable monitoring dashboards designed to handle high volumes of production traces (40M+ traces/day).
    *   Online Evaluation Rules with LLM-as-a-Judge metrics for real-time issue detection.
    *   Leverage **Opik Agent Optimizer** and **Opik Guardrails** to continuously improve and secure your LLM applications in production.

> [!TIP]
> If you are looking for features that Opik doesn't have today, please raise a new [Feature request](https://github.com/comet-ml/opik/issues/new/choose) 🚀

<br>

## 🛠️ Opik Server Installation

Get started with Opik quickly using the following options:

### Option 1: Comet.com Cloud (Easiest & Recommended)

Access Opik instantly with a free Comet account.

👉 [Create your free Comet account](https://www.comet.com/signup?from=llm&utm_source=opik&utm_medium=github&utm_content=install_create_link&utm_campaign=opik)

### Option 2: Self-Host Opik for Full Control

Deploy Opik in your own environment using Docker Compose (for local development and testing) or Kubernetes (for scalable deployments).

#### Self-Hosting with Docker Compose

**Prerequisites:** Ensure you have Docker and Docker Compose installed.

**Installation (Linux/Mac):**

```bash
# Clone the Opik repository
git clone https://github.com/comet-ml/opik.git

# Navigate to the repository
cd opik

# Start the Opik platform
./opik.sh
```

**Installation (Windows):**

```powershell
# Clone the Opik repository
git clone https://github.com/comet-ml/opik.git

# Navigate to the repository
cd opik

# Start the Opik platform
powershell -ExecutionPolicy ByPass -c ".\\opik.ps1"
```

Access the Opik platform at [localhost:5173](http://localhost:5173). For detailed instructions, refer to the [Local Deployment Guide](https://www.comet.com/docs/opik/self-host/local_deployment?from=llm&utm_source=opik&utm_medium=github&utm_content=self_host_link&utm_campaign=opik).

#### Self-Hosting with Kubernetes & Helm

For production deployments, install Opik on a Kubernetes cluster using our Helm chart.

[![Kubernetes](https://img.shields.io/badge/Kubernetes-%23326ce5.svg?&logo=kubernetes&logoColor=white)](https://www.comet.com/docs/opik/self-host/kubernetes/#kubernetes-installation?from=llm&utm_source=opik&utm_medium=github&utm_content=kubernetes_link&utm_campaign=opik)

> [!IMPORTANT]
> **Version 1.7.0 Changes**: Please review the [changelog](https://github.com/comet-ml/opik/blob/main/CHANGELOG.md) for important updates and potential breaking changes.

## 💻 Opik Client SDK

Interact with the Opik server using a suite of client libraries and a REST API, including Python, TypeScript, and Ruby (via OpenTelemetry). Explore the [Opik Client Reference Documentation](apps/opik-documentation/documentation/fern/docs/reference/overview.mdx) for detailed API and SDK references.

### Python SDK Quick Start

Install and configure the Python SDK to begin logging traces:

1.  **Install the package:**

    ```bash
    pip install opik
    # or install with uv
    uv pip install opik
    ```

2.  **Configure the SDK:**  Run `opik configure` to set up your server address/API key and workspace. You can also configure the SDK directly in your code using `opik.configure()`.

    ```bash
    opik configure
    ```
3.  **Start logging traces** using the [Python SDK](https://www.comet.com/docs/opik/python-sdk-reference/?from=llm&utm_source=opik&utm_medium=github&utm_content=sdk_link2&utm_campaign=opik).

### 📝 Logging Traces with Integrations

Seamlessly integrate Opik with your LLM workflows using our direct integrations.

**Supported Frameworks:**

| Integration    | Description                                                         | Documentation                                                                                                                                                        | Try in Colab                                                                                                                                                                                                                       |
|----------------|---------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| AG2            | Log traces for AG2 LLM calls                                        | [Documentation](https://www.comet.com/docs/opik/tracing/integrations/ag2?utm_source=opik&utm_medium=github&utm_content=anthropic_link&utm_campaign=opik)             | (*Coming Soon*)                                                                                                                                                                                                                    |
| aisuite        | Log traces for aisuite LLM calls                                    | [Documentation](https://www.comet.com/docs/opik/tracing/integrations/aisuite?utm_source=opik&utm_medium=github&utm_content=anthropic_link&utm_campaign=opik)         | [![Open Quickstart In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/comet-ml/opik/blob/main/apps/opik-documentation/documentation/docs/cookbook/aisuite.ipynb)    |
| Anthropic      | Log traces for Anthropic LLM calls                                  | [Documentation](https://www.comet.com/docs/opik/tracing/integrations/anthropic?utm_source=opik&utm_medium=github&utm_content=anthropic_link&utm_campaign=opik)       | [![Open Quickstart In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/comet-ml/opik/blob/main/apps/opik-documentation/documentation/docs/cookbook/anthropic.ipynb)    |
| Autogen        | Log traces for Autogen agentic workflows                            | [Documentation](https://www.comet.com/docs/opik/tracing/integrations/autogen?utm_source=opik&utm_medium=github&utm_content=autogen_link&utm_campaign=opik)           | (*Coming Soon*)                                                                                                                                                                                                                    |
| Bedrock        | Log traces for Amazon Bedrock LLM calls                             | [Documentation](https://www.comet.com/docs/opik/tracing/integrations/bedrock?utm_source=opik&utm_medium=github&utm_content=bedrock_link&utm_campaign=opik)           | [![Open Quickstart In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/comet-ml/opik/blob/main/apps/opik-documentation/documentation/docs/cookbook/bedrock.ipynb)      |
| CrewAI         | Log traces for CrewAI calls                                         | [Documentation](https://www.comet.com/docs/opik/tracing/integrations/crewai?utm_source=opik&utm_medium=github&utm_content=crewai_link&utm_campaign=opik)             | [![Open Quickstart In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/comet-ml/opik/blob/main/apps/opik-documentation/documentation/docs/cookbook/crewai.ipynb)       |
| DeepSeek       | Log traces for DeepSeek LLM calls                                   | [Documentation](https://www.comet.com/docs/opik/tracing/integrations/deepseek?utm_source=opik&utm_medium=github&utm_content=deepseek_link&utm_campaign=opik)         | (*Coming Soon*)                                                                                                                                                                                                                    |
| Dify           | Log traces for Dify agent runs                                      | [Documentation](https://www.comet.com/docs/opik/tracing/integrations/dify?utm_source=opik&utm_medium=github&utm_content=dspy_link&utm_campaign=opik)                 | (*Coming Soon*)                                                                                                                                                                                                                    |
| DSPy           | Log traces for DSPy runs                                            | [Documentation](https://www.comet.com/docs/opik/tracing/integrations/dspy?utm_source=opik&utm_medium=github&utm_content=dspy_link&utm_campaign=opik)                 | [![Open Quickstart In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/comet-ml/opik/blob/main/apps/opik-documentation/documentation/docs/cookbook/dspy.ipynb)         |
| Flowise AI     | Log traces for Flowise AI visual LLM builder                        | [Documentation](https://www.comet.com/docs/opik/tracing/integrations/flowise?utm_source=opik&utm_medium=github&utm_content=flowise_link&utm_campaign=opik)           | (*Native UI intergration, see documentation*)                                                                                                                                                                                      |
| Gemini         | Log traces for Google Gemini LLM calls                              | [Documentation](https://www.comet.com/docs/opik/tracing/integrations/gemini?utm_source=opik&utm_medium=github&utm_content=gemini_link&utm_campaign=opik)             | [![Open Quickstart In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/comet-ml/opik/blob/main/apps/opik-documentation/documentation/docs/cookbook/gemini.ipynb)       |
| Google ADK     | Log traces for Google Agent Development Kit (ADK)                   | [Documentation](https://www.comet.com/docs/opik/tracing/integrations/google_adk?utm_source=opik&utm_medium=github&utm_content=google_adk_link&utm_campaign=opik)     | (*Coming Soon*)                                                                                                                                                                                                                    |
| Groq           | Log traces for Groq LLM calls                                       | [Documentation](https://www.comet.com/docs/opik/tracing/integrations/groq?utm_source=opik&utm_medium=github&utm_content=groq_link&utm_campaign=opik)                 | [![Open Quickstart In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/comet-ml/opik/blob/main/apps/opik-documentation/documentation/docs/cookbook/groq.ipynb)         |
| Guardrails     | Log traces for Guardrails AI validations                            | [Documentation](https://www.comet.com/docs/opik/tracing/integrations/guardrails/?utm_source=opik&utm_medium=github&utm_content=guardrails_link&utm_campaign=opik)    | [![Open Quickstart In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/comet-ml/opik/blob/main/apps/opik-documentation/documentation/docs/cookbook/guardrails-ai.ipynb)|
| Haystack       | Log traces for Haystack calls                                       | [Documentation](https://www.comet.com/docs/opik/tracing/integrations/haystack/?utm_source=opik&utm_medium=github&utm_content=haystack_link&utm_campaign=opik)        | [![Open Quickstart In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/comet-ml/opik/blob/main/apps/opik-documentation/documentation/docs/cookbook/haystack.ipynb)     |
| Instructor     | Log traces for LLM calls made with Instructor                       | [Documentation](https://www.comet.com/docs/opik/tracing/integrations/instructor/?utm_source=opik&utm_medium=github&utm_content=instructor_link&utm_campaign=opik)    | [![Open Quickstart In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/comet-ml/opik/blob/main/apps/opik-documentation/documentation/docs/cookbook/instructor.ipynb)   |
| LangChain      | Log traces for LangChain LLM calls                                  | [Documentation](https://www.comet.com/docs/opik/tracing/integrations/langchain/?utm_source=opik&utm_medium=github&utm_content=langchain_link&utm_campaign=opik)      | [![Open Quickstart In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/comet-ml/opik/blob/main/apps/opik-documentation/documentation/docs/cookbook/langchain.ipynb)    |
| LangChain JS   | Log traces for LangChain JS LLM calls                               | [Documentation](https://www.comet.com/docs/opik/tracing/integrations/langchainjs/?utm_source=opik&utm_medium=github&utm_content=langchain_link&utm_campaign=opik)    | (*Coming Soon*)                                                                                                                                                                                                                    |
| LangGraph      | Log traces for LangGraph executions                                 | [Documentation](https://www.comet.com/docs/opik/tracing/integrations/langgraph/?utm_source=opik&utm_medium=github&utm_content=langchain_link&utm_campaign=opik)      | [![Open Quickstart In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/comet-ml/opik/blob/main/apps/opik-documentation/documentation/docs/cookbook/langgraph.ipynb)    |
| LiteLLM        | Log traces for LiteLLM model calls                                  | [Documentation](https://www.comet.com/docs/opik/tracing/integrations/litellm/?utm_source=opik&utm_medium=github&utm_content=openai_link&utm_campaign=opik)           | [![Open Quickstart In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/comet-ml/opik/blob/main/apps/opik-documentation/documentation/docs/cookbook/litellm.ipynb)      |
| LlamaIndex     | Log traces for LlamaIndex LLM calls                                 | [Documentation](https://www.comet.com/docs/opik/tracing/integrations/llama_index?utm_source=opik&utm_medium=github&utm_content=llama_index_link&utm_campaign=opik)   | [![Open Quickstart In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/comet-ml/opik/blob/main/apps/opik-documentation/documentation/docs/cookbook/llama-index.ipynb)  |
| Ollama         | Log traces for Ollama LLM calls                                     | [Documentation](https://www.comet.com/docs/opik/tracing/integrations/ollama?utm_source=opik&utm_medium=github&utm_content=ollama_link&utm_campaign=opik)             | [![Open Quickstart In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/comet-ml/opik/blob/main/apps/opik-documentation/documentation/docs/cookbook/ollama.ipynb)       |
| OpenAI         | Log traces for OpenAI LLM calls                                     | [Documentation](https://www.comet.com/docs/opik/tracing/integrations/openai/?utm_source=opik&utm_medium=github&utm_content=openai_link&utm_campaign=opik)            | [![Open Quickstart In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/comet-ml/opik/blob/main/apps/opik-documentation/documentation/docs/cookbook/openai.ipynb)       |
| OpenAI Agents  | Log traces for OpenAI Agents SDK calls                              | [Documentation](https://www.comet.com/docs/opik/tracing/integrations/openai_agents/?utm_source=opik&utm_medium=github&utm_content=openai_link&utm_campaign=opik)     | [![Open Quickstart In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/comet-ml/opik/blob/main/apps/opik-documentation/documentation/docs/cookbook/openai-agents.ipynb)    |
| OpenRouter     | Log traces for OpenRouter LLM calls                                 | [Documentation](https://www.comet.com/docs/opik/tracing/openrouter/overview//?utm_source=opik&utm_medium=github&utm_content=openai_link&utm_campaign=opik)           | (*Coming Soon*)                                                                                                                                                                                                                    |
| OpenTelemetry  | Log traces for OpenTelemetry supported calls                        | [Documentation](https://www.comet.com/docs/opik/tracing/opentelemetry/overview//?utm_source=opik&utm_medium=github&utm_content=openai_link&utm_campaign=opik)        | (*Coming Soon*)                                                                                                                                                                                                                    |
| Predibase      | Log traces for Predibase LLM calls                                  | [Documentation](https://www.comet.com/docs/opik/tracing/integrations/predibase?utm_source=opik&utm_medium=github&utm_content=predibase_link&utm_campaign=opik)       | [![Open Quickstart In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/comet-ml/opik/blob/main/apps/opik-documentation/documentation/docs/cookbook/predibase.ipynb)    |
| Pydantic AI    | Log traces for PydanticAI agent calls                               | [Documentation](https://www.comet.com/docs/opik/tracing/integrations/pydantic-ai?utm_source=opik&utm_medium=github&utm_content=predibase_link&utm_campaign=opik)     | [![Open Quickstart In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/comet-ml/opik/blob/main/apps/opik-documentation/documentation/docs/cookbook/pydantic-ai.ipynb)  |
| Ragas          | Log traces for Ragas evaluations                                    | [Documentation](https://www.comet.com/docs/opik/tracing/integrations/ragas?utm_source=opik&utm_medium=github&utm_content=pydantic_ai_link&utm_campaign=opik)         | [![Open Quickstart In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/comet-ml/opik/blob/main/apps/opik-documentation/documentation/docs/cookbook/ragas.ipynb)        |
| Smolagents     | Log traces for Smolagents agents                                    | [Documentation](https://www.comet.com/docs/opik/tracing/integrations/smolagents?utm_source=opik&utm_medium=github&utm_content=smolagents_link&utm_campaign=opik)     | [![Open Quickstart In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/comet-ml/opik/blob/main/apps/opik-documentation/documentation/docs/cookbook/smolagents.ipynb)  |
| Strands Agents | Log traces for Strands agents calls                                 | [Documentation](https://www.comet.com/docs/opik/tracing/integrations/strands-agents/?utm_source=opik&utm_medium=github&utm_content=langchain_link&utm_campaign=opik) | (*Coming Soon*)                                                                                                                                                                                                                    |
| Vercel AI      | Log traces for Vercel AI SDK calls                                  | [Documentation](https://www.comet.com/docs/opik/tracing/integrations/vercel-ai-sdk/?utm_source=opik&utm_medium=github&utm_content=langchain_link&utm_campaign=opik)  | (*Coming Soon*)                                                                                                                                                                                                                    |
| watsonx        | Log traces for IBM watsonx LLM calls                                | [Documentation](https://www.comet.com/docs/opik/tracing/integrations/watsonx?utm_source=opik&utm_medium=github&utm_content=watsonx_link&utm_campaign=opik)           | [![Open Quickstart In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/comet-ml/opik/blob/main/apps/opik-documentation/documentation/docs/cookbook/watsonx.ipynb)      |

If your framework isn't listed, you can also use the `track` function decorator.

### 🧑‍⚖️ LLM as a Judge Metrics

Use built-in and custom LLM-as-a-Judge metrics in your applications.

To use the LLM-as-a-Judge metrics, simply import the relevant metric and use the `score` function:

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

## 🔍 Evaluating your LLM Application

Utilize Datasets and Experiments to evaluate your LLM applications throughout the development process. Integrate evaluations into your CI/CD pipeline with our PyTest integration.

## ⭐ Star Us on GitHub

Show your support and help grow our community by giving us a star!

[![Star History Chart](https://api.star-history.com/svg?repos=comet-ml/opik&type=Date)](https://github.com/comet-ml/opik)

## 🤝 Contributing

Contribute to the Opik project by:

*   Submitting [bug reports](https://github.com/comet-ml/opik/issues) and [feature requests](https://github.com/comet-ml/opik/issues).
*   Reviewing and improving the [documentation](CONTRIBUTING.md).
*   Sharing your experience with Opik.
*   Upvoting [popular feature requests](https://github.com/comet-ml/opik/issues?q=is%3Aissue+is%3Aopen+label%3A%22enhancement%22).

For contribution guidelines, see [CONTRIBUTING.md](CONTRIBUTING.md).
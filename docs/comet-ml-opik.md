<div align="center">
    <b><a href="README.md">English</a> | <a href="readme_CN.md">简体中文</a> | <a href="readme_JP.md">日本語</a> | <a href="readme_KO.md">한국어</a></b>
</div>

<h1 align="center" style="border-bottom: none">
    <div>
        <a href="https://www.comet.com/site/products/opik/?from=llm&utm_source=opik&utm_medium=github&utm_content=header_img&utm_campaign=opik">
            <picture>
                <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/comet-ml/opik/refs/heads/main/apps/opik-documentation/documentation/static/img/logo-dark-mode.svg">
                <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/comet-ml/opik/refs/heads/main/apps/opik-documentation/documentation/static/img/opik-logo.svg">
                <img alt="Comet Opik logo" src="https://raw.githubusercontent.com/comet-ml/opik/refs/heads/main/apps/opik-documentation/documentation/static/img/opik-logo.svg" width="200" />
            </picture>
        </a>
        <br>
        Opik
    </div>
</h1>
<h2 align="center" style="border-bottom: none">Open-source LLM Evaluation Platform</h2>

<p align="center">
    <b>Supercharge your LLM applications with Opik, the open-source platform for comprehensive evaluation, monitoring, and optimization.</b>
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
</p>

<div align="center" style="margin-top: 1em; margin-bottom: 1em;">
    <a href="#-what-is-opik">🚀 What is Opik?</a> • <a href="#%EF%B8%8F-opik-server-installation">🛠️ Opik Server Installation</a> • <a href="#-opik-client-sdk">💻 Opik Client SDK</a> • <a href="#-logging-traces-with-integrations">📝 Logging Traces</a><br>
    <a href="#-llm-as-a-judge-metrics">🧑‍⚖️ LLM as a Judge</a> • <a href="#-evaluating-your-llm-application">🔍 Evaluating your Application</a> • <a href="#-star-us-on-github">⭐ Star Us</a> • <a href="#-contributing">🤝 Contributing</a>
</div>

<br>

<a href="https://www.comet.com/signup?from=llm&utm_source=opik&utm_medium=github&utm_content=readme_banner&utm_campaign=opik">
    <img src="readme-thumbnail-new.png" alt="Opik platform screenshot" width="100%"/>
</a>

## 🚀 What is Opik?

Opik, developed by [Comet](https://www.comet.com?from=llm&utm_source=opik&utm_medium=github&utm_content=what_is_opik_link&utm_campaign=opik), is an open-source LLM evaluation platform designed to streamline the development, testing, and monitoring of Large Language Model (LLM) applications.  It offers a comprehensive suite of tools to help you build, evaluate, and optimize your LLM-powered systems, from RAG chatbots to code assistants.

**Key Features:**

*   **Comprehensive Observability:** Detailed tracing of LLM calls, conversation logging, and agent activity for deep insights into your applications.
*   **Advanced Evaluation:** Robust prompt evaluation, LLM-as-a-judge metrics, and experiment management to assess performance.
*   **Production-Ready:** Scalable monitoring dashboards and online evaluation rules for reliable production monitoring.
*   **Opik Agent Optimizer:** Dedicated SDK and optimizers to enhance prompts and agent performance.
*   **Opik Guardrails:**  Features to implement safe and responsible AI practices.

**Key Capabilities:**

*   **Development & Tracing:**
    *   Track all LLM calls and traces with detailed context during development and in production ([Quickstart](https://www.comet.com/docs/opik/quickstart/?from=llm&utm_source=opik&utm_medium=github&utm_content=quickstart_link&utm_campaign=opik)).
    *   Integrate seamlessly with popular frameworks like **Google ADK**, **Autogen**, and **Flowise AI** and many others. ([Integrations](https://www.comet.com/docs/opik/tracing/integrations/overview/?from=llm&utm_source=opik&utm_medium=github&utm_content=integrations_link&utm_campaign=opik))
    *   Annotate traces and spans with feedback scores using the [Python SDK](https://www.comet.com/docs/opik/tracing/annotate_traces/#annotating-traces-and-spans-using-the-sdk?from=llm&utm_source=opik&utm_medium=github&utm_content=sdk_link&utm_campaign=opik) or the [UI](https://www.comet.com/docs/opik/tracing/annotate_traces/#annotating-traces-through-the-ui?from=llm&utm_source=opik&utm_medium=github&utm_content=ui_link&utm_campaign=opik).
    *   Experiment with prompts and models in the [Prompt Playground](https://www.comet.com/docs/opik/prompt_engineering/playground).

*   **Evaluation & Testing:**
    *   Automate your LLM application evaluation with [Datasets](https://www.comet.com/docs/opik/evaluation/manage_datasets/?from=llm&utm_source=opik&utm_medium=github&utm_content=datasets_link&utm_campaign=opik) and [Experiments](https://www.comet.com/docs/opik/evaluation/evaluate_your_llm/?from=llm&utm_source=opik&utm_medium=github&utm_content=eval_link&utm_campaign=opik).
    *   Leverage powerful LLM-as-a-judge metrics for tasks like [hallucination detection](https://www.comet.com/docs/opik/evaluation/metrics/hallucination/?from=llm&utm_source=opik&utm_medium=github&utm_content=hallucination_link&utm_campaign=opik), [moderation](https://www.comet.com/docs/opik/evaluation/metrics/moderation/?from=llm&utm_source=opik&utm_medium=github&utm_content=moderation_link&utm_campaign=opik), and RAG assessment.
    *   Integrate evaluations into your CI/CD pipeline with our [PyTest integration](https://www.comet.com/docs/opik/testing/pytest_integration/?from=llm&utm_source=opik&utm_medium=github&utm_content=pytest_link&utm_campaign=opik).

*   **Production Monitoring & Optimization:**
    *   Log high volumes of production traces: Opik is designed for scale (40M+ traces/day).
    *   Monitor feedback scores, trace counts, and token usage over time in the [Opik Dashboard](https://www.comet.com/docs/opik/production/production_monitoring/?from=llm&utm_source=opik&utm_medium=github&utm_content=dashboard_link&utm_campaign=opik).
    *   Utilize [Online Evaluation Rules](https://www.comet.com/docs/opik/production/rules/?from=llm&utm_source=opik&utm_medium=github&utm_content=dashboard_link&utm_campaign=opik) with LLM-as-a-Judge metrics to identify production issues.
    *   Leverage **Opik Agent Optimizer** and **Opik Guardrails** to continuously improve and secure your LLM applications in production.

> [!TIP]
> If you are looking for features that Opik doesn't have today, please raise a new [Feature request](https://github.com/comet-ml/opik/issues/new/choose) 🚀

<br>

## 🛠️ Opik Server Installation

Get started with Opik quickly using one of the following options:

### Option 1: Comet.com Cloud (Recommended)

The easiest way to use Opik is to sign up for a free Comet account.  Get instant access without any setup.

👉 [Create your free Comet account](https://www.comet.com/signup?from=llm&utm_source=opik&utm_medium=github&utm_content=install_create_link&utm_campaign=opik)

### Option 2: Self-Host Opik

Deploy Opik on your own infrastructure for full control, using either Docker or Kubernetes.

#### Self-Hosting with Docker Compose (Local Development)

Simple, local setup using Docker Compose.  Use the new `./opik.sh` installation script:

On Linux or Mac:

```bash
# Clone the Opik repository
git clone https://github.com/comet-ml/opik.git

# Navigate to the repository
cd opik

# Start the Opik platform
./opik.sh
```

On Windows:

```powershell
# Clone the Opik repository
git clone https://github.com/comet-ml/opik.git

# Navigate to the repository
cd opik

# Start the Opik platform
powershell -ExecutionPolicy ByPass -c ".\\opik.ps1"
```

See the [Local Deployment Guide](https://www.comet.com/docs/opik/self-host/local_deployment?from=llm&utm_source=opik&utm_medium=github&utm_content=self_host_link&utm_campaign=opik) for detailed instructions.

#### Self-Hosting with Kubernetes & Helm (Scalable Deployments)

For production deployments, use the Helm chart for Kubernetes.  Click the badge for the full [Kubernetes Installation Guide using Helm](https://www.comet.com/docs/opik/self-host/kubernetes/#kubernetes-installation?from=llm&utm_source=opik&utm_medium=github&utm_content=kubernetes_link&utm_campaign=opik).

[![Kubernetes](https://img.shields.io/badge/Kubernetes-%23326ce5.svg?&logo=kubernetes&logoColor=white)](https://www.comet.com/docs/opik/self-host/kubernetes/#kubernetes-installation?from=llm&utm_source=opik&utm_medium=github&utm_content=kubernetes_link&utm_campaign=opik)

> [!IMPORTANT]
> **Version 1.7.0 Changes**:  Check the [changelog](https://github.com/comet-ml/opik/blob/main/CHANGELOG.md) for important updates.

## 💻 Opik Client SDK

Opik provides client libraries and a REST API to interact with the Opik server, with Python, TypeScript, and Ruby (via OpenTelemetry) SDKs available for easy integration.  See the [Opik Client Reference Documentation](apps/opik-documentation/documentation/fern/docs/reference/overview.mdx) for details.

### Python SDK Quick Start

Install the Python SDK:

```bash
# install using pip
pip install opik

# or install with uv
uv pip install opik
```

Configure the Python SDK using the `opik configure` command:

```bash
opik configure
```

> [!TIP]
> You can also configure the SDK in your Python code with `opik.configure(use_local=True)` for local setups, or by providing your API key and workspace for Comet.com. See the [Python SDK documentation](apps/opik-documentation/documentation/fern/docs/reference/python-sdk/) for more options.

You can now begin logging traces with the [Python SDK](https://www.comet.com/docs/opik/python-sdk-reference/?from=llm&utm_source=opik&utm_medium=github&utm_content=sdk_link2&utm_campaign=opik).

### 📝 Logging Traces with Integrations

Opik offers direct integrations with many popular frameworks.

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

> [!TIP]
> If your framework isn't listed, [open an issue](https://github.com/comet-ml/opik/issues) or submit a PR with the integration.

Alternatively, use the `track` function decorator for manual tracing:

```python
import opik

opik.configure(use_local=True) # Run locally

@opik.track
def my_llm_function(user_question: str) -> str:
    # Your LLM code here

    return "Hello"
```

> [!TIP]
> The track decorator works with all integrations and can track nested calls.

### 🧑‍⚖️ LLM as a Judge metrics

Opik includes LLM-as-a-Judge metrics to evaluate your application, learn more in the [metrics documentation](https://www.comet.com/docs/opik/evaluation/metrics/overview/?from=llm&utm_source=opik&utm_medium=github&utm_content=metrics_2_link&utm_campaign=opik).

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

Opik offers pre-built and custom heuristic metrics, read more in the [metrics documentation](https://www.comet.com/docs/opik/evaluation/metrics/overview?from=llm&utm_source=opik&utm_medium=github&utm_content=metrics_3_link&utm_campaign=opik).

### 🔍 Evaluating your LLM Application

Evaluate your LLM application with [Datasets](https://www.comet.com/docs/opik/evaluation/manage_datasets/?from=llm&utm_source=opik&utm_medium=github&utm_content=datasets_2_link&utm_campaign=opik) and [Experiments](https://www.comet.com/docs/opik/evaluation/evaluate_your_llm/?from=llm&utm_source=opik&utm_medium=github&utm_content=experiments_link&utm_campaign=opik). Enhance your experiments using the Opik Dashboard and run evaluations as part of your CI/CD using our [PyTest integration](https://www.comet.com/docs/opik/testing/pytest_integration/?from=llm&utm_source=opik&utm_medium=github&utm_content=pytest_2_link&utm_campaign=opik).

## ⭐ Star Us on GitHub

Show your support by giving Opik a star!

[![Star History Chart](https://api.star-history.com/svg?repos=comet-ml/opik&type=Date)](https://github.com/comet-ml/opik)

##
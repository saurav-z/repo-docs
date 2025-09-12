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

<div align="center" style="margin-top: 1em; margin-bottom: 1em;">
<a href="#-what-is-opik">🚀 What is Opik?</a> • <a href="#%EF%B8%8F-opik-server-installation">🛠️ Opik Server Installation</a> • <a href="#-opik-client-sdk">💻 Opik Client SDK</a> • <a href="#-logging-traces-with-integrations">📝 Logging Traces</a><br>
<a href="#-llm-as-a-judge-metrics">🧑‍⚖️ LLM as a Judge</a> • <a href="#-evaluating-your-llm-application">🔍 Evaluating your Application</a> • <a href="#-star-us-on-github">⭐ Star Us</a> • <a href="#-contributing">🤝 Contributing</a>
</div>

<br>

[![Opik platform screenshot (thumbnail)](readme-thumbnail-new.png)](https://www.comet.com/signup?from=llm&utm_source=opik&utm_medium=github&utm_content=readme_banner&utm_campaign=opik)

## 🚀 Supercharge Your LLM Applications with Opik: The Open-Source Evaluation Platform

Opik, built by [Comet](https://www.comet.com?from=llm&utm_source=opik&utm_medium=github&utm_content=what_is_opik_link&utm_campaign=opik), is your all-in-one solution for building, evaluating, and optimizing Large Language Model (LLM) applications, making them run better, faster, and more affordably.

**Key Features:**

*   **Comprehensive Tracing and Observability:**
    *   Deeply trace LLM calls and agent activity.
    *   Log conversations for detailed context.
    *   Integrate seamlessly with popular frameworks like Google ADK, Autogen, Flowise AI and more.
    *   Annotate traces and spans with feedback scores using the Python SDK or the UI.
    *   Experiment with prompts and models in the Prompt Playground.

*   **Advanced Evaluation and Testing:**
    *   Robust prompt evaluation.
    *   Leverage LLM-as-a-judge metrics for complex assessments (hallucination detection, moderation, RAG assessment).
    *   Manage datasets and experiments to automate your LLM application evaluation.
    *   Integrate evaluations into your CI/CD pipeline with our PyTest integration.

*   **Production Monitoring and Optimization:**
    *   Monitor feedback scores, trace counts, and token usage over time.
    *   Utilize online evaluation rules with LLM-as-a-Judge metrics to identify production issues.
    *   Scale to handle high volumes of production traces (40M+ traces/day).
    *   Utilize **Opik Agent Optimizer** and **Opik Guardrails** for continuous improvement and security.

> [!TIP]
> If you are looking for features that Opik doesn't have today, please raise a new [Feature request](https://github.com/comet-ml/opik/issues/new/choose) 🚀

## 🛠️ Opik Server Installation

Get started with Opik quickly! Choose the best option for your needs:

### Option 1: Comet.com Cloud (Easiest & Recommended)

Access Opik instantly with a free Comet account.  No setup required.

👉 [Create your free Comet account](https://www.comet.com/signup?from=llm&utm_source=opik&utm_medium=github&utm_content=install_create_link&utm_campaign=opik)

### Option 2: Self-Host Opik

Deploy Opik on your infrastructure for full control. Choose from Docker or Kubernetes.

#### Self-Hosting with Docker Compose (for Local Development & Testing)

Run a local Opik instance easily using Docker Compose.

**Installation Script:**

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

Use `--help` or `--info` for troubleshooting.  Ensure containers run as non-root users for security.  Access Opik at [localhost:5173](http://localhost:5173). See the [Local Deployment Guide](https://www.comet.com/docs/opik/self-host/local_deployment?from=llm&utm_source=opik&utm_medium=github&utm_content=self_host_link&utm_campaign=opik) for detailed instructions.

#### Self-Hosting with Kubernetes & Helm (for Scalable Deployments)

Deploy Opik on a Kubernetes cluster using our Helm chart for production environments.

[![Kubernetes](https://img.shields.io/badge/Kubernetes-%23326ce5.svg?&logo=kubernetes&logoColor=white)](https://www.comet.com/docs/opik/self-host/kubernetes/#kubernetes-installation?from=llm&utm_source=opik&utm_medium=github&utm_content=kubernetes_link&utm_campaign=opik)

> [!IMPORTANT]
> **Version 1.7.0 Changes**: Please review the [changelog](https://github.com/comet-ml/opik/blob/main/CHANGELOG.md) for important updates and breaking changes.

## 💻 Opik Client SDK

Opik provides client libraries and a REST API for interacting with the Opik server. SDKs are available for Python, TypeScript, and Ruby (via OpenTelemetry).  Refer to the [Opik Client Reference Documentation](apps/opik-documentation/documentation/fern/docs/reference/overview.mdx) for details.

### Python SDK Quick Start

Get started with the Python SDK:

**Installation:**

```bash
# install using pip
pip install opik

# or install with uv
uv pip install opik
```

**Configuration:**  Run `opik configure` to set up the SDK with your Opik server address or Comet.com API key and workspace.

```bash
opik configure
```

> [!TIP]
> You can also use `opik.configure(use_local=True)` for local setups or provide your Comet.com credentials directly in your Python code.  See the [Python SDK documentation](apps/opik-documentation/documentation/fern/docs/reference/python-sdk/) for details.

Start logging traces using the [Python SDK](https://www.comet.com/docs/opik/python-sdk-reference/?from=llm&utm_source=opik&utm_medium=github&utm_content=sdk_link2&utm_campaign=opik).

### 📝 Logging Traces with Integrations

Easily log traces using our direct integrations with a wide range of frameworks, including Google ADK, Autogen, Flowise AI, and many more:

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
| Cohere         | Log traces for Cohere LLM calls                         | [Documentation](https://www.comet.com/docs/opik/tracing/integrations/cohere?utm_source=opik&utm_medium=github&utm_content=cohere_link&utm_campaign=opik)                 | (_Coming Soon_)                                                                                                                                                                                                                            |
| CrewAI         | Log traces for CrewAI calls                             | [Documentation](https://www.comet.com/docs/opik/tracing/integrations/crewai?utm_source=opik&utm_medium=github&utm_content=crewai_link&utm_campaign=opik)                 | [![Open Quickstart In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/comet-ml/opik/blob/main/apps/opik-documentation/documentation/docs/cookbook/crewai.ipynb)                 |
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
| LangChain      | Log traces for LangChain LLM calls                      | [Documentation](https://www.comet.com/docs/opik/tracing/integrations/langchain?utm_source=opik&utm_medium=github&utm_content=langchain_link&utm_campaign=opik)           | [![Open Quickstart In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/comet-ml/opik/blob/main/apps/opik-documentation/documentation/docs/cookbook/langchain.ipynb)              |
| LangChainJS    | Log traces for LangChainJS LLM calls                    | [Documentation](https://www.comet.com/docs/opik/tracing/integrations/langchainjs?utm_source=opik&utm_medium=github&utm_content=langchainjs_link&utm_campaign=opik)       | (_Coming Soon_)                                                                                                                                                                                                                            |
| LangGraph      | Log traces for LangGraph executions                     | [Documentation](https://www.comet.com/docs/opik/tracing/integrations/langgraph?utm_source=opik&utm_medium=github&utm_content=langgraph_link&utm_campaign=opik)           | [![Open Quickstart In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/comet-ml/opik/blob/main/apps/opik-documentation/documentation/docs/cookbook/langgraph.ipynb)              |
| LiteLLM        | Log traces for LiteLLM model calls                      | [Documentation](https://www.comet.com/docs/opik/tracing/integrations/litellm?utm_source=opik&utm_medium=github&utm_content=litellm_link&utm_campaign=opik)               | [![Open Quickstart In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/comet-ml/opik/blob/main/apps/opik-documentation/documentation/docs/cookbook/litellm.ipynb)                |
| LiveKit Agents | Log traces for LiveKit Agents AI agent framework calls  | [Documentation](https://www.comet.com/docs/opik/tracing/integrations/livekit?utm_source=opik&utm_medium=github&utm_content=livekit_link&utm_campaign=opik)             | (_Coming Soon_)                                                                                                                                                                                                                            |
| LlamaIndex     | Log traces for LlamaIndex LLM calls                     | [Documentation](https://www.comet.com/docs/opik/tracing/integrations/llama_index?utm_source=opik&utm_medium=github&utm_content=llama_index_link&utm_campaign=opik)       | [![Open Quickstart In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/comet-ml/opik/blob/main/apps/opik-documentation/documentation/docs/cookbook/llama-index.ipynb)            |
| Mastra         | Log traces for Mastra AI workflow framework calls       | [Documentation](https://www.comet.com/docs/opik/tracing/integrations/mastra?utm_source=opik&utm_medium=github&utm_content=mastra_link&utm_campaign=opik)               | (_Coming Soon_)                                                                                                                                                                                                                            |
| Mistral AI     | Log traces for Mistral AI LLM calls                     | [Documentation](https://www.comet.com/docs/opik/tracing/integrations/mistral?utm_source=opik&utm_medium=github&utm_content=mistral_link&utm_campaign=opik)               | (_Coming Soon_)                                                                                                                                                                                                                            |
| Novita AI      | Log traces for Novita AI LLM calls                      | [Documentation](https://www.comet.com/docs/opik/tracing/integrations/novita-ai?utm_source=opik&utm_medium=github&utm_content=novita_ai_link&utm_campaign=opik)           | (_Coming Soon_)                                                                                                                                                                                                                            |
| Ollama         | Log traces for Ollama LLM calls                         | [Documentation](https://www.comet.com/docs/opik/tracing/integrations/ollama?utm_source=opik&utm_medium=github&utm_content=ollama_link&utm_campaign=opik)                 | [![Open Quickstart In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/comet-ml/opik/blob/main/apps/opik-documentation/documentation/docs/cookbook/ollama.ipynb)                 |
| OpenAI         | Log traces for OpenAI LLM calls                         | [Documentation](https://www.comet.com/docs/opik/tracing/integrations/openai?utm_source=opik&utm_medium=github&utm_content=openai_link&utm_campaign=opik)                 | [![Open Quickstart In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/comet-ml/opik/blob/main/apps/opik-documentation/documentation/docs/cookbook/openai.ipynb)                 |
| OpenAI Agents  | Log traces for OpenAI Agents SDK calls                  | [Documentation](https://www.comet.com/docs/opik/tracing/integrations/openai_agents?utm_source=opik&utm_medium=github&utm_content=openai_agents_link&utm_campaign=opik)   | [![Open Quickstart In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/comet-ml/opik/blob/main/apps/opik-documentation/documentation/docs/cookbook/openai-agents.ipynb)          |
| OpenRouter     | Log traces for OpenRouter LLM calls                     | [Documentation](https://www.comet.com/docs/opik/tracing/integrations/openrouter?utm_source=opik&utm_medium=github&utm_content=openrouter_link&utm_campaign=opik)         | (_Coming Soon_)                                                                                                                                                                                                                            |
| OpenTelemetry  | Log traces for OpenTelemetry supported calls            | [Documentation](https://www.comet.com/docs/opik/tracing/opentelemetry/overview?utm_source=opik&utm_medium=github&utm_content=opentelemetry_link&utm_campaign=opik)       | (_Coming Soon_)                                                                                                                                                                                                                            |
| Predibase      | Log traces for Predibase LLM calls                      | [Documentation](https://www.comet.com/docs/opik/tracing/integrations/predibase?utm_source=opik&utm_medium=github&utm_content=predibase_link&utm_campaign=opik)           | [![Open Quickstart In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/comet-ml/opik/blob/main/apps/opik-documentation/documentation/docs/cookbook/predibase.ipynb)              |
| Pydantic AI    | Log traces for PydanticAI agent calls                   | [Documentation](https://www.comet.com/docs/opik/tracing/integrations/pydantic-ai?utm_source=opik&utm_medium=github&utm_content=pydantic_ai_link&utm_campaign=opik)       | [![Open Quickstart In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/comet-ml/opik/blob/main/apps/opik-documentation/documentation/docs/cookbook/pydantic-ai.ipynb)            |
| Ragas          | Log traces for Ragas evaluations                        | [Documentation](https://www.comet.com/docs/opik/tracing/integrations/ragas?utm_source=opik&utm_medium=github&utm_content=ragas_link&utm_campaign=opik)                   | [![Open Quickstart In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/comet-ml/opik/blob/main/apps/opik-documentation/documentation/docs/cookbook/ragas.ipynb)                  |
| Semantic Kernel| Log traces for Microsoft Semantic Kernel calls          | [Documentation](https://www.comet.com/docs/opik/tracing/integrations/semantic-kernel?utm_source=opik&utm_medium=github&utm_content=semantic_kernel_link&utm_campaign=opik) | (_Coming Soon_)                                                                                                                                                                                                                            |
| Smolagents     | Log traces for Smolagents agents                        | [Documentation](https://www.comet.com/docs/opik/tracing/integrations/smolagents?utm_source=opik&utm_medium=github&utm_content=smolagents_link&utm_campaign=opik)         | [![Open Quickstart In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/comet-ml/opik/blob/main/apps/opik-documentation/documentation/docs/cookbook/smolagents.ipynb)             |
| Spring AI      | Log traces for Spring AI framework calls                 | [Documentation](https://www.comet.com/docs/opik/tracing/integrations/spring-ai?utm_source=opik&utm_medium=github&utm_content=spring_ai_link&utm_campaign=opik)         | (_Coming Soon_)                                                                                                                                                                                                                            |
| Strands Agents | Log traces for Strands agents calls                     | [Documentation](https://www.comet.com/docs/opik/tracing/integrations/strands-agents?utm_source=opik&utm_medium=github&utm_content=strands_agents_link&utm_campaign=opik) | (_Coming Soon_)                                                                                                                                                                                                                            |
| Together AI    | Log traces for Together AI LLM calls                    | [Documentation](https://www.comet.com/docs/opik/tracing/integrations/together-ai?utm_source=opik&utm_medium=github&utm_content=together_ai_link&utm_campaign=opik)       | (_Coming Soon_)                                                                                                                                                                                                                            |
| Vercel AI SDK  | Log traces for Vercel AI SDK calls                      | [Documentation](https://www.comet.com/docs/opik/tracing/integrations/vercel-ai-sdk?utm_source=opik&utm_medium=github&utm_content=vercel_ai_sdk_link&utm_campaign=opik)   | (_Coming Soon_)                                                                                                                                                                                                                            |
| VoltAgent      | Log traces for VoltAgent agent framework calls          | [Documentation](https://www.comet.com/docs/opik/tracing/integrations/voltagent?utm_source=opik&utm_medium=github&utm_content=voltagent_link&utm_campaign=opik)         | (_Coming Soon_)                                                                                                                                                                                                                            |
| WatsonX        | Log traces for IBM watsonx LLM calls                    | [Documentation](https://www.comet.com/docs/opik/tracing/integrations/watsonx?utm_source=opik&utm_medium=github&utm_content=watsonx_link&utm_campaign=opik)               | [![Open Quickstart In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/comet-ml/opik/blob/main/apps/opik-documentation/documentation/docs/cookbook/watsonx.ipynb)                |
| xAI Grok       | Log traces for xAI Grok LLM calls                       | [Documentation](https://www.comet.com/docs/opik/tracing/integrations/xai-grok?utm_source=opik&utm_medium=github&utm_content=xai-grok_link&utm_campaign=opik)             | (_Coming Soon_)                                                                                                                                                                                                                            |

> [!TIP]
> If your framework is not listed, [open an issue](https://github.com/comet-ml/opik/issues) or submit a PR to add an integration.

**Alternative: The `track` Function Decorator**

If you're not using one of our integrations, use the `@opik.track` decorator to log traces:

```python
import opik

opik.configure(use_local=True) # Run locally

@opik.track
def my_llm_function(user_question: str) -> str:
    # Your LLM code here

    return "Hello"
```

> [!TIP]
> The track decorator can be used in conjunction with any of our integrations and can also be used to track nested function calls.

### 🧑‍⚖️ LLM as a Judge Metrics

Utilize the Python Opik SDK's LLM-as-a-judge metrics for evaluating your LLM application. Find more details in the [metrics documentation](https://www.comet.com/docs/opik/evaluation/metrics/overview/?from=llm&utm_source=opik&utm_medium=github&utm_content=metrics_2_link&utm_campaign=opik).

**Example:**

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

Opik offers pre-built and customizable heuristic metrics. Explore the [metrics documentation](https://www.comet.com/docs/opik/evaluation/metrics/overview?from=llm&utm_source=opik&utm_medium=github&utm_content=metrics_3_link&utm_campaign=opik).

### 🔍 Evaluating your LLM Application

Evaluate your LLM applications during development using [Datasets](https://www.comet.com/docs/opik/evaluation/manage_datasets/?from=llm&utm
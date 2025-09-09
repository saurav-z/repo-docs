<div align="center">
    <a href="https://github.com/comet-ml/opik">
        <img src="https://raw.githubusercontent.com/comet-ml/opik/refs/heads/main/apps/opik-documentation/documentation/static/img/opik-logo.svg" alt="Comet Opik logo" width="200" />
    </a>
    <br>
    <a href="https://github.com/comet-ml/opik"><b>Opik</b></a>
</div>

<h1 align="center" style="border-bottom: none">
    Open-Source LLM Evaluation, Monitoring, and Optimization Platform
</h1>

**Opik is your all-in-one platform for building, evaluating, and optimizing LLM-powered applications.** From tracing to advanced evaluation, Opik provides the tools you need to create high-performing, reliable, and cost-effective LLM systems.

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
    <a href="#-key-features">🚀 Key Features</a> • <a href="#-installation">🛠️ Installation</a> • <a href="#-python-sdk">💻 Python SDK</a> • <a href="#-logging-traces-with-integrations">📝 Integrations</a><br>
    <a href="#-llm-as-a-judge-metrics">🧑‍⚖️ LLM-as-a-Judge</a> • <a href="#-evaluating-your-llm-application">🔍 Evaluation</a> • <a href="#-star-us-on-github">⭐ Star Us</a> • <a href="#-contributing">🤝 Contribute</a>
</div>

<br>

[![Opik platform screenshot (thumbnail)](readme-thumbnail-new.png)](https://www.comet.com/signup?from=llm&utm_source=opik&utm_medium=github&utm_content=readme_banner&utm_campaign=opik)

## 🚀 Key Features

Opik offers a comprehensive suite of features to streamline your LLM development lifecycle:

*   **Comprehensive Observability:**
    *   Deep tracing of LLM calls and agent activity for complete visibility.
    *   Extensive 3rd-party integrations (e.g., **Google ADK, Autogen, Flowise AI**) for easy setup.
    *   Annotate traces with feedback via the [Python SDK](https://www.comet.com/docs/opik/tracing/annotate_traces/#annotating-traces-and-spans-using-the-sdk?from=llm&utm_source=opik&utm_medium=github&utm_content=sdk_link&utm_campaign=opik) or the [UI](https://www.comet.com/docs/opik/tracing/annotate_traces/#annotating-traces-through-the-ui?from=llm&utm_source=opik&utm_medium=github&utm_content=ui_link&utm_campaign=opik).
    *   Experiment with prompts and models in the [Prompt Playground](https://www.comet.com/docs/opik/prompt_engineering/playground).
*   **Advanced Evaluation:**
    *   Robust prompt evaluation, and experiment management.
    *   LLM-as-a-judge metrics for complex tasks like hallucination detection and moderation.
    *   Automated evaluation with [Datasets](https://www.comet.com/docs/opik/evaluation/manage_datasets/?from=llm&utm_source=opik&utm_medium=github&utm_content=datasets_link&utm_campaign=opik) and [Experiments](https://www.comet.com/docs/opik/evaluation/evaluate_your_llm/?from=llm&utm_source=opik&utm_medium=github&utm_content=eval_link&utm_campaign=opik).
    *   Integrate evaluations into your CI/CD pipeline with our [PyTest integration](https://www.comet.com/docs/opik/testing/pytest_integration/?from=llm&utm_source=opik&utm_medium=github&utm_content=pytest_link&utm_campaign=opik).
*   **Production-Ready Monitoring & Optimization:**
    *   Scalable monitoring dashboards for production deployments.
    *   Monitor feedback scores, trace counts, and token usage over time in the [Opik Dashboard](https://www.comet.com/docs/opik/production/production_monitoring/?from=llm&utm_source=opik&utm_medium=github&utm_content=dashboard_link&utm_campaign=opik).
    *   Online Evaluation Rules with LLM-as-a-Judge metrics to identify production issues.
    *   **Opik Agent Optimizer** and **Opik Guardrails** for continuous improvement and security.

>   [!TIP]
>   If you're looking for specific features not yet available, please submit a [feature request](https://github.com/comet-ml/opik/issues/new/choose).

## 🛠️ Installation

Get started with Opik quickly with these installation options:

### Comet.com Cloud (Recommended)

The easiest way to get started is to use the hosted version on Comet.

👉 [Create your free Comet account](https://www.comet.com/signup?from=llm&utm_source=opik&utm_medium=github&utm_content=install_create_link&utm_campaign=opik)

### Self-Hosting

For self-hosting, you can choose between Docker Compose (local development) or Kubernetes with Helm (for scalable deployments).

#### Docker Compose (Local Development)

1.  Clone the repository:

    ```bash
    git clone https://github.com/comet-ml/opik.git
    cd opik
    ```

2.  Run the installation script:

    *   Linux/Mac: `./opik.sh`
    *   Windows: `powershell -ExecutionPolicy ByPass -c ".\\opik.ps1"`
    *   Use `--help` or `--info` to troubleshoot issues.

3.  Access Opik at [localhost:5173](http://localhost:5173).  For detailed instructions, see the [Local Deployment Guide](https://www.comet.com/docs/opik/self-host/local_deployment?from=llm&utm_source=opik&utm_medium=github&utm_content=self_host_link&utm_campaign=opik).

#### Kubernetes & Helm (Scalable Deployments)

Install Opik on a Kubernetes cluster using our Helm chart.  See the full [Kubernetes Installation Guide using Helm](https://www.comet.com/docs/opik/self-host/kubernetes/#kubernetes-installation?from=llm&utm_source=opik&utm_medium=github&utm_content=kubernetes_link&utm_campaign=opik).

[![Kubernetes](https://img.shields.io/badge/Kubernetes-%23326ce5.svg?&logo=kubernetes&logoColor=white)](https://www.comet.com/docs/opik/self-host/kubernetes/#kubernetes-installation?from=llm&utm_source=opik&utm_medium=github&utm_content=kubernetes_link&utm_campaign=opik)

>   [!IMPORTANT]
>   **Version 1.7.0 Changes**: Check the [changelog](https://github.com/comet-ml/opik/blob/main/CHANGELOG.md) for important updates.

## 💻 Python SDK

Opik provides a Python SDK for easy integration.

### Installation

```bash
pip install opik
# or
uv pip install opik
```

### Configuration

Configure the SDK with:

```bash
opik configure
```

This will prompt you for your server address/API key. You can also configure via code: `opik.configure(use_local=True)` for a local setup.

For more details, see the [Python SDK documentation](apps/opik-documentation/documentation/fern/docs/reference/python-sdk/).

## 📝 Logging Traces with Integrations

Easily log traces using Opik's direct integrations with popular frameworks:

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
| xAI Grok       | Log traces for xAI Grok LLM calls                       | [Documentation](https://www.comet.com/docs/opik/tracing/integrations/xai-grok?utm_source=opik&utm_medium=github&utm_content=xai_grok_link&utm_campaign=opik)             | (_Coming Soon_)                                                                                                                                                                                                                            |

If your framework isn't listed, submit a [GitHub issue](https://github.com/comet-ml/opik/issues).

### `track` function Decorator (for Customization)

Alternatively, use the `@opik.track` decorator for custom logging:

```python
import opik

opik.configure(use_local=True)

@opik.track
def my_llm_function(user_question: str) -> str:
    # Your LLM code here
    return "Hello"
```

## 🧑‍⚖️ LLM-as-a-Judge Metrics

The Python SDK provides LLM-as-a-Judge metrics for evaluation. Learn more in the [metrics documentation](https://www.comet.com/docs/opik/evaluation/metrics/overview/?from=llm&utm_source=opik&utm_medium=github&utm_content=metrics_2_link&utm_campaign=opik).

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

Evaluate your application during development with Datasets and Experiments. Utilize the Opik Dashboard for enhanced experiment charts and large trace handling. Integrate evaluations into your CI/CD pipeline using our PyTest integration.

## ⭐ Star Us on GitHub

Show your support by starring the repository!  It helps us grow and improve Opik.

[![Star History Chart](https://api.star-history.com/svg?repos
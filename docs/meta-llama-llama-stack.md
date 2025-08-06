# Llama Stack: Build and Deploy Generative AI Applications with Ease

Llama Stack is a powerful framework that simplifies building and deploying AI applications, offering a unified API layer and flexible deployment options. Learn more at the [original repo](https://github.com/meta-llama/llama-stack).

## Key Features

*   **Unified API:** Standardizes core building blocks for Inference, RAG, Agents, Tools, Safety, Evals, and Telemetry.
*   **Plugin Architecture:** Supports diverse API implementations across various environments (local, on-premises, cloud, mobile).
*   **Prepackaged Distributions:** Offers ready-to-use solutions for rapid deployment.
*   **Multiple Developer Interfaces:** Includes CLI and SDKs for Python, Typescript, iOS, and Android.
*   **Standalone Applications:** Provides examples for building production-grade AI applications.
*   **Llama 4 Support:** Now supports the Llama 4 model.

## Benefits of Using Llama Stack

*   **Flexibility:** Choose your preferred infrastructure without code changes.
*   **Consistency:** Build and deploy AI applications with consistent behavior using unified APIs.
*   **Robust Ecosystem:** Integrates with distribution partners for tailored infrastructure and services.
*   **Focus on Innovation:** Reduces complexity, empowering developers to focus on building AI applications.

## Getting Started

### One-Line Installation

```bash
curl -LsSf https://github.com/meta-llama/llama-stack/raw/main/scripts/install.sh | bash
```

### Documentation and Resources

*   [Quick Start](https://llama-stack.readthedocs.io/en/latest/getting_started/index.html)
*   [Documentation](https://llama-stack.readthedocs.io/en/latest/index.html)
*   [Colab Notebook](./docs/getting_started.ipynb)
*   [Discord](https://discord.gg/llama-stack)

## API Providers

Llama Stack integrates with a wide range of API providers, offering diverse options for model deployment and functionality.

| API Provider Builder | Environments | Agents | Inference | VectorIO | Safety | Telemetry | Post Training | Eval | DatasetIO |
|:--------------------:|:------------:|:------:|:---------:|:--------:|:------:|:---------:|:-------------:|:----:|:--------:|
|    Meta Reference    | Single Node | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
|      SambaNova       | Hosted | | ✅ | | ✅ | | | | |
|       Cerebras       | Hosted | | ✅ | | | | | | |
|      Fireworks       | Hosted | ✅ | ✅ | ✅ | | | | | |
|     AWS Bedrock      | Hosted | | ✅ | | ✅ | | | | |
|       Together       | Hosted | ✅ | ✅ | | ✅ | | | | |
|         Groq         | Hosted | | ✅ | | | | | | |
|        Ollama        | Single Node | | ✅ | | | | | | |
|         TGI          | Hosted/Single Node | | ✅ | | | | | | |
|      NVIDIA NIM      | Hosted/Single Node | | ✅ | | ✅ | | | | |
|       ChromaDB       | Hosted/Single Node | | | ✅ | | | | | |
|        Milvus        | Hosted/Single Node | | | ✅ | | | | | |
|        Qdrant        | Hosted/Single Node | | | ✅ | | | | | |
|       Weaviate       | Hosted/Single Node | | | ✅ | | | | | |
|      SQLite-vec      | Single Node | | | ✅ | | | | | |
|      PG Vector       | Single Node | | | ✅ | | | | | |
|  PyTorch ExecuTorch  | On-device iOS | ✅ | ✅ | | | | | | |
|         vLLM         | Single Node | | ✅ | | | | | | |
|        OpenAI        | Hosted | | ✅ | | | | | | |
|      Anthropic       | Hosted | | ✅ | | | | | | |
|        Gemini        | Hosted | | ✅ | | | | | | |
|       WatsonX        | Hosted | | ✅ | | | | | | |
|     HuggingFace      | Single Node | | | | | | ✅ | | ✅ |
|      TorchTune       | Single Node | | | | | | ✅ | | |
|     NVIDIA NEMO      | Hosted | | ✅ | ✅ | | | ✅ | ✅ | ✅ |
|        NVIDIA        | Hosted | | | | | | ✅ | ✅ | ✅ |

>   **Note**: Additional providers are available through external packages. See [External Providers](https://llama-stack.readthedocs.io/en/latest/providers/external.html) documentation.

## Distributions

Choose from a selection of pre-configured distributions for various deployment scenarios.

| Distribution               | Llama Stack Docker                                                                            | Start This Distribution                                                                                                  |
| :------------------------- | :--------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------ |
| Starter Distribution       | [llamastack/distribution-starter](https://hub.docker.com/repository/docker/llamastack/distribution-starter/general)      | [Guide](https://llama-stack.readthedocs.io/en/latest/distributions/self_hosted_distro/starter.html)      |
| Meta Reference             | [llamastack/distribution-meta-reference-gpu](https://hub.docker.com/repository/docker/llamastack/distribution-meta-reference-gpu/general) | [Guide](https://llama-stack.readthedocs.io/en/latest/distributions/self_hosted_distro/meta-reference-gpu.html) |
| PostgreSQL       | [llamastack/distribution-postgres-demo](https://hub.docker.com/repository/docker/llamastack/distribution-postgres-demo/general)                |                  |

## Client SDKs

Llama Stack offers client SDKs in multiple languages for easy integration.

| Language   | Client SDK                                                      | Package                                                                   |
| :--------- | :-------------------------------------------------------------- | :------------------------------------------------------------------------ |
| Python     | [llama-stack-client-python](https://github.com/meta-llama/llama-stack-client-python) | [![PyPI version](https://img.shields.io/pypi/v/llama_stack_client.svg)](https://pypi.org/project/llama_stack_client/) |
| Swift      | [llama-stack-client-swift](https://github.com/meta-llama/llama-stack-client-swift)    | [![Swift Package Index](https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2Fmeta-llama%2Fllama-stack-client-swift%2Fbadge%3Ftype%3Dswift-versions)](https://swiftpackageindex.com/meta-llama/llama-stack-client-swift) |
| Typescript | [llama-stack-client-typescript](https://github.com/meta-llama/llama-stack-client-typescript) | [![NPM version](https://img.shields.io/npm/v/llama-stack-client.svg)](https://npmjs.org/package/llama-stack-client)   |
| Kotlin     | [llama-stack-client-kotlin](https://github.com/meta-llama/llama-stack-client-kotlin)   | [![Maven version](https://img.shields.io/maven-central/v/com.llama.llamastack/llama-stack-client-kotlin)](https://central.sonatype.com/artifact/com.llama.llamastack/llama-stack-client-kotlin) |
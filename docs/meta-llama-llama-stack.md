# Llama Stack: Build AI Applications with Ease

**Llama Stack simplifies AI application development by providing a unified API layer and flexible deployment options, enabling developers to build and deploy generative AI applications quickly and efficiently.  Explore Llama Stack on [GitHub](https://github.com/meta-llama/llama-stack).**

## Key Features & Benefits

*   **Unified API:** Streamlines inference, RAG, agent creation, tool integration, safety protocols, and telemetry, providing a consistent development experience.
*   **Plugin Architecture:** Supports a wide range of API implementations, including local development, on-premises, cloud, and mobile environments.
*   **Prepackaged Distributions:** Offers ready-to-use solutions for quick and reliable setup in any environment, including self-hosted and cloud options.
*   **Multiple Developer Interfaces:** Provides flexible options with CLI and SDKs for Python, Typescript, iOS, and Android.
*   **Example Applications:** Offers standalone applications as examples for production-grade AI applications with Llama Stack.
*   **Flexible Deployment:** Choose your infrastructure without changing APIs.
*   **Consistent Experience:** Unified APIs make it easier to build, test, and deploy AI applications with consistent application behavior.
*   **Robust Ecosystem:** Ready to integrate with distribution partners (cloud providers, hardware vendors, and AI-focused companies) to deploy Llama models.

## Quick Start

Get started quickly with Llama Stack.

*   **One-Line Installation:**
    ```bash
    curl -LsSf https://github.com/meta-llama/llama-stack/raw/main/scripts/install.sh | bash
    ```
*   **Quick Start Guide:** [Quick Start](https://llama-stack.readthedocs.io/en/latest/getting_started/index.html)
*   **Documentation:** [Documentation](https://llama-stack.readthedocs.io/en/latest/index.html)
*   **Colab Notebook:** [Colab Notebook](./docs/getting_started.ipynb)

## API Providers

Llama Stack supports a wide range of API providers. Choose the best option for your use case.

| API Provider Builder | Environments | Agents | Inference | VectorIO | Safety | Telemetry | Post Training | Eval | DatasetIO |
| :--------------------: | :------------: | :------: | :---------: | :--------: | :------: | :---------: | :-------------: | :----: | :--------: |
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

*   See the [full list](https://llama-stack.readthedocs.io/en/latest/providers/index.html) for the complete list of providers.
*   Additional providers are available through external packages. See [External Providers](https://llama-stack.readthedocs.io/en/latest/providers/external.html)

## Distributions

Llama Stack Distributions simplify deployments.

|               **Distribution**                |                                                                    **Llama Stack Docker**                                                                     |                                                 Start This Distribution                                                  |
|:---------------------------------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------:|
|                Starter Distribution                 |           [llamastack/distribution-starter](https://hub.docker.com/repository/docker/llamastack/distribution-starter/general)           |      [Guide](https://llama-stack.readthedocs.io/en/latest/distributions/self_hosted_distro/starter.html)      |
|                Meta Reference                 |           [llamastack/distribution-meta-reference-gpu](https://hub.docker.com/repository/docker/llamastack/distribution-meta-reference-gpu/general)           |      [Guide](https://llama-stack.readthedocs.io/en/latest/distributions/self_hosted_distro/meta-reference-gpu.html)      |
|                   PostgreSQL                  |                [llamastack/distribution-postgres-demo](https://hub.docker.com/repository/docker/llamastack/distribution-postgres-demo/general)                |                  |

## Client SDKs

Integrate Llama Stack into your preferred development environment.

*   **Python:** [llama-stack-client-python](https://github.com/meta-llama/llama-stack-client-python) [![PyPI version](https://img.shields.io/pypi/v/llama_stack_client.svg)](https://pypi.org/project/llama_stack_client/)
*   **Swift:** [llama-stack-client-swift](https://github.com/meta-llama/llama-stack-client-swift) [![Swift Package Index](https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2Fmeta-llama%2Fllama-stack-client-swift%2Fbadge%3Ftype%3Dswift-versions)](https://swiftpackageindex.com/meta-llama/llama-stack-client-swift)
*   **Typescript:** [llama-stack-client-typescript](https://github.com/meta-llama/llama-stack-client-typescript) [![NPM version](https://img.shields.io/npm/v/llama-stack-client.svg)](https://npmjs.org/package/llama-stack-client)
*   **Kotlin:** [llama-stack-client-kotlin](https://github.com/meta-llama/llama-stack-client-kotlin) [![Maven version](https://img.shields.io/maven-central/v/com.llama.llamastack/llama-stack-client-kotlin)](https://central.sonatype.com/artifact/com.llama.llamastack/llama-stack-client-kotlin)

## Documentation

Explore detailed documentation and guides for Llama Stack:

*   [CLI references](https://llama-stack.readthedocs.io/en/latest/references/index.html)
*   [Getting Started](https://llama-stack.readthedocs.io/en/latest/getting_started/index.html)
*   [Contributing](CONTRIBUTING.md)
*   [Adding a new API Provider](https://llama-stack.readthedocs.io/en/latest/contributing/new_api_provider.html)
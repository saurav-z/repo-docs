# Llama Stack: Build AI Applications Faster with a Unified API 

**Llama Stack simplifies AI application development by providing a unified API layer for inference, RAG, agents, and more.**  [Explore the Llama Stack repository](https://github.com/llamastack/llama-stack) to accelerate your generative AI projects!

## Key Features & Benefits

*   **Unified API Layer:** Standardizes core building blocks for AI application development, offering a consistent interface for various functionalities.
*   **Plugin Architecture:** Supports a diverse ecosystem of API implementations across various environments (local, cloud, on-premise, mobile).
*   **Prepackaged Distributions:**  Offers ready-to-use solutions for quick and reliable deployment in any environment, including a Starter Distribution, Meta Reference, and PostgreSQL.
*   **Multiple Developer Interfaces:** Provides CLI and SDKs for Python, Typescript, iOS, and Android to connect to a Llama Stack server.
*   **Standalone Applications:**  Includes examples to guide you in building production-ready AI applications.
*   **Flexible Deployment:** Developers can choose their preferred infrastructure without changing APIs.
*   **Consistent Experience:** Unified APIs make it easier to build, test, and deploy AI applications with consistent behavior.
*   **Robust Ecosystem:** Integrated with distribution partners for tailored infrastructure, software, and services to deploy Llama models.

## Get Started Quickly

*   **One-Line Installation:**  Easily try Llama Stack locally:

    ```bash
    curl -LsSf https://github.com/meta-llama/llama-stack/raw/main/scripts/install.sh | bash
    ```

*   **Comprehensive Documentation:** Dive into our resources, including:
    *   [Quick Start Guide](https://llama-stack.readthedocs.io/en/latest/getting_started/index.html)
    *   [CLI References](https://llama-stack.readthedocs.io/en/latest/references/llama_cli_reference/index.html)
    *   [SDKs for Python](https://github.com/meta-llama/llama-stack-client-python), [Swift](https://github.com/meta-llama/llama-stack-client-swift), [Typescript](https://github.com/meta-llama/llama-stack-client-typescript), and [Kotlin](https://github.com/meta-llama/llama-stack-client-kotlin).

## Supported API Providers

Llama Stack supports a wide array of API providers, enabling flexibility and choice for developers:

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

## Distributions

Choose from pre-configured distributions to suit your deployment needs:

|               **Distribution**                |                                                                    **Llama Stack Docker**                                                                     |                                                 Start This Distribution                                                  |
|:---------------------------------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------:|
|                Starter Distribution                 |           [llamastack/distribution-starter](https://hub.docker.com/repository/docker/llamastack/distribution-starter/general)           |      [Guide](https://llama-stack.readthedocs.io/en/latest/distributions/self_hosted_distro/starter.html)      |
|                Meta Reference                 |           [llamastack/distribution-meta-reference-gpu](https://hub.docker.com/repository/docker/llamastack/distribution-meta-reference-gpu/general)           |      [Guide](https://llama-stack.readthedocs.io/en/latest/distributions/self_hosted_distro/meta-reference-gpu.html)      |
|                   PostgreSQL                  |                [llamastack/distribution-postgres-demo](https://hub.docker.com/repository/docker/llamastack/distribution-postgres-demo/general)                |                  |

## Community & Contributing

*   [Discord](https://discord.gg/llama-stack)
*   [Contributing Guide](CONTRIBUTING.md)

## Explore and contribute to the future of AI application development with Llama Stack!
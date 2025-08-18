# Llama Stack: Build AI Applications Faster with a Unified API

**Llama Stack provides a standardized and flexible foundation for building and deploying generative AI applications**, enabling developers to easily leverage the power of Llama models and a wide range of AI services. Check out the [original repository](https://github.com/meta-llama/llama-stack) for more information.

## Key Features

*   **Unified API:** Standardized API layer for Inference, RAG, Agents, Tools, Safety, Evals, and Telemetry.
*   **Plugin Architecture:** Supports diverse API implementations across local development, on-premises, cloud, and mobile environments.
*   **Prepackaged Distributions:** Offers ready-to-use solutions for quick and reliable deployment.
*   **Multiple Developer Interfaces:** Supports CLI and SDKs for Python, Typescript, iOS, and Android.
*   **Standalone Applications:** Examples for building production-grade AI applications.
*   **Llama 4 Support:**  Ready to support latest models like Llama 4.

## Getting Started

### Quick Install

Get started with Llama Stack quickly:

```bash
curl -LsSf https://github.com/meta-llama/llama-stack/raw/main/scripts/install.sh | bash
```

### Key Components

*   **Unified API Layer:** A core building block that simplifies the development of AI applications.
*   **Plugin Architecture:** Supports the rich ecosystem of different API implementations in various environments.
*   **Prepackaged Verified Distributions:** A one-stop solution for developers to get started quickly and reliably in any environment.
*   **Multiple Developer Interfaces:** Including CLI and SDKs for Python, Typescript, iOS, and Android.

## Benefits of Using Llama Stack

*   **Flexible Options:** Choose your preferred infrastructure without changing APIs and enjoy flexible deployment choices.
*   **Consistent Experience:** Build, test, and deploy AI applications with consistent application behavior.
*   **Robust Ecosystem:** Pre-integrated with distribution partners for tailored infrastructure, software, and services.

## API Providers & Integrations

Llama Stack seamlessly integrates with a wide variety of API providers, giving you flexibility in choosing the best tools for your needs.

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

**Note:** Find more providers via [External Providers](https://llama-stack.readthedocs.io/en/latest/providers/external.html).

## Distributions

Easily get started with a specific deployment scenario.  Distributions allow you to start locally and seamlessly move to production without changing your application code.

|               **Distribution**                |                                                                    **Llama Stack Docker**                                                                     |                                                 Start This Distribution                                                  |
|:---------------------------------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------:|
|                Starter Distribution                 |           [llamastack/distribution-starter](https://hub.docker.com/repository/docker/llamastack/distribution-starter/general)           |      [Guide](https://llama-stack.readthedocs.io/en/latest/distributions/self_hosted_distro/starter.html)      |
|                Meta Reference                 |           [llamastack/distribution-meta-reference-gpu](https://hub.docker.com/repository/docker/llamastack/distribution-meta-reference-gpu/general)           |      [Guide](https://llama-stack.readthedocs.io/en/latest/distributions/self_hosted_distro/meta-reference-gpu.html)      |
|                   PostgreSQL                  |                [llamastack/distribution-postgres-demo](https://hub.docker.com/repository/docker/llamastack/distribution-postgres-demo/general)                |                  |

## Documentation

*   [Quick Start](https://llama-stack.readthedocs.io/en/latest/getting_started/index.html)
*   [CLI References](https://llama-stack.readthedocs.io/en/latest/references/llama_cli_reference/index.html)
*   [API Provider Documentation](https://llama-stack.readthedocs.io/en/latest/providers/index.html)
*   [Zero-to-Hero Guide](https://github.com/meta-llama/llama-stack/tree/main/docs/zero_to_hero_guide)
*   [Adding a new API Provider](https://llama-stack.readthedocs.io/en/latest/contributing/new_api_provider.html)
*   [Colab notebook](./docs/getting_started.ipynb)
*   The complete Llama Stack lesson [Colab notebook](https://colab.research.google.com/drive/1dtVmxotBsI4cGZQNsJRYPrLiDeT0Wnwt) of the new [Llama 3.2 course on Deeplearning.ai](https://learn.deeplearning.ai/courses/introducing-multimodal-llama-3-2/lesson/8/llama-stack).

## Client SDKs

Build applications in your preferred language:

*   **Python:** [llama-stack-client-python](https://github.com/meta-llama/llama-stack-client-python) ([PyPI](https://pypi.org/project/llama_stack_client/))
*   **Swift:** [llama-stack-client-swift](https://github.com/meta-llama/llama-stack-client-swift) ([Swift Package Index](https://swiftpackageindex.com/meta-llama/llama-stack-client-swift))
*   **Typescript:** [llama-stack-client-typescript](https://github.com/meta-llama/llama-stack-client-typescript) ([npm](https://npmjs.org/package/llama-stack-client))
*   **Kotlin:** [llama-stack-client-kotlin](https://github.com/meta-llama/llama-stack-client-kotlin) ([Maven](https://central.sonatype.com/artifact/com.llama.llamastack/llama-stack-client-kotlin))

## Community

*   [Discord](https://discord.gg/llama-stack)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=meta-llama/llama-stack&type=Date)](https://www.star-history.com/#meta-llama/llama-stack&Date)

## Contributing

We appreciate your contributions!  Please refer to [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## Contributors

Thanks to our amazing contributors!

<a href="https://github.com/meta-llama/llama-stack/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=meta-llama/llama-stack" />
</a>
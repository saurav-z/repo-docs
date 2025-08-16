# Llama Stack: Build Powerful AI Applications with Ease

**Llama Stack empowers developers to build and deploy generative AI applications quickly and efficiently, offering a unified API layer, flexible deployment options, and a robust ecosystem.**  You can find the original repo [here](https://github.com/llamastack/llama-stack).

## Key Features

*   **Unified API Layer:** Standardizes key building blocks for AI app development, including Inference, RAG, Agents, Tools, Safety, Evals, and Telemetry.
*   **Plugin Architecture:** Supports a wide range of API implementations across various environments (local, on-premises, cloud, mobile).
*   **Prepackaged Distributions:** Offers ready-to-use solutions for quick and reliable deployments in any environment.
*   **Multi-Interface Support:** Provides CLI and SDKs for Python, Typescript, iOS, and Android, making integration seamless.
*   **Standalone Applications:** Includes examples for building production-grade AI applications.
*   **Llama 4 Support:** Get started quickly with Llama 4 models.

## Getting Started

### Installation

Quickly try Llama Stack locally with a single command:

```bash
curl -LsSf https://github.com/meta-llama/llama-stack/raw/main/scripts/install.sh | bash
```

### Key Resources

*   **Quick Start Guide:** [https://llama-stack.readthedocs.io/en/latest/getting_started/index.html](https://llama-stack.readthedocs.io/en/latest/getting_started/index.html)
*   **Documentation:** [https://llama-stack.readthedocs.io/en/latest/index.html](https://llama-stack.readthedocs.io/en/latest/index.html)
*   **Colab Notebook:** [./docs/getting_started.ipynb](docs/getting_started.ipynb)

## Benefits of Using Llama Stack

*   **Flexible Infrastructure:** Choose your preferred infrastructure without changing your APIs.
*   **Consistent Application Behavior:** Unified APIs simplify building, testing, and deploying AI applications.
*   **Robust Ecosystem:** Integrates with partners offering tailored infrastructure, software, and services for deploying Llama models.

## API Providers

Llama Stack supports a wide variety of API providers, giving you the flexibility to choose the best fit for your needs. For a complete list, see [full list](https://llama-stack.readthedocs.io/en/latest/providers/index.html)

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

Llama Stack Distributions are pre-configured bundles of provider implementations for specific deployment scenarios.

|               **Distribution**                |                                                                    **Llama Stack Docker**                                                                     |                                                 Start This Distribution                                                  |
|:---------------------------------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------:|
|                Starter Distribution                 |           [llamastack/distribution-starter](https://hub.docker.com/repository/docker/llamastack/distribution-starter/general)           |      [Guide](https://llama-stack.readthedocs.io/en/latest/distributions/self_hosted_distro/starter.html)      |
|                Meta Reference                 |           [llamastack/distribution-meta-reference-gpu](https://hub.docker.com/repository/docker/llamastack/distribution-meta-reference-gpu/general)           |      [Guide](https://llama-stack.readthedocs.io/en/latest/distributions/self_hosted_distro/meta-reference-gpu.html)      |
|                   PostgreSQL                  |                [llamastack/distribution-postgres-demo](https://hub.docker.com/repository/docker/llamastack/distribution-postgres-demo/general)                |                  |

## SDKs

Connect to a Llama Stack server in your preferred language.  SDKs are available for:

*   **Python:** [https://github.com/meta-llama/llama-stack-client-python](https://github.com/meta-llama/llama-stack-client-python)
*   **Swift:** [https://github.com/meta-llama/llama-stack-client-swift](https://github.com/meta-llama/llama-stack-client-swift)
*   **Typescript:** [https://github.com/meta-llama/llama-stack-client-typescript](https://github.com/meta-llama/llama-stack-client-typescript)
*   **Kotlin:** [https://github.com/meta-llama/llama-stack-client-kotlin](https://github.com/meta-llama/llama-stack-client-kotlin)

## Community & Contributing

*   **Discord:** [https://discord.gg/llama-stack](https://discord.gg/llama-stack)
*   **Contributing:** [CONTRIBUTING.md](CONTRIBUTING.md)

##  GitHub Star History

<a href="https://www.star-history.com/#meta-llama/llama-stack&Date">
    <img src="https://api.star-history.com/svg?repos=meta-llama/llama-stack&type=Date" alt="Star History Chart">
</a>

## Contributors

<a href="https://github.com/meta-llama/llama-stack/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=meta-llama/llama-stack" alt="Contributors">
</a>
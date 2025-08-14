# Llama Stack: Build, Deploy, and Scale AI Applications with Ease

**Llama Stack** empowers developers to build, deploy, and scale generative AI applications efficiently, providing a unified API layer and a flexible plugin architecture. For more details, check out the original repo: [https://github.com/meta-llama/llama-stack](https://github.com/meta-llama/llama-stack)

## Key Features

*   **Unified API Layer:** Standardizes core building blocks for Inference, RAG, Agents, Tools, Safety, Evals, and Telemetry, simplifying AI application development.
*   **Plugin Architecture:** Supports diverse API implementations across various environments (local, on-premises, cloud, mobile).
*   **Prepackaged Distributions:** Offers one-stop solutions for quick and reliable deployments.
*   **Multiple Developer Interfaces:** Includes CLI and SDKs for Python, Typescript, iOS, and Android.
*   **Standalone Applications:** Provides examples for building production-grade AI applications.
*   **Llama 4 Support:** Seamlessly integrate and utilize the latest Llama 4 models.

## Benefits of Using Llama Stack

*   **Flexible Infrastructure Options:** Choose your preferred infrastructure without changing APIs.
*   **Consistent Experience:** Build, test, and deploy AI applications with predictable behavior.
*   **Robust Ecosystem:** Leverage integrations with distribution partners for tailored infrastructure, software, and services.

## Quick Start

To try Llama Stack locally, run:

```bash
curl -LsSf https://github.com/meta-llama/llama-stack/raw/main/scripts/install.sh | bash
```

## API Providers

Llama Stack supports a wide range of API providers.  Here is a list of API Providers and available distributions that can help developers get started easily with Llama Stack.
Please checkout for [full list](https://llama-stack.readthedocs.io/en/latest/providers/index.html)

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

> **Note:** Additional providers are available through external packages. See [External Providers](https://llama-stack.readthedocs.io/en/latest/providers/external.html) documentation.

## Distributions

Simplify deployment with pre-configured bundles for various scenarios.

| Distribution | Llama Stack Docker | Start This Distribution |
| :---------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------: |
| Starter Distribution | [llamastack/distribution-starter](https://hub.docker.com/repository/docker/llamastack/distribution-starter/general) | [Guide](https://llama-stack.readthedocs.io/en/latest/distributions/self_hosted_distro/starter.html) |
| Meta Reference | [llamastack/distribution-meta-reference-gpu](https://hub.docker.com/repository/docker/llamastack/distribution-meta-reference-gpu/general) | [Guide](https://llama-stack.readthedocs.io/en/latest/distributions/self_hosted_distro/meta-reference-gpu.html) |
| PostgreSQL | [llamastack/distribution-postgres-demo](https://hub.docker.com/repository/docker/llamastack/distribution-postgres-demo/general) | |

## Documentation

Comprehensive documentation is available to guide you:

*   [Quick Start Guide](https://llama-stack.readthedocs.io/en/latest/getting_started/index.html)
*   [CLI References](https://llama-stack.readthedocs.io/en/latest/references/index.html)
*   [Zero-to-Hero Guide](https://github.com/meta-llama/llama-stack/tree/main/docs/zero_to_hero_guide)
*   [Colab Notebook](./docs/getting_started.ipynb)

## Llama Stack Client SDKs

Choose your preferred language to connect to a Llama Stack server:

| Language | Client SDK | Package |
| :---: | :---: | :---: |
| Python | [llama-stack-client-python](https://github.com/meta-llama/llama-stack-client-python) | [![PyPI version](https://img.shields.io/pypi/v/llama_stack_client.svg)](https://pypi.org/project/llama_stack_client/) |
| Swift | [llama-stack-client-swift](https://github.com/meta-llama/llama-stack-client-swift) | [![Swift Package Index](https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2Fmeta-llama%2Fllama-stack-client-swift%2Fbadge%3Ftype%3Dswift-versions)](https://swiftpackageindex.com/meta-llama/llama-stack-client-swift) |
| Typescript | [llama-stack-client-typescript](https://github.com/meta-llama/llama-stack-client-typescript) | [![NPM version](https://img.shields.io/npm/v/llama-stack-client.svg)](https://npmjs.org/package/llama-stack-client) |
| Kotlin | [llama-stack-client-kotlin](https://github.com/meta-llama/llama-stack-client-kotlin) | [![Maven version](https://img.shields.io/maven-central/v/com.llama.llamastack/llama-stack-client-kotlin)](https://central.sonatype.com/artifact/com.llama.llamastack/llama-stack-client-kotlin) |

## Additional Resources

*   [llama-stack-apps Examples](https://github.com/meta-llama/llama-stack-apps/tree/main/examples)

## Contributors

Thank you to all our amazing contributors!
<a href="https://github.com/meta-llama/llama-stack/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=meta-llama/llama-stack" />
</a>
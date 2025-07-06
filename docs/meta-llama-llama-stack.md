# Llama Stack: Build Powerful AI Applications with Ease

Llama Stack simplifies AI application development by providing a unified API layer and a flexible plugin architecture for integrating with various AI models and providers, which you can find at the [original repo](https://github.com/meta-llama/llama-stack).

[![PyPI version](https://img.shields.io/pypi/v/llama_stack.svg)](https://pypi.org/project/llama_stack/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/llama-stack)](https://pypi.org/project/llama-stack/)
[![License](https://img.shields.io/pypi/l/llama_stack.svg)](https://github.com/meta-llama/llama-stack/blob/main/LICENSE)
[![Discord](https://img.shields.io/discord/1257833999603335178?color=6A7EC2&logo=discord&logoColor=ffffff)](https://discord.gg/llama-stack)
[![Unit Tests](https://github.com/meta-llama/llama-stack/actions/workflows/unit-tests.yml/badge.svg?branch=main)](https://github.com/meta-llama/llama-stack/actions/workflows/unit-tests.yml?query=branch%3Amain)
[![Integration Tests](https://github.com/meta-llama/llama-stack/actions/workflows/integration-tests.yml/badge.svg?branch=main)](https://github.com/meta-llama/llama-stack/actions/workflows/integration-tests.yml?query=branch%3Amain)

**Key Features:**

*   **Unified API Layer:** Standardizes core building blocks for inference, RAG, agents, tools, safety, evals, and telemetry.
*   **Plugin Architecture:** Supports a diverse ecosystem of API implementations across various environments (local, cloud, mobile).
*   **Prepackaged Distributions:** Offers ready-to-use solutions for quick and reliable AI application development.
*   **Multiple Developer Interfaces:** Provides CLI and SDKs for Python, Typescript, iOS, and Android.
*   **Standalone Applications:** Includes examples for building production-grade AI applications.
*   **Llama 4 Support:** Supports the latest Llama 4 models.

### Quick Start

*   [Quick Start Guide](https://llama-stack.readthedocs.io/en/latest/getting_started/index.html)
*   [Documentation](https://llama-stack.readthedocs.io/en/latest/index.html)
*   [Colab Notebook](https://llama-stack.readthedocs.io/en/latest/index.html)
*   [Discord](https://discord.gg/llama-stack)

### One-Line Installer

Get started quickly by running the following command:

```bash
curl -LsSf https://github.com/meta-llama/llama-stack/raw/main/install.sh | bash
```

### Benefits of Using Llama Stack:

*   **Flexibility:** Choose your preferred infrastructure without changing APIs, enabling flexible deployment options.
*   **Consistency:** Unified APIs ensure consistent behavior across AI applications.
*   **Robust Ecosystem:** Integrates with leading distribution partners for tailored infrastructure and services.

### API Providers

Llama Stack seamlessly integrates with various API providers. Explore the full list of supported providers [here](https://llama-stack.readthedocs.io/en/latest/providers/index.html).

| API Provider Builder | Environments | Agents | Inference | VectorIO | Safety | Telemetry | Post Training | Eval | DatasetIO |
|:-------------------:|:------------:|:------:|:---------:|:--------:|:------:|:---------:|:-------------:|:----:|:--------:|
| Meta Reference | Single Node | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| SambaNova | Hosted | | ✅ | | ✅ | | | | |
| Cerebras | Hosted | | ✅ | | | | | | |
| Fireworks | Hosted | ✅ | ✅ | ✅ | | | | | |
| AWS Bedrock | Hosted | | ✅ | | ✅ | | | | |
| Together | Hosted | ✅ | ✅ | | ✅ | | | | |
| Groq | Hosted | | ✅ | | | | | | |
| Ollama | Single Node | | ✅ | | | | | | |
| TGI | Hosted/Single Node | | ✅ | | | | | | |
| NVIDIA NIM | Hosted/Single Node | | ✅ | | ✅ | | | | |
| ChromaDB | Hosted/Single Node | | | ✅ | | | | | |
| PG Vector | Single Node | | | ✅ | | | | | |
| PyTorch ExecuTorch | On-device iOS | ✅ | ✅ | | | | | | |
| vLLM | Single Node | | ✅ | | | | | | |
| OpenAI | Hosted | | ✅ | | | | | | |
| Anthropic | Hosted | | ✅ | | | | | | |
| Gemini | Hosted | | ✅ | | | | | | |
| WatsonX | Hosted | | ✅ | | | | | | |
| HuggingFace | Single Node | | | | | | ✅ | | ✅ |
| TorchTune | Single Node | | | | | | ✅ | | |
| NVIDIA NEMO | Hosted | | ✅ | ✅ | | | ✅ | ✅ | ✅ |
| NVIDIA | Hosted | | | | | | ✅ | ✅ | ✅ |

> **Note**: Additional providers are available through external packages. See [External Providers](https://llama-stack.readthedocs.io/en/latest/providers/external.html) documentation.

### Distributions

Llama Stack Distributions offer pre-configured bundles for various deployment scenarios.

|               **Distribution**                |                                                                    **Llama Stack Docker**                                                                     |                                                 Start This Distribution                                                  |
|:---------------------------------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------:|
|                Starter Distribution                 |           [llamastack/distribution-starter](https://hub.docker.com/repository/docker/llamastack/distribution-starter/general)           |      [Guide](https://llama-stack.readthedocs.io/en/latest/distributions/self_hosted_distro/starter.html)      |
|                Meta Reference                 |           [llamastack/distribution-meta-reference-gpu](https://hub.docker.com/repository/docker/llamastack/distribution-meta-reference-gpu/general)           |      [Guide](https://llama-stack.readthedocs.io/en/latest/distributions/self_hosted_distro/meta-reference-gpu.html)      |
|                   PostgreSQL                  |                [llamastack/distribution-postgres-demo](https://hub.docker.com/repository/docker/llamastack/distribution-postgres-demo/general)                |                  |

### Documentation

Access comprehensive documentation for detailed information:

*   [CLI References](https://llama-stack.readthedocs.io/en/latest/references/index.html)
*   [Getting Started Guides](https://llama-stack.readthedocs.io/en/latest/getting_started/index.html)
*   [Contributing Guide](CONTRIBUTING.md)

### Client SDKs

Choose from a variety of client SDKs:

|  **Language** |  **Client SDK** | **Package** |
| :----: | :----: | :----: |
| Python |  [llama-stack-client-python](https://github.com/meta-llama/llama-stack-client-python) | [![PyPI version](https://img.shields.io/pypi/v/llama_stack_client.svg)](https://pypi.org/project/llama_stack_client/)
| Swift  | [llama-stack-client-swift](https://github.com/meta-llama/llama-stack-client-swift) | [![Swift Package Index](https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2Fmeta-llama%2Fllama-stack-client-swift%2Fbadge%3Ftype%3Dswift-versions)](https://swiftpackageindex.com/meta-llama/llama-stack-client-swift)
| Typescript   | [llama-stack-client-typescript](https://github.com/meta-llama/llama-stack-client-typescript) | [![NPM version](https://img.shields.io/npm/v/llama-stack-client.svg)](https://npmjs.org/package/llama-stack-client)
| Kotlin | [llama-stack-client-kotlin](https://github.com/meta-llama/llama-stack-client-kotlin) | [![Maven version](https://img.shields.io/maven-central/v/com.llama.llamastack/llama-stack-client-kotlin)](https://central.sonatype.com/artifact/com.llama.llamastack/llama-stack-client-kotlin)

Explore example scripts with client SDKs in the [llama-stack-apps](https://github.com/meta-llama/llama-stack-apps/tree/main/examples) repository.
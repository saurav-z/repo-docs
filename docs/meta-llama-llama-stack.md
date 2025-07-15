# Llama Stack: Build, Deploy, and Scale Your AI Applications

**Simplify AI application development with Llama Stack, a powerful framework providing unified APIs and a flexible architecture for deploying and scaling your AI models. [Explore the Llama Stack Repository](https://github.com/meta-llama/llama-stack)**

[![PyPI version](https://img.shields.io/pypi/v/llama_stack.svg)](https://pypi.org/project/llama_stack/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/llama-stack)](https://pypi.org/project/llama-stack/)
[![License](https://img.shields.io/pypi/l/llama_stack.svg)](https://github.com/meta-llama/llama-stack/blob/main/LICENSE)
[![Discord](https://img.shields.io/discord/1257833999603335178?color=6A7EC2&logo=discord&logoColor=ffffff)](https://discord.gg/llama-stack)
[![Unit Tests](https://github.com/meta-llama/llama-stack/actions/workflows/unit-tests.yml/badge.svg?branch=main)](https://github.com/meta-llama/llama-stack/actions/workflows/unit-tests.yml?query=branch%3Amain)
[![Integration Tests](https://github.com/meta-llama/llama-stack/actions/workflows/integration-tests.yml/badge.svg?branch=main)](https://github.com/meta-llama/llama-stack/actions/workflows/integration-tests.yml?query=branch%3Amain)

[**Quick Start**](https://llama-stack.readthedocs.io/en/latest/getting_started/index.html) | [**Documentation**](https://llama-stack.readthedocs.io/en/latest/index.html) | [**Colab Notebook**](./docs/getting_started.ipynb) | [**Discord**](https://discord.gg/llama-stack)

## Key Features

*   **Unified API Layer:** Standardizes APIs for Inference, RAG, Agents, Tools, Safety, Evals, and Telemetry.
*   **Plugin Architecture:** Supports a wide ecosystem of API implementations across various environments (local, on-premise, cloud, mobile).
*   **Prepackaged Distributions:** Offers ready-to-use solutions for rapid and reliable deployment.
*   **Multiple Developer Interfaces:** Includes CLI and SDKs for Python, Typescript, iOS, and Android.
*   **Example Applications:** Provides standalone applications as blueprints for production-ready AI applications.
*   **Llama 4 Support:** Seamlessly integrates with the latest Llama 4 models.

##  🚀 Get Started Quickly 🚀

### One-Line Installation

```bash
curl -LsSf https://github.com/meta-llama/llama-stack/raw/main/scripts/install.sh | bash
```

## Overview

Llama Stack simplifies AI application development by providing a standardized foundation and building blocks. It promotes best practices across the Llama ecosystem, offering a flexible and streamlined approach to building, deploying, and scaling generative AI applications.

<div style="text-align: center;">
  <img
    src="https://github.com/user-attachments/assets/33d9576d-95ea-468d-95e2-8fa233205a50"
    width="480"
    title="Llama Stack"
    alt="Llama Stack"
  />
</div>

## Benefits of Using Llama Stack

*   **Deployment Flexibility:** Choose your preferred infrastructure without changing APIs.
*   **Consistent Experience:** Unified APIs ensure consistent application behavior during development, testing, and deployment.
*   **Robust Ecosystem:** Integrations with various partners (cloud providers, hardware vendors, and AI-focused companies) offer tailored solutions for deploying Llama models.

## API Providers

Llama Stack integrates with numerous API providers, offering diverse options for inference, agents, and more.

[Full list](https://llama-stack.readthedocs.io/en/latest/providers/index.html)

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

## Distributions

Llama Stack Distributions ("distros") provide pre-configured bundles of provider implementations for each API component, simplifying deployment.

|               **Distribution**                |                                                                    **Llama Stack Docker**                                                                     |                                                 Start This Distribution                                                  |
|:---------------------------------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------:|
|                Starter Distribution                 |           [llamastack/distribution-starter](https://hub.docker.com/repository/docker/llamastack/distribution-starter/general)           |      [Guide](https://llama-stack.readthedocs.io/en/latest/distributions/self_hosted_distro/starter.html)      |
|                Meta Reference                 |           [llamastack/distribution-meta-reference-gpu](https://hub.docker.com/repository/docker/llamastack/distribution-meta-reference-gpu/general)           |      [Guide](https://llama-stack.readthedocs.io/en/latest/distributions/self_hosted_distro/meta-reference-gpu.html)      |
|                   PostgreSQL                  |                [llamastack/distribution-postgres-demo](https://hub.docker.com/repository/docker/llamastack/distribution-postgres-demo/general)                |                  |

## Documentation

*   **CLI References:**
    *   [llama (server-side) CLI Reference](https://llama-stack.readthedocs.io/en/latest/references/llama_cli_reference/index.html): Using the `llama` CLI for Llama model management and distribution building/starting.
    *   [llama (client-side) CLI Reference](https://llama-stack.readthedocs.io/en/latest/references/llama_stack_client_cli_reference.html): Using the `llama-stack-client` CLI to query distribution information.
*   **Getting Started:**
    *   [Quick Start Guide](https://llama-stack.readthedocs.io/en/latest/getting_started/index.html)
    *   [Jupyter Notebook](docs/getting_started.ipynb)
    *   [Llama 3.2 Colab Notebook](https://colab.research.google.com/drive/1dtVmxotBsI4cGZQNsJRYPrLiDeT0Wnwt)
    *   [Zero-to-Hero Guide](https://github.com/meta-llama/llama-stack/tree/main/docs/zero_to_hero_guide)
*   [Contributing](CONTRIBUTING.md)
    *   [Adding a new API Provider](https://llama-stack.readthedocs.io/en/latest/contributing/new_api_provider.html)

## Llama Stack Client SDKs

|  **Language** |  **Client SDK** | **Package** |
| :----: | :----: | :----: |
| Python |  [llama-stack-client-python](https://github.com/meta-llama/llama-stack-client-python) | [![PyPI version](https://img.shields.io/pypi/v/llama_stack_client.svg)](https://pypi.org/project/llama_stack_client/)
| Swift  | [llama-stack-client-swift](https://github.com/meta-llama/llama-stack-client-swift) | [![Swift Package Index](https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2Fmeta-llama%2Fllama-stack-client-swift%2Fbadge%3Ftype%3Dswift-versions)](https://swiftpackageindex.com/meta-llama/llama-stack-client-swift)
| Typescript   | [llama-stack-client-typescript](https://github.com/meta-llama/llama-stack-client-typescript) | [![NPM version](https://img.shields.io/npm/v/llama-stack-client.svg)](https://npmjs.org/package/llama-stack-client)
| Kotlin | [llama-stack-client-kotlin](https://github.com/meta-llama/llama-stack-client-kotlin) | [![Maven version](https://img.shields.io/maven-central/v/com.llama.llamastack/llama-stack-client-kotlin)](https://central.sonatype.com/artifact/com.llama.llamastack/llama-stack-client-kotlin)

Connect to Llama Stack servers using client SDKs for Python, Typescript, Swift, and Kotlin; explore example applications in the [llama-stack-apps](https://github.com/meta-llama/llama-stack-apps/tree/main/examples) repository.
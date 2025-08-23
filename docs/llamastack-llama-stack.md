# Llama Stack: Build and Deploy AI Applications with Ease

**Llama Stack simplifies AI application development by providing a unified API layer and pre-packaged distributions for seamless deployment; explore the original repo [here](https://github.com/llamastack/llama-stack).**

Llama Stack is your one-stop solution for building and deploying generative AI applications. It standardizes the core building blocks of AI application development, offering flexibility, a consistent experience, and a robust ecosystem.

## Key Features:

*   **Unified API Layer:** Standardized APIs for Inference, RAG, Agents, Tools, Safety, Evals, and Telemetry.
*   **Plugin Architecture:** Supports a rich ecosystem of API implementations across various environments (local, on-premises, cloud, mobile).
*   **Pre-packaged Distributions:** Get started quickly and reliably with pre-configured bundles for different deployment scenarios.
*   **Multiple Developer Interfaces:** CLI and SDKs for Python, Typescript, iOS, and Android.
*   **Standalone Applications:** Example applications to guide production-grade AI application development.
*   **Llama 4 Support:** Comprehensive support for the latest Llama 4 models.

## Llama Stack Benefits:

*   **Flexible Infrastructure:** Choose your preferred infrastructure without changing APIs.
*   **Consistent Application Behavior:** Unified APIs make building, testing, and deploying AI applications easier.
*   **Robust Ecosystem:** Integration with distribution partners (cloud providers, hardware vendors, AI-focused companies) offering tailored infrastructure, software, and services for deploying Llama models.

## Getting Started

### Quick Installation

To try Llama Stack locally, run:

```bash
curl -LsSf https://github.com/meta-llama/llama-stack/raw/main/scripts/install.sh | bash
```

### Documentation and Resources:

*   **[Quick Start](https://llama-stack.readthedocs.io/en/latest/getting_started/index.html)**
*   **[Documentation](https://llama-stack.readthedocs.io/en/latest/index.html)**
*   **[Colab Notebook](./docs/getting_started.ipynb)**
*   **[Discord](https://discord.gg/llama-stack)**
*   **[CLI References](https://llama-stack.readthedocs.io/en/latest/references/llama_cli_reference/index.html)**
*   **[SDK Client References](https://llama-stack.readthedocs.io/en/latest/references/llama_stack_client_cli_reference.html)**
*   **[Zero-to-Hero Guide](https://github.com/meta-llama/llama-stack/tree/main/docs/zero_to_hero_guide)**
*   **[Adding a new API Provider](https://llama-stack.readthedocs.io/en/latest/contributing/new_api_provider.html)**

### Llama Stack Client SDKs

|  **Language** |  **Client SDK** | **Package** |
| :----: | :----: | :----: |
| Python |  [llama-stack-client-python](https://github.com/meta-llama/llama-stack-client-python) | [![PyPI version](https://img.shields.io/pypi/v/llama_stack_client.svg)](https://pypi.org/project/llama_stack_client/)
| Swift  | [llama-stack-client-swift](https://github.com/meta-llama/llama-stack-client-swift) | [![Swift Package Index](https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2Fmeta-llama%2Fllama-stack-client-swift%2Fbadge%3Ftype%3Dswift-versions)](https://swiftpackageindex.com/meta-llama/llama-stack-client-swift)
| Typescript   | [llama-stack-client-typescript](https://github.com/meta-llama/llama-stack-client-typescript) | [![NPM version](https://img.shields.io/npm/v/llama-stack-client.svg)](https://npmjs.org/package/llama-stack-client)
| Kotlin | [llama-stack-client-kotlin](https://github.com/meta-llama/llama-stack-client-kotlin) | [![Maven version](https://img.shields.io/maven-central/v/com.llama.llamastack/llama-stack-client-kotlin)](https://central.sonatype.com/artifact/com.llama.llamastack/llama-stack-client-kotlin)

Use our client SDKs for integrating with a Llama Stack server in your preferred language, you can choose from [python](https://github.com/meta-llama/llama-stack-client-python), [typescript](https://github.com/meta-llama/llama-stack-client-typescript), [swift](https://github.com/meta-llama/llama-stack-client-swift), and [kotlin](https://github.com/meta-llama/llama-stack-client-kotlin) programming languages.

## API Providers

Llama Stack supports various API providers and distributions. See [full list](https://llama-stack.readthedocs.io/en/latest/providers/index.html)

## Distributions

Llama Stack Distributions are pre-configured bundles of provider implementations. They make it easy to get started with a specific deployment scenario

|               **Distribution**                |                                                                    **Llama Stack Docker**                                                                     |                                                 Start This Distribution                                                  |
|:---------------------------------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------:|
|                Starter Distribution                 |           [llamastack/distribution-starter](https://hub.docker.com/repository/docker/llamastack/distribution-starter/general)           |      [Guide](https://llama-stack.readthedocs.io/en/latest/distributions/self_hosted_distro/starter.html)      |
|                Meta Reference                 |           [llamastack/distribution-meta-reference-gpu](https://hub.docker.com/repository/docker/llamastack/distribution-meta-reference-gpu/general)           |      [Guide](https://llama-stack.readthedocs.io/en/latest/distributions/self_hosted_distro/meta-reference-gpu.html)      |
|                   PostgreSQL                  |                [llamastack/distribution-postgres-demo](https://hub.docker.com/repository/docker/llamastack/distribution-postgres-demo/general)                |                  |

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=meta-llama/llama-stack&type=Date)](https://www.star-history.com/#meta-llama/llama-stack&Date)

## Contributors

Thanks to all our amazing contributors!

<a href="https://github.com/meta-llama/llama-stack/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=meta-llama/llama-stack" />
</a>
# Llama Stack: Build Powerful AI Applications with Ease

Llama Stack simplifies the development of AI applications by providing a unified API layer and flexible deployment options. ([View the original repo](https://github.com/meta-llama/llama-stack))

## Key Features

*   **Unified API Layer:** Standardizes core building blocks for Inference, RAG, Agents, Tools, Safety, Evals, and Telemetry.
*   **Plugin Architecture:** Supports a diverse ecosystem of API implementations across various environments (local, on-premises, cloud, mobile).
*   **Prepackaged Distributions:** Offers ready-to-use solutions for quick and reliable application development.
*   **Multiple Developer Interfaces:** Provides CLI and SDKs for Python, Typescript, iOS, and Android.
*   **Standalone Applications:** Includes example applications for building production-grade AI solutions.
*   **Llama 4 Support:** Get started with Llama 4 models using our latest release.

## Getting Started

### Quick Installation

Get started with Llama Stack locally with a single command:

```bash
curl -LsSf https://github.com/meta-llama/llama-stack/raw/main/scripts/install.sh | bash
```

### Core Benefits

*   **Flexibility:** Choose your preferred infrastructure without changing APIs.
*   **Consistency:** Build, test, and deploy AI applications with unified APIs.
*   **Ecosystem:** Integrates with leading providers for tailored infrastructure and services.

## API Providers

Llama Stack supports a wide range of API providers, enabling you to deploy your AI applications on the infrastructure that best suits your needs.

*   **Meta Reference:** Single Node with Agents, Inference, VectorIO, Safety, Telemetry, Post Training, Eval, and DatasetIO
*   **SambaNova:** Hosted with Inference and Safety
*   **Cerebras:** Hosted with Inference
*   **Fireworks:** Hosted with Agents, Inference, and VectorIO
*   **AWS Bedrock:** Hosted with Inference and Safety
*   **Together:** Hosted with Agents, Inference, and Safety
*   **Groq:** Hosted with Inference
*   **Ollama:** Single Node with Inference
*   **TGI:** Hosted/Single Node with Inference
*   **NVIDIA NIM:** Hosted/Single Node with Inference and Safety
*   **ChromaDB:** Hosted/Single Node with VectorIO
*   **PG Vector:** Single Node with VectorIO
*   **PyTorch ExecuTorch:** On-device iOS with Agents and Inference
*   **vLLM:** Single Node with Inference
*   **OpenAI:** Hosted with Inference
*   **Anthropic:** Hosted with Inference
*   **Gemini:** Hosted with Inference
*   **WatsonX:** Hosted with Inference
*   **HuggingFace:** Single Node with Post Training and DatasetIO
*   **TorchTune:** Single Node with Post Training
*   **NVIDIA NEMO:** Hosted with Inference, VectorIO, Post Training, Eval, and DatasetIO
*   **NVIDIA:** Hosted with Post Training, Eval, and DatasetIO

>   **Note:** Explore additional providers through external packages. See [External Providers](https://llama-stack.readthedocs.io/en/latest/providers/external.html) for more details.

## Distributions

Llama Stack Distributions offer pre-configured bundles of provider implementations for various deployment scenarios.

| **Distribution**            | **Llama Stack Docker**                                                                    | **Start This Distribution**                                                                    |
| :-------------------------- | :--------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------- |
| Starter Distribution         | [llamastack/distribution-starter](https://hub.docker.com/repository/docker/llamastack/distribution-starter/general) | [Guide](https://llama-stack.readthedocs.io/en/latest/distributions/self_hosted_distro/starter.html) |
| Meta Reference              | [llamastack/distribution-meta-reference-gpu](https://hub.docker.com/repository/docker/llamastack/distribution-meta-reference-gpu/general) | [Guide](https://llama-stack.readthedocs.io/en/latest/distributions/self_hosted_distro/meta-reference-gpu.html) |
| PostgreSQL                  | [llamastack/distribution-postgres-demo](https://hub.docker.com/repository/docker/llamastack/distribution-postgres-demo/general)                |                                                                                              |

## Documentation

*   **CLI References:**
    *   [llama CLI Reference](https://llama-stack.readthedocs.io/en/latest/references/llama_cli_reference/index.html)
    *   [llama-stack-client CLI Reference](https://llama-stack.readthedocs.io/en/latest/references/llama_stack_client_cli_reference.html)
*   **Getting Started:**
    *   [Quick Start Guide](https://llama-stack.readthedocs.io/en/latest/getting_started/index.html)
    *   [Jupyter Notebook](./docs/getting_started.ipynb)
    *   [Colab Notebook](https://colab.research.google.com/drive/1dtVmxotBsI4cGZQNsJRYPrLiDeT0Wnwt)
    *   [Zero-to-Hero Guide](https://github.com/meta-llama/llama-stack/tree/main/docs/zero_to_hero_guide)
*   [Contributing](CONTRIBUTING.md)
    *   [Adding a new API Provider](https://llama-stack.readthedocs.io/en/latest/contributing/new_api_provider.html)

## Client SDKs

Choose from our client SDKs for seamless integration in your preferred programming language.

| **Language**  | **Client SDK**                                                                                         | Package                                                                                                              |
| :-----------: | :----------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------- |
|    Python     | [llama-stack-client-python](https://github.com/meta-llama/llama-stack-client-python)                   | [![PyPI version](https://img.shields.io/pypi/v/llama_stack_client.svg)](https://pypi.org/project/llama_stack_client/) |
|     Swift     | [llama-stack-client-swift](https://github.com/meta-llama/llama-stack-client-swift)                      | [![Swift Package Index](https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2Fmeta-llama%2Fllama-stack-client-swift%2Fbadge%3Ftype%3Dswift-versions)](https://swiftpackageindex.com/meta-llama/llama-stack-client-swift) |
|  Typescript   | [llama-stack-client-typescript](https://github.com/meta-llama/llama-stack-client-typescript)           | [![NPM version](https://img.shields.io/npm/v/llama-stack-client.svg)](https://npmjs.org/package/llama-stack-client)     |
|    Kotlin     | [llama-stack-client-kotlin](https://github.com/meta-llama/llama-stack-client-kotlin)                   | [![Maven version](https://img.shields.io/maven-central/v/com.llama.llamastack/llama-stack-client-kotlin)](https://central.sonatype.com/artifact/com.llama.llamastack/llama-stack-client-kotlin) |

Explore example scripts in the [llama-stack-apps](https://github.com/meta-llama/llama-stack-apps/tree/main/examples) repo to get started with our client SDKs.
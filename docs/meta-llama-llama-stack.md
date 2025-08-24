# Llama Stack: Build Powerful AI Applications with Ease

Llama Stack simplifies AI application development by providing a unified API layer and flexible deployment options, allowing developers to focus on innovation. **[Explore the Llama Stack GitHub repository](https://github.com/meta-llama/llama-stack) to get started.**

[![PyPI version](https://img.shields.io/pypi/v/llama_stack.svg)](https://pypi.org/project/llama-stack/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/llama-stack)](https://pypi.org/project/llama-stack/)
[![License](https://img.shields.io/pypi/l/llama_stack.svg)](https://github.com/meta-llama/llama-stack/blob/main/LICENSE)
[![Discord](https://img.shields.io/discord/1257833999603335178?color=6A7EC2&logo=discord&logoColor=ffffff)](https://discord.gg/llama-stack)
[![Unit Tests](https://github.com/meta-llama/llama-stack/actions/workflows/unit-tests.yml/badge.svg?branch=main)](https://github.com/meta-llama/llama-stack/actions/workflows/unit-tests.yml?query=branch%3Amain)
[![Integration Tests](https://github.com/meta-llama/llama-stack/actions/workflows/integration-tests.yml/badge.svg?branch=main)](https://github.com/meta-llama/llama-stack/actions/workflows/integration-tests.yml?query=branch%3Amain)

## Key Features of Llama Stack:

*   **Unified API Layer:** Standardizes core building blocks for inference, RAG, agents, tools, safety, evals, and telemetry.
*   **Plugin Architecture:** Supports a diverse ecosystem of API implementations across various environments (local, on-premises, cloud, mobile).
*   **Prepackaged Distributions:** Offers ready-to-use solutions for quick and reliable deployments in any environment.
*   **Multiple Developer Interfaces:** Provides CLI and SDKs for Python, Typescript, iOS, and Android, streamlining development.
*   **Standalone Applications:** Includes example applications demonstrating how to build production-grade AI applications.
*   **Llama 4 Support:**  Supports the latest Llama 4 models.

## Getting Started

*   [**Quick Start**](https://llama-stack.readthedocs.io/en/latest/getting_started/index.html)
*   [**Documentation**](https://llama-stack.readthedocs.io/en/latest/index.html)
*   [**Colab Notebook**](./docs/getting_started.ipynb)

## Quick Installation

Install Llama Stack locally with a single command:

```bash
curl -LsSf https://github.com/meta-llama/llama-stack/raw/main/scripts/install.sh | bash
```

## Benefits of Using Llama Stack:

*   **Flexibility:** Choose your preferred infrastructure without changing APIs, with versatile deployment options.
*   **Consistency:** Build, test, and deploy AI applications with a consistent experience through unified APIs.
*   **Robust Ecosystem:** Benefit from integrations with cloud providers, hardware vendors, and AI companies offering tailored infrastructure and services.

## Supported API Providers

Llama Stack integrates with a wide range of API providers, providing developers with numerous options:

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

Llama Stack Distributions streamline deployments:

|               **Distribution**                |                                                                    **Llama Stack Docker**                                                                     |                                                 Start This Distribution                                                  |
|:---------------------------------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------:|
|                Starter Distribution                 |           [llamastack/distribution-starter](https://hub.docker.com/repository/docker/llamastack/distribution-starter/general)           |      [Guide](https://llama-stack.readthedocs.io/en/latest/distributions/self_hosted_distro/starter.html)      |
|                Meta Reference                 |           [llamastack/distribution-meta-reference-gpu](https://hub.docker.com/repository/docker/llamastack/distribution-meta-reference-gpu/general)           |      [Guide](https://llama-stack.readthedocs.io/en/latest/distributions/self_hosted_distro/meta-reference-gpu.html)      |
|                   PostgreSQL                  |                [llamastack/distribution-postgres-demo](https://hub.docker.com/repository/docker/llamastack/distribution-postgres-demo/general)                |                  |

## Llama Stack Client SDKs

Connect to a Llama Stack server using your preferred language:

|  **Language** |  **Client SDK** | **Package** |
| :----: | :----: | :----: |
| Python |  [llama-stack-client-python](https://github.com/meta-llama/llama-stack-client-python) | [![PyPI version](https://img.shields.io/pypi/v/llama_stack_client.svg)](https://pypi.org/project/llama_stack_client/)
| Swift  | [llama-stack-client-swift](https://github.com/meta-llama/llama-stack-client-swift) | [![Swift Package Index](https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2Fmeta-llama%2Fllama-stack-client-swift%2Fbadge%3Ftype%3Dswift-versions)](https://swiftpackageindex.com/meta-llama/llama-stack-client-swift)
| Typescript   | [llama-stack-client-typescript](https://github.com/meta-llama/llama-stack-client-typescript) | [![NPM version](https://img.shields.io/npm/v/llama-stack-client.svg)](https://npmjs.org/package/llama-stack-client)
| Kotlin | [llama-stack-client-kotlin](https://github.com/meta-llama/llama-stack-client-kotlin) | [![Maven version](https://img.shields.io/maven-central/v/com.llama.llamastack/llama-stack-client-kotlin)](https://central.sonatype.com/artifact/com.llama.llamastack/llama-stack-client-kotlin)

## Documentation & Resources

*   [**CLI References:**](https://llama-stack.readthedocs.io/en/latest/references/llama_cli_reference/index.html) and [Client CLI Reference](https://llama-stack.readthedocs.io/en/latest/references/llama_stack_client_cli_reference.html)
*   [**Zero-to-Hero Guide:**](https://github.com/meta-llama/llama-stack/tree/main/docs/zero_to_hero_guide)
*   [**Adding a new API Provider:**](https://llama-stack.readthedocs.io/en/latest/contributing/new_api_provider.html)
*   [**Complete Llama Stack course on Deeplearning.ai:**](https://learn.deeplearning.ai/courses/introducing-multimodal-llama-3-2/lesson/8/llama-stack).

## 🌟 GitHub Star History
## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=meta-llama/llama-stack&type=Date)](https://www.star-history.com/#meta-llama/llama-stack&Date)

## ✨ Contributors

Thanks to all of our amazing contributors!

<a href="https://github.com/meta-llama/llama-stack/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=meta-llama/llama-stack" />
</a>
```
Key improvements and SEO optimizations:

*   **Clear, concise title:**  "Llama Stack: Build Powerful AI Applications with Ease" clearly communicates the project's purpose and value proposition.
*   **Strong hook:** The first sentence immediately grabs attention and highlights the key benefit: simplifying AI app development.
*   **Keyword Integration:**  Uses relevant keywords like "AI applications," "unified API," and "deployment options."
*   **Header Structure:**  Uses headers (H2s) to organize content logically, making it easier to scan and understand.
*   **Bulleted Lists:** Uses bullet points to emphasize key features and benefits, which helps with readability and SEO.
*   **Clear Calls to Action:** Includes links to the documentation and quick start guide.
*   **Concise Language:** Avoids unnecessary jargon and uses straightforward language.
*   **Visual appeal:** The use of images helps to break up the text and provides visual interest.
*   **Focus on Value:** The description focuses on what users can *do* with Llama Stack.
*   **Updated Sections:** Includes updated sections that are useful for users.
*   **Optimized Content:** Includes the full range of useful information.
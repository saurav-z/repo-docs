# Llama Stack: Build and Deploy Generative AI Applications with Ease

Llama Stack empowers developers to streamline the development and deployment of generative AI applications, offering a unified API and flexible infrastructure options.  [Learn more on GitHub](https://github.com/meta-llama/llama-stack).

## Key Features

*   **Unified API Layer:** Standardizes inference, RAG, agents, tools, safety, evaluations, and telemetry for consistent behavior.
*   **Plugin Architecture:** Supports a rich ecosystem of API implementations across various environments (local, on-premises, cloud, mobile).
*   **Prepackaged Distributions:** Offers a one-stop solution for quick and reliable deployments in any environment.
*   **Multiple Developer Interfaces:** Provides CLI and SDKs for Python, Typescript, iOS, and Android, making it easy to integrate.
*   **Standalone Applications:** Includes examples for building production-grade AI applications with Llama Stack.
*   **Llama 4 Support:**  Supports Llama 4 models, expanding options for developers.

## Benefits of Using Llama Stack

*   **Flexible Infrastructure:** Choose your preferred infrastructure without changing your API code.
*   **Consistent Application Behavior:** Unified APIs simplify building, testing, and deploying AI applications.
*   **Robust Ecosystem:** Integration with distribution partners provides tailored infrastructure, software, and services for deploying Llama models.

## Getting Started

### One-Line Installation

Quickly try Llama Stack locally with a single command:

```bash
curl -LsSf https://github.com/meta-llama/llama-stack/raw/main/scripts/install.sh | bash
```

### Quick Links

*   [**Quick Start Guide**](https://llama-stack.readthedocs.io/en/latest/getting_started/index.html)
*   [**Documentation**](https://llama-stack.readthedocs.io/en/latest/index.html)
*   [**Colab Notebook**](docs/getting_started.ipynb)
*   [**Discord Community**](https://discord.gg/llama-stack)

##  Llama Stack Components

*   **API Providers:** Access a variety of AI providers like Meta, SambaNova, Fireworks, and more. See [API Providers](#api-providers) section for details and [full list](https://llama-stack.readthedocs.io/en/latest/providers/index.html).
*   **Distributions:** Pre-configured bundles for specific deployment scenarios, like a local development setup or production. See [Distributions](#distributions) section.

## API Providers

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

|               **Distribution**                |                                                                    **Llama Stack Docker**                                                                     |                                                 Start This Distribution                                                  |
|:---------------------------------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------:|
|                Starter Distribution                 |           [llamastack/distribution-starter](https://hub.docker.com/repository/docker/llamastack/distribution-starter/general)           |      [Guide](https://llama-stack.readthedocs.io/en/latest/distributions/self_hosted_distro/starter.html)      |
|                Meta Reference                 |           [llamastack/distribution-meta-reference-gpu](https://hub.docker.com/repository/docker/llamastack/distribution-meta-reference-gpu/general)           |      [Guide](https://llama-stack.readthedocs.io/en/latest/distributions/self_hosted_distro/meta-reference-gpu.html)      |
|                   PostgreSQL                  |                [llamastack/distribution-postgres-demo](https://hub.docker.com/repository/docker/llamastack/distribution-postgres-demo/general)                |                  |

## Client SDKs

Connect to Llama Stack servers in your preferred language:

|  **Language** |  **Client SDK** | **Package** |
| :----: | :----: | :----: |
| Python |  [llama-stack-client-python](https://github.com/meta-llama/llama-stack-client-python) | [![PyPI version](https://img.shields.io/pypi/v/llama_stack_client.svg)](https://pypi.org/project/llama_stack_client/)
| Swift  | [llama-stack-client-swift](https://github.com/meta-llama/llama-stack-client-swift) | [![Swift Package Index](https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2Fmeta-llama%2Fllama-stack-client-swift%2Fbadge%3Ftype%3Dswift-versions)](https://swiftpackageindex.com/meta-llama/llama-stack-client-swift)
| Typescript   | [llama-stack-client-typescript](https://github.com/meta-llama/llama-stack-client-typescript) | [![NPM version](https://img.shields.io/npm/v/llama-stack-client.svg)](https://npmjs.org/package/llama-stack-client)
| Kotlin | [llama-stack-client-kotlin](https://github.com/meta-llama/llama-stack-client-kotlin) | [![Maven version](https://img.shields.io/maven-central/v/com.llama.llamastack/llama-stack-client-kotlin)](https://central.sonatype.com/artifact/com.llama.llamastack/llama-stack-client-kotlin)

## Further Resources

*   [**Documentation**](https://llama-stack.readthedocs.io/en/latest/index.html)
*   [CLI References](https://llama-stack.readthedocs.io/en/latest/references/index.html)
*   [Zero-to-Hero Guide](https://github.com/meta-llama/llama-stack/tree/main/docs/zero_to_hero_guide)
*   [Contributing](CONTRIBUTING.md)

## Community

*   [**Discord**](https://discord.gg/llama-stack)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=meta-llama/llama-stack&type=Date)](https://www.star-history.com/#meta-llama/llama-stack&Date)

## ✨ Contributors

Thanks to all of our amazing contributors!

<a href="https://github.com/meta-llama/llama-stack/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=meta-llama/llama-stack" />
</a>
```
Key improvements and SEO considerations:

*   **Clear, Concise, and Action-Oriented Headline:**  "Llama Stack: Build and Deploy Generative AI Applications with Ease" immediately tells the user what the project does.
*   **One-Sentence Hook:** Positions Llama Stack as a solution.
*   **Keyword Optimization:** Includes relevant keywords like "Generative AI," "API," "Deployment," "Applications."
*   **Organized Structure:** Uses clear headings and subheadings for readability.
*   **Bulleted Key Features:** Makes it easy for users to quickly scan and understand the main benefits.
*   **Emphasis on Benefits:** Highlights the advantages of using Llama Stack.
*   **Call to Actions:**  Includes direct links to the Quick Start, Documentation, and Discord community.
*   **Comprehensive Table of Contents** The updated information has been reorganized.
*   **Clear Instructions:** Presents the one-line installation command prominently.
*   **Emphasis on Key Features:** Focuses on the key functionalities and benefits of Llama Stack.
*   **Includes "Star History"** Useful for showing how popular the repo is.
*   **Contributors Section:**  Acknowledges and thanks the contributors.
*   **Direct links** to relevant resources.
# Llama Stack: Build & Deploy AI Applications with Ease

**Llama Stack is your all-in-one solution for simplifying AI application development, offering a unified API layer, plugin architecture, and prepackaged distributions for a seamless experience.** ([See the original repo](https://github.com/llamastack/llama-stack))

## Key Features of Llama Stack

*   **Unified API Layer:** Standardizes Inference, RAG, Agents, Tools, Safety, Evals, and Telemetry for a consistent development experience.
*   **Plugin Architecture:** Supports a rich ecosystem of API implementations, including local, on-premises, cloud, and mobile environments.
*   **Prepackaged Distributions:** Offers a one-stop solution for developers to get started quickly and reliably in any environment.
*   **Multiple Developer Interfaces:** Provides CLI and SDKs for Python, Typescript, iOS, and Android, offering flexibility in how you build.
*   **Standalone Applications:** Includes examples for building production-grade AI applications with Llama Stack.

## Why Choose Llama Stack?

Llama Stack empowers developers to build powerful generative AI applications by:

*   **Flexible Deployment Options:** Choose your preferred infrastructure without changing APIs, enabling flexible deployment choices.
*   **Consistent Experience:**  Unified APIs make it easier to build, test, and deploy AI applications with consistent application behavior.
*   **Robust Ecosystem:** Integrations with cloud providers, hardware vendors, and AI-focused companies that offer tailored infrastructure, software, and services for deploying Llama models.

## Quick Start: One-Line Installation

Get started with Llama Stack locally using a single command:

```bash
curl -LsSf https://github.com/meta-llama/llama-stack/raw/main/scripts/install.sh | bash
```

## Llama Stack Benefits

By reducing friction and complexity, Llama Stack empowers developers to focus on building transformative generative AI applications.

## API Providers

Llama Stack supports a wide range of API providers, giving you flexibility in your AI development.  Explore the [full list of providers](https://llama-stack.readthedocs.io/en/latest/providers/index.html) in our documentation.

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

Llama Stack Distributions simplify deployment.  Choose a distribution that fits your needs.

|               **Distribution**                |                                                                    **Llama Stack Docker**                                                                     |                                                 Start This Distribution                                                  |
|:---------------------------------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------:|
|                Starter Distribution                 |           [llamastack/distribution-starter](https://hub.docker.com/repository/docker/llamastack/distribution-starter/general)           |      [Guide](https://llama-stack.readthedocs.io/en/latest/distributions/self_hosted_distro/starter.html)      |
|                Meta Reference                 |           [llamastack/distribution-meta-reference-gpu](https://hub.docker.com/repository/docker/llamastack/distribution-meta-reference-gpu/general)           |      [Guide](https://llama-stack.readthedocs.io/en/latest/distributions/self_hosted_distro/meta-reference-gpu.html)      |
|                   PostgreSQL                  |                [llamastack/distribution-postgres-demo](https://hub.docker.com/repository/docker/llamastack/distribution-postgres-demo/general)                |                  |

## Documentation

Explore comprehensive documentation to get the most out of Llama Stack:

*   [Quick Start](https://llama-stack.readthedocs.io/en/latest/getting_started/index.html)
*   [CLI References](https://llama-stack.readthedocs.io/en/latest/references/llama_cli_reference/index.html) and [CLI Client](https://llama-stack.readthedocs.io/en/latest/references/llama_stack_client_cli_reference.html)
*   [Zero-to-Hero Guide](https://github.com/meta-llama/llama-stack/tree/main/docs/zero_to_hero_guide)
*   [Contributing Guide](CONTRIBUTING.md) & [New API Provider](https://llama-stack.readthedocs.io/en/latest/contributing/new_api_provider.html)

## Llama Stack Client SDKs

Choose your preferred language for building client applications:

|  **Language** |  **Client SDK** | **Package** |
| :----: | :----: | :----: |
| Python |  [llama-stack-client-python](https://github.com/meta-llama/llama-stack-client-python) | [![PyPI version](https://img.shields.io/pypi/v/llama_stack_client.svg)](https://pypi.org/project/llama_stack_client/)
| Swift  | [llama-stack-client-swift](https://github.com/meta-llama/llama-stack-client-swift) | [![Swift Package Index](https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2Fmeta-llama%2Fllama-stack-client-swift%2Fbadge%3Ftype%3Dswift-versions)](https://swiftpackageindex.com/meta-llama/llama-stack-client-swift)
| Typescript   | [llama-stack-client-typescript](https://github.com/meta-llama/llama-stack-client-typescript) | [![NPM version](https://img.shields.io/npm/v/llama-stack-client.svg)](https://npmjs.org/package/llama-stack-client)
| Kotlin | [llama-stack-client-kotlin](https://github.com/meta-llama/llama-stack-client-kotlin) | [![Maven version](https://img.shields.io/maven-central/v/com.llama.llamastack/llama-stack-client-kotlin)](https://central.sonatype.com/artifact/com.llama.llamastack/llama-stack-client-kotlin)

## Resources

*   [Llama Stack Apps Examples](https://github.com/meta-llama/llama-stack-apps/tree/main/examples)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=meta-llama/llama-stack&type=Date)](https://www.star-history.com/#meta-llama/llama-stack&Date)

## Contributors

Thank you to our amazing contributors!

<a href="https://github.com/meta-llama/llama-stack/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=meta-llama/llama-stack" />
</a>
```
Key improvements:

*   **SEO-Friendly Title and Hook:**  Uses a clear, concise title and a strong one-sentence hook.
*   **Structured Headings:** Uses clear headings for better readability and organization.
*   **Bulleted Lists:** Uses bulleted lists for key features and benefits, making them easy to scan.
*   **Keyword Optimization:** Includes relevant keywords throughout the text (e.g., "AI application development," "generative AI," "unified API," "plugin architecture").
*   **Clear Calls to Action:** Encourages users to explore the documentation, and use the one-line installation command.
*   **Focus on Benefits:** Highlights the value proposition for developers.
*   **Concise Language:**  Rephrases text for better clarity and conciseness.
*   **Complete Coverage:** Includes all relevant information from the original README.
*   **Visual Enhancement:** Keep the images.
*   **Links:** Maintains and improves the links.
*   **Removed extraneous elements:** Removed unnecessary formatting and promotional language (e.g., "✨🎉").
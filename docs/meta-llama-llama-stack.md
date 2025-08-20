# Llama Stack: Build AI Applications with Ease 🚀

**Llama Stack simplifies AI application development by providing a unified API layer and a plugin architecture, enabling flexible deployment options and a consistent user experience.**  Visit the [Llama Stack GitHub Repository](https://github.com/meta-llama/llama-stack) for more details.

**Key Features:**

*   **Unified API Layer:** Standardizes inference, RAG, agents, tools, safety, evaluations, and telemetry.
*   **Plugin Architecture:** Supports a wide range of API implementations across various environments (local, on-premises, cloud, mobile).
*   **Prepackaged Distributions:** Offers one-stop solutions for quick and reliable deployments in any environment.
*   **Multiple Developer Interfaces:** Provides CLI and SDKs for Python, Typescript, iOS, and Android.
*   **Standalone Applications:** Includes examples to build production-ready AI applications.
*   **Llama 4 Support:**  Supports Llama 4 models.

**Llama Stack Benefits:**

*   **Flexible Infrastructure:** Choose your preferred infrastructure without changing APIs.
*   **Consistent Experience:** Build, test, and deploy AI applications with consistent behavior.
*   **Robust Ecosystem:** Integrates with leading cloud providers, hardware vendors, and AI-focused companies.

**Get Started Quickly:**

*   **One-Line Installation:**
    ```bash
    curl -LsSf https://github.com/meta-llama/llama-stack/raw/main/scripts/install.sh | bash
    ```
*   **Quick Start:** [Quick Start Guide](https://llama-stack.readthedocs.io/en/latest/getting_started/index.html)
*   **Colab Notebook:** [Getting Started Notebook](./docs/getting_started.ipynb)

**API Providers:**

Llama Stack supports a growing list of API providers, including:

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

For a complete list, see the [full list](https://llama-stack.readthedocs.io/en/latest/providers/index.html).

**Distributions:**

Llama Stack distributions offer pre-configured bundles. Supported distributions include:

*   [Starter Distribution](https://llama-stack.readthedocs.io/en/latest/distributions/self_hosted_distro/starter.html)
*   [Meta Reference GPU](https://llama-stack.readthedocs.io/en/latest/distributions/self_hosted_distro/meta-reference-gpu.html)

**Documentation:**

*   [Documentation](https://llama-stack.readthedocs.io/en/latest/index.html)
*   CLI References: [llama (server-side)](https://llama-stack.readthedocs.io/en/latest/references/llama_cli_reference/index.html) and [llama-stack-client](https://llama-stack.readthedocs.io/en/latest/references/llama_stack_client_cli_reference.html)
*   [Zero-to-Hero Guide](https://github.com/meta-llama/llama-stack/tree/main/docs/zero_to_hero_guide)
*   [Adding a new API Provider](https://llama-stack.readthedocs.io/en/latest/contributing/new_api_provider.html)

**Llama Stack Client SDKs:**

Use the client SDKs in your preferred language to connect to a Llama Stack server:

|  **Language** |  **Client SDK** | **Package** |
| :----: | :----: | :----: |
| Python |  [llama-stack-client-python](https://github.com/meta-llama/llama-stack-client-python) | [![PyPI version](https://img.shields.io/pypi/v/llama_stack_client.svg)](https://pypi.org/project/llama_stack_client/)
| Swift  | [llama-stack-client-swift](https://github.com/meta-llama/llama-stack-client-swift) | [![Swift Package Index](https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2Fmeta-llama%2Fllama-stack-client-swift%2Fbadge%3Ftype%3Dswift-versions)](https://swiftpackageindex.com/meta-llama/llama-stack-client-swift)
| Typescript   | [llama-stack-client-typescript](https://github.com/meta-llama/llama-stack-client-typescript) | [![NPM version](https://img.shields.io/npm/v/llama-stack-client.svg)](https://npmjs.org/package/llama-stack-client)
| Kotlin | [llama-stack-client-kotlin](https://github.com/meta-llama/llama-stack-client-kotlin) | [![Maven version](https://img.shields.io/maven-central/v/com.llama.llamastack/llama-stack-client-kotlin)](https://central.sonatype.com/artifact/com.llama.llamastack/llama-stack-client-kotlin)

**Contributors:**

[Contributors Image](https://contrib.rocks/image?repo=meta-llama/llama-stack)

**Star History:**

[Star History Chart](https://api.star-history.com/svg?repos=meta-llama/llama-stack&type=Date)
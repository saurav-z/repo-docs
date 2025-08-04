# Llama Stack: Build and Deploy Generative AI Applications with Ease

Llama Stack is a powerful framework designed to streamline the development and deployment of generative AI applications, providing a unified API and flexible infrastructure options.  Explore the original repo at [https://github.com/meta-llama/llama-stack](https://github.com/meta-llama/llama-stack).

**Key Features:**

*   **Unified API Layer:** Simplify AI application development with a consistent interface for inference, RAG, agents, tools, safety, evaluations, and telemetry.
*   **Plugin Architecture:** Supports diverse API implementations across various environments (local, on-premises, cloud, mobile).
*   **Prepackaged Distributions:** Get started quickly and reliably with pre-configured solutions.
*   **Multiple Developer Interfaces:**  Access Llama Stack through CLI and SDKs for Python, TypeScript, iOS, and Android.
*   **Standalone Applications:**  Leverage example applications to build production-ready AI solutions.
*   **Llama 4 Support:**  Seamlessly integrate and run Llama 4 models.

**Benefits:**

*   **Flexible Infrastructure:** Choose your preferred infrastructure without changing APIs, with flexible deployment options.
*   **Consistent Application Behavior:** Easily build, test, and deploy AI applications with a unified API.
*   **Robust Ecosystem:** Benefit from integrations with cloud providers, hardware vendors, and AI-focused companies.

**Quick Start:**

```bash
curl -LsSf https://github.com/meta-llama/llama-stack/raw/main/scripts/install.sh | bash
```

**API Providers:**

Llama Stack supports a wide range of API providers, enabling you to select the best solution for your needs.

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

> **Note**: Additional providers are available through external packages. See [External Providers](https://llama-stack.readthedocs.io/en/latest/providers/external.html) documentation.

**Distributions:**

Llama Stack Distributions offer pre-configured bundles for easy deployment.

|               **Distribution**                |                                                                    **Llama Stack Docker**                                                                     |                                                 Start This Distribution                                                  |
|:---------------------------------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------:|
|                Starter Distribution                 |           [llamastack/distribution-starter](https://hub.docker.com/repository/docker/llamastack/distribution-starter/general)           |      [Guide](https://llama-stack.readthedocs.io/en/latest/distributions/self_hosted_distro/starter.html)      |
|                Meta Reference                 |           [llamastack/distribution-meta-reference-gpu](https://hub.docker.com/repository/docker/llamastack/distribution-meta-reference-gpu/general)           |      [Guide](https://llama-stack.readthedocs.io/en/latest/distributions/self_hosted_distro/meta-reference-gpu.html)      |
|                   PostgreSQL                  |                [llamastack/distribution-postgres-demo](https://hub.docker.com/repository/docker/llamastack/distribution-postgres-demo/general)                |                  |

**Documentation:**

*   [Quick Start](https://llama-stack.readthedocs.io/en/latest/getting_started/index.html)
*   [CLI References](https://llama-stack.readthedocs.io/en/latest/references/llama_cli_reference/index.html)
*   [SDKs](https://llama-stack.readthedocs.io/en/latest/sdk/index.html)

**SDKs:**

Choose your preferred language to build applications with Llama Stack:

*   **Python:** [llama-stack-client-python](https://github.com/meta-llama/llama-stack-client-python) ([PyPI](https://pypi.org/project/llama_stack_client/))
*   **Swift:** [llama-stack-client-swift](https://github.com/meta-llama/llama-stack-client-swift) ([Swift Package Index](https://swiftpackageindex.com/meta-llama/llama-stack-client-swift))
*   **TypeScript:** [llama-stack-client-typescript](https://github.com/meta-llama/llama-stack-client-typescript) ([NPM](https://npmjs.org/package/llama-stack-client))
*   **Kotlin:** [llama-stack-client-kotlin](https://github.com/meta-llama/llama-stack-client-kotlin) ([Maven](https://central.sonatype.com/artifact/com.llama.llamastack/llama-stack-client-kotlin))

**Examples:**

Find example scripts to interact with the Llama Stack server in the [llama-stack-apps](https://github.com/meta-llama/llama-stack-apps/tree/main/examples) repository.
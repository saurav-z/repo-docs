# Llama Stack: Build Production-Ready AI Applications with Ease

**Llama Stack** is a comprehensive framework that simplifies the development and deployment of AI applications, providing a unified API layer and pre-configured distributions. [Explore the Llama Stack repository](https://github.com/llamastack/llama-stack) to get started!

## Key Features

*   **Unified API Layer:** Standardizes inference, RAG, agents, tools, safety, evaluations, and telemetry for consistent application behavior.
*   **Plugin Architecture:** Supports a rich ecosystem of API implementations across diverse environments (local, on-premises, cloud, mobile).
*   **Prepackaged Distributions:** Offers one-stop solutions for quick and reliable deployments in various environments, from local development to production.
*   **Multiple Developer Interfaces:** Provides CLI and SDKs for Python, Typescript, iOS, and Android, streamlining development workflows.
*   **Standalone Applications:** Includes example applications to build production-grade AI applications.
*   **Llama 4 Support:** Enables users to run Llama 4 models.

## Benefits of Using Llama Stack

*   **Flexible Infrastructure:** Choose your preferred infrastructure without code changes, allowing for flexible deployment options.
*   **Consistent Experience:** Unified APIs simplify building, testing, and deploying AI applications with consistent behavior.
*   **Robust Ecosystem:** Integrations with distribution partners offer tailored infrastructure, software, and services for deploying Llama models.

## Getting Started

*   **Quick Installation:** Install Llama Stack locally with a single command:
    ```bash
    curl -LsSf https://github.com/meta-llama/llama-stack/raw/main/scripts/install.sh | bash
    ```
*   **Quick Start:**
    *   **[Quick Start](https://llama-stack.readthedocs.io/en/latest/getting_started/index.html)**
    *   **[Documentation](https://llama-stack.readthedocs.io/en/latest/index.html)**
    *   **[Colab Notebook](./docs/getting_started.ipynb)**
    *   **[Discord](https://discord.gg/llama-stack)**

## Llama Stack for Llama 4 Models
Llama Stack now supports the Llama 4 models. The models can be found at llama.com.

### CLI Example

```bash
# Run a chat completion
MODEL="Llama-4-Scout-17B-16E-Instruct"

llama-stack-client --endpoint http://localhost:8321 \
inference chat-completion \
--model-id meta-llama/$MODEL \
--message "write a haiku for meta's llama 4 models"

ChatCompletionResponse(
    completion_message=CompletionMessage(content="Whispers in code born\nLlama's gentle, wise heartbeat\nFuture's soft unfold", role='assistant', stop_reason='end_of_turn', tool_calls=[]),
    logprobs=None,
    metrics=[Metric(metric='prompt_tokens', value=21.0, unit=None), Metric(metric='completion_tokens', value=28.0, unit=None), Metric(metric='total_tokens', value=49.0, unit=None)]
)
```

### Python SDK Example
```python
from llama_stack_client import LlamaStackClient

client = LlamaStackClient(base_url=f"http://localhost:8321")

model_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
prompt = "Write a haiku about coding"

print(f"User> {prompt}")
response = client.inference.chat_completion(
    model_id=model_id,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ],
)
print(f"Assistant> {response.completion_message.content}")
```

## API Providers

Llama Stack integrates with various API providers, offering diverse options for model inference, vector storage, and more.

| API Provider Builder | Environments | Agents | Inference | VectorIO | Safety | Telemetry | Post Training | Eval | DatasetIO |
| :--------------------: | :------------: | :------: | :---------: | :--------: | :------: | :---------: | :-------------: | :----: | :--------: |
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
| Milvus | Hosted/Single Node | | | ✅ | | | | | |
| Qdrant | Hosted/Single Node | | | ✅ | | | | | |
| Weaviate | Hosted/Single Node | | | ✅ | | | | | |
| SQLite-vec | Single Node | | | ✅ | | | | | |
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

*   **[Full List of Providers](https://llama-stack.readthedocs.io/en/latest/providers/index.html)**

## Distributions

Llama Stack Distributions provide pre-configured setups for specific deployment scenarios.

| Distribution | Llama Stack Docker | Start This Distribution |
| :-----------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------: |
| Starter Distribution | [llamastack/distribution-starter](https://hub.docker.com/repository/docker/llamastack/distribution-starter/general) | [Guide](https://llama-stack.readthedocs.io/en/latest/distributions/self_hosted_distro/starter.html) |
| Meta Reference | [llamastack/distribution-meta-reference-gpu](https://hub.docker.com/repository/docker/llamastack/distribution-meta-reference-gpu/general) | [Guide](https://llama-stack.readthedocs.io/en/latest/distributions/self_hosted_distro/meta-reference-gpu.html) |
| PostgreSQL | [llamastack/distribution-postgres-demo](https://hub.docker.com/repository/docker/llamastack/distribution-postgres-demo/general) |  |

## Documentation

*   **CLI References:**
    *   [llama (server-side) CLI Reference](https://llama-stack.readthedocs.io/en/latest/references/llama_cli_reference/index.html)
    *   [llama (client-side) CLI Reference](https://llama-stack.readthedocs.io/en/latest/references/llama_stack_client_cli_reference.html)
*   **Getting Started:**
    *   [Quick guide to start a Llama Stack server](https://llama-stack.readthedocs.io/en/latest/getting_started/index.html)
    *   [Jupyter notebook](./docs/getting_started.ipynb)
    *   [Colab notebook](https://colab.research.google.com/drive/1dtVmxotBsI4cGZQNsJRYPrLiDeT0Wnwt)
    *   [Zero-to-Hero Guide](https://github.com/meta-llama/llama-stack/tree/main/docs/zero_to_hero_guide)
*   [Contributing](CONTRIBUTING.md)
    *   [Adding a new API Provider](https://llama-stack.readthedocs.io/en/latest/contributing/new_api_provider.html)

## Llama Stack Client SDKs

Choose from various SDKs to connect to a Llama Stack server in your favorite language:

| **Language** | **Client SDK** | **Package** |
| :----: | :----: | :----: |
| Python |  [llama-stack-client-python](https://github.com/meta-llama/llama-stack-client-python) | [![PyPI version](https://img.shields.io/pypi/v/llama_stack_client.svg)](https://pypi.org/project/llama_stack_client/)
| Swift  | [llama-stack-client-swift](https://github.com/meta-llama/llama-stack-client-swift) | [![Swift Package Index](https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2Fmeta-llama%2Fllama-stack-client-swift%2Fbadge%3Ftype%3Dswift-versions)](https://swiftpackageindex.com/meta-llama/llama-stack-client-swift)
| Typescript   | [llama-stack-client-typescript](https://github.com/meta-llama/llama-stack-client-typescript) | [![NPM version](https://img.shields.io/npm/v/llama-stack-client.svg)](https://npmjs.org/package/llama-stack-client)
| Kotlin | [llama-stack-client-kotlin](https://github.com/meta-llama/llama-stack-client-kotlin) | [![Maven version](https://img.shields.io/maven-central/v/com.llama.llamastack/llama-stack-client-kotlin)](https://central.sonatype.com/artifact/com.llama.llamastack/llama-stack-client-kotlin)

Find more example scripts using client SDKs in the [llama-stack-apps](https://github.com/meta-llama/llama-stack-apps/tree/main/examples) repository.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=meta-llama/llama-stack&type=Date)](https://www.star-history.com/#meta-llama/llama-stack&Date)

## Contributors

Thanks to all our amazing contributors!

<a href="https://github.com/meta-llama/llama-stack/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=meta-llama/llama-stack" />
</a>
```
Key improvements and SEO optimizations:

*   **Concise Hook:** The introductory sentence clearly and concisely summarizes the value proposition.
*   **Keyword Integration:**  Includes relevant keywords like "AI application development," "unified API," and specific mentions of the Llama ecosystem throughout the document.
*   **Clear Headings:**  Uses descriptive and keyword-rich headings to improve readability and SEO.
*   **Bulleted Lists:** Uses bulleted lists to highlight key features and benefits, making the content easier to scan.
*   **Internal Linking:**  Links to other sections of the documentation, which is good for SEO.
*   **Call to Action:** Encourages users to explore the repository.
*   **Markdown Formatting:** Proper use of Markdown ensures good readability.
*   **Updated Information:** Includes the new information of Llama 4 models.
*   **Star History and Contributors:** Includes the star history and contributors information.
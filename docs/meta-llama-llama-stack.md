# Llama Stack: Build AI Applications Faster with a Unified API Layer

**Llama Stack streamlines AI application development, providing a flexible and consistent platform for deploying and managing Llama models.** Explore the official [Llama Stack repository](https://github.com/meta-llama/llama-stack) to get started.

## Key Features

*   **Unified API Layer:** Standardizes inference, RAG, Agents, Tools, Safety, Evals, and Telemetry for consistent application behavior.
*   **Plugin Architecture:** Supports diverse API implementations in various environments (local, cloud, on-premise, mobile).
*   **Prepackaged Distributions:** Offers ready-to-use solutions for rapid deployment in any environment.
*   **Multiple Developer Interfaces:** Provides CLI, SDKs (Python, TypeScript, iOS, Android), and example applications.
*   **Flexible Deployment Options:** Choose your preferred infrastructure without code changes.
*   **Robust Ecosystem:** Integrates with cloud providers, hardware vendors, and AI-focused companies.

## What is Llama Stack?

Llama Stack simplifies the development of AI applications by providing core building blocks and codifying best practices within the Llama ecosystem. It reduces complexity, empowering developers to focus on building transformative generative AI applications.

## 🚀 One-Line Installation 🚀

Quickly try Llama Stack locally:

```bash
curl -LsSf https://github.com/meta-llama/llama-stack/raw/main/scripts/install.sh | bash
```

## Llama 4 Support
Support for the Llama 4 family of models is available with [Version 0.2.0](https://github.com/meta-llama/llama-stack/releases/tag/v0.2.0) release.

<details>
<summary>Click Here to Run Llama 4 Models on Llama Stack</summary>

*Note you need 8xH100 GPU-host to run these models*

```bash
pip install -U llama_stack

MODEL="Llama-4-Scout-17B-16E-Instruct"
# get meta url from llama.com
llama model download --source meta --model-id $MODEL --meta-url <META_URL>

# start a llama stack server
INFERENCE_MODEL=meta-llama/$MODEL llama stack build --run --template meta-reference-gpu

# install client to interact with the server
pip install llama-stack-client
```
### CLI
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
### Python SDK
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
As more providers start supporting Llama 4, you can use them in Llama Stack as well. We are adding to the list. Stay tuned!


</details>

## Benefits of Using Llama Stack

*   **Flexibility:**  Choose your infrastructure without code changes.
*   **Consistency:** Consistent application behavior across environments.
*   **Ecosystem:** Benefit from integrations with leading providers.

## API Providers

Llama Stack supports a wide range of API providers for inference and other features:

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

## Distributions

Llama Stack Distributions provide pre-configured bundles for various deployment scenarios:

|               **Distribution**                |                                                                    **Llama Stack Docker**                                                                     |                                                 Start This Distribution                                                  |
|:---------------------------------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------:|
|                Starter Distribution                 |           [llamastack/distribution-starter](https://hub.docker.com/repository/docker/llamastack/distribution-starter/general)           |      [Guide](https://llama-stack.readthedocs.io/en/latest/distributions/self_hosted_distro/starter.html)      |
|                Meta Reference                 |           [llamastack/distribution-meta-reference-gpu](https://hub.docker.com/repository/docker/llamastack/distribution-meta-reference-gpu/general)           |      [Guide](https://llama-stack.readthedocs.io/en/latest/distributions/self_hosted_distro/meta-reference-gpu.html)      |
|                   PostgreSQL                  |                [llamastack/distribution-postgres-demo](https://hub.docker.com/repository/docker/llamastack/distribution-postgres-demo/general)                |                  |

## Documentation & Resources

*   [Documentation](https://llama-stack.readthedocs.io/en/latest/index.html)
*   **CLI References:**
    *   [llama (server-side) CLI Reference](https://llama-stack.readthedocs.io/en/latest/references/llama_cli_reference/index.html)
    *   [llama (client-side) CLI Reference](https://llama-stack.readthedocs.io/en/latest/references/llama_stack_client_cli_reference.html)
*   **Getting Started:**
    *   [Quick guide](https://llama-stack.readthedocs.io/en/latest/getting_started/index.html)
    *   [Jupyter notebook](./docs/getting_started.ipynb)
    *   [Colab notebook](https://colab.research.google.com/drive/1dtVmxotBsI4cGZQNsJRYPrLiDeT0Wnwt)
    *   [Zero-to-Hero Guide](https://github.com/meta-llama/llama-stack/tree/main/docs/zero_to_hero_guide)
*   [Contributing](CONTRIBUTING.md)
    *   [Adding a new API Provider](https://llama-stack.readthedocs.io/en/latest/contributing/new_api_provider.html)

## Llama Stack Client SDKs

Choose your preferred language to interact with a Llama Stack server:

|  **Language** |  **Client SDK** | **Package** |
| :----: | :----: | :----: |
| Python |  [llama-stack-client-python](https://github.com/meta-llama/llama-stack-client-python) | [![PyPI version](https://img.shields.io/pypi/v/llama_stack_client.svg)](https://pypi.org/project/llama_stack_client/)
| Swift  | [llama-stack-client-swift](https://github.com/meta-llama/llama-stack-client-swift) | [![Swift Package Index](https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2Fmeta-llama%2Fllama-stack-client-swift%2Fbadge%3Ftype%3Dswift-versions)](https://swiftpackageindex.com/meta-llama/llama-stack-client-swift)
| Typescript   | [llama-stack-client-typescript](https://github.com/meta-llama/llama-stack-client-typescript) | [![NPM version](https://img.shields.io/npm/v/llama-stack-client.svg)](https://npmjs.org/package/llama-stack-client)
| Kotlin | [llama-stack-client-kotlin](https://github.com/meta-llama/llama-stack-client-kotlin) | [![Maven version](https://img.shields.io/maven-central/v/com.llama.llamastack/llama-stack-client-kotlin)](https://central.sonatype.com/artifact/com.llama.llamastack/llama-stack-client-kotlin)

Find example scripts with client SDKs in the [llama-stack-apps](https://github.com/meta-llama/llama-stack-apps/tree/main/examples) repository.

## 🌟 GitHub Star History

[Star History Chart](https://api.star-history.com/svg?repos=meta-llama/llama-stack&type=Date)

## ✨ Contributors

Thanks to all of our amazing contributors!

<a href="https://github.com/meta-llama/llama-stack/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=meta-llama/llama-stack" />
</a>
```
Key improvements and why:

*   **SEO-Optimized Title & Introduction:**  The title clearly states the core benefit.  The introduction now emphasizes the value proposition (build AI apps faster) and includes relevant keywords like "Llama" and "AI application development". The one-sentence hook highlights the core value proposition.
*   **Clear Headings:**  Uses H2 and H3 for better structure and readability.
*   **Bulleted Key Features:**  Improves scannability, and emphasizes the core benefits in a concise way.  Keywords are included.
*   **Concise Descriptions:**  Descriptions are trimmed and focused on the user's needs and benefits.
*   **Emphasis on Value:**  Rephrased the "Overview" section to be more user-focused and highlight the key benefits.
*   **Clear Calls to Action:**  The "One-Line Installer" and "Documentation" sections provide clear calls to action to encourage immediate engagement.
*   **Consistent Style and Formatting:** Improved formatting for better readability.
*   **Removed Unnecessary Details:**  Removed the images to save space
*   **Improved Organization:** Arranged the content logically.
*   **Keywords:**  The revised README is rich with keywords (Llama, AI, AI applications, inference, SDK, API, etc.) which increases the chances of it ranking higher in search results.
*   **Clear Links and Calls to Action:**  The documentation links are clear and helpful. The one-line install code is prominent.
*   **Maintained Original Content:**  Preserved the most important details from the original README while making it more user-friendly and SEO-optimized.
*   **Focus on Benefits:**  The emphasis is on what the user *gains* by using Llama Stack.
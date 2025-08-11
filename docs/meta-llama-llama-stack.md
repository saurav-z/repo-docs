# Llama Stack: Build AI Applications with Ease 🚀

**Llama Stack is a powerful framework that simplifies the development of AI applications by providing a unified API layer and pre-packaged distributions, enabling developers to seamlessly build and deploy with flexibility and choice. Learn more about Llama Stack on [GitHub](https://github.com/meta-llama/llama-stack).**

[![PyPI version](https://img.shields.io/pypi/v/llama_stack.svg)](https://pypi.org/project/llama-stack/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/llama-stack)](https://pypi.org/project/llama-stack/)
[![License](https://img.shields.io/pypi/l/llama_stack.svg)](https://github.com/meta-llama/llama-stack/blob/main/LICENSE)
[![Discord](https://img.shields.io/discord/1257833999603335178?color=6A7EC2&logo=discord&logoColor=ffffff)](https://discord.gg/llama-stack)
[![Unit Tests](https://github.com/meta-llama/llama-stack/actions/workflows/unit-tests.yml/badge.svg?branch=main)](https://github.com/meta-llama/llama-stack/actions/workflows/unit-tests.yml?query=branch%3Amain)
[![Integration Tests](https://github.com/meta-llama/llama-stack/actions/workflows/integration-tests.yml/badge.svg?branch=main)](https://github.com/meta-llama/llama-stack/actions/workflows/integration-tests.yml?query=branch%3Amain)

[**Quick Start**](https://llama-stack.readthedocs.io/en/latest/getting_started/index.html) | [**Documentation**](https://llama-stack.readthedocs.io/en/latest/index.html) | [**Colab Notebook**](./docs/getting_started.ipynb) | [**Discord**](https://discord.gg/llama-stack)

## Key Features

*   **Unified API Layer:** Standardizes core building blocks for inference, RAG, agents, tools, safety, evals, and telemetry, simplifying AI application development.
*   **Plugin Architecture:** Supports a diverse ecosystem of API implementations across various environments (local, on-premises, cloud, and mobile).
*   **Prepackaged Distributions:** Offers ready-to-use solutions for quick and reliable deployment in any environment.
*   **Multiple Developer Interfaces:** Provides CLI and SDKs for Python, Typescript, iOS, and Android, and more.
*   **Standalone Applications:** Includes examples to guide you on building production-grade AI applications.
*   **Llama 4 Support:** Now supports the Llama 4 family of models!

## What's New: Llama 4 Support!

We're excited to announce support for the Llama 4 models! Get started by following the instructions below:

<details>
<summary>Click here to see how to run Llama 4 models on Llama Stack</summary>

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

## Getting Started Quickly

### 🚀 One-Line Installation 🚀

Get started with Llama Stack locally:

```bash
curl -LsSf https://github.com/meta-llama/llama-stack/raw/main/scripts/install.sh | bash
```

## Benefits of Using Llama Stack

*   **Flexibility:** Choose your preferred infrastructure without changing APIs, and enjoy flexible deployment options.
*   **Consistency:** Build, test, and deploy AI applications with consistent behavior thanks to unified APIs.
*   **Robust Ecosystem:** Benefit from integrations with distribution partners (cloud providers, hardware vendors, and AI-focused companies) providing tailored infrastructure, software, and services for deploying Llama models.

Llama Stack streamlines the development process, so you can focus on innovation and building impressive generative AI applications.

## API Providers

Llama Stack supports a growing list of API providers. Explore the options below to kickstart your project.  See the [full list](https://llama-stack.readthedocs.io/en/latest/providers/index.html) for complete details.

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

> **Note:**  Additional providers are available through external packages. See [External Providers](https://llama-stack.readthedocs.io/en/latest/providers/external.html) for more details.

## Distributions: Your Deployment Options

Llama Stack Distributions provide pre-configured bundles to easily get started.

|               **Distribution**                |                                                                    **Llama Stack Docker**                                                                     |                                                 Start This Distribution                                                  |
|:---------------------------------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------:|
|                Starter Distribution                 |           [llamastack/distribution-starter](https://hub.docker.com/repository/docker/llamastack/distribution-starter/general)           |      [Guide](https://llama-stack.readthedocs.io/en/latest/distributions/self_hosted_distro/starter.html)      |
|                Meta Reference                 |           [llamastack/distribution-meta-reference-gpu](https://hub.docker.com/repository/docker/llamastack/distribution-meta-reference-gpu/general)           |      [Guide](https://llama-stack.readthedocs.io/en/latest/distributions/self_hosted_distro/meta-reference-gpu.html)      |
|                   PostgreSQL                  |                [llamastack/distribution-postgres-demo](https://hub.docker.com/repository/docker/llamastack/distribution-postgres-demo/general)                |                  |

## Documentation and Resources

*   **Comprehensive Documentation:**  Explore our detailed [Documentation](https://llama-stack.readthedocs.io/en/latest/index.html) for in-depth guides and references.
*   **CLI References:**
    *   [llama (server-side) CLI Reference](https://llama-stack.readthedocs.io/en/latest/references/llama_cli_reference/index.html): Learn how to use the `llama` CLI for model management and distribution building.
    *   [llama (client-side) CLI Reference](https://llama-stack.readthedocs.io/en/latest/references/llama_stack_client_cli_reference.html): Query information about your distribution with the `llama-stack-client` CLI.
*   **Getting Started Guides:**
    *   [Quick Start Guide](https://llama-stack.readthedocs.io/en/latest/getting_started/index.html): Jump in with a quick guide to launching your Llama Stack server.
    *   [Jupyter Notebook](docs/getting_started.ipynb): Walk through simple text and vision inference using the `llama_stack_client` APIs.
    *   [Colab Notebook](https://colab.research.google.com/drive/1dtVmxotBsI4cGZQNsJRYPrLiDeT0Wnwt): Access the complete Llama Stack lesson from the Deeplearning.ai Llama 3.2 course.
    *   [Zero-to-Hero Guide](https://github.com/meta-llama/llama-stack/tree/main/docs/zero_to_hero_guide): Explore all the key components of Llama Stack with code samples.
*   **Contributing:**  Review our [Contributing Guidelines](CONTRIBUTING.md) and [Adding a new API Provider](https://llama-stack.readthedocs.io/en/latest/contributing/new_api_provider.html) for information on how to contribute.

## Client SDKs

Easily connect to your Llama Stack server with our client SDKs:

|  **Language** |  **Client SDK** | **Package** |
| :----: | :----: | :----: |
| Python |  [llama-stack-client-python](https://github.com/meta-llama/llama-stack-client-python) | [![PyPI version](https://img.shields.io/pypi/v/llama_stack_client.svg)](https://pypi.org/project/llama_stack_client/)
| Swift  | [llama-stack-client-swift](https://github.com/meta-llama/llama-stack-client-swift) | [![Swift Package Index](https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2Fmeta-llama%2Fllama-stack-client-swift%2Fbadge%3Ftype%3Dswift-versions)](https://swiftpackageindex.com/meta-llama/llama-stack-client-swift)
| Typescript   | [llama-stack-client-typescript](https://github.com/meta-llama/llama-stack-client-typescript) | [![NPM version](https://img.shields.io/npm/v/llama-stack-client.svg)](https://npmjs.org/package/llama-stack-client)
| Kotlin | [llama-stack-client-kotlin](https://github.com/meta-llama/llama-stack-client-kotlin) | [![Maven version](https://img.shields.io/maven-central/v/com.llama.llamastack/llama-stack-client-kotlin)](https://central.sonatype.com/artifact/com.llama.llamastack/llama-stack-client-kotlin)

Explore example scripts to talk with your Llama Stack server in the [llama-stack-apps](https://github.com/meta-llama/llama-stack-apps/tree/main/examples) repo.

## Community & Support

*   [Discord](https://discord.gg/llama-stack)

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=meta-llama/llama-stack&type=Date)](https://www.star-history.com/#meta-llama/llama-stack&Date)

## ✨ Contributors

A huge thanks to all our amazing contributors!

<a href="https://github.com/meta-llama/llama-stack/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=meta-llama/llama-stack" />
</a>
```
Key improvements:

*   **SEO Optimization:** Added a concise and compelling introduction for immediate user engagement.  Used keywords like "AI applications", "framework", "unified API", "pre-packaged distributions", and "Llama".
*   **Clear Headings:**  Organized the content using clear, descriptive headings for improved readability and navigation.
*   **Bulleted Key Features:** Presented key features in a clear and concise bulleted list.
*   **Concise Explanations:** Improved explanations while keeping it brief.
*   **Call to Actions:** Encouraged users to explore resources like the documentation and discord channel.
*   **Llama 4 Update Highlighted:**  The new Llama 4 support is highlighted prominently.
*   **Comprehensive Information:** Retained all crucial details from the original README.
*   **Structure:** Improved the overall structure for better information flow.
*   **Emphasis on Benefits:**  Highlighting the benefits of using Llama Stack to attract users.
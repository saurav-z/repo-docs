# Llama Stack: Build AI Applications Faster with a Unified API (Meta)

**Llama Stack simplifies AI application development by providing a unified API layer and flexible deployment options.** Explore the power of Llama Stack and [discover how it can revolutionize your AI projects!](https://github.com/meta-llama/llama-stack)

[![PyPI version](https://img.shields.io/pypi/v/llama_stack.svg)](https://pypi.org/project/llama_stack/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/llama-stack)](https://pypi.org/project/llama-stack/)
[![License](https://img.shields.io/pypi/l/llama_stack.svg)](https://github.com/meta-llama/llama-stack/blob/main/LICENSE)
[![Discord](https://img.shields.io/discord/1257833999603335178?color=6A7EC2&logo=discord&logoColor=ffffff)](https://discord.gg/llama-stack)
[![Unit Tests](https://github.com/meta-llama/llama-stack/actions/workflows/unit-tests.yml/badge.svg?branch=main)](https://github.com/meta-llama/llama-stack/actions/workflows/unit-tests.yml?query=branch%3Amain)
[![Integration Tests](https://github.com/meta-llama/llama-stack/actions/workflows/integration-tests.yml/badge.svg?branch=main)](https://github.com/meta-llama/llama-stack/actions/workflows/integration-tests.yml?query=branch%3Amain)

## Key Features

*   **Unified API for AI**: Standardizes core building blocks for inference, RAG, agents, tools, safety, and telemetry.
*   **Modular Plugin Architecture**: Supports a wide range of API implementations across various environments (local, cloud, on-premise, mobile).
*   **Pre-packaged Distributions**: Offers ready-to-use solutions for rapid and reliable deployments in any environment.
*   **Multi-Language SDKs**: Provides versatile developer interfaces like CLI and SDKs in Python, TypeScript, iOS, and Android.
*   **Production-Ready Examples**: Includes standalone applications to guide the development of production-grade AI applications.

## What's New?

### ✨ Llama 4 Support 🎉

Llama Stack now supports the Meta's Llama 4 family of models.

<details>
<summary>Run Llama 4 Models with Llama Stack</summary>

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

## Quick Start

### 🚀 One-Line Installation 🚀

Get started quickly with Llama Stack by running the following command:

```bash
curl -LsSf https://github.com/meta-llama/llama-stack/raw/main/scripts/install.sh | bash
```

## Overview: Streamlining AI Application Development

Llama Stack is designed to streamline the development of AI applications. It provides a unified API layer and plugin architecture for flexibility, pre-packaged distributions for ease of use, and developer interfaces to suit various needs. This allows developers to focus on building innovative AI applications rather than getting bogged down by infrastructure complexities.

<div style="text-align: center;">
  <img
    src="https://github.com/user-attachments/assets/33d9576d-95ea-468d-95e2-8fa233205a50"
    width="480"
    title="Llama Stack"
    alt="Llama Stack"
  />
</div>

## Benefits of Using Llama Stack

*   **Flexible Infrastructure Choices**: Seamlessly switch between infrastructure providers without changing your application code.
*   **Consistent Application Behavior**: Achieve consistent results across different deployments with unified APIs.
*   **Robust Ecosystem Integration**: Benefit from integrations with leading cloud providers, hardware vendors, and AI-focused companies.

## API Providers

Llama Stack supports a variety of API providers to meet different needs.  See [full list](https://llama-stack.readthedocs.io/en/latest/providers/index.html) for the latest.

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

> **Note**: Explore additional providers in our [External Providers](https://llama-stack.readthedocs.io/en/latest/providers/external.html) documentation.

## Distributions

Llama Stack Distributions provide pre-configured bundles for quick and easy setup.

| **Distribution**                | **Llama Stack Docker** | Start Distribution |
|:-----------------------------:|:-------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------:|
| Starter Distribution | [llamastack/distribution-starter](https://hub.docker.com/repository/docker/llamastack/distribution-starter/general) | [Guide](https://llama-stack.readthedocs.io/en/latest/distributions/self_hosted_distro/starter.html) |
| Meta Reference   | [llamastack/distribution-meta-reference-gpu](https://hub.docker.com/repository/docker/llamastack/distribution-meta-reference-gpu/general)   | [Guide](https://llama-stack.readthedocs.io/en/latest/distributions/self_hosted_distro/meta-reference-gpu.html)   |
| PostgreSQL   | [llamastack/distribution-postgres-demo](https://hub.docker.com/repository/docker/llamastack/distribution-postgres-demo/general)   |   |

## Documentation & Resources

*   **[Quick Start](https://llama-stack.readthedocs.io/en/latest/getting_started/index.html)**: Get started with a Llama Stack server.
*   **[Documentation](https://llama-stack.readthedocs.io/en/latest/index.html)**: Detailed documentation and guides.
*   **CLI References:**
    *   [llama (server-side) CLI Reference](https://llama-stack.readthedocs.io/en/latest/references/llama_cli_reference/index.html)
    *   [llama (client-side) CLI Reference](https://llama-stack.readthedocs.io/en/latest/references/llama_stack_client_cli_reference.html)
*   **[Getting Started](https://llama-stack.readthedocs.io/en/latest/getting_started/index.html)**
*   **[Jupyter notebook](./docs/getting_started.ipynb)**: Walk-through simple text and vision inference llama_stack_client APIs
*   **[Colab notebook](https://colab.research.google.com/drive/1dtVmxotBsI4cGZQNsJRYPrLiDeT0Wnwt)**: Learn from the Llama 3.2 course on Deeplearning.ai
*   **[Zero-to-Hero Guide](https://github.com/meta-llama/llama-stack/tree/main/docs/zero_to_hero_guide)**: Guide through all the key components with code samples.
*   **[Contributing](CONTRIBUTING.md)**: Learn how to contribute.
*   **[Adding a new API Provider](https://llama-stack.readthedocs.io/en/latest/contributing/new_api_provider.html)**

## Llama Stack Client SDKs

|  **Language** |  **Client SDK** | Package |
| :----: | :----: | :----: |
| Python |  [llama-stack-client-python](https://github.com/meta-llama/llama-stack-client-python) | [![PyPI version](https://img.shields.io/pypi/v/llama_stack_client.svg)](https://pypi.org/project/llama_stack_client/)
| Swift  | [llama-stack-client-swift](https://github.com/meta-llama/llama-stack-client-swift) | [![Swift Package Index](https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2Fmeta-llama%2Fllama-stack-client-swift%2Fbadge%3Ftype%3Dswift-versions)](https://swiftpackageindex.com/meta-llama/llama-stack-client-swift)
| Typescript   | [llama-stack-client-typescript](https://github.com/meta-llama/llama-stack-client-typescript) | [![NPM version](https://img.shields.io/npm/v/llama-stack-client.svg)](https://npmjs.org/package/llama-stack-client)
| Kotlin | [llama-stack-client-kotlin](https://github.com/meta-llama/llama-stack-client-kotlin) | [![Maven version](https://img.shields.io/maven-central/v/com.llama.llamastack/llama-stack-client-kotlin)](https://central.sonatype.com/artifact/com.llama.llamastack/llama-stack-client-kotlin)

Choose from our client SDKs: [python](https://github.com/meta-llama/llama-stack-client-python), [typescript](https://github.com/meta-llama/llama-stack-client-typescript), [swift](https://github.com/meta-llama/llama-stack-client-swift), and [kotlin](https://github.com/meta-llama/llama-stack-client-kotlin).  Find example scripts in our [llama-stack-apps](https://github.com/meta-llama/llama-stack-apps/tree/main/examples) repo.

## 🌟 GitHub Star History

[![Star History Chart](https://api.star-history.com/svg?repos=meta-llama/llama-stack&type=Date)](https://www.star-history.com/#meta-llama/llama-stack&Date)

## ✨ Contributors

A big thank you to all our amazing contributors!

<a href="https://github.com/meta-llama/llama-stack/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=meta-llama/llama-stack" />
</a>
```
Key improvements and SEO considerations:

*   **Clear Title and Hook**:  Strong title with keyword "Llama Stack" and a concise, impactful first sentence.
*   **Keyword Optimization**:  Incorporates relevant keywords (AI application, Meta, unified API, etc.) throughout.
*   **Structured Content**: Uses headings, subheadings, and lists to improve readability and SEO ranking.
*   **Bulleted Key Features**:  Highlights the most important aspects in an easy-to-scan format.
*   **Focus on Benefits**:  Emphasizes the value proposition for developers (speed, flexibility, consistency).
*   **Internal Linking**: Includes links to other sections within the README.
*   **External Links**: Includes links to the project's GitHub repo and other resources.
*   **Concise and Focused**: Streamlines the information to be more impactful.
*   **SEO-Friendly Markdown**: Uses proper markdown formatting for headings, lists, and emphasis.
*   **Contributor Section**: Includes a contributor section to build trust and community.
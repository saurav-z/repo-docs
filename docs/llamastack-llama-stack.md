# Llama Stack: Build Powerful AI Applications with Ease

**Llama Stack simplifies AI application development by providing a unified API and a flexible, extensible architecture.**  Explore the [original repository](https://github.com/llamastack/llama-stack) for more details.

[![PyPI version](https://img.shields.io/pypi/v/llama_stack.svg)](https://pypi.org/project/llama_stack/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/llama-stack)](https://pypi.org/project/llama-stack/)
[![License](https://img.shields.io/pypi/l/llama_stack.svg)](https://github.com/meta-llama/llama-stack/blob/main/LICENSE)
[![Discord](https://img.shields.io/discord/1257833999603335178?color=6A7EC2&logo=discord&logoColor=ffffff)](https://discord.gg/llama-stack)
[![Unit Tests](https://github.com/meta-llama/llama-stack/actions/workflows/unit-tests.yml/badge.svg?branch=main)](https://github.com/meta-llama/llama-stack/actions/workflows/unit-tests.yml?query=branch%3Amain)
[![Integration Tests](https://github.com/meta-llama/llama-stack/actions/workflows/integration-tests.yml/badge.svg?branch=main)](https://github.com/meta-llama/llama-stack/actions/workflows/integration-tests.yml?query=branch%3Amain)

*   [**Quick Start**](https://llama-stack.readthedocs.io/en/latest/getting_started/index.html) |
    [**Documentation**](https://llama-stack.readthedocs.io/en/latest/index.html) |
    [**Colab Notebook**](./docs/getting_started.ipynb) |
    [**Discord**](https://discord.gg/llama-stack)

## Key Features

*   **Unified API:** Standardizes core building blocks for inference, RAG, agents, tools, safety, and more, simplifying your AI development.
*   **Plugin Architecture:**  Supports a vast ecosystem of API implementations across various environments, including local development, on-premises, cloud, and mobile.
*   **Prepackaged Distributions:** Offers ready-to-use solutions to quickly get started with consistent behavior across environments.
*   **Multi-Language SDKs:**  Provides CLI and SDKs in Python, Typescript, iOS, and Android, empowering developers to use their preferred tools.
*   **Standalone Applications:**  Includes example applications demonstrating how to build production-grade AI applications.
*   **Llama 4 Support:**  Includes support for the Llama 4 models, enabling you to build on the latest advancements (See below).

## Llama 4 Support
Llama Stack offers support for Meta's Llama 4 models. To run these models on Llama Stack, follow these steps:
1.  Install Llama Stack: `pip install -U llama_stack`
2.  Download the model using `llama model download --source meta --model-id $MODEL --meta-url <META_URL>` where META_URL can be found from llama.com
3.  Build and run a Llama Stack server: `INFERENCE_MODEL=meta-llama/$MODEL llama stack build --run --template meta-reference-gpu`
4.  Install the client `pip install llama-stack-client`
5.  Use CLI, or Python SDK:

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

## One-Line Installation

Get started quickly with a local Llama Stack instance:

```bash
curl -LsSf https://github.com/meta-llama/llama-stack/raw/main/scripts/install.sh | bash
```

## Overview

Llama Stack is designed to simplify AI application development, by codifying best practices across the Llama ecosystem. It offers:

*   **Unified API Layer:** For Inference, RAG, Agents, Tools, Safety, Evals, and Telemetry.
*   **Plugin Architecture:** Supporting diverse API implementations (local, on-premise, cloud, mobile).
*   **Prepackaged Verified Distributions:** A one-stop solution to quickly get started.
*   **Multiple Developer Interfaces:** CLI and SDKs for Python, Typescript, iOS, and Android.
*   **Standalone Applications:** Example applications for building production-grade AI applications.

<div style="text-align: center;">
  <img
    src="https://github.com/user-attachments/assets/33d9576d-95ea-468d-95e2-8fa233205a50"
    width="480"
    title="Llama Stack"
    alt="Llama Stack"
  />
</div>

## Benefits of Using Llama Stack

*   **Flexible Options:** Choose your preferred infrastructure without changing APIs and enjoy flexible deployment choices.
*   **Consistent Experience:**  Build, test, and deploy AI applications with consistent application behavior.
*   **Robust Ecosystem:**  Integrated with distribution partners (cloud providers, hardware vendors, and AI-focused companies).

## API Providers

Llama Stack supports a wide range of API providers.  See the [full list](https://llama-stack.readthedocs.io/en/latest/providers/index.html) for the most up-to-date information.

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

>   **Note:** Additional providers are available through external packages. See [External Providers](https://llama-stack.readthedocs.io/en/latest/providers/external.html) documentation.

## Distributions

Llama Stack Distributions ("distros") bundle provider implementations for specific deployment scenarios.

|               **Distribution**                |                                                                    **Llama Stack Docker**                                                                     |                                                 Start This Distribution                                                  |
|:---------------------------------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------:|
|                Starter Distribution                 |           [llamastack/distribution-starter](https://hub.docker.com/repository/docker/llamastack/distribution-starter/general)           |      [Guide](https://llama-stack.readthedocs.io/en/latest/distributions/self_hosted_distro/starter.html)      |
|                Meta Reference                 |           [llamastack/distribution-meta-reference-gpu](https://hub.docker.com/repository/docker/llamastack/distribution-meta-reference-gpu/general)           |      [Guide](https://llama-stack.readthedocs.io/en/latest/distributions/self_hosted_distro/meta-reference-gpu.html)      |
|                   PostgreSQL                  |                [llamastack/distribution-postgres-demo](https://hub.docker.com/repository/docker/llamastack/distribution-postgres-demo/general)                |                  |

## Documentation

Comprehensive documentation is available on the [Documentation](https://llama-stack.readthedocs.io/en/latest/index.html) page.

*   **CLI References:**
    *   [llama (server-side) CLI Reference](https://llama-stack.readthedocs.io/en/latest/references/llama_cli_reference/index.html)
    *   [llama (client-side) CLI Reference](https://llama-stack.readthedocs.io/en/latest/references/llama_stack_client_cli_reference.html)
*   **Getting Started:**
    *   [Quick Start Guide](https://llama-stack.readthedocs.io/en/latest/getting_started/index.html).
    *   [Jupyter notebook](./docs/getting_started.ipynb)
    *   [Colab notebook](https://colab.research.google.com/drive/1dtVmxotBsI4cGZQNsJRYPrLiDeT0Wnwt)
    *   [Zero-to-Hero Guide](https://github.com/meta-llama/llama-stack/tree/main/docs/zero_to_hero_guide)
*   [Contributing](CONTRIBUTING.md)
    *   [Adding a new API Provider](https://llama-stack.readthedocs.io/en/latest/contributing/new_api_provider.html)

## Llama Stack Client SDKs

Choose your preferred language for connecting to a Llama Stack server:

|  **Language** |  **Client SDK** | **Package** |
| :----: | :----: | :----: |
| Python |  [llama-stack-client-python](https://github.com/meta-llama/llama-stack-client-python) | [![PyPI version](https://img.shields.io/pypi/v/llama_stack_client.svg)](https://pypi.org/project/llama_stack_client/)
| Swift  | [llama-stack-client-swift](https://github.com/meta-llama/llama-stack-client-swift) | [![Swift Package Index](https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2Fmeta-llama%2Fllama-stack-client-swift%2Fbadge%3Ftype%3Dswift-versions)](https://swiftpackageindex.com/meta-llama/llama-stack-client-swift)
| Typescript   | [llama-stack-client-typescript](https://github.com/meta-llama/llama-stack-client-typescript) | [![NPM version](https://img.shields.io/npm/v/llama-stack-client.svg)](https://npmjs.org/package/llama-stack-client)
| Kotlin | [llama-stack-client-kotlin](https://github.com/meta-llama/llama-stack-client-kotlin) | [![Maven version](https://img.shields.io/maven-central/v/com.llama.llamastack/llama-stack-client-kotlin)](https://central.sonatype.com/artifact/com.llama.llamastack/llama-stack-client-kotlin)

Find more example scripts in the [llama-stack-apps](https://github.com/meta-llama/llama-stack-apps/tree/main/examples) repository.

## 🌟 GitHub Star History
## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=meta-llama/llama-stack&type=Date)](https://www.star-history.com/#meta-llama/llama-stack&Date)

## ✨ Contributors

Thank you to all our amazing contributors!

<a href="https://github.com/meta-llama/llama-stack/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=meta-llama/llama-stack" />
</a>
# Llama Stack: Build Production-Ready Generative AI Applications

**Llama Stack simplifies and standardizes the core components for building robust AI applications, enabling developers to focus on innovation.**  Get started with the [Llama Stack](https://github.com/meta-llama/llama-stack) project.

[![PyPI version](https://img.shields.io/pypi/v/llama_stack.svg)](https://pypi.org/project/llama_stack/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/llama-stack)](https://pypi.org/project/llama-stack/)
[![License](https://img.shields.io/pypi/l/llama_stack.svg)](https://github.com/meta-llama/llama-stack/blob/main/LICENSE)
[![Discord](https://img.shields.io/discord/1257833999603335178?color=6A7EC2&logo=discord&logoColor=ffffff)](https://discord.gg/llama-stack)
[![Unit Tests](https://github.com/meta-llama/llama-stack/actions/workflows/unit-tests.yml/badge.svg?branch=main)](https://github.com/meta-llama/llama-stack/actions/workflows/unit-tests.yml?query=branch%3Amain)
[![Integration Tests](https://github.com/meta-llama/llama-stack/actions/workflows/integration-tests.yml/badge.svg?branch=main)](https://github.com/meta-llama/llama-stack/actions/workflows/integration-tests.yml?query=branch%3Amain)
![coverage badge](./coverage.svg)

*   [**Quick Start**](https://llama-stack.readthedocs.io/en/latest/getting_started/index.html) | [**Documentation**](https://llama-stack.readthedocs.io/en/latest/index.html) | [**Colab Notebook**](./docs/getting_started.ipynb) | [**Discord**](https://discord.gg/llama-stack)

## Key Features

*   **Unified API Layer:** Standardizes APIs for Inference, RAG, Agents, Tools, Safety, Evals, and Telemetry.
*   **Plugin Architecture:** Supports a broad ecosystem of API implementations across various environments (local, cloud, on-premise, mobile).
*   **Prepackaged Distributions:** Offers a one-stop solution for developers to quickly and reliably get started.
*   **Multiple Developer Interfaces:** Provides CLI and SDKs for Python, Typescript, iOS, and Android.
*   **Standalone Applications:** Includes examples for building production-grade AI applications.
*   **Llama 4 Support:** Supports the Llama 4 model family (see details below).

## Llama 4 Support

Llama Stack now offers support for the Llama 4 family of models. Version 0.2.0 released with support for the models released by Meta.

<details>
<summary>👋 Click here to see how to run Llama 4 models on Llama Stack </summary>

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

Get started with Llama Stack quickly using a single command:

```bash
curl -LsSf https://github.com/meta-llama/llama-stack/raw/main/scripts/install.sh | bash
```

## Overview

Llama Stack provides the core building blocks to simplify AI application development, codifying best practices across the Llama ecosystem.

<div style="text-align: center;">
  <img
    src="https://github.com/user-attachments/assets/33d9576d-95ea-468d-95e2-8fa233205a50"
    width="480"
    title="Llama Stack"
    alt="Llama Stack"
  />
</div>

## Benefits of Using Llama Stack

*   **Flexible Infrastructure Options:**  Choose your preferred infrastructure without changing APIs, along with flexible deployment choices.
*   **Consistent Application Behavior:** Unified APIs make it easier to build, test, and deploy AI applications.
*   **Robust Ecosystem:** Integrated with distribution partners offering tailored infrastructure, software, and services for deploying Llama models.

## API Providers

Llama Stack supports a variety of API providers. See the [full list](https://llama-stack.readthedocs.io/en/latest/providers/index.html) and get started easily.

| API Provider Builder | Environments | Agents | Inference | VectorIO | Safety | Telemetry | Post Training | Eval | DatasetIO |
|:-------------------:|:------------:|:------:|:---------:|:--------:|:------:|:---------:|:-------------:|:----:|:--------:|
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

> **Note**: Additional providers are available through external packages. See [External Providers](https://llama-stack.readthedocs.io/en/latest/providers/external.html) documentation.

## Distributions

Llama Stack Distributions (or "distros") are pre-configured bundles of provider implementations for each API component.

|               **Distribution**                |                                                                    **Llama Stack Docker**                                                                     |                                                 Start This Distribution                                                  |
|:---------------------------------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------:|
|                Starter Distribution                 |           [llamastack/distribution-starter](https://hub.docker.com/repository/docker/llamastack/distribution-starter/general)           |      [Guide](https://llama-stack.readthedocs.io/en/latest/distributions/self_hosted_distro/starter.html)      |
|                Meta Reference                 |           [llamastack/distribution-meta-reference-gpu](https://hub.docker.com/repository/docker/llamastack/distribution-meta-reference-gpu/general)           |      [Guide](https://llama-stack.readthedocs.io/en/latest/distributions/self_hosted_distro/meta-reference-gpu.html)      |
|                   PostgreSQL                  |                [llamastack/distribution-postgres-demo](https://hub.docker.com/repository/docker/llamastack/distribution-postgres-demo/general)                |                  |

## Documentation

Explore comprehensive documentation:

*   **CLI References:**
    *   [`llama` CLI Reference](https://llama-stack.readthedocs.io/en/latest/references/llama_cli_reference/index.html) - guide for working with Llama models.
    *   [`llama-stack-client` CLI Reference](https://llama-stack.readthedocs.io/en/latest/references/llama_stack_client_cli_reference.html) - guide for querying information about the distribution.
*   **Getting Started:**
    *   [Quick Start Guide](https://llama-stack.readthedocs.io/en/latest/getting_started/index.html)
    *   [Jupyter Notebook](docs/getting_started.ipynb)
    *   [Colab Notebook](https://colab.research.google.com/drive/1dtVmxotBsI4cGZQNsJRYPrLiDeT0Wnwt)
    *   [Zero-to-Hero Guide](https://github.com/meta-llama/llama-stack/tree/main/docs/zero_to_hero_guide)
*   **Contributing:**
    *   [Contributing Guide](CONTRIBUTING.md)
    *   [Adding a New API Provider](https://llama-stack.readthedocs.io/en/latest/contributing/new_api_provider.html)

## Llama Stack Client SDKs

Connect to your Llama Stack server using these client SDKs:

|  **Language** |  **Client SDK** | **Package** |
| :----: | :----: | :----: |
| Python |  [llama-stack-client-python](https://github.com/meta-llama/llama-stack-client-python) | [![PyPI version](https://img.shields.io/pypi/v/llama_stack_client.svg)](https://pypi.org/project/llama_stack_client/)
| Swift  | [llama-stack-client-swift](https://github.com/meta-llama/llama-stack-client-swift) | [![Swift Package Index](https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2Fmeta-llama%2Fllama-stack-client-swift%2Fbadge%3Ftype%3Dswift-versions)](https://swiftpackageindex.com/meta-llama/llama-stack-client-swift)
| Typescript   | [llama-stack-client-typescript](https://github.com/meta-llama/llama-stack-client-typescript) | [![NPM version](https://img.shields.io/npm/v/llama-stack-client.svg)](https://npmjs.org/package/llama-stack-client)
| Kotlin | [llama-stack-client-kotlin](https://github.com/meta-llama/llama-stack-client-kotlin) | [![Maven version](https://img.shields.io/maven-central/v/com.llama.llamastack/llama-stack-client-kotlin)](https://central.sonatype.com/artifact/com.llama.llamastack/llama-stack-client-kotlin)

Find example scripts in the [llama-stack-apps](https://github.com/meta-llama/llama-stack-apps/tree/main/examples) repository.
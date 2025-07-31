# Llama Stack: Build and Deploy AI Applications with Ease

**Llama Stack simplifies AI application development by providing a unified API layer and pre-configured distributions, making it easier than ever to build, test, and deploy your AI projects.** (Find the original repository [here](https://github.com/meta-llama/llama-stack).)

[![PyPI version](https://img.shields.io/pypi/v/llama_stack.svg)](https://pypi.org/project/llama_stack/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/llama-stack)](https://pypi.org/project/llama-stack/)
[![License](https://img.shields.io/pypi/l/llama_stack.svg)](https://github.com/meta-llama/llama-stack/blob/main/LICENSE)
[![Discord](https://img.shields.io/discord/1257833999603335178?color=6A7EC2&logo=discord&logoColor=ffffff)](https://discord.gg/llama-stack)
[![Unit Tests](https://github.com/meta-llama/llama-stack/actions/workflows/unit-tests.yml/badge.svg?branch=main)](https://github.com/meta-llama/llama-stack/actions/workflows/unit-tests.yml?query=branch%3Amain)
[![Integration Tests](https://github.com/meta-llama/llama-stack/actions/workflows/integration-tests.yml/badge.svg?branch=main)](https://github.com/meta-llama/llama-stack/actions/workflows/integration-tests.yml?query=branch%3Amain)
![coverage badge](./coverage.svg)

*   [**Quick Start**](https://llama-stack.readthedocs.io/en/latest/getting_started/index.html)
*   [**Documentation**](https://llama-stack.readthedocs.io/en/latest/index.html)
*   [**Colab Notebook**](./docs/getting_started.ipynb)
*   [**Discord**](https://discord.gg/llama-stack)

## Key Features

*   **Unified API Layer:** Standardizes inference, RAG, agents, tools, safety, evaluations, and telemetry.
*   **Plugin Architecture:** Supports a wide range of API implementations across diverse environments.
*   **Prepackaged Distributions:** Offers ready-to-use solutions for various deployment scenarios.
*   **Multiple Developer Interfaces:** Provides CLI and SDKs for Python, Typescript, iOS, and Android.
*   **Standalone Applications:** Includes examples for building production-grade AI applications.

## What's New?

### ✨🎉 Llama 4 Support 🎉✨

Llama Stack now supports the latest Llama 4 models released by Meta.

<details>

<summary>👋 Click here to see how to run Llama 4 models on Llama Stack </summary>

\
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

Get started quickly by running:

```bash
curl -LsSf https://github.com/meta-llama/llama-stack/raw/main/scripts/install.sh | bash
```

## Overview

Llama Stack streamlines AI application development by codifying best practices across the Llama ecosystem. It provides a comprehensive set of tools and features designed to simplify the development process.

### Llama Stack Benefits

*   **Flexibility:** Choose your preferred infrastructure without changing APIs.
*   **Consistency:** Build, test, and deploy AI applications with consistent behavior via unified APIs.
*   **Robust Ecosystem:** Integrates with partners offering tailored infrastructure, software, and services.

Llama Stack empowers developers to focus on building innovative generative AI applications by reducing friction and complexity.

## API Providers

Llama Stack supports a wide range of API providers. Here's a partial list:

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

> **Note:** Additional providers are available via external packages. See [External Providers](https://llama-stack.readthedocs.io/en/latest/providers/external.html) documentation.

## Distributions

Llama Stack Distributions (or "distros") are pre-configured bundles of provider implementations. They make it easy to get started with a specific deployment scenario.

|               **Distribution**                |                                                                    **Llama Stack Docker**                                                                     |                                                 Start This Distribution                                                  |
|:---------------------------------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------:|
|                Starter Distribution                 |           [llamastack/distribution-starter](https://hub.docker.com/repository/docker/llamastack/distribution-starter/general)           |      [Guide](https://llama-stack.readthedocs.io/en/latest/distributions/self_hosted_distro/starter.html)      |
|                Meta Reference                 |           [llamastack/distribution-meta-reference-gpu](https://hub.docker.com/repository/docker/llamastack/distribution-meta-reference-gpu/general)           |      [Guide](https://llama-stack.readthedocs.io/en/latest/distributions/self_hosted_distro/meta-reference-gpu.html)      |
|                   PostgreSQL                  |                [llamastack/distribution-postgres-demo](https://hub.docker.com/repository/docker/llamastack/distribution-postgres-demo/general)                |                  |

## Documentation

Explore comprehensive documentation for Llama Stack:

*   [CLI Reference](https://llama-stack.readthedocs.io/en/latest/references/llama_cli_reference/index.html) & [Client CLI Reference](https://llama-stack.readthedocs.io/en/latest/references/llama_stack_client_cli_reference.html): Detailed guides for using the CLI tools.
*   [Getting Started](https://llama-stack.readthedocs.io/en/latest/getting_started/index.html): Quick start guides and a Jupyter notebook.
*   [Llama 3.2 Colab Notebook](https://colab.research.google.com/drive/1dtVmxotBsI4cGZQNsJRYPrLiDeT0Wnwt):  Learn more about how to use Llama Stack.
*   [Zero-to-Hero Guide](https://github.com/meta-llama/llama-stack/tree/main/docs/zero_to_hero_guide): Comprehensive guide with code samples.
*   [Contributing Guide](CONTRIBUTING.md): Learn how to contribute and add new API providers.

## Llama Stack Client SDKs

Choose your preferred language:

|  **Language** |  **Client SDK** | **Package** |
| :----: | :----: | :----: |
| Python |  [llama-stack-client-python](https://github.com/meta-llama/llama-stack-client-python) | [![PyPI version](https://img.shields.io/pypi/v/llama_stack_client.svg)](https://pypi.org/project/llama_stack_client/)
| Swift  | [llama-stack-client-swift](https://github.com/meta-llama/llama-stack-client-swift) | [![Swift Package Index](https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2Fmeta-llama%2Fllama-stack-client-swift%2Fbadge%3Ftype%3Dswift-versions)](https://swiftpackageindex.com/meta-llama/llama-stack-client-swift)
| Typescript   | [llama-stack-client-typescript](https://github.com/meta-llama/llama-stack-client-typescript) | [![NPM version](https://img.shields.io/npm/v/llama-stack-client.svg)](https://npmjs.org/package/llama-stack-client)
| Kotlin | [llama-stack-client-kotlin](https://github.com/meta-llama/llama-stack-client-kotlin) | [![Maven version](https://img.shields.io/maven-central/v/com.llama.llamastack/llama-stack-client-kotlin)](https://central.sonatype.com/artifact/com.llama.llamastack/llama-stack-client-kotlin)

Explore client SDKs in Python, Typescript, Swift, and Kotlin. Check out the [llama-stack-apps](https://github.com/meta-llama/llama-stack-apps/tree/main/examples) repo for more examples.
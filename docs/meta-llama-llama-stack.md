# Llama Stack: Build and Deploy Generative AI Applications with Ease

**Llama Stack simplifies AI application development by providing a standardized and flexible platform for building, deploying, and scaling generative AI applications.**  [Explore the Llama Stack Repository](https://github.com/meta-llama/llama-stack)

[![PyPI version](https://img.shields.io/pypi/v/llama_stack.svg)](https://pypi.org/project/llama_stack/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/llama-stack)](https://pypi.org/project/llama-stack/)
[![License](https://img.shields.io/pypi/l/llama_stack.svg)](https://github.com/meta-llama/llama-stack/blob/main/LICENSE)
[![Discord](https://img.shields.io/discord/1257833999603335178?color=6A7EC2&logo=discord&logoColor=ffffff)](https://discord.gg/llama-stack)
[![Unit Tests](https://github.com/meta-llama/llama-stack/actions/workflows/unit-tests.yml/badge.svg?branch=main)](https://github.com/meta-llama/llama-stack/actions/workflows/unit-tests.yml?query=branch%3Amain)
[![Integration Tests](https://github.com/meta-llama/llama-stack/actions/workflows/integration-tests.yml/badge.svg?branch=main)](https://github.com/meta-llama/llama-stack/actions/workflows/integration-tests.yml?query=branch%3Amain)
![coverage badge](./coverage.svg)

[**Quick Start**](https://llama-stack.readthedocs.io/en/latest/getting_started/index.html) | [**Documentation**](https://llama-stack.readthedocs.io/en/latest/index.html) | [**Colab Notebook**](./docs/getting_started.ipynb) | [**Discord**](https://discord.gg/llama-stack)

## Key Features

*   **Unified API Layer:** Standardized APIs for Inference, RAG, Agents, Tools, Safety, Evals, and Telemetry, promoting consistent behavior.
*   **Plugin Architecture:** Supports various API implementations across different environments (local, cloud, mobile).
*   **Prepackaged Distributions:** Ready-to-use bundles for rapid deployment in diverse scenarios.
*   **Multi-language SDKs:** Including CLI and SDKs for Python, Typescript, iOS, and Android.
*   **Standalone Applications:** Example applications to showcase production-grade AI application development.
*   **Llama 4 Support:**  Supports the latest Llama 4 models.

## What's New: Llama 4 Support!

Version 0.2.0 brings exciting support for the Llama 4 family of models.

<details>
<summary>👋 Click here to see how to run Llama 4 models on Llama Stack </summary>

```bash
# Install and configure the Llama Stack client with Llama 4 models
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

</details>

## Quick Installation

Get started locally with a single command:

```bash
curl -LsSf https://github.com/meta-llama/llama-stack/raw/main/scripts/install.sh | bash
```

## Overview

Llama Stack streamlines AI application development by standardizing core components and codifying best practices within the Llama ecosystem.  It provides a comprehensive solution for:

*   **Unified API:** Ensures consistent application behavior.
*   **Plugin Architecture:** Supports diverse API implementations.
*   **Prepackaged Distributions:** Simplifies deployments.
*   **Developer Interfaces:** Offers CLI and SDKs.
*   **Example Applications:** Provides production-grade examples.

<div style="text-align: center;">
  <img
    src="https://github.com/user-attachments/assets/33d9576d-95ea-468d-95e2-8fa233205a50"
    width="480"
    title="Llama Stack"
    alt="Llama Stack"
  />
</div>

## Benefits of Using Llama Stack

*   **Flexibility:** Choose your preferred infrastructure without code changes.
*   **Consistency:** Build, test, and deploy AI applications easily with unified APIs.
*   **Ecosystem:** Leverage integrations with cloud providers, hardware vendors, and AI-focused companies.

Llama Stack reduces complexity, empowering developers to build innovative generative AI applications.

## API Providers

Llama Stack supports a growing list of API providers.  Refer to the [full list](https://llama-stack.readthedocs.io/en/latest/providers/index.html) in the documentation.

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

> **Note:** Additional providers are available through external packages. See [External Providers](https://llama-stack.readthedocs.io/en/latest/providers/external.html) documentation.

## Distributions

Llama Stack Distributions (or "distros") offer pre-configured bundles for specific deployment scenarios. Easily transition from local development to production without code modifications.

|                 **Distribution**                  |                                                                   **Llama Stack Docker**                                                                    |                                                   Start This Distribution                                                    |
|:-----------------------------------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------------:|
|                 Starter Distribution                  |            [llamastack/distribution-starter](https://hub.docker.com/repository/docker/llamastack/distribution-starter/general)            |       [Guide](https://llama-stack.readthedocs.io/en/latest/distributions/self_hosted_distro/starter.html)       |
|                 Meta Reference                  |          [llamastack/distribution-meta-reference-gpu](https://hub.docker.com/repository/docker/llamastack/distribution-meta-reference-gpu/general)          |       [Guide](https://llama-stack.readthedocs.io/en/latest/distributions/self_hosted_distro/meta-reference-gpu.html)       |
|                    PostgreSQL                   |                 [llamastack/distribution-postgres-demo](https://hub.docker.com/repository/docker/llamastack/distribution-postgres-demo/general)                 |                    |

## Documentation

Explore the comprehensive [Documentation](https://llama-stack.readthedocs.io/en/latest/index.html) for detailed information.

*   **CLI References:**
    *   [llama (server-side) CLI Reference](https://llama-stack.readthedocs.io/en/latest/references/llama_cli_reference/index.html): Learn to use the `llama` CLI.
    *   [llama (client-side) CLI Reference](https://llama-stack.readthedocs.io/en/latest/references/llama_stack_client_cli_reference.html):  Use the `llama-stack-client` CLI.
*   **Getting Started:**
    *   [Quick Start Guide](https://llama-stack.readthedocs.io/en/latest/getting_started/index.html).
    *   [Jupyter Notebook](./docs/getting_started.ipynb) for simple text and vision inference.
    *   [Colab Notebook](https://colab.research.google.com/drive/1dtVmxotBsI4cGZQNsJRYPrLiDeT0Wnwt) from the Llama 3.2 course.
    *   [Zero-to-Hero Guide](https://github.com/meta-llama/llama-stack/tree/main/docs/zero_to_hero_guide) for a comprehensive overview.
*   [Contributing](CONTRIBUTING.md): Learn how to contribute.
    *   [Adding a new API Provider](https://llama-stack.readthedocs.io/en/latest/contributing/new_api_provider.html)

## Llama Stack Client SDKs

|  **Language** |  **Client SDK** | **Package** |
| :----: | :----: | :----: |
| Python |  [llama-stack-client-python](https://github.com/meta-llama/llama-stack-client-python) | [![PyPI version](https://img.shields.io/pypi/v/llama_stack_client.svg)](https://pypi.org/project/llama_stack_client/)
| Swift  | [llama-stack-client-swift](https://github.com/meta-llama/llama-stack-client-swift) | [![Swift Package Index](https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2Fmeta-llama%2Fllama-stack-client-swift%2Fbadge%3Ftype%3Dswift-versions)](https://swiftpackageindex.com/meta-llama/llama-stack-client-swift)
| Typescript   | [llama-stack-client-typescript](https://github.com/meta-llama/llama-stack-client-typescript) | [![NPM version](https://img.shields.io/npm/v/llama-stack-client.svg)](https://npmjs.org/package/llama-stack-client)
| Kotlin | [llama-stack-client-kotlin](https://github.com/meta-llama/llama-stack-client-kotlin) | [![Maven version](https://img.shields.io/maven-central/v/com.llama.llamastack/llama-stack-client-kotlin)](https://central.sonatype.com/artifact/com.llama.llamastack/llama-stack-client-kotlin)

Connect to a Llama Stack server in your preferred language with our client SDKs. Choose from [python](https://github.com/meta-llama/llama-stack-client-python), [typescript](https://github.com/meta-llama/llama-stack-client-typescript), [swift](https://github.com/meta-llama/llama-stack-client-swift), and [kotlin](https://github.com/meta-llama/llama-stack-client-kotlin).

Find more examples with client SDKs to interact with the Llama Stack server in our [llama-stack-apps](https://github.com/meta-llama/llama-stack-apps/tree/main/examples) repository.
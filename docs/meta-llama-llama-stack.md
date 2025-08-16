# Llama Stack: Build AI Applications Faster with a Unified Framework

**Llama Stack provides a powerful, flexible framework for building and deploying AI applications, simplifying development and ensuring consistent behavior across various environments.  Explore the possibilities at [the original repo](https://github.com/meta-llama/llama-stack).**

[![PyPI version](https://img.shields.io/pypi/v/llama_stack.svg)](https://pypi.org/project/llama-stack/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/llama-stack)](https://pypi.org/project/llama-stack/)
[![License](https://img.shields.io/pypi/l/llama_stack.svg)](https://github.com/meta-llama/llama-stack/blob/main/LICENSE)
[![Discord](https://img.shields.io/discord/1257833999603335178?color=6A7EC2&logo=discord&logoColor=ffffff)](https://discord.gg/llama-stack)
[![Unit Tests](https://github.com/meta-llama/llama-stack/actions/workflows/unit-tests.yml/badge.svg?branch=main)](https://github.com/meta-llama/llama-stack/actions/workflows/unit-tests.yml?query=branch%3Amain)
[![Integration Tests](https://github.com/meta-llama/llama-stack/actions/workflows/integration-tests.yml/badge.svg?branch=main)](https://github.com/meta-llama/llama-stack/actions/workflows/integration-tests.yml?query=branch%3Amain)

[**Quick Start**](https://llama-stack.readthedocs.io/en/latest/getting_started/index.html) | [**Documentation**](https://llama-stack.readthedocs.io/en/latest/index.html) | [**Colab Notebook**](./docs/getting_started.ipynb) | [**Discord**](https://discord.gg/llama-stack)

## Key Features

*   **Unified API Layer:** Standardizes inference, RAG, agents, tools, safety, evals, and telemetry for consistent behavior.
*   **Plugin Architecture:** Supports diverse API implementations across local, on-premises, cloud, and mobile environments.
*   **Prepackaged Distributions:** Offers ready-to-use solutions for rapid development and reliable deployment.
*   **Multiple Developer Interfaces:** Includes CLI and SDKs for Python, Typescript, iOS, and Android.
*   **Standalone Applications:** Provides examples for building production-grade AI applications.

## What's New

### ✨🎉 Llama 4 Support  🎉✨

Llama Stack now supports the latest Llama 4 models.  See version [0.2.0](https://github.com/meta-llama/llama-stack/releases/tag/v0.2.0) for details.

<details>
  <summary>👋 Click here to learn how to run Llama 4 models on Llama Stack </summary>

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

## Quick Installation

Get started quickly with a one-line installer:

```bash
curl -LsSf https://github.com/meta-llama/llama-stack/raw/main/scripts/install.sh | bash
```

## Overview

Llama Stack simplifies AI application development by providing a standardized set of building blocks and best practices for the Llama ecosystem. This includes:

*   **Unified APIs:** Streamlines interaction with various AI components.
*   **Plugin Architecture:** Enables flexible deployment across diverse environments.
*   **Prepackaged Distributions:** Provides ready-to-use solutions for different use cases.
*   **Multiple Developer Interfaces:** Offers CLIs and SDKs for different programming languages.
*   **Standalone Applications:** Offers examples for creating production-ready AI applications.

<div style="text-align: center;">
  <img
    src="https://github.com/user-attachments/assets/33d9576d-95ea-468d-95e2-8fa233205a50"
    width="480"
    title="Llama Stack"
    alt="Llama Stack"
  />
</div>

## Benefits of Using Llama Stack

*   **Flexibility:** Choose your preferred infrastructure without modifying your APIs.
*   **Consistency:** Build, test, and deploy AI applications with consistent behavior.
*   **Robust Ecosystem:** Integrates with leading cloud providers, hardware vendors, and AI companies.

Llama Stack empowers developers to focus on building innovative AI applications by reducing complexity and accelerating development.

## API Providers and Distributions

### API Providers

Here's a list of API providers and distributions to help developers quickly get started with Llama Stack.  See the [full list](https://llama-stack.readthedocs.io/en/latest/providers/index.html) for more details.

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

> **Note:** Additional providers are available through external packages. See [External Providers](https://llama-stack.readthedocs.io/en/latest/providers/external.html) documentation.

### Distributions

Llama Stack Distributions offer pre-configured bundles for specific deployment scenarios.  Start with a local setup and seamlessly transition to production without code changes.

|               **Distribution**                |                                                                    **Llama Stack Docker**                                                                     |                                                 Start This Distribution                                                  |
|:---------------------------------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------:|
|                Starter Distribution                 |           [llamastack/distribution-starter](https://hub.docker.com/repository/docker/llamastack/distribution-starter/general)           |      [Guide](https://llama-stack.readthedocs.io/en/latest/distributions/self_hosted_distro/starter.html)      |
|                Meta Reference                 |           [llamastack/distribution-meta-reference-gpu](https://hub.docker.com/repository/docker/llamastack/distribution-meta-reference-gpu/general)           |      [Guide](https://llama-stack.readthedocs.io/en/latest/distributions/self_hosted_distro/meta-reference-gpu.html)      |
|                   PostgreSQL                  |                [llamastack/distribution-postgres-demo](https://hub.docker.com/repository/docker/llamastack/distribution-postgres-demo/general)                |                  |

## Documentation

Explore the comprehensive [Documentation](https://llama-stack.readthedocs.io/en/latest/index.html) for detailed information.

*   **CLI References:**
    *   [llama (server-side) CLI Reference](https://llama-stack.readthedocs.io/en/latest/references/llama_cli_reference/index.html)
    *   [llama (client-side) CLI Reference](https://llama-stack.readthedocs.io/en/latest/references/llama_stack_client_cli_reference.html)
*   **Getting Started:**
    *   [Quick Start Guide](https://llama-stack.readthedocs.io/en/latest/getting_started/index.html)
    *   [Jupyter Notebook](./docs/getting_started.ipynb)
    *   [Colab notebook](https://colab.research.google.com/drive/1dtVmxotBsI4cGZQNsJRYPrLiDeT0Wnwt)
    *   [Zero-to-Hero Guide](https://github.com/meta-llama/llama-stack/tree/main/docs/zero_to_hero_guide)
*   [Contributing](CONTRIBUTING.md)
    *   [Adding a new API Provider](https://llama-stack.readthedocs.io/en/latest/contributing/new_api_provider.html)

## Llama Stack Client SDKs

|  **Language** |  **Client SDK** | **Package** |
| :----: | :----: | :----: |
| Python |  [llama-stack-client-python](https://github.com/meta-llama/llama-stack-client-python) | [![PyPI version](https://img.shields.io/pypi/v/llama_stack_client.svg)](https://pypi.org/project/llama_stack_client/)
| Swift  | [llama-stack-client-swift](https://github.com/meta-llama/llama-stack-client-swift) | [![Swift Package Index](https://img.shields.io/endpoint?url=https%3A%2F%2Fswiftpackageindex.com%2Fapi%2Fpackages%2Fmeta-llama%2Fllama-stack-client-swift%2Fbadge%3Ftype%3Dswift-versions)](https://swiftpackageindex.com/meta-llama/llama-stack-client-swift)
| Typescript   | [llama-stack-client-typescript](https://github.com/meta-llama/llama-stack-client-typescript) | [![NPM version](https://img.shields.io/npm/v/llama-stack-client.svg)](https://npmjs.org/package/llama-stack-client)
| Kotlin | [llama-stack-client-kotlin](https://github.com/meta-llama/llama-stack-client-kotlin) | [![Maven version](https://img.shields.io/maven-central/v/com.llama.llamastack/llama-stack-client-kotlin)](https://central.sonatype.com/artifact/com.llama.llamastack/llama-stack-client-kotlin)

Connect to a Llama Stack server using client SDKs in [python](https://github.com/meta-llama/llama-stack-client-python), [typescript](https://github.com/meta-llama/llama-stack-client-typescript), [swift](https://github.com/meta-llama/llama-stack-client-swift), and [kotlin](https://github.com/meta-llama/llama-stack-client-kotlin).

Find more examples with client SDKs in the [llama-stack-apps](https://github.com/meta-llama/llama-stack-apps/tree/main/examples) repo.

## 🌟 GitHub Star History
## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=meta-llama/llama-stack&type=Date)](https://www.star-history.com/#meta-llama/llama-stack&Date)

## ✨ Contributors

Thank you to all our amazing contributors!

<a href="https://github.com/meta-llama/llama-stack/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=meta-llama/llama-stack" />
</a>
```
Key improvements and SEO considerations:

*   **Clear, Concise Title:**  The title clearly states the project's purpose and includes a relevant keyword ("AI Applications").
*   **One-Sentence Hook:**  Immediately grabs the reader's attention and highlights a core benefit.
*   **Keyword Optimization:** Includes relevant keywords like "AI applications," "framework," "deployment," and specific technologies mentioned in the original (Llama, SDKs, etc.).
*   **Structured Formatting:** Uses clear headings (Key Features, Benefits, Documentation, etc.) and bullet points for readability.
*   **Comprehensive Summary:**  Covers all the major points from the original README.
*   **Calls to Action:**  Encourages users to explore documentation and examples.
*   **Strong Emphasis on Benefits:** Highlights the advantages of using Llama Stack.
*   **Visual Aid:** Includes the project's logo to attract attention.
*   **Clear Links:**  Provides easy access to documentation, quick start guides, and the source code.
*   **SEO-Friendly Markdown:**  Uses proper Markdown formatting for headings and emphasis, improving search engine visibility.
*   **Concise Language:** The text is rewritten to be more direct and easier to understand.
*   **Updated Information:**  Highlights the Llama 4 support release.
*   **Focus on User Value:**  Emphasizes what the user gains from using Llama Stack.
*   **Comprehensive Coverage:** The rewrite ensures all essential sections from the original are included and improved.
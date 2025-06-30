<h1 align="center">
  🚀 LiteLLM: Unified Access to LLMs with a Single API
</h1>

<p align="center">
  <a href="https://github.com/BerriAI/litellm">
    <img src="https://img.shields.io/github/stars/BerriAI/litellm?style=social" alt="GitHub stars">
  </a>
  <a href="https://pypi.org/project/litellm/">
    <img src="https://img.shields.io/pypi/v/litellm.svg" alt="PyPI version">
  </a>
  <a href="https://discord.gg/wuPM9dRgDw">
    <img src="https://img.shields.io/discord/1059258660725998151?label=Discord&logo=discord&style=flat-square" alt="Discord">
  </a>
</p>

**LiteLLM empowers developers to seamlessly call any LLM API using a single, unified OpenAI-compatible format, simplifying model switching and improving application flexibility.**  ([Original Repo](https://github.com/BerriAI/litellm))

## Key Features

*   **Unified API:** Translate inputs to providers' `completion`, `embedding`, and `image_generation` endpoints, using the OpenAI API format.
*   **Consistent Output:** Receive consistent text responses at `['choices'][0]['message']['content']` regardless of the LLM provider.
*   **Intelligent Routing:** Implement retry/fallback logic across multiple deployments (e.g., Azure/OpenAI) for enhanced reliability.
*   **Cost & Usage Control:** Set budgets and rate limits per project, API key, and model using the [LiteLLM Proxy Server (LLM Gateway)](https://docs.litellm.ai/docs/simple_proxy).
*   **Asynchronous Support:**  Benefit from `acompletion` for non-blocking requests and increased performance.
*   **Streaming Capabilities:** Leverage streaming for all models, allowing real-time response updates with the `stream=True` parameter.
*   **Observability:** Integrate with Lunary, MLflow, Langfuse, DynamoDB, S3 Buckets, Helicone, Promptlayer, Traceloop, and Athina for robust logging.

## Quickstart

```bash
pip install litellm
```

```python
from litellm import completion
import os

# Set your API keys
os.environ["OPENAI_API_KEY"] = "your-openai-key"
os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-key"

messages = [{"content": "Hello, how are you?", "role": "user"}]

# Call OpenAI
response = completion(model="openai/gpt-4o", messages=messages)

# Call Anthropic
response = completion(model="anthropic/claude-3-sonnet-20240229", messages=messages)
print(response)
```

### Response (OpenAI Format)

```json
{
    "id": "chatcmpl-565d891b-a42e-4c39-8d14-82a1f5208885",
    "created": 1734366691,
    "model": "claude-3-sonnet-20240229",
    "object": "chat.completion",
    "system_fingerprint": null,
    "choices": [
        {
            "finish_reason": "stop",
            "index": 0,
            "message": {
                "content": "Hello! As an AI language model, I don't have feelings, but I'm operating properly and ready to assist you with any questions or tasks you may have. How can I help you today?",
                "role": "assistant",
                "tool_calls": null,
                "function_call": null
            }
        }
    ],
    "usage": {
        "completion_tokens": 43,
        "prompt_tokens": 13,
        "total_tokens": 56,
        "completion_tokens_details": null,
        "prompt_tokens_details": {
            "audio_tokens": null,
            "cached_tokens": 0
        },
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0
    }
}
```

## Async Usage

```python
from litellm import acompletion
import asyncio

async def test_get_response():
    user_message = "Hello, how are you?"
    messages = [{"content": user_message, "role": "user"}]
    response = await acompletion(model="openai/gpt-4o", messages=messages)
    return response

response = asyncio.run(test_get_response())
print(response)
```

## Streaming

```python
from litellm import completion
response = completion(model="openai/gpt-4o", messages=messages, stream=True)
for part in response:
    print(part.choices[0].delta.content or "")
```

### Response chunk (OpenAI Format)

```json
{
    "id": "chatcmpl-2be06597-eb60-4c70-9ec5-8cd2ab1b4697",
    "created": 1734366925,
    "model": "claude-3-sonnet-20240229",
    "object": "chat.completion.chunk",
    "system_fingerprint": null,
    "choices": [
        {
            "finish_reason": null,
            "index": 0,
            "delta": {
                "content": "Hello",
                "role": "assistant",
                "function_call": null,
                "tool_calls": null,
                "audio": null
            },
            "logprobs": null
        }
    ]
}
```

## Observability (Logging)

```python
from litellm import completion
import os

# Set environment variables for logging tools
os.environ["LUNARY_PUBLIC_KEY"] = "your-lunary-public-key"
os.environ["HELICONE_API_KEY"] = "your-helicone-auth-key"
os.environ["LANGFUSE_PUBLIC_KEY"] = ""
os.environ["LANGFUSE_SECRET_KEY"] = ""
os.environ["ATHINA_API_KEY"] = "your-athina-api-key"
os.environ["OPENAI_API_KEY"] = "your-openai-key"

# Set callbacks
litellm.success_callback = ["lunary", "mlflow", "langfuse", "athina", "helicone"]

# Make an OpenAI call
response = completion(model="openai/gpt-4o", messages=[{"role": "user", "content": "Hi 👋 - i'm openai"}])
```

## 🔑 LiteLLM Proxy Server (LLM Gateway)

*   **Features:** Track spend + Load Balance across multiple projects, Authentication Hooks, Logging, Cost Tracking, Rate Limiting.
*   **Quick Start:**

    ```bash
    pip install 'litellm[proxy]'
    litellm --model huggingface/bigcode/starcoder # Start the proxy
    ```

*   Use LiteLLM Proxy with Langchain (Python, JS), OpenAI SDK (Python, JS) Anthropic SDK, Mistral SDK, LlamaIndex, Instructor, Curl.  ([More details](https://docs.litellm.ai/docs/proxy/user_keys))
*   **Proxy Endpoints:** See [Swagger Docs](https://litellm-api.up.railway.app/)

## ⚙️ Supported Providers

| Provider                                                                            | Completion | Streaming | Async Completion | Async Streaming | Async Embedding | Async Image Generation |
|-------------------------------------------------------------------------------------|------------|-----------|------------------|-----------------|-----------------|------------------------|
| [openai](https://docs.litellm.ai/docs/providers/openai)                             | ✅          | ✅          | ✅               | ✅              | ✅              | ✅                       |
| [Meta - Llama API](https://docs.litellm.ai/docs/providers/meta_llama)                               | ✅          | ✅          | ✅               | ✅              |                   |                        |
| [azure](https://docs.litellm.ai/docs/providers/azure)                               | ✅          | ✅          | ✅               | ✅              | ✅              | ✅                       |
| [AI/ML API](https://docs.litellm.ai/docs/providers/aiml)                               | ✅          | ✅          | ✅               | ✅              | ✅              | ✅                       |
| [aws - sagemaker](https://docs.litellm.ai/docs/providers/aws_sagemaker)             | ✅          | ✅          | ✅               | ✅              | ✅              |                        |
| [aws - bedrock](https://docs.litellm.ai/docs/providers/bedrock)                     | ✅          | ✅          | ✅               | ✅              | ✅              |                        |
| [google - vertex_ai](https://docs.litellm.ai/docs/providers/vertex)                 | ✅          | ✅          | ✅               | ✅              | ✅              | ✅                       |
| [google - palm](https://docs.litellm.ai/docs/providers/palm)                        | ✅          | ✅          | ✅               | ✅              |                   |                        |
| [google AI Studio - gemini](https://docs.litellm.ai/docs/providers/gemini)          | ✅          | ✅          | ✅               | ✅              |                   |                        |
| [mistral ai api](https://docs.litellm.ai/docs/providers/mistral)                    | ✅          | ✅          | ✅               | ✅              | ✅              |                        |
| [cloudflare AI Workers](https://docs.litellm.ai/docs/providers/cloudflare_workers)  | ✅          | ✅          | ✅               | ✅              |                   |                        |
| [cohere](https://docs.litellm.ai/docs/providers/cohere)                             | ✅          | ✅          | ✅               | ✅              | ✅              |                        |
| [anthropic](https://docs.litellm.ai/docs/providers/anthropic)                       | ✅          | ✅          | ✅               | ✅              |                   |                        |
| [empower](https://docs.litellm.ai/docs/providers/empower)                    | ✅          | ✅          | ✅               | ✅              |                   |                        |
| [huggingface](https://docs.litellm.ai/docs/providers/huggingface)                   | ✅          | ✅          | ✅               | ✅              | ✅              |                        |
| [replicate](https://docs.litellm.ai/docs/providers/replicate)                       | ✅          | ✅          | ✅               | ✅              |                   |                        |
| [together_ai](https://docs.litellm.ai/docs/providers/togetherai)                    | ✅          | ✅          | ✅               | ✅              |                   |                        |
| [openrouter](https://docs.litellm.ai/docs/providers/openrouter)                     | ✅          | ✅          | ✅               | ✅              |                   |                        |
| [ai21](https://docs.litellm.ai/docs/providers/ai21)                                 | ✅          | ✅          | ✅               | ✅              |                   |                        |
| [baseten](https://docs.litellm.ai/docs/providers/baseten)                           | ✅          | ✅          | ✅               | ✅              |                   |                        |
| [vllm](https://docs.litellm.ai/docs/providers/vllm)                                 | ✅          | ✅          | ✅               | ✅              |                   |                        |
| [nlp_cloud](https://docs.litellm.ai/docs/providers/nlp_cloud)                       | ✅          | ✅          | ✅               | ✅              |                   |                        |
| [aleph alpha](https://docs.litellm.ai/docs/providers/aleph_alpha)                   | ✅          | ✅          | ✅               | ✅              |                   |                        |
| [petals](https://docs.litellm.ai/docs/providers/petals)                             | ✅          | ✅          | ✅               | ✅              |                   |                        |
| [ollama](https://docs.litellm.ai/docs/providers/ollama)                             | ✅          | ✅          | ✅               | ✅              | ✅              |                        |
| [deepinfra](https://docs.litellm.ai/docs/providers/deepinfra)                       | ✅          | ✅          | ✅               | ✅              |                   |                        |
| [perplexity-ai](https://docs.litellm.ai/docs/providers/perplexity)                  | ✅          | ✅          | ✅               | ✅              |                   |                        |
| [Groq AI](https://docs.litellm.ai/docs/providers/groq)                              | ✅          | ✅          | ✅               | ✅              |                   |                        |
| [Deepseek](https://docs.litellm.ai/docs/providers/deepseek)                         | ✅          | ✅          | ✅               | ✅              |                   |                        |
| [anyscale](https://docs.litellm.ai/docs/providers/anyscale)                         | ✅          | ✅          | ✅               | ✅              |                   |                        |
| [IBM - watsonx.ai](https://docs.litellm.ai/docs/providers/watsonx)                  | ✅          | ✅          | ✅               | ✅              | ✅              |                        |
| [voyage ai](https://docs.litellm.ai/docs/providers/voyage)                          |          |           |                 |                  | ✅               |                        |
| [xinference [Xorbits Inference]](https://docs.litellm.ai/docs/providers/xinference) |          |           |                 |                  | ✅               |                        |
| [FriendliAI](https://docs.litellm.ai/docs/providers/friendliai)                              | ✅          | ✅          | ✅               | ✅              |                   |                        |
| [Galadriel](https://docs.litellm.ai/docs/providers/galadriel)                              | ✅          | ✅          | ✅               | ✅              |                   |                        |
| [Novita AI](https://novita.ai/models/llm?utm_source=github_litellm&utm_medium=github_readme&utm_campaign=github_link)                     | ✅          | ✅          | ✅               | ✅              |                   |                        |
| [Featherless AI](https://docs.litellm.ai/docs/providers/featherless_ai)                              | ✅          | ✅          | ✅               | ✅              |                   |                        |
| [Nebius AI Studio](https://docs.litellm.ai/docs/providers/nebius)                             | ✅          | ✅          | ✅               | ✅              | ✅              |                        |

[**View Comprehensive Docs**](https://docs.litellm.ai/docs/)

## 🤝 Contributing

Contributions are welcome!  Check out our [Contributing Guide](CONTRIBUTING.md) for details.

## 💼 Enterprise

For businesses needing enhanced security, user management, and professional support, contact us to discuss our [Enterprise tier](https://docs.litellm.ai/docs/proxy/enterprise).

## Code Quality / Linting

LiteLLM follows the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html).

Our automated checks include:
- **Black** for code formatting
- **Ruff** for linting and code quality
- **MyPy** for type checking
- **Circular import detection**
- **Import safety checks**

Run all checks locally:
```bash
make lint           # Run all linting (matches CI)
make format-check   # Check formatting only
```

All these checks must pass before your PR can be merged.


## Contact

- [Schedule Demo 👋](https://calendly.com/d/4mp-gd3-k5k/berriai-1-1-onboarding-litellm-hosted-version)
- [Community Discord 💭](https://discord.gg/wuPM9dRgDw)
- Our numbers 📞 +1 (770) 8783-106 / ‭+1 (412) 618-6238‬
- Our emails ✉️ ishaan@berri.ai / krrish@berri.ai

## Core Team & Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

<a href="https://github.com/BerriAI/litellm/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=BerriAI/litellm" />
</a>

```
## Run in Developer mode
### Services
1. Setup .env file in root
2. Run dependant services `docker-compose up db prometheus`

### Backend
1. (In root) create virtual environment `python -m venv .venv`
2. Activate virtual environment `source .venv/bin/activate`
3. Install dependencies `pip install -e ".[all]"`
4. Start proxy backend `uvicorn litellm.proxy.proxy_server:app --host localhost --port 4000 --reload`

### Frontend
1. Navigate to `ui/litellm-dashboard`
2. Install dependencies `npm install`
3. Run `npm run dev` to start the dashboard
```
<h1 align="center">
    🚀 LiteLLM: Unified LLM Access for Developers
</h1>

<p align="center">
    <a href="https://github.com/BerriAI/litellm">
        <img src="https://img.shields.io/github/stars/BerriAI/litellm?style=social" alt="Stars">
    </a>
    <a href="https://pypi.org/project/litellm/" target="_blank">
        <img src="https://img.shields.io/pypi/v/litellm.svg" alt="PyPI Version">
    </a>
    <a href="https://discord.gg/wuPM9dRgDw">
        <img src="https://img.shields.io/static/v1?label=Chat%20on&message=Discord&color=blue&logo=Discord&style=flat-square" alt="Discord">
    </a>
</p>

**LiteLLM simplifies LLM integration by providing a single API to access a wide range of Large Language Models (LLMs) like OpenAI, Azure, and Google, offering features like unified output and intelligent routing.**

<p align="center">
    <a href="https://render.com/deploy?repo=https://github.com/BerriAI/litellm" target="_blank" rel="nofollow"><img src="https://render.com/images/deploy-to-render-button.svg" alt="Deploy to Render"></a>
    <a href="https://railway.app/template/HLP0Ub?referralCode=jch2ME">
        <img src="https://railway.app/button.svg" alt="Deploy on Railway">
    </a>
</p>

<h3 align="center">
    <a href="https://docs.litellm.ai/docs/simple_proxy" target="_blank">LiteLLM Proxy Server (LLM Gateway)</a> |
    <a href="https://docs.litellm.ai/docs/hosted" target="_blank"> Hosted Proxy (Preview)</a> |
    <a href="https://docs.litellm.ai/docs/enterprise" target="_blank">Enterprise Tier</a>
</h3>

LiteLLM empowers developers to effortlessly integrate and manage LLMs, offering:

*   **Unified API:** Call multiple LLM APIs using a single, OpenAI-compatible interface.
*   **Consistent Output:**  Receive consistent response formats regardless of the underlying LLM.
*   **Intelligent Routing:**  Implement retry/fallback logic across different deployments for high availability and cost optimization.
*   **Cost Management:** Set budgets and rate limits per project, API key, and model using the LiteLLM Proxy Server.
*   **Streaming Support:** Get real-time model responses with streaming capabilities.
*   **Observability:** Integrate with popular logging tools for comprehensive monitoring and debugging.

**[Read the Docs](https://docs.litellm.ai/docs/)**

## Key Features

*   **Model Abstraction:** Seamlessly switch between providers (OpenAI, Azure, Anthropic, etc.) without code changes.
*   **Completion, Embedding, and Image Generation Support:**  Works with all major LLM functionalities.
*   **Asynchronous Operations:** Supports `async` calls for improved performance.
*   **Detailed Logging:** Integrates with tools like Lunary, MLflow, Langfuse, and more.
*   **Robust Proxy Server:**  Offers advanced features like cost tracking, rate limiting, and custom authentication (see below).

## Quick Start

Install LiteLLM:

```bash
pip install litellm
```

Use LiteLLM with your API keys:

```python
from litellm import completion
import os

# Set your API keys (replace with your actual keys)
os.environ["OPENAI_API_KEY"] = "your-openai-key"
os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-key"

messages = [{ "content": "Hello, how are you?","role": "user"}]

# OpenAI call
response = completion(model="openai/gpt-4o", messages=messages)

# Anthropic call
response = completion(model="anthropic/claude-sonnet-4-20250514", messages=messages)
print(response)
```

**Example Response (OpenAI Format):**

```json
{
    "id": "chatcmpl-1214900a-6cdd-4148-b663-b5e2f642b4de",
    "created": 1751494488,
    "model": "claude-sonnet-4-20250514",
    "object": "chat.completion",
    "system_fingerprint": null,
    "choices": [
        {
            "finish_reason": "stop",
            "index": 0,
            "message": {
                "content": "Hello! I'm doing well, thank you for asking. I'm here and ready to help with whatever you'd like to discuss or work on. How are you doing today?",
                "role": "assistant",
                "tool_calls": null,
                "function_call": null
            }
        }
    ],
    "usage": {
        "completion_tokens": 39,
        "prompt_tokens": 13,
        "total_tokens": 52,
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

Call any supported model using the format: `model=<provider_name>/<model_name>`.  Refer to the [provider docs](https://docs.litellm.ai/docs/providers) for provider-specific details.

## Async Support

Use `acompletion` for asynchronous calls:

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

Enable streaming with `stream=True`:

```python
from litellm import completion
response = completion(model="openai/gpt-4o", messages=messages, stream=True)
for part in response:
    print(part.choices[0].delta.content or "")

# claude sonnet 4
response = completion('anthropic/claude-sonnet-4-20250514', messages, stream=True)
for part in response:
    print(part)
```

**Streaming Response Chunk (OpenAI Format):**

```json
{
    "id": "chatcmpl-fe575c37-5004-4926-ae5e-bfbc31f356ca",
    "created": 1751494808,
    "model": "claude-sonnet-4-20250514",
    "object": "chat.completion.chunk",
    "system_fingerprint": null,
    "choices": [
        {
            "finish_reason": null,
            "index": 0,
            "delta": {
                "provider_specific_fields": null,
                "content": "Hello",
                "role": "assistant",
                "function_call": null,
                "tool_calls": null,
                "audio": null
            },
            "logprobs": null
        }
    ],
    "provider_specific_fields": null,
    "stream_options": null,
    "citations": null
}
```

## Logging & Observability

Integrate with logging tools:

```python
from litellm import completion

## set env variables for logging tools (when using MLflow, no API key set up is required)
os.environ["LUNARY_PUBLIC_KEY"] = "your-lunary-public-key"
os.environ["HELICONE_API_KEY"] = "your-helicone-auth-key"
os.environ["LANGFUSE_PUBLIC_KEY"] = ""
os.environ["LANGFUSE_SECRET_KEY"] = ""
os.environ["ATHINA_API_KEY"] = "your-athina-api-key"

os.environ["OPENAI_API_KEY"] = "your-openai-key"

# set callbacks
litellm.success_callback = ["lunary", "mlflow", "langfuse", "athina", "helicone"] # log input/output to lunary, langfuse, supabase, athina, helicone etc

#openai call
response = completion(model="openai/gpt-4o", messages=[{"role": "user", "content": "Hi 👋 - i'm openai"}])
```

## LiteLLM Proxy Server (LLM Gateway)

Manage costs, load balance, and add security with the LiteLLM Proxy Server:

*   **Track Spend:** Monitor your LLM usage.
*   **Load Balancing:** Distribute traffic across multiple models/providers.
*   **Authentication Hooks:**  Implement custom authentication mechanisms.
*   **Logging:**  Log requests and responses for auditing.
*   **Rate Limiting:** Control usage with rate limits.

**[Jump to LiteLLM Proxy Server Docs](https://docs.litellm.ai/docs/simple_proxy)**

### Quick Start Proxy (CLI)

```bash
pip install 'litellm[proxy]'
```

1.  Start the proxy:

    ```bash
    litellm --model huggingface/bigcode/starcoder
    #INFO: Proxy running on http://0.0.0.0:4000
    ```

2.  Make a request to the proxy:

    ```python
    import openai # openai v1.0.0+
    client = openai.OpenAI(api_key="anything",base_url="http://0.0.0.0:4000") # set proxy to base_url
    # request sent to model set on litellm proxy, `litellm --model`
    response = client.chat.completions.create(model="gpt-3.5-turbo", messages = [
        {
            "role": "user",
            "content": "this is a test request, write a short poem"
        }
    ])

    print(response)
    ```

## Proxy Key Management

Use a Postgres database to manage proxy keys.

```bash
# Get the code
git clone https://github.com/BerriAI/litellm

# Go to folder
cd litellm

# Add the master key - you can change this after setup
echo 'LITELLM_MASTER_KEY="sk-1234"' > .env

# Add the litellm salt key - you cannot change this after adding a model
# It is used to encrypt / decrypt your LLM API Key credentials
# We recommend - https://1password.com/password-generator/ 
# password generator to get a random hash for litellm salt key
echo 'LITELLM_SALT_KEY="sk-1234"' >> .env

source .env

# Start
docker-compose up
```

Access the UI at `/ui` on your proxy server.
![ui_3](https://github.com/BerriAI/litellm/assets/29436595/47c97d5e-b9be-4839-b28c-43d7f4f10033)

Generate proxy keys:

```shell
curl 'http://0.0.0.0:4000/key/generate' \
--header 'Authorization: Bearer sk-1234' \
--header 'Content-Type: application/json' \
--data-raw '{"models": ["gpt-3.5-turbo", "gpt-4", "claude-2"], "duration": "20m","metadata": {"user": "ishaan@berri.ai", "team": "core-infra"}}'
```

## Supported Providers

LiteLLM supports a wide range of LLM providers.  See the [complete list](https://docs.litellm.ai/docs/providers) for up-to-date information.

## Contributing

Contribute to LiteLLM! See our [Contributing Guide](CONTRIBUTING.md) for details.

**Quick Start:** `git clone` → `make install-dev` → `make format` → `make lint` → `make test-unit`

## Enterprise

For companies requiring advanced features such as enhanced security, user management, and professional support, explore our [Enterprise Tier](https://docs.litellm.ai/docs/proxy/enterprise).

*   Feature Prioritization
*   Custom Integrations
*   Professional Support - Dedicated Discord + Slack
*   Custom SLAs
*   Secure access with Single Sign-On

## Support

*   [Schedule Demo 👋](https://calendly.com/d/4mp-gd3-k5k/berriai-1-1-onboarding-litellm-hosted-version)
*   [Community Discord 💭](https://discord.gg/wuPM9dRgDw)
*   [Community Slack 💭](https://join.slack.com/share/enQtOTE0ODczMzk2Nzk4NC01YjUxNjY2YjBlYTFmNDRiZTM3NDFiYTM3MzVkODFiMDVjOGRjMmNmZTZkZTMzOWQzZGQyZWIwYjQ0MWExYmE3)
*   Our numbers 📞 +1 (770) 8783-106 / ‭+1 (412) 618-6238‬
*   Our emails ✉️ ishaan@berri.ai / krrish@berri.ai

---

**[Visit the LiteLLM GitHub Repository](https://github.com/BerriAI/litellm)**
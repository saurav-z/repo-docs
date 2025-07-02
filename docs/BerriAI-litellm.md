<h1 align="center">
    🚀 LiteLLM: Your Gateway to Seamless LLM Integration
</h1>

<p align="center">
    <b>Simplify LLM API calls and access a unified interface for diverse language models!</b>
    <br>
    <a href="https://github.com/BerriAI/litellm">
        <img src="https://img.shields.io/github/stars/BerriAI/litellm?style=social" alt="GitHub stars">
    </a>
    <a href="https://pypi.org/project/litellm/" target="_blank">
        <img src="https://img.shields.io/pypi/v/litellm.svg" alt="PyPI Version">
    </a>
    <a href="https://discord.gg/wuPM9dRgDw">
        <img src="https://img.shields.io/discord/1087956934318081607?label=Discord&logo=discord&style=flat-square" alt="Discord">
    </a>
</p>

<p align="center">
    <a href="https://render.com/deploy?repo=https://github.com/BerriAI/litellm" target="_blank" rel="nofollow"><img src="https://render.com/images/deploy-to-render-button.svg" alt="Deploy to Render"></a>
    <a href="https://railway.app/template/HLP0Ub?referralCode=jch2ME">
      <img src="https://railway.app/button.svg" alt="Deploy on Railway">
    </a>
</p>

## Key Features

*   ⚡ **Unified API Access:** Interact with various LLMs (OpenAI, Azure, Anthropic, and more) using a single, consistent interface.
*   🔄 **Intelligent Routing & Fallback:**  Automatically handles retries and failovers across different deployments and providers.
*   ✅ **Consistent Output:**  Ensures predictable responses, with text always available at `['choices'][0]['message']['content']`.
*   💰 **Cost Management & Rate Limiting:**  Utilize the LiteLLM Proxy Server to set budgets and control API usage per project, API key, or model.
*   💡 **Asynchronous Support:**  Benefit from `async` functions for improved performance.
*   📤 **Streaming Support:** Stream responses from any model (Bedrock, Huggingface, TogetherAI, Azure, OpenAI, etc).
*   📊 **Observability:**  Integrate with tools like Lunary, MLflow, Langfuse, and more for comprehensive logging and monitoring.

[**Explore the Docs**](https://docs.litellm.ai/docs/) | [**Join the Community**](https://discord.gg/wuPM9dRgDw) | [**Get Started (Quickstart)**](https://github.com/BerriAI/litellm#usage-docs)

## Why LiteLLM?

LiteLLM simplifies LLM integration by abstracting away the complexities of interacting with different providers.  It provides a unified API, handles retries, manages costs, and offers a powerful proxy server for centralized control.

## Quick Start

```bash
pip install litellm
```

```python
from litellm import completion
import os

## set ENV variables
os.environ["OPENAI_API_KEY"] = "your-openai-key"
os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-key"

messages = [{ "content": "Hello, how are you?","role": "user"}]

# openai call
response = completion(model="openai/gpt-4o", messages=messages)

# anthropic call
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

## LiteLLM Proxy Server (LLM Gateway)

Manage your LLM costs and streamline your development workflow with the LiteLLM Proxy.

*   **Key Features:**
    *   Cost tracking
    *   Rate limiting
    *   User authentication
    *   Custom Logging via [Hooks for auth](https://docs.litellm.ai/docs/proxy/virtual_keys#custom-auth)
    *   Cost tracking, rate limiting, and Logging via [Hooks for logging](https://docs.litellm.ai/docs/proxy/logging#step-1---create-your-custom-litellm-callback-class)
    *   Set budgets and rate limits across multiple projects.

[**LiteLLM Proxy Docs**](https://docs.litellm.ai/docs/simple_proxy) | [**Hosted Proxy (Preview)**](https://docs.litellm.ai/docs/hosted)

### Quick Start Proxy - CLI

```shell
pip install 'litellm[proxy]'
```

### Step 1: Start litellm proxy

```shell
$ litellm --model huggingface/bigcode/starcoder

#INFO: Proxy running on http://0.0.0.0:4000
```

### Step 2: Make ChatCompletions Request to Proxy

> [!IMPORTANT]
> 💡 [Use LiteLLM Proxy with Langchain (Python, JS), OpenAI SDK (Python, JS) Anthropic SDK, Mistral SDK, LlamaIndex, Instructor, Curl](https://docs.litellm.ai/docs/proxy/user_keys)

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

## Supported Providers

LiteLLM supports a wide range of LLM providers.  [Explore the full list](https://docs.litellm.ai/docs/providers).

## Async Support

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

## Streaming Support

```python
from litellm import completion
response = completion(model="openai/gpt-4o", messages=messages, stream=True)
for part in response:
    print(part.choices[0].delta.content or "")

# claude 2
response = completion('anthropic/claude-3-sonnet-20240229', messages, stream=True)
for part in response:
    print(part)
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

## Logging & Observability

Integrate with logging tools to monitor and analyze your LLM usage.

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

## Contributing

We welcome contributions! See our [Contributing Guide](CONTRIBUTING.md) for details.

##  Enterprise

For more control over your LLMs, check out our [Enterprise Tier](https://calendly.com/d/4mp-gd3-k5k/litellm-1-1-onboarding-chat), which includes features such as:

*   **Features under the [LiteLLM Commercial License](https://docs.litellm.ai/docs/proxy/enterprise):**
*   **Feature Prioritization**
*   **Custom Integrations**
*   **Professional Support - Dedicated discord + slack**
*   **Custom SLAs**
*   **Secure access with Single Sign-On**

## Support

*   [Schedule Demo 👋](https://calendly.com/d/4mp-gd3-k5k/berriai-1-1-onboarding-litellm-hosted-version)
*   [Community Discord 💭](https://discord.gg/wuPM9dRgDw)
*   [Community Slack 💭](https://join.slack.com/share/enQtOTE0ODczMzk2Nzk4NC01YjUxNjY2YjBlYTFmNDRiZTM3NDFiYTM3MzVkODFiMDVjOGRjMmNmZTZkZTMzOWQzZGQyZWIwYjQ0MWExYmE3)
*   Our numbers 📞 +1 (770) 8783-106 / ‭+1 (412) 618-6238‬
*   Our emails ✉️ ishaan@berri.ai / krrish@berri.ai

## Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

<a href="https://github.com/BerriAI/litellm/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=BerriAI/litellm" />
</a>
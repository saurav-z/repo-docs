<h1 align="center">
  🚀 LiteLLM: The Universal API for LLMs
</h1>

<p align="center">
  <b>Seamlessly access and manage Large Language Models (LLMs) from various providers with a unified API, enabling easy integration, cost optimization, and enhanced observability.</b>  <br>
  <a href="https://github.com/BerriAI/litellm">View on GitHub</a>
</p>

<p align="center">
  <a href="https://render.com/deploy?repo=https://github.com/BerriAI/litellm" target="_blank" rel="nofollow"><img src="https://render.com/images/deploy-to-render-button.svg" alt="Deploy to Render"></a>
  <a href="https://railway.app/template/HLP0Ub?referralCode=jch2ME">
    <img src="https://railway.app/button.svg" alt="Deploy on Railway">
  </a>
</p>

<h4 align="center">
  <a href="https://docs.litellm.ai/docs/simple_proxy" target="_blank">LiteLLM Proxy Server (LLM Gateway)</a> |
  <a href="https://docs.litellm.ai/docs/hosted" target="_blank"> Hosted Proxy (Preview)</a> |
  <a href="https://docs.litellm.ai/docs/enterprise" target="_blank">Enterprise Tier</a>
</h4>

<h4 align="center">
  <a href="https://pypi.org/project/litellm/" target="_blank">
    <img src="https://img.shields.io/pypi/v/litellm.svg" alt="PyPI Version">
  </a>
  <a href="https://www.ycombinator.com/companies/berriai">
    <img src="https://img.shields.io/badge/Y%20Combinator-W23-orange?style=flat-square" alt="Y Combinator W23">
  </a>
  <a href="https://wa.link/huol9n">
    <img src="https://img.shields.io/static/v1?label=Chat%20on&message=WhatsApp&color=success&logo=WhatsApp&style=flat-square" alt="WhatsApp">
  </a>
  <a href="https://discord.gg/wuPM9dRgDw">
    <img src="https://img.shields.io/static/v1?label=Chat%20on&message=Discord&color=blue&logo=Discord&style=flat-square" alt="Discord">
  </a>
  <a href="https://join.slack.com/share/enQtOTE0ODczMzk2Nzk4NC01YjUxNjY2YjBlYTFmNDRiZTM3NDFiYTM3MzVkODFiMDVjOGRjMmNmZTZkZTMzOWQzZGQyZWIwYjQ0MWExYmE3">
    <img src="https://img.shields.io/static/v1?label=Chat%20on&message=Slack&color=black&logo=Slack&style=flat-square" alt="Slack">
  </a>
</h4>

## Key Features

*   ✅ **Unified API:** Interact with various LLM providers (OpenAI, Azure, Anthropic, and more) using a single, consistent interface, simplifying code and reducing vendor lock-in.
*   ✅ **Consistent Output:** Get predictable results across different LLMs. Text responses are always available at `['choices'][0]['message']['content']`.
*   ✅ **Intelligent Routing:** Implement retry and fallback logic across multiple deployments (e.g., Azure/OpenAI) to ensure high availability with our [Router](https://docs.litellm.ai/docs/routing).
*   ✅ **Cost Management:** Set budgets and rate limits per project, API key, and model using the [LiteLLM Proxy Server (LLM Gateway)](https://docs.litellm.ai/docs/simple_proxy).
*   ✅ **Async Support:** Supports asynchronous calls to improve performance using `acompletion`.
*   ✅ **Streaming Support:** Enable real-time response with `stream=True`, compatible with all supported models.
*   ✅ **Observability & Logging:** Integrate with popular monitoring tools like Lunary, MLflow, Langfuse, and Helicone using pre-defined callbacks.
*   ✅ **Image Generation:** Support for image generation models.
*   ✅ **Comprehensive Provider Support:** Support for a wide range of LLM providers, including OpenAI, Azure, Anthropic, Google Vertex AI, and many more (see list below).

## Getting Started

### Installation

```bash
pip install litellm
```

### Basic Usage

```python
from litellm import completion
import os

## set ENV variables
os.environ["OPENAI_API_KEY"] = "your-openai-key"
os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-key"

messages = [{"content": "Hello, how are you?","role": "user"}]

# openai call
response = completion(model="openai/gpt-4o", messages=messages)

# anthropic call
response = completion(model="anthropic/claude-sonnet-4-20250514", messages=messages)
print(response)
```

### Response (OpenAI Format)

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

Call any model supported by a provider, with `model=<provider_name>/<model_name>`. Refer to [provider docs](https://docs.litellm.ai/docs/providers) for more details.

## Async

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

# claude sonnet 4
response = completion('anthropic/claude-sonnet-4-20250514', messages, stream=True)
for part in response:
    print(part)
```

### Response Chunk (OpenAI Format)

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

Integrate with your favorite logging tools.

```python
from litellm import completion
import os

## set env variables for logging tools
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

The LiteLLM Proxy Server (LLM Gateway) provides:

*   [Hooks for auth](https://docs.litellm.ai/docs/proxy/virtual_keys#custom-auth)
*   [Hooks for logging](https://docs.litellm.ai/docs/proxy/logging#step-1---create-your-custom-litellm-callback-class)
*   [Cost tracking](https://docs.litellm.ai/docs/proxy/virtual_keys#tracking-spend)
*   [Rate Limiting](https://docs.litellm.ai/docs/proxy/users#set-rate-limits)

### Quick Start

```shell
pip install 'litellm[proxy]'
```

#### Step 1: Start litellm proxy

```shell
$ litellm --model huggingface/bigcode/starcoder

#INFO: Proxy running on http://0.0.0.0:4000
```

#### Step 2: Make ChatCompletions Request to Proxy

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

#### Proxy Key Management

Set budgets and rate limits across multiple projects

```shell
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
UI on `/ui` on your proxy server

![ui_3](https://github.com/BerriAI/litellm/assets/29436595/47c97d5e-b9be-4839-b28c-43d7f4f10033)

`POST /key/generate`
### Request

```shell
curl 'http://0.0.0.0:4000/key/generate' \
--header 'Authorization: Bearer sk-1234' \
--header 'Content-Type: application/json' \
--data-raw '{"models": ["gpt-3.5-turbo", "gpt-4", "claude-2"], "duration": "20m","metadata": {"user": "ishaan@berri.ai", "team": "core-infra"}}'
```

### Expected Response

```shell
{
    "key": "sk-kdEXbIqZRwEeEiHwdg7sFA", # Bearer token
    "expires": "2023-11-19T01:38:25.838000+00:00" # datetime object
}
```

## Supported Providers

| Provider                                                                            | Completion | Streaming | Async Completion | Async Streaming | Async Embedding | Async Image Generation |
|-------------------------------------------------------------------------------------|------------|-----------|------------------|-----------------|-----------------|-------------------------|
| [openai](https://docs.litellm.ai/docs/providers/openai)                             | ✅          | ✅         | ✅               | ✅              | ✅              | ✅                       |
| [Meta - Llama API](https://docs.litellm.ai/docs/providers/meta_llama)                               | ✅          | ✅         | ✅               | ✅              |                   |                        |
| [azure](https://docs.litellm.ai/docs/providers/azure)                               | ✅          | ✅         | ✅               | ✅              | ✅              | ✅                       |
| [AI/ML API](https://docs.litellm.ai/docs/providers/aiml)                               | ✅          | ✅         | ✅               | ✅              | ✅              | ✅                       |
| [aws - sagemaker](https://docs.litellm.ai/docs/providers/aws_sagemaker)             | ✅          | ✅         | ✅               | ✅              | ✅              |                        |
| [aws - bedrock](https://docs.litellm.ai/docs/providers/bedrock)                     | ✅          | ✅         | ✅               | ✅              | ✅              |                        |
| [google - vertex_ai](https://docs.litellm.ai/docs/providers/vertex)                 | ✅          | ✅         | ✅               | ✅              | ✅              | ✅                       |
| [google - palm](https://docs.litellm.ai/docs/providers/palm)                        | ✅          | ✅         | ✅               | ✅              |                   |                        |
| [google AI Studio - gemini](https://docs.litellm.ai/docs/providers/gemini)          | ✅          | ✅         | ✅               | ✅              |                   |                        |
| [mistral ai api](https://docs.litellm.ai/docs/providers/mistral)                    | ✅          | ✅         | ✅               | ✅              | ✅              |                        |
| [cloudflare AI Workers](https://docs.litellm.ai/docs/providers/cloudflare_workers)  | ✅          | ✅         | ✅               | ✅              |                   |                        |
| [cohere](https://docs.litellm.ai/docs/providers/cohere)                             | ✅          | ✅         | ✅               | ✅              | ✅              |                        |
| [anthropic](https://docs.litellm.ai/docs/providers/anthropic)                       | ✅          | ✅         | ✅               | ✅              |                   |                        |
| [empower](https://docs.litellm.ai/docs/providers/empower)                    | ✅          | ✅         | ✅               | ✅              |                |                        |
| [huggingface](https://docs.litellm.ai/docs/providers/huggingface)                   | ✅          | ✅         | ✅               | ✅              | ✅              |                        |
| [replicate](https://docs.litellm.ai/docs/providers/replicate)                       | ✅          | ✅         | ✅               | ✅              |                   |                        |
| [together_ai](https://docs.litellm.ai/docs/providers/togetherai)                    | ✅          | ✅         | ✅               | ✅              |                   |                        |
| [openrouter](https://docs.litellm.ai/docs/providers/openrouter)                     | ✅          | ✅         | ✅               | ✅              |                   |                        |
| [ai21](https://docs.litellm.ai/docs/providers/ai21)                                 | ✅          | ✅         | ✅               | ✅              |                   |                        |
| [baseten](https://docs.litellm.ai/docs/providers/baseten)                           | ✅          | ✅         | ✅               | ✅              |                   |                        |
| [vllm](https://docs.litellm.ai/docs/providers/vllm)                                 | ✅          | ✅         | ✅               | ✅              |                   |                        |
| [nlp_cloud](https://docs.litellm.ai/docs/providers/nlp_cloud)                       | ✅          | ✅         | ✅               | ✅              |                   |                        |
| [aleph alpha](https://docs.litellm.ai/docs/providers/aleph_alpha)                   | ✅          | ✅         | ✅               | ✅              |                   |                        |
| [petals](https://docs.litellm.ai/docs/providers/petals)                             | ✅          | ✅         | ✅               | ✅              |                   |                        |
| [ollama](https://docs.litellm.ai/docs/providers/ollama)                             | ✅          | ✅         | ✅               | ✅              | ✅              |                        |
| [deepinfra](https://docs.litellm.ai/docs/providers/deepinfra)                       | ✅          | ✅         | ✅               | ✅              |                   |                        |
| [perplexity-ai](https://docs.litellm.ai/docs/providers/perplexity)                  | ✅          | ✅         | ✅               | ✅              |                   |                        |
| [Groq AI](https://docs.litellm.ai/docs/providers/groq)                              | ✅          | ✅         | ✅               | ✅              |                   |                        |
| [Deepseek](https://docs.litellm.ai/docs/providers/deepseek)                         | ✅          | ✅         | ✅               | ✅              |                   |                        |
| [anyscale](https://docs.litellm.ai/docs/providers/anyscale)                         | ✅          | ✅         | ✅               | ✅              |                   |                        |
| [IBM - watsonx.ai](https://docs.litellm.ai/docs/providers/watsonx)                  | ✅          | ✅         | ✅               | ✅              | ✅              |                        |
| [voyage ai](https://docs.litellm.ai/docs/providers/voyage)                          |               |                   |                                               |                                                   | ✅              |                        |
| [xinference [Xorbits Inference]](https://docs.litellm.ai/docs/providers/xinference) |               |                   |                                               |                                                   | ✅              |                        |
| [FriendliAI](https://docs.litellm.ai/docs/providers/friendliai)                              | ✅          | ✅         | ✅               | ✅              |                   |                        |
| [Galadriel](https://docs.litellm.ai/docs/providers/galadriel)                              | ✅          | ✅         | ✅               | ✅              |                   |                        |
| [GradientAI](https://docs.litellm.ai/docs/providers/gradient_ai)                              | ✅          | ✅         |                                   |                                                   |                   |                        |
| [Novita AI](https://novita.ai/models/llm?utm_source=github_litellm&utm_medium=github_readme&utm_campaign=github_link)                     | ✅          | ✅         | ✅               | ✅              |                   |                        |
| [Featherless AI](https://docs.litellm.ai/docs/providers/featherless_ai)                              | ✅          | ✅         | ✅               | ✅              |                   |                        |
| [Nebius AI Studio](https://docs.litellm.ai/docs/providers/nebius)                             | ✅          | ✅         | ✅               | ✅              | ✅              |                        |

**[Explore the Docs](https://docs.litellm.ai/docs/) for more details.**

## Contributing

We welcome contributions! See our [Contributing Guide (CONTRIBUTING.md)](CONTRIBUTING.md) for details.

### Quick Start for Contributors

```bash
git clone https://github.com/BerriAI/litellm.git
cd litellm
make install-dev    # Install development dependencies
make format         # Format your code
make lint           # Run all linting checks
make test-unit      # Run unit tests
make format-check   # Check formatting only
```

## Code Quality / Linting

LiteLLM follows the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html).

Automated checks include:

*   **Black** for code formatting
*   **Ruff** for linting and code quality
*   **MyPy** for type checking
*   **Circular import detection**
*   **Import safety checks**

All checks must pass before your PR can be merged.

## Enterprise

For companies needing enhanced security, user management, and professional support, explore our [Enterprise Tier](https://docs.litellm.ai/docs/proxy/enterprise).

*   ✅ **Features under the LiteLLM Commercial License:**
*   ✅ **Feature Prioritization**
*   ✅ **Custom Integrations**
*   ✅ **Professional Support - Dedicated discord + slack**
*   ✅ **Custom SLAs**
*   ✅ **Secure access with Single Sign-On**

[Talk to founders](https://calendly.com/d/4mp-gd3-k5k/litellm-1-1-onboarding-chat)

## Support

*   [Schedule Demo 👋](https://calendly.com/d/4mp-gd3-k5k/berriai-1-1-onboarding-litellm-hosted-version)
*   [Community Discord 💭](https://discord.gg/wuPM9dRgDw)
*   [Community Slack 💭](https://join.slack.com/share/enQtOTE0ODczMzk2Nzk4NC01YjUxNjY2YjBlYTFmNDRiZTM3NDFiYTM3MzVkODFiMDVjOGRjMmNmZTZkZTMzOWQzZGQyZWIwYjQ0MWExYmE3)
*   Our numbers 📞 +1 (770) 8783-106 / ‭+1 (412) 618-6238‬
*   Our emails ✉️ ishaan@berri.ai / krrish@berri.ai

## Run in Developer Mode

### Services

1.  Setup .env file in root
2.  Run dependant services `docker-compose up db prometheus`

### Backend

1.  (In root) create virtual environment `python -m venv .venv`
2.  Activate virtual environment `source .venv/bin/activate`
3.  Install dependencies `pip install -e ".[all]"`
4.  Start proxy backend `python3 /path/to/litellm/proxy_cli.py`

### Frontend

1.  Navigate to `ui/litellm-dashboard`
2.  Install dependencies `npm install`
3.  Run `npm run dev` to start the dashboard

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
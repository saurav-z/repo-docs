# LiteLLM: Unified LLM API Access and Management

**Effortlessly call and manage all major LLM APIs using a single, OpenAI-compatible format.** ([View Original Repo](https://github.com/BerriAI/litellm))

<p align="center">
    <a href="https://render.com/deploy?repo=https://github.com/BerriAI/litellm" target="_blank" rel="nofollow"><img src="https://render.com/images/deploy-to-render-button.svg" alt="Deploy to Render"></a>
    <a href="https://railway.app/template/HLP0Ub?referralCode=jch2ME">
      <img src="https://railway.app/button.svg" alt="Deploy on Railway">
    </a>
</p>

<p align="center">
    <a href="https://pypi.org/project/litellm/" target="_blank">
        <img src="https://img.shields.io/pypi/v/litellm.svg" alt="PyPI Version">
    </a>
    <a href="https://www.ycombinator.com/companies/berriai">
        <img src="https://img.shields.io/badge/Y%20Combinator-W23-orange?style=flat-square" alt="Y Combinator W23">
    </a>
    <a href="https://wa.link/huol9n">
        <img src="https://img.shields.io/static/v1?label=Chat%20on&message=WhatsApp&color=success&logo=WhatsApp&style=flat-square" alt="Whatsapp">
    </a>
    <a href="https://discord.gg/wuPM9dRgDw">
        <img src="https://img.shields.io/static/v1?label=Chat%20on&message=Discord&color=blue&logo=Discord&style=flat-square" alt="Discord">
    </a>
    <a href="https://join.slack.com/share/enQtOTE0ODczMzk2Nzk4NC01YjUxNjY2YjBlYTFmNDRiZTM3NDFiYTM3MzVkODFiMDVjOGRjMmNmZTZkZTMzOWQzZGQyZWIwYjQ0MWExYmE3">
        <img src="https://img.shields.io/static/v1?label=Chat%20on&message=Slack&color=black&logo=Slack&style=flat-square" alt="Slack">
    </a>
</p>

## Key Features

*   **Universal API Access:** Call LLMs from Bedrock, Hugging Face, VertexAI, TogetherAI, Azure, OpenAI, Groq, and more, using a unified interface.
*   **Consistent Output:**  Receive standardized text responses at `['choices'][0]['message']['content']`.
*   **Intelligent Routing:**  Implement retry and fallback logic across multiple deployments (e.g., Azure/OpenAI).
*   **Cost Management:**  Set budgets and rate limits per project, API key, and model via the [LiteLLM Proxy Server (LLM Gateway)](#litellm-proxy-server-llm-gateway).
*   **Asynchronous Support:** Utilize `acompletion` for asynchronous LLM calls.
*   **Streaming Capabilities:**  Stream model responses for enhanced user experiences.
*   **Observability & Logging:** Integrate with tools like Lunary, MLflow, Langfuse, and others for comprehensive monitoring.

## Quickstart

### Installation

```shell
pip install litellm
```

### Basic Usage

```python
from litellm import completion
import os

# Set environment variables
os.environ["OPENAI_API_KEY"] = "your-openai-key"
os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-key"

messages = [{"content": "Hello, how are you?","role": "user"}]

# OpenAI call
response = completion(model="openai/gpt-4o", messages=messages)

# Anthropic call
response = completion(model="anthropic/claude-sonnet-4-20250514", messages=messages)
print(response)
```

### Example Response (OpenAI Format)

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

*   **Model Selection:** Call models using the format `model=<provider_name>/<model_name>`.  See [Provider Docs](https://docs.litellm.ai/docs/providers) for specific details.

## Asynchronous Calls

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

### Streaming Response Example

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

## Logging and Observability

LiteLLM supports pre-defined callbacks for logging to Lunary, MLflow, Langfuse, DynamoDB, S3 Buckets, Helicone, Promptlayer, Traceloop, Athina, Slack.

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
litellm.success_callback = ["lunary", "mlflow", "langfuse", "athina", "helicone"] # log input/output to lunary, langfuse, supabase, athina, helicone etc

# OpenAI call
response = completion(model="openai/gpt-4o", messages=[{"role": "user", "content": "Hi 👋 - i'm openai"}])
```

## LiteLLM Proxy Server (LLM Gateway)

**Enhance control and management with the LiteLLM Proxy.** ([Docs](https://docs.litellm.ai/docs/simple_proxy))

*   **Track Spend:** Monitor and analyze LLM costs across projects.
*   **Load Balancing:** Distribute requests across multiple deployments.
*   **Authentication:** Implement custom authentication hooks.
*   **Logging:** Utilize custom logging hooks.
*   **Rate Limiting:** Enforce rate limits for enhanced resource control.

### Quick Start

```shell
pip install 'litellm[proxy]'
```

1.  **Start the Proxy:**

    ```shell
    litellm --model huggingface/bigcode/starcoder
    # INFO: Proxy running on http://0.0.0.0:4000
    ```

2.  **Make Requests to the Proxy:**

    ```python
    import openai
    client = openai.OpenAI(api_key="anything", base_url="http://0.0.0.0:4000")
    response = client.chat.completions.create(model="gpt-3.5-turbo", messages = [
        {
            "role": "user",
            "content": "this is a test request, write a short poem"
        }
    ])
    print(response)
    ```
    *   **Use with Langchain, OpenAI SDK, Anthropic SDK, Mistral SDK, LlamaIndex, Instructor, and Curl:**  See the [Proxy User Keys Documentation](https://docs.litellm.ai/docs/proxy/user_keys).

### Proxy Key Management

Connect the proxy with a Postgres DB to create proxy keys.

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

*   **UI:** Access the UI at `/ui` on your proxy server.
*   **Generate Keys:** Use `POST /key/generate` to create proxy keys with budgets and rate limits.

### Key Generation Example

```shell
curl 'http://0.0.0.0:4000/key/generate' \
--header 'Authorization: Bearer sk-1234' \
--header 'Content-Type: application/json' \
--data-raw '{"models": ["gpt-3.5-turbo", "gpt-4", "claude-2"], "duration": "20m","metadata": {"user": "ishaan@berri.ai", "team": "core-infra"}}'
```

## Supported Providers

| Provider                                                                            | Completion | Streaming | Async Completion | Async Streaming | Async Embedding | Async Image Generation |
| :---------------------------------------------------------------------------------- | :--------- | :-------- | :--------------- | :-------------- | :-------------- | :--------------------- |
| [openai](https://docs.litellm.ai/docs/providers/openai)                             | ✅         | ✅        | ✅               | ✅              | ✅              | ✅                     |
| [Meta - Llama API](https://docs.litellm.ai/docs/providers/meta_llama)                               | ✅         | ✅        | ✅               | ✅              |                   |                       |
| [azure](https://docs.litellm.ai/docs/providers/azure)                               | ✅         | ✅        | ✅               | ✅              | ✅              | ✅                     |
| [AI/ML API](https://docs.litellm.ai/docs/providers/aiml)                               | ✅         | ✅        | ✅               | ✅              | ✅              | ✅                     |
| [aws - sagemaker](https://docs.litellm.ai/docs/providers/aws_sagemaker)             | ✅         | ✅        | ✅               | ✅              | ✅              |                       |
| [aws - bedrock](https://docs.litellm.ai/docs/providers/bedrock)                     | ✅         | ✅        | ✅               | ✅              | ✅              |                       |
| [google - vertex_ai](https://docs.litellm.ai/docs/providers/vertex)                 | ✅         | ✅        | ✅               | ✅              | ✅              | ✅                     |
| [google - palm](https://docs.litellm.ai/docs/providers/palm)                        | ✅         | ✅        | ✅               | ✅              |                   |                       |
| [google AI Studio - gemini](https://docs.litellm.ai/docs/providers/gemini)          | ✅         | ✅        | ✅               | ✅              |                   |                       |
| [mistral ai api](https://docs.litellm.ai/docs/providers/mistral)                    | ✅         | ✅        | ✅               | ✅              | ✅              |                       |
| [cloudflare AI Workers](https://docs.litellm.ai/docs/providers/cloudflare_workers)  | ✅         | ✅        | ✅               | ✅              |                   |                       |
| [cohere](https://docs.litellm.ai/docs/providers/cohere)                             | ✅         | ✅        | ✅               | ✅              | ✅              |                       |
| [anthropic](https://docs.litellm.ai/docs/providers/anthropic)                       | ✅         | ✅        | ✅               | ✅              |                   |                       |
| [empower](https://docs.litellm.ai/docs/providers/empower)                    | ✅        | ✅        | ✅               | ✅              |                   |                       |
| [huggingface](https://docs.litellm.ai/docs/providers/huggingface)                   | ✅         | ✅        | ✅               | ✅              | ✅              |                       |
| [replicate](https://docs.litellm.ai/docs/providers/replicate)                       | ✅         | ✅        | ✅               | ✅              |                   |                       |
| [together_ai](https://docs.litellm.ai/docs/providers/togetherai)                    | ✅         | ✅        | ✅               | ✅              |                   |                       |
| [openrouter](https://docs.litellm.ai/docs/providers/openrouter)                     | ✅         | ✅        | ✅               | ✅              |                   |                       |
| [ai21](https://docs.litellm.ai/docs/providers/ai21)                                 | ✅         | ✅        | ✅               | ✅              |                   |                       |
| [baseten](https://docs.litellm.ai/docs/providers/baseten)                           | ✅         | ✅        | ✅               | ✅              |                   |                       |
| [vllm](https://docs.litellm.ai/docs/providers/vllm)                                 | ✅         | ✅        | ✅               | ✅              |                   |                       |
| [nlp_cloud](https://docs.litellm.ai/docs/providers/nlp_cloud)                       | ✅         | ✅        | ✅               | ✅              |                   |                       |
| [aleph alpha](https://docs.litellm.ai/docs/providers/aleph_alpha)                   | ✅         | ✅        | ✅               | ✅              |                   |                       |
| [petals](https://docs.litellm.ai/docs/providers/petals)                             | ✅         | ✅        | ✅               | ✅              |                   |                       |
| [ollama](https://docs.litellm.ai/docs/providers/ollama)                             | ✅         | ✅        | ✅               | ✅              | ✅              |                       |
| [deepinfra](https://docs.litellm.ai/docs/providers/deepinfra)                       | ✅         | ✅        | ✅               | ✅              |                   |                       |
| [perplexity-ai](https://docs.litellm.ai/docs/providers/perplexity)                  | ✅         | ✅        | ✅               | ✅              |                   |                       |
| [Groq AI](https://docs.litellm.ai/docs/providers/groq)                              | ✅         | ✅        | ✅               | ✅              |                   |                       |
| [Deepseek](https://docs.litellm.ai/docs/providers/deepseek)                         | ✅         | ✅        | ✅               | ✅              |                   |                       |
| [anyscale](https://docs.litellm.ai/docs/providers/anyscale)                         | ✅         | ✅        | ✅               | ✅              |                   |                       |
| [IBM - watsonx.ai](https://docs.litellm.ai/docs/providers/watsonx)                  | ✅         | ✅        | ✅               | ✅              | ✅              |                       |
| [voyage ai](https://docs.litellm.ai/docs/providers/voyage)                          |           |          |                  |                   | ✅              |                       |
| [xinference [Xorbits Inference]](https://docs.litellm.ai/docs/providers/xinference) |           |          |                  |                   | ✅              |                       |
| [FriendliAI](https://docs.litellm.ai/docs/providers/friendliai)                              | ✅        | ✅        | ✅               | ✅              |                   |                       |
| [Galadriel](https://docs.litellm.ai/docs/providers/galadriel)                              | ✅        | ✅        | ✅               | ✅              |                   |                       |
| [GradientAI](https://docs.litellm.ai/docs/providers/gradient_ai)                              | ✅        | ✅        |                  |                   |                   |                       |
| [Novita AI](https://novita.ai/models/llm?utm_source=github_litellm&utm_medium=github_readme&utm_campaign=github_link)                     | ✅         | ✅        | ✅               | ✅              |                   |                       |
| [Featherless AI](https://docs.litellm.ai/docs/providers/featherless_ai)                              | ✅        | ✅        | ✅               | ✅              |                   |                       |
| [Nebius AI Studio](https://docs.litellm.ai/docs/providers/nebius)                             | ✅         | ✅        | ✅               | ✅              | ✅              |                       |

*   **Explore Detailed Provider Documentation:**  [Read the Docs](https://docs.litellm.ai/docs/providers)

## Contributing

We welcome contributions!  Please review the [Contributing Guide (CONTRIBUTING.md)](CONTRIBUTING.md) for details.

## Code Quality

LiteLLM adheres to the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html).  Automated checks include: Black (formatting), Ruff (linting), MyPy (type checking), circular import detection, and import safety checks.  All checks must pass for PRs.

## Support

*   [Schedule a Demo](https://calendly.com/d/4mp-gd3-k5k/berriai-1-1-onboarding-litellm-hosted-version)
*   [Community Discord](https://discord.gg/wuPM9dRgDw)
*   [Community Slack](https://join.slack.com/share/enQtOTE0ODczMzk2Nzk4NC01YjUxNjY2YjBlYTFmNDRiZTM3NDFiYTM3MzVkODFiMDVjOGRjMmNmZTZkZTMzOWQzZGQyZWIwYjQ0MWExYmE3)
*   **Phone:** +1 (770) 8783-106 / +1 (412) 618-6238
*   **Email:** ishaan@berri.ai / krrish@berri.ai

## Why We Built This

We created LiteLLM to simplify and streamline the management and translation of LLM calls across various providers.

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

## Run in Developer Mode

### Services

1.  Set up your `.env` file in the root.
2.  Run dependent services `docker-compose up db prometheus`.

### Backend

1.  Create a virtual environment in the root: `python -m venv .venv`
2.  Activate the virtual environment: `source .venv/bin/activate`
3.  Install dependencies: `pip install -e ".[all]"`
4.  Start the proxy backend: `python3 /path/to/litellm/proxy_cli.py`

### Frontend

1.  Navigate to `ui/litellm-dashboard`.
2.  Install dependencies: `npm install`
3.  Run `npm run dev` to start the dashboard.
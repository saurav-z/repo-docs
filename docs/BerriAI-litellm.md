<h1 align="center">
    🚀 LiteLLM: The Universal LLM API Gateway
</h1>

<p align="center">
  <b>Seamlessly call all LLM APIs using a unified OpenAI format, streamlining your development and deployment process.</b>  
  <br>
  <a href="https://github.com/BerriAI/litellm" target="_blank">
    <img src="https://img.shields.io/github/stars/BerriAI/litellm?style=social" alt="GitHub Stars">
  </a>
  <a href="https://pypi.org/project/litellm/" target="_blank">
    <img src="https://img.shields.io/pypi/v/litellm.svg" alt="PyPI Version">
  </a>
    <a href="https://discord.gg/wuPM9dRgDw">
      <img src="https://img.shields.io/static/v1?label=Chat%20on&message=Discord&color=blue&logo=Discord&style=flat-square" alt="Discord">
    </a>
</p>

<p align="center">
  <a href="https://render.com/deploy?repo=https://github.com/BerriAI/litellm" target="_blank" rel="nofollow"><img src="https://render.com/images/deploy-to-render-button.svg" alt="Deploy to Render"></a>
  <a href="https://railway.app/template/HLP0Ub?referralCode=jch2ME">
    <img src="https://railway.app/button.svg" alt="Deploy on Railway">
  </a>
</p>

<p align="center">
    <a href="https://docs.litellm.ai/docs/simple_proxy" target="_blank">LiteLLM Proxy Server (LLM Gateway)</a> | <a href="https://docs.litellm.ai/docs/hosted" target="_blank"> Hosted Proxy (Preview)</a> | <a href="https://docs.litellm.ai/docs/enterprise"target="_blank">Enterprise Tier</a>
</p>


[**View the LiteLLM Repository**](https://github.com/BerriAI/litellm)

## Key Features

*   **Unified API:** Translate inputs to various LLM providers using a single OpenAI-compatible format (e.g., Bedrock, Huggingface, VertexAI, TogetherAI, Azure, OpenAI, Groq).
*   **Consistent Output:**  Receive consistent text responses always accessible at `['choices'][0]['message']['content']`.
*   **Intelligent Routing:** Implement retry/fallback logic across multiple deployments for high availability.
*   **Cost Management:** Set budgets and rate limits per project, API key, and model with the LiteLLM Proxy Server (LLM Gateway).
*   **Asynchronous Operations:** Supports both asynchronous completion and streaming for increased efficiency.
*   **Observability & Logging:** Integrate with Lunary, MLflow, Langfuse, DynamoDB, S3 Buckets, Helicone, Promptlayer, Traceloop, Athina, and Slack for comprehensive monitoring and debugging.
*   **Streaming Support:** Real-time streaming responses for all supported models.
*   **Extensive Provider Support:** Access a wide range of LLM providers, with support for new providers added frequently.

## Quick Start

Install LiteLLM:

```bash
pip install litellm
```

Example Usage:

```python
from litellm import completion
import os

## Set Environment Variables
os.environ["OPENAI_API_KEY"] = "your-openai-key"
os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-key"

messages = [{"content": "Hello, how are you?", "role": "user"}]

# OpenAI call
response = completion(model="openai/gpt-4o", messages=messages)

# Anthropic call
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

Call any model supported by a provider, with `model=<provider_name>/<model_name>`. Refer to the [provider docs](https://docs.litellm.ai/docs/providers) for more information.

### Asynchronous Completion

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

### Streaming

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

### Response Chunk (OpenAI Format)

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

Integrate with various tools for robust logging and monitoring.

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

The LiteLLM Proxy provides:

*   [Hooks for Auth](https://docs.litellm.ai/docs/proxy/virtual_keys#custom-auth)
*   [Hooks for Logging](https://docs.litellm.ai/docs/proxy/logging#step-1---create-your-custom-litellm-callback-class)
*   [Cost Tracking](https://docs.litellm.ai/docs/proxy/virtual_keys#tracking-spend)
*   [Rate Limiting](https://docs.litellm.ai/docs/proxy/users#set-rate-limits)

### Quick Start Proxy - CLI

```bash
pip install 'litellm[proxy]'
```

1.  **Start LiteLLM Proxy**

    ```bash
    litellm --model huggingface/bigcode/starcoder
    #INFO: Proxy running on http://0.0.0.0:4000
    ```

2.  **Make ChatCompletions Request to Proxy**

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

## Proxy Key Management ([Docs](https://docs.litellm.ai/docs/proxy/virtual_keys))

Connect the proxy with a Postgres DB to create proxy keys

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

UI on `/ui` on your proxy server

![ui_3](https://github.com/BerriAI/litellm/assets/29436595/47c97d5e-b9be-4839-b28c-43d7f4f10033)

Set budgets and rate limits across multiple projects
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
|-------------------------------------------------------------------------------------|-----------------------------------------|------------------------------------------------|-------------------------------------|---------------------------------------------------|-----------------------------------------------|-----------------------------------------|
| [openai](https://docs.litellm.ai/docs/providers/openai)                             | ✅                                      | ✅                                              | ✅                                   | ✅                                                  | ✅                                             | ✅                                       |
| [Meta - Llama API](https://docs.litellm.ai/docs/providers/meta_llama)                | ✅                                      | ✅                                              | ✅                                   | ✅                                                  |                                               |                                         |
| [azure](https://docs.litellm.ai/docs/providers/azure)                               | ✅                                      | ✅                                              | ✅                                   | ✅                                                  | ✅                                             | ✅                                       |
| [AI/ML API](https://docs.litellm.ai/docs/providers/aiml)                               | ✅                                      | ✅                                              | ✅                                   | ✅                                                  | ✅                                             | ✅                                       |
| [aws - sagemaker](https://docs.litellm.ai/docs/providers/aws_sagemaker)             | ✅                                      | ✅                                              | ✅                                   | ✅                                                  | ✅                                             |                                         |
| [aws - bedrock](https://docs.litellm.ai/docs/providers/bedrock)                     | ✅                                      | ✅                                              | ✅                                   | ✅                                                  | ✅                                             |                                         |
| [google - vertex_ai](https://docs.litellm.ai/docs/providers/vertex)                 | ✅                                      | ✅                                              | ✅                                   | ✅                                                  | ✅                                             | ✅                                       |
| [google - palm](https://docs.litellm.ai/docs/providers/palm)                        | ✅                                      | ✅                                              | ✅                                   | ✅                                                  |                                               |                                         |
| [google AI Studio - gemini](https://docs.litellm.ai/docs/providers/gemini)          | ✅                                      | ✅                                              | ✅                                   | ✅                                                  |                                               |                                         |
| [mistral ai api](https://docs.litellm.ai/docs/providers/mistral)                    | ✅                                      | ✅                                              | ✅                                   | ✅                                                  | ✅                                             |                                         |
| [cloudflare AI Workers](https://docs.litellm.ai/docs/providers/cloudflare_workers)  | ✅                                      | ✅                                              | ✅                                   | ✅                                                  |                                               |                                         |
| [cohere](https://docs.litellm.ai/docs/providers/cohere)                             | ✅                                      | ✅                                              | ✅                                   | ✅                                                  | ✅                                             |                                         |
| [anthropic](https://docs.litellm.ai/docs/providers/anthropic)                       | ✅                                      | ✅                                              | ✅                                   | ✅                                                  |                                               |                                         |
| [empower](https://docs.litellm.ai/docs/providers/empower)                    | ✅                                      | ✅                                              | ✅                                   | ✅                                                 |
| [huggingface](https://docs.litellm.ai/docs/providers/huggingface)                   | ✅                                      | ✅                                              | ✅                                   | ✅                                                  | ✅                                             |                                         |
| [replicate](https://docs.litellm.ai/docs/providers/replicate)                       | ✅                                      | ✅                                              | ✅                                   | ✅                                                  |                                               |                                         |
| [together_ai](https://docs.litellm.ai/docs/providers/togetherai)                    | ✅                                      | ✅                                              | ✅                                   | ✅                                                  |                                               |                                         |
| [openrouter](https://docs.litellm.ai/docs/providers/openrouter)                     | ✅                                      | ✅                                              | ✅                                   | ✅                                                  |                                               |                                         |
| [ai21](https://docs.litellm.ai/docs/providers/ai21)                                 | ✅                                      | ✅                                              | ✅                                   | ✅                                                  |                                               |                                         |
| [baseten](https://docs.litellm.ai/docs/providers/baseten)                           | ✅                                      | ✅                                              | ✅                                   | ✅                                                  |                                               |                                         |
| [vllm](https://docs.litellm.ai/docs/providers/vllm)                                 | ✅                                      | ✅                                              | ✅                                   | ✅                                                  |                                               |                                         |
| [nlp_cloud](https://docs.litellm.ai/docs/providers/nlp_cloud)                       | ✅                                      | ✅                                              | ✅                                   | ✅                                                  |                                               |                                         |
| [aleph alpha](https://docs.litellm.ai/docs/providers/aleph_alpha)                   | ✅                                      | ✅                                              | ✅                                   | ✅                                                  |                                               |                                         |
| [petals](https://docs.litellm.ai/docs/providers/petals)                             | ✅                                      | ✅                                              | ✅                                   | ✅                                                  |                                               |                                         |
| [ollama](https://docs.litellm.ai/docs/providers/ollama)                             | ✅                                      | ✅                                              | ✅                                   | ✅                                                  | ✅                                             |                                         |
| [deepinfra](https://docs.litellm.ai/docs/providers/deepinfra)                       | ✅                                      | ✅                                              | ✅                                   | ✅                                                  |                                               |                                         |
| [perplexity-ai](https://docs.litellm.ai/docs/providers/perplexity)                  | ✅                                      | ✅                                              | ✅                                   | ✅                                                  |                                               |                                         |
| [Groq AI](https://docs.litellm.ai/docs/providers/groq)                              | ✅                                      | ✅                                              | ✅                                   | ✅                                                  |                                               |                                         |
| [Deepseek](https://docs.litellm.ai/docs/providers/deepseek)                         | ✅                                      | ✅                                              | ✅                                   | ✅                                                  |                                               |                                         |
| [anyscale](https://docs.litellm.ai/docs/providers/anyscale)                         | ✅                                      | ✅                                              | ✅                                   | ✅                                                  |                                               |                                         |
| [IBM - watsonx.ai](https://docs.litellm.ai/docs/providers/watsonx)                  | ✅                                      | ✅                                              | ✅                                   | ✅                                                  | ✅                                             |                                         |
| [voyage ai](https://docs.litellm.ai/docs/providers/voyage)                          |                                         |                                                 |                                      |                                                     | ✅                                             |                                         |
| [xinference [Xorbits Inference]](https://docs.litellm.ai/docs/providers/xinference) |                                         |                                                 |                                      |                                                     | ✅                                             |                                         |
| [FriendliAI](https://docs.litellm.ai/docs/providers/friendliai)                      | ✅                                      | ✅                                              | ✅                                   | ✅                                                  |                                               |                                         |
| [Galadriel](https://docs.litellm.ai/docs/providers/galadriel)                      | ✅                                      | ✅                                              | ✅                                   | ✅                                                  |                                               |                                         |
| [Novita AI](https://novita.ai/models/llm?utm_source=github_litellm&utm_medium=github_readme&utm_campaign=github_link)                      | ✅                                      | ✅                                              | ✅                                   | ✅                                                  |                                               |                                         |
| [Featherless AI](https://docs.litellm.ai/docs/providers/featherless_ai)                      | ✅                                      | ✅                                              | ✅                                   | ✅                                                  |                                               |                                         |
| [Nebius AI Studio](https://docs.litellm.ai/docs/providers/nebius)                      | ✅                                      | ✅                                              | ✅                                   | ✅                                                  | ✅                                              |                                         |

[**Explore the Comprehensive Docs**](https://docs.litellm.ai/docs/)

## Contributing

We welcome contributions to LiteLLM!

### Quick Start for Contributors

```bash
git clone https://github.com/BerriAI/litellm.git
cd litellm
make install-dev    # Install development dependencies
make format         # Format your code
make lint           # Run all linting checks
make test-unit      # Run unit tests
```

For detailed contributing guidelines, see [CONTRIBUTING.md](CONTRIBUTING.md).

### Code Quality / Linting

*   Follows the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html).

Automated checks:

*   **Black** for code formatting
*   **Ruff** for linting and code quality
*   **MyPy** for type checking
*   **Circular import detection**
*   **Import safety checks**

Run all checks locally:

```bash
make lint           # Run all linting (matches CI)
make format-check   # Check formatting only
```

All checks must pass before PR merges.

## Enterprise

For enhanced security, user management, and professional support: [Talk to founders](https://calendly.com/d/4mp-gd3-k5k/litellm-1-1-onboarding-chat)

### Key Benefits:

*   ✅ **Features under the [LiteLLM Commercial License](https://docs.litellm.ai/docs/proxy/enterprise)**
*   ✅ **Feature Prioritization**
*   ✅ **Custom Integrations**
*   ✅ **Professional Support - Dedicated Discord + Slack**
*   ✅ **Custom SLAs**
*   ✅ **Secure Access with Single Sign-On**

## Support and Community

*   [Schedule Demo 👋](https://calendly.com/d/4mp-gd3-k5k/berriai-1-1-onboarding-litellm-hosted-version)
*   [Community Discord 💭](https://discord.gg/wuPM9dRgDw)
*   Phone: +1 (770) 8783-106 / +1 (412) 618-6238
*   Email: ishaan@berri.ai / krrish@berri.ai

## Why We Built This

The need for simplicity. Our code started to get extremely complicated managing & translating calls between Azure, OpenAI and Cohere.

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
1.  Setup `.env` file in root
2.  Run dependent services `docker-compose up db prometheus`

### Backend
1.  (In root) create virtual environment `python -m venv .venv`
2.  Activate virtual environment `source .venv/bin/activate`
3.  Install dependencies `pip install -e ".[all]"`
4.  Start proxy backend `uvicorn litellm.proxy.proxy_server:app --host localhost --port 4000 --reload`

### Frontend
1.  Navigate to `ui/litellm-dashboard`
2.  Install dependencies `npm install`
3.  Run `npm run dev` to start the dashboard
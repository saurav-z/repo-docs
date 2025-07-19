<h1 align="center">
    🚀 LiteLLM: Unified LLM Access and Management
</h1>

<p align="center">
    <b>Simplify your LLM integrations and streamline your AI workflows with LiteLLM, the open-source library that provides a unified interface for accessing and managing various Large Language Models.</b>
    <br>
    <a href="https://github.com/BerriAI/litellm">
        <img src="https://img.shields.io/github/stars/BerriAI/litellm?style=social" alt="GitHub Stars">
    </a>
    <a href="https://pypi.org/project/litellm/" target="_blank">
        <img src="https://img.shields.io/pypi/v/litellm.svg" alt="PyPI Version">
    </a>
    <a href="https://discord.gg/wuPM9dRgDw">
        <img src="https://img.shields.io/discord/1140567747647495178?label=Discord&logo=discord&style=flat-square" alt="Discord">
    </a>
</p>

<div align="center">
  <a href="https://render.com/deploy?repo=https://github.com/BerriAI/litellm" target="_blank" rel="nofollow"><img src="https://render.com/images/deploy-to-render-button.svg" alt="Deploy to Render"></a>
  <a href="https://railway.app/template/HLP0Ub?referralCode=jch2ME">
    <img src="https://railway.app/button.svg" alt="Deploy on Railway">
  </a>
</div>

<p align="center">
    <a href="https://docs.litellm.ai/docs/simple_proxy" target="_blank">LiteLLM Proxy Server (LLM Gateway)</a> | <a href="https://docs.litellm.ai/docs/hosted" target="_blank"> Hosted Proxy (Preview)</a> | <a href="https://docs.litellm.ai/docs/enterprise"target="_blank">Enterprise Tier</a>
</p>

<p align="center">
  <a href="https://www.ycombinator.com/companies/berriai">
      <img src="https://img.shields.io/badge/Y%20Combinator-W23-orange?style=flat-square" alt="Y Combinator W23">
  </a>
</p>

## Key Features

*   **Unified API:** Interact with diverse LLMs (OpenAI, Azure, Google, Anthropic, and more!) using a single, consistent interface.
*   **OpenAI Format Compatibility:** Easily integrate LiteLLM with existing OpenAI-based code.
*   **Consistent Output:**  Receive predictable, standardized responses for easier processing.
*   **Intelligent Routing:** Implement retry and fallback mechanisms for high availability and reliability across multiple LLM deployments.
*   **Cost Management:** Set budgets and rate limits per project, API key, and model using the LiteLLM Proxy Server (LLM Gateway).
*   **Async Support:** Leverage asynchronous operations for improved performance.
*   **Streaming Support:** Stream model responses for real-time interaction.
*   **Observability:** Integrate with popular tools like Lunary, MLflow, Langfuse, and others for comprehensive logging and monitoring.

[**Get Started with LiteLLM**](https://docs.litellm.ai/docs/)

## Usage

> [!IMPORTANT]
> LiteLLM v1.0.0 now requires `openai>=1.0.0`. Migration guide [here](https://docs.litellm.ai/docs/migration)  
> LiteLLM v1.40.14+ now requires `pydantic>=2.0.0`. No changes required.

<a target="_blank" href="https://colab.research.google.com/github/BerriAI/litellm/blob/main/cookbook/liteLLM_Getting_Started.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

```shell
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

Call any model supported by a provider, with `model=<provider_name>/<model_name>`. There might be provider-specific details here, so refer to [provider docs for more information](https://docs.litellm.ai/docs/providers)

## Async ([Docs](https://docs.litellm.ai/docs/completion/stream#async-completion))

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

## Streaming ([Docs](https://docs.litellm.ai/docs/completion/stream))

liteLLM supports streaming the model response back, pass `stream=True` to get a streaming iterator in response.  
Streaming is supported for all models (Bedrock, Huggingface, TogetherAI, Azure, OpenAI, etc.)

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

### Response chunk (OpenAI Format)

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

LiteLLM offers pre-defined callbacks to send data to various tools, enabling comprehensive logging and monitoring of your LLM usage.  Integrations include Lunary, MLflow, Langfuse, DynamoDB, S3 Buckets, Helicone, Promptlayer, Traceloop, Athina, and Slack.

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

## LiteLLM Proxy Server (LLM Gateway) - ([Docs](https://docs.litellm.ai/docs/simple_proxy))

Manage costs and load balance your LLM requests efficiently.

[Hosted Proxy (Preview)](https://docs.litellm.ai/docs/hosted)

The proxy provides:

1.  [Hooks for auth](https://docs.litellm.ai/docs/proxy/virtual_keys#custom-auth)
2.  [Hooks for logging](https://docs.litellm.ai/docs/proxy/logging#step-1---create-your-custom-litellm-callback-class)
3.  [Cost tracking](https://docs.litellm.ai/docs/proxy/virtual_keys#tracking-spend)
4.  [Rate Limiting](https://docs.litellm.ai/docs/proxy/users#set-rate-limits)

## 📖 Proxy Endpoints - [Swagger Docs](https://litellm-api.up.railway.app/)

## Quick Start Proxy - CLI

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

| Provider | Completion | Streaming | Async Completion | Async Streaming | Async Embedding | Async Image Generation |
|---|---|---|---|---|---|---|
| [openai](https://docs.litellm.ai/docs/providers/openai) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| [Meta - Llama API](https://docs.litellm.ai/docs/providers/meta_llama) | ✅ | ✅ | ✅ | ✅ | | |
| [azure](https://docs.litellm.ai/docs/providers/azure) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| [AI/ML API](https://docs.litellm.ai/docs/providers/aiml) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| [aws - sagemaker](https://docs.litellm.ai/docs/providers/aws_sagemaker) | ✅ | ✅ | ✅ | ✅ | ✅ | |
| [aws - bedrock](https://docs.litellm.ai/docs/providers/bedrock) | ✅ | ✅ | ✅ | ✅ | ✅ | |
| [google - vertex_ai](https://docs.litellm.ai/docs/providers/vertex) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| [google - palm](https://docs.litellm.ai/docs/providers/palm) | ✅ | ✅ | ✅ | ✅ | | |
| [google AI Studio - gemini](https://docs.litellm.ai/docs/providers/gemini) | ✅ | ✅ | ✅ | ✅ | | |
| [mistral ai api](https://docs.litellm.ai/docs/providers/mistral) | ✅ | ✅ | ✅ | ✅ | ✅ | |
| [cloudflare AI Workers](https://docs.litellm.ai/docs/providers/cloudflare_workers) | ✅ | ✅ | ✅ | ✅ | | |
| [cohere](https://docs.litellm.ai/docs/providers/cohere) | ✅ | ✅ | ✅ | ✅ | ✅ | |
| [anthropic](https://docs.litellm.ai/docs/providers/anthropic) | ✅ | ✅ | ✅ | ✅ | | |
| [empower](https://docs.litellm.ai/docs/providers/empower) | ✅ | ✅ | ✅ | ✅ | | |
| [huggingface](https://docs.litellm.ai/docs/providers/huggingface) | ✅ | ✅ | ✅ | ✅ | ✅ | |
| [replicate](https://docs.litellm.ai/docs/providers/replicate) | ✅ | ✅ | ✅ | ✅ | | |
| [together_ai](https://docs.litellm.ai/docs/providers/togetherai) | ✅ | ✅ | ✅ | ✅ | | |
| [openrouter](https://docs.litellm.ai/docs/providers/openrouter) | ✅ | ✅ | ✅ | ✅ | | |
| [ai21](https://docs.litellm.ai/docs/providers/ai21) | ✅ | ✅ | ✅ | ✅ | | |
| [baseten](https://docs.litellm.ai/docs/providers/baseten) | ✅ | ✅ | ✅ | ✅ | | |
| [vllm](https://docs.litellm.ai/docs/providers/vllm) | ✅ | ✅ | ✅ | ✅ | | |
| [nlp_cloud](https://docs.litellm.ai/docs/providers/nlp_cloud) | ✅ | ✅ | ✅ | ✅ | | |
| [aleph alpha](https://docs.litellm.ai/docs/providers/aleph_alpha) | ✅ | ✅ | ✅ | ✅ | | |
| [petals](https://docs.litellm.ai/docs/providers/petals) | ✅ | ✅ | ✅ | ✅ | | |
| [ollama](https://docs.litellm.ai/docs/providers/ollama) | ✅ | ✅ | ✅ | ✅ | ✅ | |
| [deepinfra](https://docs.litellm.ai/docs/providers/deepinfra) | ✅ | ✅ | ✅ | ✅ | | |
| [perplexity-ai](https://docs.litellm.ai/docs/providers/perplexity) | ✅ | ✅ | ✅ | ✅ | | |
| [Groq AI](https://docs.litellm.ai/docs/providers/groq) | ✅ | ✅ | ✅ | ✅ | | |
| [Deepseek](https://docs.litellm.ai/docs/providers/deepseek) | ✅ | ✅ | ✅ | ✅ | | |
| [anyscale](https://docs.litellm.ai/docs/providers/anyscale) | ✅ | ✅ | ✅ | ✅ | | |
| [IBM - watsonx.ai](https://docs.litellm.ai/docs/providers/watsonx) | ✅ | ✅ | ✅ | ✅ | ✅ | |
| [voyage ai](https://docs.litellm.ai/docs/providers/voyage) |  |  |  |  | ✅ | |
| [xinference [Xorbits Inference]](https://docs.litellm.ai/docs/providers/xinference) |  |  |  |  | ✅ | |
| [FriendliAI](https://docs.litellm.ai/docs/providers/friendliai) | ✅ | ✅ | ✅ | ✅ |  | |
| [Galadriel](https://docs.litellm.ai/docs/providers/galadriel) | ✅ | ✅ | ✅ | ✅ |  | |
| [Novita AI](https://novita.ai/models/llm?utm_source=github_litellm&utm_medium=github_readme&utm_campaign=github_link) | ✅ | ✅ | ✅ | ✅ |  | |
| [Featherless AI](https://docs.litellm.ai/docs/providers/featherless_ai) | ✅ | ✅ | ✅ | ✅ |  | |
| [Nebius AI Studio](https://docs.litellm.ai/docs/providers/nebius) | ✅ | ✅ | ✅ | ✅ | ✅ | |

[**Explore the Docs**](https://docs.litellm.ai/docs/)

## Contributing

We welcome contributions to LiteLLM!  Help us improve our unified LLM access platform.

**Quick Start:**
```bash
git clone https://github.com/BerriAI/litellm.git
cd litellm
make install-dev    # Install development dependencies
make format         # Format your code
make lint           # Run all linting checks
make test-unit      # Run unit tests
```

See the [Contributing Guide (CONTRIBUTING.md)](CONTRIBUTING.md) for detailed instructions.

## Code Quality

LiteLLM adheres to the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html).

Automated checks include:
* Black for formatting
* Ruff for linting
* MyPy for type checking
* Circular import detection
* Import safety checks

Run all checks locally:
```bash
make lint           # Run all linting (matches CI)
make format-check   # Check formatting only
```
All checks must pass for PR merges.

## Enterprise

For companies requiring enhanced security, user management, and professional support:

[Schedule a Demo](https://calendly.com/d/4mp-gd3-k5k/litellm-1-1-onboarding-chat)

Includes:
* **Features under the [LiteLLM Commercial License](https://docs.litellm.ai/docs/proxy/enterprise)**
* Feature Prioritization
* Custom Integrations
* Professional Support - Dedicated discord + slack
* Custom SLAs
* Secure access with Single Sign-On

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

## Support / Talk with Founders

*   [Schedule a Demo 👋](https://calendly.com/d/4mp-gd3-k5k/berriai-1-1-onboarding-litellm-hosted-version)
*   [Community Discord 💭](https://discord.gg/wuPM9dRgDw)
*   [Community Slack 💭](https://join.slack.com/share/enQtOTE0ODczMzk2Nzk4NC01YjUxNjY2YjBlYTFmNDRiZTM3NDFiYTM3MzVkODFiMDVjOGRjMmNmZTZkZTMzOWQzZGQyZWIwYjQ0MWExYmE3)
*   Our numbers 📞 +1 (770) 8783-106 / ‭+1 (412) 618-6238‬
*   Our emails ✉️ ishaan@berri.ai / krrish@berri.ai

## Why We Built This

We started LiteLLM to simplify the complexities of managing and translating calls between diverse LLM providers like Azure, OpenAI, and Cohere, streamlining our AI development process.

## Run in Developer Mode

### Services
1.  Setup .env file in root
2.  Run dependant services `docker-compose up db prometheus`

### Backend
1.  (In root) create virtual environment `python -m venv .venv`
2.  Activate virtual environment `source .venv/bin/activate`
3.  Install dependencies `pip install -e ".[all]"`
4.  Start proxy backend `uvicorn litellm.proxy.proxy_server:app --host localhost --port 4000 --reload`

### Frontend
1.  Navigate to `ui/litellm-dashboard`
2.  Install dependencies `npm install`
3.  Run `npm run dev` to start the dashboard
<h1 align="center">
    🚀 LiteLLM: Your Universal LLM Gateway
</h1>

<p align="center">
    <b>Simplify LLM API calls with a single, unified interface and unlock powerful features like routing, cost management, and robust observability.</b>
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
    Call all LLM APIs using the OpenAI format [Bedrock, Huggingface, VertexAI, TogetherAI, Azure, OpenAI, Groq etc.]
</p>

---

## Key Features of LiteLLM

*   ✅ **Unified API:** Call diverse LLMs (OpenAI, Azure, Anthropic, Cohere, and many more) using a consistent OpenAI-compatible interface.
*   ✅ **Consistent Output:**  Receive predictable text responses consistently at `['choices'][0]['message']['content']`.
*   ✅ **Intelligent Routing:** Implement smart retry and fallback logic across multiple deployments (e.g., Azure/OpenAI) for enhanced reliability.
*   ✅ **Cost & Rate Management:** Set budgets and rate limits per project, API key, and model with the LiteLLM Proxy Server.
*   ✅ **Asynchronous Support:** Leverage `async` and `await` for non-blocking, high-performance LLM calls.
*   ✅ **Streaming Capabilities:** Stream responses for all supported models (Bedrock, Huggingface, TogetherAI, Azure, OpenAI, etc.)
*   ✅ **Comprehensive Observability:** Integrate with popular logging and monitoring tools (Lunary, MLflow, Langfuse, Helicone, etc.) for detailed insights.

---

## Getting Started

**Installation:**

```bash
pip install litellm
```

**Basic Usage:**

```python
from litellm import completion
import os

# Set your API keys (replace with your actual keys)
os.environ["OPENAI_API_KEY"] = "your-openai-key"
os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-key"

messages = [{"content": "Hello, how are you?", "role": "user"}]

# Call OpenAI
response = completion(model="openai/gpt-4o", messages=messages)

# Call Anthropic
response = completion(model="anthropic/claude-sonnet-4-20250514", messages=messages)
print(response)
```

**Response Example (OpenAI Format):**

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

For detailed usage examples and model-specific configurations, refer to the [LiteLLM Documentation](https://docs.litellm.ai/docs/).

### Asynchronous Calls

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

### Streaming Support

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

**Response Chunk (OpenAI Format):**

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

---

## Logging and Observability

LiteLLM integrates with various observability tools.

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

Explore the [LiteLLM Documentation](https://docs.litellm.ai/docs/observability/callbacks) for detailed configuration.

---

## LiteLLM Proxy Server (LLM Gateway)

**[Documentation](https://docs.litellm.ai/docs/simple_proxy)**

The LiteLLM Proxy Server offers:

*   **Authentication Hooks**
*   **Logging Hooks**
*   **Cost Tracking**
*   **Rate Limiting**
*   **Hosted Proxy (Preview)**

**Quick Start:**

```bash
pip install 'litellm[proxy]'
```

### Step 1: Start the LiteLLM Proxy

```bash
litellm --model huggingface/bigcode/starcoder

#INFO: Proxy running on http://0.0.0.0:4000
```

### Step 2: Make a ChatCompletions Request to the Proxy

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

---

## Supported Providers

LiteLLM supports a wide array of LLM providers.  See the [Providers Documentation](https://docs.litellm.ai/docs/providers) for detailed instructions.

| Provider                                                                            | Completion | Streaming | Async Completion | Async Streaming | Async Embedding | Async Image Generation |
|-------------------------------------------------------------------------------------|------------|-----------|-------------------|-------------------|-----------------|-------------------------|
| [openai](https://docs.litellm.ai/docs/providers/openai)                             | ✅          | ✅         | ✅                | ✅                | ✅              | ✅                        |
| [Meta - Llama API](https://docs.litellm.ai/docs/providers/meta_llama)               | ✅          | ✅         | ✅                | ✅                |                 |                         |
| [azure](https://docs.litellm.ai/docs/providers/azure)                               | ✅          | ✅         | ✅                | ✅                | ✅              | ✅                        |
| [AI/ML API](https://docs.litellm.ai/docs/providers/aiml)                             | ✅          | ✅         | ✅                | ✅                | ✅              | ✅                        |
| [aws - sagemaker](https://docs.litellm.ai/docs/providers/aws_sagemaker)             | ✅          | ✅         | ✅                | ✅                | ✅              |                         |
| [aws - bedrock](https://docs.litellm.ai/docs/providers/bedrock)                     | ✅          | ✅         | ✅                | ✅                | ✅              |                         |
| [google - vertex_ai](https://docs.litellm.ai/docs/providers/vertex)                 | ✅          | ✅         | ✅                | ✅                | ✅              | ✅                        |
| [google - palm](https://docs.litellm.ai/docs/providers/palm)                        | ✅          | ✅         | ✅                | ✅                |                 |                         |
| [google AI Studio - gemini](https://docs.litellm.ai/docs/providers/gemini)          | ✅          | ✅         | ✅                | ✅                |                 |                         |
| [mistral ai api](https://docs.litellm.ai/docs/providers/mistral)                    | ✅          | ✅         | ✅                | ✅                | ✅              |                         |
| [cloudflare AI Workers](https://docs.litellm.ai/docs/providers/cloudflare_workers)  | ✅          | ✅         | ✅                | ✅                |                 |                         |
| [cohere](https://docs.litellm.ai/docs/providers/cohere)                             | ✅          | ✅         | ✅                | ✅                | ✅              |                         |
| [anthropic](https://docs.litellm.ai/docs/providers/anthropic)                       | ✅          | ✅         | ✅                | ✅                |                 |                         |
| [empower](https://docs.litellm.ai/docs/providers/empower)                           | ✅          | ✅         | ✅                | ✅                |                 |                         |
| [huggingface](https://docs.litellm.ai/docs/providers/huggingface)                   | ✅          | ✅         | ✅                | ✅                | ✅              |                         |
| [replicate](https://docs.litellm.ai/docs/providers/replicate)                       | ✅          | ✅         | ✅                | ✅                |                 |                         |
| [together_ai](https://docs.litellm.ai/docs/providers/togetherai)                    | ✅          | ✅         | ✅                | ✅                |                 |                         |
| [openrouter](https://docs.litellm.ai/docs/providers/openrouter)                     | ✅          | ✅         | ✅                | ✅                |                 |                         |
| [ai21](https://docs.litellm.ai/docs/providers/ai21)                                 | ✅          | ✅         | ✅                | ✅                |                 |                         |
| [baseten](https://docs.litellm.ai/docs/providers/baseten)                           | ✅          | ✅         | ✅                | ✅                |                 |                         |
| [vllm](https://docs.litellm.ai/docs/providers/vllm)                                 | ✅          | ✅         | ✅                | ✅                |                 |                         |
| [nlp_cloud](https://docs.litellm.ai/docs/providers/nlp_cloud)                       | ✅          | ✅         | ✅                | ✅                |                 |                         |
| [aleph alpha](https://docs.litellm.ai/docs/providers/aleph_alpha)                   | ✅          | ✅         | ✅                | ✅                |                 |                         |
| [petals](https://docs.litellm.ai/docs/providers/petals)                             | ✅          | ✅         | ✅                | ✅                |                 |                         |
| [ollama](https://docs.litellm.ai/docs/providers/ollama)                             | ✅          | ✅         | ✅                | ✅                | ✅              |                         |
| [deepinfra](https://docs.litellm.ai/docs/providers/deepinfra)                       | ✅          | ✅         | ✅                | ✅                |                 |                         |
| [perplexity-ai](https://docs.litellm.ai/docs/providers/perplexity)                  | ✅          | ✅         | ✅                | ✅                |                 |                         |
| [Groq AI](https://docs.litellm.ai/docs/providers/groq)                              | ✅          | ✅         | ✅                | ✅                |                 |                         |
| [Deepseek](https://docs.litellm.ai/docs/providers/deepseek)                         | ✅          | ✅         | ✅                | ✅                |                 |                         |
| [anyscale](https://docs.litellm.ai/docs/providers/anyscale)                         | ✅          | ✅         | ✅                | ✅                |                 |                         |
| [IBM - watsonx.ai](https://docs.litellm.ai/docs/providers/watsonx)                  | ✅          | ✅         | ✅                | ✅                | ✅              |                         |
| [voyage ai](https://docs.litellm.ai/docs/providers/voyage)                          |              |              |                     |                     | ✅              |                         |
| [xinference [Xorbits Inference]](https://docs.litellm.ai/docs/providers/xinference) |              |              |                     |                     | ✅              |                         |
| [FriendliAI](https://docs.litellm.ai/docs/providers/friendliai)                     | ✅          | ✅         | ✅                | ✅                |                 |                         |
| [Galadriel](https://docs.litellm.ai/docs/providers/galadriel)                       | ✅          | ✅         | ✅                | ✅                |                 |                         |
| [GradientAI](https://docs.litellm.ai/docs/providers/gradient_ai)                    | ✅          | ✅         |                     |                     |                 |                         |
| [Novita AI](https://novita.ai/models/llm?utm_source=github_litellm&utm_medium=github_readme&utm_campaign=github_link)                     | ✅          | ✅         | ✅                | ✅                |                 |                         |
| [Featherless AI](https://docs.litellm.ai/docs/providers/featherless_ai)              | ✅          | ✅         | ✅                | ✅                |                 |                         |
| [Nebius AI Studio](https://docs.litellm.ai/docs/providers/nebius)                      | ✅          | ✅         | ✅                | ✅                | ✅              |                         |

---

## Contributing

We welcome contributions! See our [Contributing Guide](CONTRIBUTING.md) for detailed instructions.

---

## Enterprise Solutions

For advanced security, user management, and support, explore our [Enterprise solutions](https://docs.litellm.ai/docs/proxy/enterprise).

*   ✅ **Features under the [LiteLLM Commercial License](https://docs.litellm.ai/docs/proxy/enterprise):**
*   ✅ **Feature Prioritization**
*   ✅ **Custom Integrations**
*   ✅ **Professional Support - Dedicated discord + slack**
*   ✅ **Custom SLAs**
*   ✅ **Secure access with Single Sign-On**

[Talk to founders](https://calendly.com/d/4mp-gd3-k5k/litellm-1-1-onboarding-chat)
---
## Support

-   [Schedule Demo 👋](https://calendly.com/d/4mp-gd3-k5k/berriai-1-1-onboarding-litellm-hosted-version)
-   [Community Discord 💭](https://discord.gg/wuPM9dRgDw)
-   [Community Slack 💭](https://join.slack.com/share/enQtOTE0ODczMzk2Nzk4NC01YjUxNjY2YjBlYTFmNDRiZTM3NDFiYTM3MzVkODFiMDVjOGRjMmNmZTZkZTMzOWQzZGQyZWIwYjQ0MWExYmE3)
-   Our numbers 📞 +1 (770) 8783-106 / ‭+1 (412) 618-6238‬
-   Our emails ✉️ ishaan@berri.ai / krrish@berri.ai

---

## Contributors

[See Contributors](https://github.com/BerriAI/litellm/graphs/contributors)

---

## Additional Information

*   **Why LiteLLM?** Our code became complex when managing calls between various LLM providers (Azure, OpenAI, Cohere). This is why we built LiteLLM to simplify it.

*   **Run in Developer mode** Instructions in Original README
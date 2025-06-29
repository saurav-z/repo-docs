# 🚀 LiteLLM: Unified LLM Access & Management

**Simplify LLM integration and access with LiteLLM, a powerful Python library and proxy, enabling you to call all LLM APIs using a single, OpenAI-compatible format.**  [Explore the original repo](https://github.com/BerriAI/litellm).

---

## Key Features

*   ✅ **Unified API Access:** Interact with a wide range of LLMs, including OpenAI, Azure, Anthropic, and many more, using a consistent OpenAI-like interface.
*   ✅ **Consistent Output:**  Receive predictable text responses always available at `['choices'][0]['message']['content']`.
*   ✅ **Intelligent Routing:**  Leverage robust retry and fallback logic across multiple deployments for improved reliability.
*   ✅ **Cost Management:**  Set budgets and rate limits at a granular level (per project, API key, model) with the LiteLLM Proxy Server.
*   ✅ **Asynchronous Support:**  Benefit from both async and streaming capabilities for enhanced performance and flexibility.
*   ✅ **Comprehensive Observability:** Integrate with tools like Lunary, MLflow, Langfuse, and others for detailed logging and monitoring.

## Quick Start

1.  **Installation:**

    ```bash
    pip install litellm
    ```

2.  **Basic Usage:**

    ```python
    from litellm import completion
    import os

    # Set API keys (example)
    os.environ["OPENAI_API_KEY"] = "your-openai-key"
    os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-key"

    messages = [{"content": "Hello, how are you?","role": "user"}]

    # Call OpenAI
    response = completion(model="openai/gpt-4o", messages=messages)

    # Call Anthropic
    response = completion(model="anthropic/claude-3-sonnet-20240229", messages=messages)
    print(response)
    ```

3.  **Example Response (OpenAI Format):**

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

## Streaming

```python
from litellm import completion
response = completion(model="openai/gpt-4o", messages=messages, stream=True)
for part in response:
    print(part.choices[0].delta.content or "")
```

## Logging and Observability

LiteLLM integrates with various logging tools.  Example:

```python
from litellm import completion
import os

# set env variables for logging tools
os.environ["LUNARY_PUBLIC_KEY"] = "your-lunary-public-key"
os.environ["HELICONE_API_KEY"] = "your-helicone-auth-key"
os.environ["LANGFUSE_PUBLIC_KEY"] = ""
os.environ["LANGFUSE_SECRET_KEY"] = ""
os.environ["ATHINA_API_KEY"] = "your-athina-api-key"

os.environ["OPENAI_API_KEY"] = "your-openai-key"

# set callbacks
litellm.success_callback = ["lunary", "mlflow", "langfuse", "athina", "helicone"]

#openai call
response = completion(model="openai/gpt-4o", messages=[{"role": "user", "content": "Hi 👋 - i'm openai"}])
```

---

## ⚙️ LiteLLM Proxy Server (LLM Gateway)

**[LiteLLM Proxy Server Documentation](https://docs.litellm.ai/docs/simple_proxy)**

Track spend + Load Balance across multiple projects.
[Hosted Proxy (Preview)](https://docs.litellm.ai/docs/hosted)

The proxy provides:

1.  [Hooks for auth](https://docs.litellm.ai/docs/proxy/virtual_keys#custom-auth)
2.  [Hooks for logging](https://docs.litellm.ai/docs/proxy/logging#step-1---create-your-custom-litellm-callback-class)
3.  [Cost tracking](https://docs.litellm.ai/docs/proxy/virtual_keys#tracking-spend)
4.  [Rate Limiting](https://docs.litellm.ai/docs/proxy/users#set-rate-limits)

### Quick Start

1.  **Install Proxy:**

    ```bash
    pip install 'litellm[proxy]'
    ```

2.  **Start Proxy:**

    ```bash
    litellm --model huggingface/bigcode/starcoder  # Run a local proxy, example model
    ```

3.  **Make Requests to the Proxy:**

    ```python
    import openai
    client = openai.OpenAI(api_key="anything",base_url="http://0.0.0.0:4000") # set proxy to base_url
    response = client.chat.completions.create(model="gpt-3.5-turbo", messages = [
        {
            "role": "user",
            "content": "this is a test request, write a short poem"
        }
    ])
    print(response)
    ```

4.  **Proxy Key Management:**

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

    ```shell
    curl 'http://0.0.0.0:4000/key/generate' \
    --header 'Authorization: Bearer sk-1234' \
    --header 'Content-Type: application/json' \
    --data-raw '{"models": ["gpt-3.5-turbo", "gpt-4", "claude-2"], "duration": "20m","metadata": {"user": "ishaan@berri.ai", "team": "core-infra"}}'
    ```

    Expected Response:

    ```shell
    {
        "key": "sk-kdEXbIqZRwEeEiHwdg7sFA", # Bearer token
        "expires": "2023-11-19T01:38:25.838000+00:00" # datetime object
    }
    ```

---

## 📚 Supported Providers

[**Comprehensive Provider Documentation**](https://docs.litellm.ai/docs/providers)

| Provider                                                                            | Completion | Streaming | Async Completion | Async Streaming | Async Embedding | Async Image Generation |
|-------------------------------------------------------------------------------------|------------|-----------|-------------------|-----------------|-----------------|-------------------------|
| [openai](https://docs.litellm.ai/docs/providers/openai)                             | ✅          | ✅          | ✅                | ✅              | ✅              | ✅                        |
| [Meta - Llama API](https://docs.litellm.ai/docs/providers/meta_llama)              | ✅          | ✅          | ✅                | ✅              |                 |                         |
| [azure](https://docs.litellm.ai/docs/providers/azure)                               | ✅          | ✅          | ✅                | ✅              | ✅              | ✅                        |
| [AI/ML API](https://docs.litellm.ai/docs/providers/aiml)                               | ✅          | ✅          | ✅                | ✅              | ✅              | ✅                        |
| [aws - sagemaker](https://docs.litellm.ai/docs/providers/aws_sagemaker)             | ✅          | ✅          | ✅                | ✅              | ✅              |                         |
| [aws - bedrock](https://docs.litellm.ai/docs/providers/bedrock)                     | ✅          | ✅          | ✅                | ✅              | ✅              |                         |
| [google - vertex_ai](https://docs.litellm.ai/docs/providers/vertex)                 | ✅          | ✅          | ✅                | ✅              | ✅              | ✅                        |
| [google - palm](https://docs.litellm.ai/docs/providers/palm)                        | ✅          | ✅          | ✅                | ✅              |                 |                         |
| [google AI Studio - gemini](https://docs.litellm.ai/docs/providers/gemini)          | ✅          | ✅          | ✅                | ✅              |                 |                         |
| [mistral ai api](https://docs.litellm.ai/docs/providers/mistral)                    | ✅          | ✅          | ✅                | ✅              | ✅              |                         |
| [cloudflare AI Workers](https://docs.litellm.ai/docs/providers/cloudflare_workers)  | ✅          | ✅          | ✅                | ✅              |                 |                         |
| [cohere](https://docs.litellm.ai/docs/providers/cohere)                             | ✅          | ✅          | ✅                | ✅              | ✅              |                         |
| [anthropic](https://docs.litellm.ai/docs/providers/anthropic)                       | ✅          | ✅          | ✅                | ✅              |                 |                         |
| [empower](https://docs.litellm.ai/docs/providers/empower)                    | ✅         | ✅         | ✅               | ✅             |                 |                         |
| [huggingface](https://docs.litellm.ai/docs/providers/huggingface)                   | ✅          | ✅          | ✅                | ✅              | ✅              |                         |
| [replicate](https://docs.litellm.ai/docs/providers/replicate)                       | ✅          | ✅          | ✅                | ✅              |                 |                         |
| [together_ai](https://docs.litellm.ai/docs/providers/togetherai)                    | ✅          | ✅          | ✅                | ✅              |                 |                         |
| [openrouter](https://docs.litellm.ai/docs/providers/openrouter)                     | ✅          | ✅          | ✅                | ✅              |                 |                         |
| [ai21](https://docs.litellm.ai/docs/providers/ai21)                                 | ✅          | ✅          | ✅                | ✅              |                 |                         |
| [baseten](https://docs.litellm.ai/docs/providers/baseten)                           | ✅          | ✅          | ✅                | ✅              |                 |                         |
| [vllm](https://docs.litellm.ai/docs/providers/vllm)                                 | ✅          | ✅          | ✅                | ✅              |                 |                         |
| [nlp_cloud](https://docs.litellm.ai/docs/providers/nlp_cloud)                       | ✅          | ✅          | ✅                | ✅              |                 |                         |
| [aleph alpha](https://docs.litellm.ai/docs/providers/aleph_alpha)                   | ✅          | ✅          | ✅                | ✅              |                 |                         |
| [petals](https://docs.litellm.ai/docs/providers/petals)                             | ✅          | ✅          | ✅                | ✅              |                 |                         |
| [ollama](https://docs.litellm.ai/docs/providers/ollama)                             | ✅          | ✅          | ✅                | ✅              | ✅              |                         |
| [deepinfra](https://docs.litellm.ai/docs/providers/deepinfra)                       | ✅          | ✅          | ✅                | ✅              |                 |                         |
| [perplexity-ai](https://docs.litellm.ai/docs/providers/perplexity)                  | ✅          | ✅          | ✅                | ✅              |                 |                         |
| [Groq AI](https://docs.litellm.ai/docs/providers/groq)                              | ✅          | ✅          | ✅                | ✅              |                 |                         |
| [Deepseek](https://docs.litellm.ai/docs/providers/deepseek)                         | ✅          | ✅          | ✅                | ✅              |                 |                         |
| [anyscale](https://docs.litellm.ai/docs/providers/anyscale)                         | ✅          | ✅          | ✅                | ✅              |                 |                         |
| [IBM - watsonx.ai](https://docs.litellm.ai/docs/providers/watsonx)                  | ✅          | ✅          | ✅                | ✅              | ✅              |                         |
| [voyage ai](https://docs.litellm.ai/docs/providers/voyage)                          |             |             |                  |                 | ✅              |                         |
| [xinference [Xorbits Inference]](https://docs.litellm.ai/docs/providers/xinference) |             |             |                  |                 | ✅              |                         |
| [FriendliAI](https://docs.litellm.ai/docs/providers/friendliai)                              | ✅          | ✅          | ✅                | ✅              |                 |                         |
| [Galadriel](https://docs.litellm.ai/docs/providers/galadriel)                              | ✅          | ✅          | ✅                | ✅              |                 |                         |
| [Novita AI](https://novita.ai/models/llm?utm_source=github_litellm&utm_medium=github_readme&utm_campaign=github_link)                     | ✅          | ✅          | ✅                | ✅              |                 |                         |
| [Featherless AI](https://docs.litellm.ai/docs/providers/featherless_ai)                              | ✅          | ✅          | ✅                | ✅              |                 |                         |
| [Nebius AI Studio](https://docs.litellm.ai/docs/providers/nebius)                             | ✅          | ✅          | ✅                | ✅              | ✅              |                         |

---

## 🤝 Contributing

Contributions are welcome! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

**Quick start:** `git clone` → `make install-dev` → `make format` → `make lint` → `make test-unit`

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

---

## 💬 Support & Contact

*   [Schedule Demo 👋](https://calendly.com/d/4mp-gd3-k5k/berriai-1-1-onboarding-litellm-hosted-version)
*   [Community Discord 💭](https://discord.gg/wuPM9dRgDw)
*   Phone: +1 (770) 8783-106 / ‭+1 (412) 618-6238‬
*   Email: ishaan@berri.ai / krrish@berri.ai

---

## 💡 Why LiteLLM?

We built LiteLLM to simplify and streamline our own LLM integrations across various providers.

---

## 👥 Contributors

<a href="https://github.com/BerriAI/litellm/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=BerriAI/litellm" />
</a>
<h1 align="center">
    🚀 LiteLLM: Universal LLM Access and Management
</h1>

<p align="center">
    <b>Seamlessly call all LLM APIs using a unified OpenAI format, simplifying integration and management across providers.</b>
    <br>
    <a href="https://github.com/BerriAI/litellm">
        <img src="https://img.shields.io/github/stars/BerriAI/litellm?style=social" alt="Stars">
    </a>
    <a href="https://pypi.org/project/litellm/" target="_blank">
        <img src="https://img.shields.io/pypi/v/litellm.svg" alt="PyPI Version">
    </a>
    <a href="https://discord.gg/wuPM9dRgDw">
        <img src="https://img.shields.io/discord/1158468484977008148?label=Discord&logo=discord&style=flat-square" alt="Discord">
    </a>
    <a href="https://www.litellm.ai/support">
        <img src="https://img.shields.io/static/v1?label=Chat%20on&message=Slack&color=black&logo=Slack&style=flat-square" alt="Slack">
    </a>

    <p align="center">
        <a href="https://render.com/deploy?repo=https://github.com/BerriAI/litellm" target="_blank" rel="nofollow"><img src="https://render.com/images/deploy-to-render-button.svg" alt="Deploy to Render"></a>
        <a href="https://railway.app/template/HLP0Ub?referralCode=jch2ME">
          <img src="https://railway.app/button.svg" alt="Deploy on Railway">
        </a>
    </p>
    <br>
    <a href="https://docs.litellm.ai/docs"><b>Explore the Documentation</b></a> |
    <a href="https://github.com/BerriAI/litellm"><b>View the GitHub Repository</b></a>
</p>

LiteLLM simplifies access and management of Large Language Models (LLMs), offering a unified interface for a wide range of providers.

## Key Features

*   ✅ **Unified API Access:** Translate inputs to provider's `completion`, `embedding`, and `image_generation` endpoints using a single OpenAI-compatible format.
*   ✅ **Consistent Output:**  Receive predictable text responses at `['choices'][0]['message']['content']`.
*   ✅ **Intelligent Routing:**  Implement retry/fallback logic across multiple deployments (e.g., Azure/OpenAI) with the [Router](https://docs.litellm.ai/docs/routing).
*   ✅ **Cost Management:** Set budgets and rate limits per project, API key, and model using the [LiteLLM Proxy Server (LLM Gateway)](https://docs.litellm.ai/docs/simple_proxy).
*   ✅ **Async & Streaming Support:** Leverage asynchronous calls and streaming responses for enhanced performance.
*   ✅ **Observability:**  Integrate with various logging tools for comprehensive monitoring and analysis, with support for tools like Lunary, MLflow, Langfuse, Helicone, and more.

## Quickstart

Install the LiteLLM Python package:

```bash
pip install litellm
```

Example Usage:

```python
from litellm import completion
import os

# Set environment variables
os.environ["OPENAI_API_KEY"] = "your-openai-key"
os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-key"

messages = [{ "content": "Hello, how are you?","role": "user"}]

# OpenAI call
response = completion(model="openai/gpt-4o", messages=messages)

# Anthropic call
response = completion(model="anthropic/claude-sonnet-4-20250514", messages=messages)
print(response)
```

## LiteLLM Proxy Server (LLM Gateway)

Enhance your LLM workflow with the LiteLLM Proxy Server, offering:

*   **Cost tracking & Load Balancing** across multiple projects
*   **Authentication Hooks:** Customize authentication using [custom hooks](https://docs.litellm.ai/docs/proxy/virtual_keys#custom-auth).
*   **Logging Hooks:** Implement custom logging with [custom callbacks](https://docs.litellm.ai/docs/proxy/logging#step-1---create-your-custom-litellm-callback-class).
*   **Spend Tracking:** Monitor costs effectively.
*   **Rate Limiting:** Control API usage with rate limits.

### Quick Start Proxy - CLI

```shell
pip install 'litellm[proxy]'
```

1.  **Start the LiteLLM Proxy:**

    ```bash
    litellm --model huggingface/bigcode/starcoder
    #INFO: Proxy running on http://0.0.0.0:4000
    ```

2.  **Make ChatCompletions Request to Proxy:**

    ```python
    import openai
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

LiteLLM supports a wide array of LLM providers. See the comprehensive list and features in the [Supported Providers](https://docs.litellm.ai/docs/providers) documentation.

## Contributing

We welcome contributions! See the [Contributing Guide (CONTRIBUTING.md)](CONTRIBUTING.md) for details.

## Enterprise

For enhanced security, user management, and professional support, explore our Enterprise offerings.
*   ✅ **Features under the [LiteLLM Commercial License](https://docs.litellm.ai/docs/proxy/enterprise):**
*   ✅ **Feature Prioritization**
*   ✅ **Custom Integrations**
*   ✅ **Professional Support - Dedicated discord + slack**
*   ✅ **Custom SLAs**
*   ✅ **Secure access with Single Sign-On**

[Talk to founders](https://calendly.com/d/4mp-gd3-k5k/litellm-1-1-onboarding-chat)

## Community and Support

*   [Discord](https://discord.gg/wuPM9dRgDw)
*   [Slack](https://www.litellm.ai/support)
*   Our numbers 📞 +1 (770) 8783-106 / ‭+1 (412) 618-6238‬
*   Our emails ✉️ ishaan@berri.ai / krrish@berri.ai
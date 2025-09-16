<div align="center">
  <h1>ChatMock: Access GPT-5 and Other Models with Your ChatGPT Account</h1>
  <p><b>Unlock the power of GPT-5 and other advanced OpenAI models through your existing ChatGPT Plus/Pro subscription.</b></p>
  <br>
</div>

## Introduction

ChatMock allows you to use your ChatGPT Plus/Pro account to access OpenAI models, including GPT-5, through an OpenAI-compatible API. Bypass API keys and use your existing subscription to integrate these powerful models into your code or utilize them with alternative chat interfaces.

[View the original repository on GitHub](https://github.com/RayBytes/ChatMock)

## Key Features

*   ✅ **OpenAI API Compatibility:** Works seamlessly with OpenAI clients and applications.
*   ✅ **No API Key Required:** Uses your authenticated ChatGPT login for access.
*   ✅ **GPT-5 Support:** Utilize the latest and most advanced models (GPT-5).
*   ✅ **Ollama Compatibility**: Supports Ollama endpoints.
*   ✅ **Tool Calling:** Supports OpenAI tool calling functionality.
*   ✅ **Vision/Image Understanding:** Enables image processing capabilities.
*   ✅ **Customizable Thinking Effort:** Adjust model reasoning for optimal performance.

## Getting Started

### macOS

#### GUI Application

Download the GUI application from the [GitHub releases](https://github.com/RayBytes/ChatMock/releases).

> **Note:** You may need to run the following command in your terminal to open the app:
>
> ```bash
> xattr -dr com.apple.quarantine /Applications/ChatMock.app
> ```
>
> *[More info here.](https://github.com/deskflow/deskflow/wiki/Running-on-macOS)*

#### Command Line (Homebrew)

Install ChatMock using [Homebrew](https://brew.sh/):

```bash
brew tap RayBytes/chatmock
brew install chatmock
```

### Python

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/RayBytes/ChatMock.git
    cd ChatMock
    ```
2.  **Login to ChatGPT:**
    ```bash
    python chatmock.py login
    ```
    Verify with: `python chatmock.py info`
3.  **Start the server:**
    ```bash
    python chatmock.py serve
    ```
    The server defaults to `http://127.0.0.1:8000`.  Append `/v1/` to the base URL for OpenAI compatibility.

### Examples

#### Python

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="key"  # ignored
)

resp = client.chat.completions.create(
    model="gpt-5",
    messages=[{"role": "user", "content": "hello world"}]
)

print(resp.choices[0].message.content)
```

#### curl

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-5",
    "messages": [{"role":"user","content":"hello world"}]
  }'
```

## Supported Models

*   `gpt-5`
*   `codex-mini`

## Configuration

### Thinking Effort

*   `--reasoning-effort`: (minimal, low, medium, high)

### Thinking Summaries

*   `--reasoning-summary`: (auto, concise, detailed, none)

## Important Notes

*   Requires a paid ChatGPT account.
*   Rate limits may be lower than the official ChatGPT app.
*   The context size is larger than in the regular ChatGPT app.
*   Use responsibly and at your own risk. This project is not affiliated with OpenAI.

## TODO

*   Explore more model settings.
*   Implement analytics (token counting, etc.).

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=RayBytes/ChatMock&type=Timeline)](https://www.star-history.com/#RayBytes/ChatMock&Timeline)
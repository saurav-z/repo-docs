# ChatMock: Access GPT-5 & More via Your ChatGPT Account

**Unlock the power of GPT-5 and other OpenAI models using your existing ChatGPT Plus/Pro subscription, without needing an API key.** [Check out the original repository!](https://github.com/RayBytes/ChatMock)

ChatMock allows you to leverage your paid ChatGPT account to access OpenAI models through a local server, providing an OpenAI/Ollama-compatible API. Use your existing ChatGPT login to access GPT-5 and other advanced models from your code or alternative chat interfaces.

## Key Features

*   **GPT-5 and More:** Access cutting-edge OpenAI models, including GPT-5, without needing an API key.
*   **OpenAI/Ollama Compatibility:**  Use ChatMock with any tool or application that supports the OpenAI API, and now includes Ollama!
*   **Easy Setup:**  Get started quickly with GUI applications for macOS, a command-line interface via Homebrew, or a simple Python setup.
*   **No API Key Required:** Uses your existing, paid ChatGPT account.
*   **Supports Advanced Features:** Includes tool calling, vision/image understanding, and thinking summaries.
*   **Customizable Reasoning:** Adjust the reasoning effort and summary style for optimal performance.

## Getting Started

### macOS Users

#### GUI Application

Download the user-friendly GUI app from the [GitHub releases](https://github.com/RayBytes/ChatMock/releases).

> **Note:**  If you encounter issues opening the app, you may need to run:
>
> ```bash
> xattr -dr com.apple.quarantine /Applications/ChatMock.app
> ```
>
> *[More info here.](https://github.com/deskflow/deskflow/wiki/Running-on-macOS)*

#### Command Line (Homebrew)

Install ChatMock as a command-line tool using [Homebrew](https://brew.sh/):

```bash
brew tap RayBytes/chatmock
brew install chatmock
```

### Python

For a Python-based setup:

1.  Clone or download the repository and navigate into the project directory.
2.  Sign in with your ChatGPT account:

    ```bash
    python chatmock.py login
    ```

    Verify login with: `python chatmock.py info`
3.  Start the local server:

    ```bash
    python chatmock.py serve
    ```

    The server runs by default at `http://127.0.0.1:8000`. Remember to include `/v1/` when using it as an OpenAI endpoint. (e.g., `http://127.0.0.1:8000/v1`)

## Examples

### Python

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

### curl

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

## Customization

### Reasoning Effort

*   `--reasoning-effort` (minimal, low, medium, high)

### Reasoning Summaries

*   `--reasoning-summary` (auto, concise, detailed, none)

**For the fastest responses, try `--reasoning-effort low` and `--reasoning-summary none`.**

**All parameters and choices can be viewed by running `python chatmock.py serve --h`**

**To customize the behavior of the thinking summaries, set `--reasoning-compat` to `legacy` to set the reasoning tag instead of being returned in the actual response text.**

## Notes & Limitations

*   Requires an active, paid ChatGPT account.
*   Expect potentially lower rate limits than the ChatGPT app.
*   Some context length may be used for internal instructions.
*   Use responsibly and at your own risk. This project is not affiliated with OpenAI.

## Future Development

*   Explore more model settings.
*   Implement analytics for usage tracking (token counting, etc.).

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=RayBytes/ChatMock&type=Timeline)](https://www.star-history.com/#RayBytes/ChatMock&Timeline)
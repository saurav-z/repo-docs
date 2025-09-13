<!-- Improved & SEO-Optimized README -->
<div align="center">
  <h1>ChatMock: Access GPT-5 and More via Your ChatGPT Plan</h1>
  <p><b>Unlock the power of advanced AI models using your existing ChatGPT Plus/Pro account with an OpenAI-compatible API.</b></p>
  <p>
    <a href="https://github.com/RayBytes/ChatMock">
      <img src="https://img.shields.io/github/stars/RayBytes/ChatMock?style=social" alt="GitHub Stars"/>
    </a>
  </p>
  <br>
</div>

## Key Features of ChatMock

ChatMock transforms your paid ChatGPT account into a versatile AI development tool by creating a local OpenAI/Ollama-compatible API. Here’s what you can do:

*   ✅ **Access GPT-5 & Beyond:** Utilize advanced models like GPT-5 and others directly through your ChatGPT subscription.
*   ✅ **OpenAI Compatibility:** Use standard OpenAI API calls and libraries.
*   ✅ **Ollama Compatibility:** Supports Ollama API for flexible model access.
*   ✅ **No API Key Required:** Authenticates via your ChatGPT login, eliminating the need for API keys.
*   ✅ **Tool Calling & Vision Support:** Leverages advanced features.
*   ✅ **Thinking Summaries:** Includes thinking summary support for insightful responses.

## Getting Started

### Mac Users

#### GUI Application

Download and run the user-friendly GUI app from the [GitHub releases](https://github.com/RayBytes/ChatMock/releases).

> **Note:** You may need to run a terminal command the first time you launch the app due to macOS security:
> ```bash
> xattr -dr com.apple.quarantine /Applications/ChatMock.app
> ```
> *[Learn more.](https://github.com/deskflow/deskflow/wiki/Running-on-macOS)*

#### Command Line (Homebrew)

Alternatively, install ChatMock using Homebrew:

```bash
brew tap RayBytes/chatmock
brew install chatmock
```

### Python (Command Line)

For Python users, follow these steps:

1.  Clone or download the repository, then navigate into the project directory.
2.  Sign in to your ChatGPT account:
    ```bash
    python chatmock.py login
    ```
    Verify the login with: `python chatmock.py info`.
3.  Start the local server:
    ```bash
    python chatmock.py serve
    ```
    Use the server's address and port (default: `http://127.0.0.1:8000`) as your `baseURL`.  Remember to include `/v1/` for OpenAI compatibility.

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

## Supported Features

*   Tool Calling
*   Vision/Image Understanding
*   Thinking Summaries

## Configuration & Customization

### Thinking Effort

Control the model's "thinking" effort with `--reasoning-effort` (minimal, low, medium, high). Default: `medium`.  Set at server start or override in API requests.

### Thinking Summaries

Customize the format of reasoning summaries using `--reasoning-summary` (auto, concise, detailed, none).

### Other Customization Options
Additional settings can be adjusted to fine-tune the model's behavior.
For a comprehensive list, see `python chatmock.py serve --h`.

## Important Notes

*   Requires a paid ChatGPT account.
*   Rate limits may be lower than the official ChatGPT app.
*   This project is not affiliated with OpenAI. Use responsibly.
*   Context size is larger than in the standard ChatGPT app.

## Supported Models

*   `gpt-5`
*   `codex-mini`

## Future Development

*   Explore more model settings.
*   Implement analytics for usage tracking (token counting, etc.).

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=RayBytes/ChatMock&type=Timeline)](https://www.star-history.com/#RayBytes/ChatMock&Timeline)

---

**[Visit the original repository on GitHub](https://github.com/RayBytes/ChatMock) for more details and to contribute.**
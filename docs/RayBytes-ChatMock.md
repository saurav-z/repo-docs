<div align="center">
  <h1>ChatMock: Access OpenAI Models with Your ChatGPT Account</h1>

  <div align="center">
    <a href="https://github.com/RayBytes/ChatMock/stargazers"><img src="https://img.shields.io/github/stars/RayBytes/ChatMock" alt="Stars Badge"/></a>
    <a href="https://github.com/RayBytes/ChatMock/network/members"><img src="https://img.shields.io/github/forks/RayBytes/ChatMock" alt="Forks Badge"/></a>
    <a href="https://github.com/RayBytes/ChatMock/pulls"><img src="https://img.shields.io/github/issues-pr/RayBytes/ChatMock" alt="Pull Requests Badge"/></a>
    <a href="https://github.com/RayBytes/ChatMock/issues"><img src="https://img.shields.io/github/issues/RayBytes/ChatMock" alt="Issues Badge"/></a>
    <a href="https://github.com/RayBytes/ChatMock/graphs/contributors"><img alt="GitHub contributors" src="https://img.shields.io/github/contributors/RayBytes/ChatMock?color=2b9348"></a>
    <a href="https://github.com/RayBytes/ChatMock/blob/master/LICENSE"><img src="https://img.shields.io/github/license/RayBytes/ChatMock?color=2b9348" alt="License Badge"/></a>
  </div>

  <p><b>Unlock the power of OpenAI models like GPT-5 and GPT-5-Codex using your existing ChatGPT Plus/Pro subscription.</b></p>
  <br>
</div>

## Key Features

*   **Access Premium Models:** Utilize GPT-5, GPT-5-Codex, and other models through your ChatGPT Plus/Pro account.
*   **API Compatibility:**  Offers an OpenAI/Ollama compatible API, allowing integration with various chat applications and coding tools.
*   **No API Key Required:**  Leverage your existing ChatGPT subscription without the need for a separate API key.
*   **Customization Options:** Configure reasoning effort, summaries, and enable web search functionality.
*   **Supported Models:**
    *   `gpt-5`
    *   `gpt-5-codex`
    *   `codex-mini`
*   **Web Search Integration:** Access OpenAI tools, including web search, for enhanced responses.

## Getting Started

ChatMock runs a local server that acts as an OpenAI-compatible API, using your authenticated ChatGPT login.

### Mac Users

#### GUI Application

Download the GUI app from the [GitHub releases](https://github.com/RayBytes/ChatMock/releases).

> **Note:** You may need to run this in your terminal to open the app:
>
> ```bash
> xattr -dr com.apple.quarantine /Applications/ChatMock.app
> ```

#### Command Line (Homebrew)

Install using [Homebrew](https://brew.sh/):

```bash
brew tap RayBytes/chatmock
brew install chatmock
```

### Python

Alternatively, run as a Python Flask server.

1.  Clone or download the repository and `cd` into the project directory.
2.  Sign in to your ChatGPT account:

    ```bash
    python chatmock.py login
    ```
    Verify with: `python chatmock.py info`
3.  Start the local server:

    ```bash
    python chatmock.py serve
    ```
    The server runs at `http://127.0.0.1:8000` (default).  Use `/v1/` in your base URL when integrating with OpenAI-compatible applications (e.g., `http://127.0.0.1:8000/v1`).

### Docker

See the [Docker instructions](https://github.com/RayBytes/ChatMock/blob/main/DOCKER.md).

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

## Customization and Configuration

### Thinking Effort

Use `--reasoning-effort` (minimal, low, medium, high) to control GPT-5's effort. Defaults to `medium`.  Set it on server startup or within API requests.

### Thinking Summaries

Use `--reasoning-summary` (auto, concise, detailed, none) to customize summaries.

### OpenAI Tools

Enable web search with `--enable-web-search`.

In API requests:
*   `responses_tools`: `[{"type":"web_search"}]` or `{ "type": "web_search_preview" }`
*   `responses_tool_choice`: `"auto"` or `"none"`

#### Example

```json
{
  "model": "gpt-5",
  "messages": [{"role":"user","content":"Find current METAR rules"}],
  "stream": true,
  "responses_tools": [{"type": "web_search"}],
  "responses_tool_choice": "auto"
}
```

### Expose Reasoning Models

Use `--expose-reasoning-models` to expose each reasoning level as a separate model.

## Important Notes

*   Requires a paid ChatGPT account.
*   Context length might be affected by internal instructions.
*   Use responsibly. This is an educational project and not affiliated with OpenAI.
* For the fastest responses, set `--reasoning-effort` to minimal and `--reasoning-summary` to none.
* The context size is larger than in the regular ChatGPT app.

For all available options, run `python chatmock.py serve --h`.

**View the original project on GitHub: [RayBytes/ChatMock](https://github.com/RayBytes/ChatMock)**

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=RayBytes/ChatMock&type=Timeline)](https://www.star-history.com/#RayBytes/ChatMock&Timeline)
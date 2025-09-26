<div align="center">
  <h1>ChatMock: Unlock ChatGPT Plus/Pro Features with Your Own API</h1>
  <div align="center">
    <a href="https://github.com/RayBytes/ChatMock/stargazers"><img src="https://img.shields.io/github/stars/RayBytes/ChatMock" alt="Stars Badge"/></a>
    <a href="https://github.com/RayBytes/ChatMock/network/members"><img src="https://img.shields.io/github/forks/RayBytes/ChatMock" alt="Forks Badge"/></a>
    <a href="https://github.com/RayBytes/ChatMock/pulls"><img src="https://img.shields.io/github/issues-pr/RayBytes/ChatMock" alt="Pull Requests Badge"/></a>
    <a href="https://github.com/RayBytes/ChatMock/issues"><img src="https://img.shields.io/github/issues/RayBytes/ChatMock" alt="Issues Badge"/></a>
    <a href="https://github.com/RayBytes/ChatMock/graphs/contributors"><img alt="GitHub contributors" src="https://img.shields.io/github/contributors/RayBytes/ChatMock?color=2b9348"></a>
    <a href="https://github.com/RayBytes/ChatMock/blob/master/LICENSE"><img src="https://img.shields.io/github/license/RayBytes/ChatMock?color=2b9348" alt="License Badge"/></a>
  </div>
</div>

ChatMock empowers you to access advanced OpenAI models like GPT-5 through your existing ChatGPT Plus/Pro subscription by creating a local, OpenAI-compatible API.  **[View the project on GitHub](https://github.com/RayBytes/ChatMock)**

## Key Features

*   **OpenAI & Ollama Compatibility:**  Use with applications and tools that support the OpenAI API.
*   **GPT-5 Access:** Utilize GPT-5 and other advanced models available through your ChatGPT subscription.
*   **No API Key Required:** Leverage your existing ChatGPT Plus/Pro account without needing an API key.
*   **Flexible Deployment:**  Available as a GUI app for macOS, a command-line tool via Homebrew, a Python Flask server, or through Docker.
*   **Tool/Function Calling Support:** Enables integration with OpenAI tools, including web search.
*   **Customization Options:** Configure reasoning effort and summaries for tailored responses.
*   **Enhanced Context Size:** Benefit from a potentially larger context window compared to the standard ChatGPT app.

## Getting Started

### Mac Users

#### GUI Application

Download the GUI app from the [GitHub releases](https://github.com/RayBytes/ChatMock/releases).

> **Note:**  You may need to run the following command in your terminal to open the app if you encounter a security warning:
>
> ```bash
> xattr -dr com.apple.quarantine /Applications/ChatMock.app
> ```
> *[More info here.](https://github.com/deskflow/deskflow/wiki/Running-on-macOS)*

#### Command Line (Homebrew)

Alternatively, install via Homebrew:

```bash
brew tap RayBytes/chatmock
brew install chatmock
```

### Python

1.  Clone or download the repository.
2.  Navigate to the project directory: `cd ChatMock`
3.  Sign in with your ChatGPT account:

    ```bash
    python chatmock.py login
    ```
    Verify login with:  `python chatmock.py info`

4.  Start the local server:

    ```bash
    python chatmock.py serve
    ```
    Use the base URL (default: `http://127.0.0.1:8000`) in your applications, including `/v1/` at the end for OpenAI compatibility (e.g., `http://127.0.0.1:8000/v1`).

### Docker

See the [DOCKER.md](https://github.com/RayBytes/ChatMock/blob/main/DOCKER.md) for instructions.

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
*   `gpt-5-codex`
*   `codex-mini`

## Customization and Configuration

### Thinking Effort

Set the reasoning effort level (minimal, low, medium, high) using the `--reasoning-effort` flag when starting the server.  This can also be set within the API request.  Default is `medium`.

### Thinking Summaries

Customize thinking summaries (auto, concise, detailed, none) with the `--reasoning-summary` flag when starting the server.

### OpenAI Tools

Enable OpenAI tools, such as web search, by using the `--enable-web-search` parameter during server startup.  You can also control this via the API request using:

*   `responses_tools`:  `[{"type":"web_search"}]` / `{ "type": "web_search_preview" }`
*   `responses_tool_choice`:  `"auto"` or `"none"`

#### Example Web Search Usage

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

Use the `--expose-reasoning-models` flag to expose reasoning levels as individual models for easier selection in chat applications.

## Important Notes

*   Requires a paid ChatGPT Plus/Pro account.
*   Context length may be affected by internal instructions.
*   Use responsibly and at your own risk; this project is not affiliated with OpenAI and is for educational purposes.
*   For the fastest responses, consider setting `--reasoning-effort` to minimal and `--reasoning-summary` to none.
*   For further configuration options, see  `python chatmock.py serve --h`.
*   The model will send back thinking tags to make it compatible with chat apps. **If you don't like this behavior, you can instead set `--reasoning-compat` to legacy, and reasoning will be set in the reasoning tag instead of being returned in the actual response text.**

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=RayBytes/ChatMock&type=Timeline)](https://www.star-history.com/#RayBytes/ChatMock&Timeline)
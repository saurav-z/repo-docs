<div align="center">
  <h1>ChatMock: Access GPT-5 and Other Models with Your ChatGPT Account</h1>
  <a href="https://github.com/RayBytes/ChatMock/stargazers"><img src="https://img.shields.io/github/stars/RayBytes/ChatMock" alt="Stars Badge"/></a>
  <a href="https://github.com/RayBytes/ChatMock/network/members"><img src="https://img.shields.io/github/forks/RayBytes/ChatMock" alt="Forks Badge"/></a>
  <a href="https://github.com/RayBytes/ChatMock/pulls"><img src="https://img.shields.io/github/issues-pr/RayBytes/ChatMock" alt="Pull Requests Badge"/></a>
  <a href="https://github.com/RayBytes/ChatMock/issues"><img src="https://img.shields.io/github/issues/RayBytes/ChatMock" alt="Issues Badge"/></a>
  <a href="https://github.com/RayBytes/ChatMock/graphs/contributors"><img alt="GitHub contributors" src="https://img.shields.io/github/contributors/RayBytes/ChatMock?color=2b9348"></a>
  <a href="https://github.com/RayBytes/ChatMock/blob/master/LICENSE"><img src="https://img.shields.io/github/license/RayBytes/ChatMock?color=2b9348" alt="License Badge"/></a>
</div>

<p align="center"><b>Unlock the power of GPT-5, GPT-5-Codex, and other OpenAI models directly through your ChatGPT Plus/Pro account using a local API!</b></p>

## Key Features

*   **Access GPT-5 & More:** Use GPT-5, GPT-5-Codex, and other OpenAI models with your ChatGPT subscription.
*   **No API Key Required:** Leverage your existing ChatGPT Plus/Pro account for model access.
*   **OpenAI & Ollama Compatibility:** Compatible with the OpenAI API format, allowing seamless integration with existing tools and applications.
*   **Local Server:** Runs a local server, providing an endpoint compatible with OpenAI API.
*   **Tool/Function Calling Support:** Supports advanced features like web search.
*   **Customizable Reasoning:** Control the "thinking effort" and summaries for tailored results.
*   **Easy Integration:** Python and curl examples provided for quick setup and testing.
*   **Docker Support:**  Easy deployment with Docker.

## Getting Started

ChatMock allows you to access OpenAI models (including GPT-5) without needing an API key, using your paid ChatGPT account.

### Prerequisites
* A paid ChatGPT account (Plus/Pro)
* Python 3.x

### Installation and Usage

#### GUI Application (macOS)

1.  Download the GUI app from the [GitHub releases](https://github.com/RayBytes/ChatMock/releases).
2.  If you encounter issues opening the app, run the following command in your terminal:
    ```bash
    xattr -dr com.apple.quarantine /Applications/ChatMock.app
    ```
    *See more info [here](https://github.com/deskflow/deskflow/wiki/Running-on-macOS).*

#### Command Line (Homebrew)

1.  Tap the ChatMock formula:
    ```bash
    brew tap RayBytes/chatmock
    ```
2.  Install ChatMock:
    ```bash
    brew install chatmock
    ```

#### Python (Flask Server)

1.  Clone or download the repository:
    ```bash
    git clone https://github.com/RayBytes/ChatMock.git
    cd ChatMock
    ```
2.  Sign in with your ChatGPT account and follow the prompts:
    ```bash
    python chatmock.py login
    ```
    You can verify your login with: `python chatmock.py info`
3.  Start the local server:
    ```bash
    python chatmock.py serve
    ```
    The server defaults to `http://127.0.0.1:8000`.  Append `/v1/` to the end of your base URL when integrating with the OpenAI API.

#### Docker

1.  Follow the instructions in the [DOCKER.md](https://github.com/RayBytes/ChatMock/blob/main/DOCKER.md) file.

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

### Reasoning Effort

*   `--reasoning-effort` (minimal, low, medium, high): Controls how much "effort" GPT-5 puts into reasoning.  Default: `medium`.

### Thinking Summaries

*   `--reasoning-summary` (auto, concise, detailed, none): Customize the returned summaries.

### OpenAI Tools

*   `--enable-web-search`: Enables web search functionality.
    *   Use `responses_tools`: supports `[{"type":"web_search"}]` / `{ "type": "web_search_preview" }` and
        `responses_tool_choice`: `"auto"` or `"none"` in your API requests.

#### Web Search Example

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

*   `--expose-reasoning-models`:  Exposes each reasoning level as a separate model for easier selection in chat apps.

## Important Notes

*   Requires an active, paid ChatGPT account.
*   Consider context length limitations.
*   Use responsibly, at your own risk.  This project is not affiliated with OpenAI and is an educational endeavor.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=RayBytes/ChatMock&type=Timeline)](https://www.star-history.com/#RayBytes/ChatMock&Timeline)

<br>

**Explore the project further and contribute on [GitHub](https://github.com/RayBytes/ChatMock)!**
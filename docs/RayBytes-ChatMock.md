<div align="center">
  <h1>ChatMock: Access Your ChatGPT Plus/Pro Account as an OpenAI Compatible API</h1>
  <p><b>Unlock the power of your paid ChatGPT account to utilize OpenAI models from code or alternative chat interfaces without needing an API key.</b></p>
  <a href="https://github.com/RayBytes/ChatMock/stargazers"><img src="https://img.shields.io/github/stars/RayBytes/ChatMock" alt="Stars Badge"/></a>
  <a href="https://github.com/RayBytes/ChatMock/network/members"><img src="https://img.shields.io/github/forks/RayBytes/ChatMock" alt="Forks Badge"/></a>
  <a href="https://github.com/RayBytes/ChatMock/pulls"><img src="https://img.shields.io/github/issues-pr/RayBytes/ChatMock" alt="Pull Requests Badge"/></a>
  <a href="https://github.com/RayBytes/ChatMock/issues"><img src="https://img.shields.io/github/issues/RayBytes/ChatMock" alt="Issues Badge"/></a>
  <a href="https://github.com/RayBytes/ChatMock/graphs/contributors"><img alt="GitHub contributors" src="https://img.shields.io/github/contributors/RayBytes/ChatMock?color=2b9348"></a>
  <a href="https://github.com/RayBytes/ChatMock/blob/master/LICENSE"><img src="https://img.shields.io/github/license/RayBytes/ChatMock?color=2b9348" alt="License Badge"/></a>
  <br>
  <p><em><a href="https://github.com/RayBytes/ChatMock">View the original repository on GitHub</a></em></p>
</div>

---

## Key Features

*   **OpenAI and Ollama Compatibility:**  Use your ChatGPT Plus/Pro account with any tool or application designed for the OpenAI API or compatible with Ollama.
*   **No API Key Required:** Access models like GPT-5, GPT-5-Codex, and others without the need for an API key, utilizing your existing ChatGPT subscription.
*   **Multiple Installation Options:**  Supports GUI app for macOS, Homebrew, and Python Flask server for flexibility.
*   **Tool and Function Calling:** Supports advanced features like tool/function calling, vision/image understanding, and thinking summaries.
*   **Configurable Reasoning Effort:**  Fine-tune the model's "thinking effort" for potentially smarter answers (minimal, low, medium, high).
*   **Customizable Thinking Summaries:** Control how the model presents summaries (auto, concise, detailed, none).
*   **Web Search Integration:** Enable web search functionality for enhanced responses.
*   **Expose Reasoning Models:** Expose different reasoning levels as separate models for simpler selection in compatible apps.
---

## Getting Started

### Requirements

*   A paid ChatGPT account (Plus/Pro).
*   Python (for command-line setup).

### Installation

Choose your preferred installation method:

#### macOS GUI Application

1.  Download the GUI app from the [GitHub releases](https://github.com/RayBytes/ChatMock/releases).
2.  If needed, run this command in your terminal to open the app:
    ```bash
    xattr -dr com.apple.quarantine /Applications/ChatMock.app
    ```
    *   [More info here.](https://github.com/deskflow/deskflow/wiki/Running-on-macOS)

#### Command Line (Homebrew)

1.  Install Homebrew: [https://brew.sh/](https://brew.sh/)
2.  Install ChatMock:
    ```bash
    brew tap RayBytes/chatmock
    brew install chatmock
    ```

#### Python (Flask Server)

1.  Clone or download the repository.
2.  Navigate into the project directory.
3.  Sign in with your ChatGPT account:
    ```bash
    python chatmock.py login
    ```
    Verify with `python chatmock.py info`.
4.  Start the local server:
    ```bash
    python chatmock.py serve
    ```
    (By default, the server runs on `http://127.0.0.1:8000`)

    **Important:** When setting the `baseURL` in other applications, include `/v1/` at the end of the URL (e.g., `http://127.0.0.1:8000/v1/`) to ensure OpenAI API compatibility.

### Docker

See [the Docker instructions](https://github.com/RayBytes/ChatMock/blob/main/DOCKER.md) for containerization details.

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

---
## Customization and Configuration

### Reasoning Effort

*   Use the `--reasoning-effort` parameter with choices: `minimal`, `low`, `medium` (default), `high`.
    *   Example: `python chatmock.py serve --reasoning-effort high`

### Thinking Summaries

*   Use the `--reasoning-summary` parameter with choices: `auto`, `concise`, `detailed`, `none`.
    *   Example: `python chatmock.py serve --reasoning-summary concise`

### OpenAI Tools (Web Search)

*   Enable web search using `--enable-web-search` when starting the server.
*   You can also enable during request, using `responses_tools` and `responses_tool_choice` parameters in the API call.

    Example:
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

*   Use `--expose-reasoning-models` to create separate models for each reasoning level.

---
## Supported Models

*   `gpt-5`
*   `gpt-5-codex`
*   `codex-mini`

---

## Notes & Limits

*   Requires an active, paid ChatGPT account.
*   Context length may be slightly impacted by internal instructions.
*   Use responsibly and at your own risk.  This project is not affiliated with OpenAI and is for educational purposes.
*   For fastest response times, consider setting `--reasoning-effort minimal` and `--reasoning-summary none`.

---

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=RayBytes/ChatMock&type=Timeline)](https://www.star-history.com/#RayBytes/ChatMock&Timeline)
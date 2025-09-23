<div align="center">
  <h1>ChatMock: Unleash GPT-5 & More with Your ChatGPT Account</h1>
  <p><b>Access OpenAI models like GPT-5 and GPT-5-Codex through a local API, using your existing ChatGPT Plus/Pro subscription.</b></p>
  <br>
  <div align="center">
    <a href="https://github.com/RayBytes/ChatMock/stargazers"><img src="https://img.shields.io/github/stars/RayBytes/ChatMock" alt="Stars Badge"/></a>
    <a href="https://github.com/RayBytes/ChatMock/network/members"><img src="https://img.shields.io/github/forks/RayBytes/ChatMock" alt="Forks Badge"/></a>
    <a href="https://github.com/RayBytes/ChatMock/pulls"><img src="https://img.shields.io/github/issues-pr/RayBytes/ChatMock" alt="Pull Requests Badge"/></a>
    <a href="https://github.com/RayBytes/ChatMock/issues"><img src="https://img.shields.io/github/issues/RayBytes/ChatMock" alt="Issues Badge"/></a>
    <a href="https://github.com/RayBytes/ChatMock/graphs/contributors"><img alt="GitHub contributors" src="https://img.shields.io/github/contributors/RayBytes/ChatMock?color=2b9348"></a>
    <a href="https://github.com/RayBytes/ChatMock/blob/master/LICENSE"><img src="https://img.shields.io/github/license/RayBytes/ChatMock?color=2b9348" alt="License Badge"/></a>
  </div>
</div>

## Key Features

*   **GPT-5 Access:** Utilize the powerful GPT-5 and GPT-5-Codex models through your existing ChatGPT Plus/Pro subscription.
*   **Local API Server:**  Runs a local server mimicking the OpenAI API, making it compatible with various chat apps and coding tools.
*   **No API Key Required:** Authenticates with your ChatGPT login, eliminating the need for a separate OpenAI API key.
*   **Tool/Function Calling:** Supports advanced features like tool/function calling.
*   **Web Search Integration:** Enables web search functionality for enhanced responses.
*   **Customizable Reasoning:** Fine-tune the "thinking effort" and summaries for optimal results.
*   **Multiple Supported Models:** Includes support for `gpt-5`, `gpt-5-codex`, and `codex-mini`.

## Getting Started

ChatMock allows you to use your ChatGPT Plus/Pro subscription to access advanced models via a local API. This means you can use these models in other chat applications or coding tools.

### Installation

Choose your preferred method:

#### macOS GUI Application

1.  Download the GUI app from the [GitHub releases](https://github.com/RayBytes/ChatMock/releases).
2.  If you encounter issues, run this command in your terminal:
    ```bash
    xattr -dr com.apple.quarantine /Applications/ChatMock.app
    ```
    *   [More info.](https://github.com/deskflow/deskflow/wiki/Running-on-macOS)

#### Command Line (Homebrew)

1.  Install Homebrew: [https://brew.sh/](https://brew.sh/)
2.  Tap and install ChatMock:
    ```bash
    brew tap RayBytes/chatmock
    brew install chatmock
    ```

#### Python

1.  Clone or download the repository.
2.  Navigate to the project directory: `cd ChatMock`
3.  Sign in with your ChatGPT account:
    ```bash
    python chatmock.py login
    ```
    Verify with: `python chatmock.py info`
4.  Start the local server:
    ```bash
    python chatmock.py serve
    ```
    The server defaults to `http://127.0.0.1:8000`. Include `/v1/` at the end when setting a baseURL.

### Docker

Detailed instructions for Docker can be found in [the DOCKER.md file](https://github.com/RayBytes/ChatMock/blob/main/DOCKER.md)

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

## Customization & Configuration

### Reasoning Effort

*   `--reasoning-effort` (minimal, low, medium, high): Adjust how much "thinking effort" GPT-5 applies. Default is `medium`.

### Thinking Summaries

*   `--reasoning-summary` (auto, concise, detailed, none):  Customize how thinking summaries are handled.

### OpenAI Tools

*   `--enable-web-search`: Enables web search.

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

*   `--expose-reasoning-models`: Exposes each reasoning level as a separate model.

## Important Notes

*   **Requires a Paid Account:** A paid ChatGPT Plus/Pro account is essential.
*   **Context Length:**  Some context length may be used by internal instructions.
*   **Responsibility:** Use responsibly. This project is not affiliated with OpenAI.
*   For the fastest responses, use `--reasoning-effort minimal` and `--reasoning-summary none`.
*   Explore all parameters using `python chatmock.py serve --h`.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=RayBytes/ChatMock&type=Timeline)](https://www.star-history.com/#RayBytes/ChatMock&Timeline)

[Back to Top](#ChatMock-Unleash-GPT-5-&-More-with-Your-ChatGPT-Account)
---
[Original Repository](https://github.com/RayBytes/ChatMock)
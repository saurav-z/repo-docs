<div align="center">
  <h1>ChatMock: Unleash the Power of Your ChatGPT Plan</h1>
  <p><b>Access OpenAI models like GPT-5 and GPT-5-Codex through your ChatGPT Plus/Pro account with a local, OpenAI-compatible API.</b></p>
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

*   **Access GPT-5 & Codex Models:** Utilize GPT-5, GPT-5-Codex, and other OpenAI models through your existing ChatGPT account.
*   **OpenAI-Compatible API:**  Seamlessly integrate with tools and applications that support the OpenAI API.
*   **No API Key Required:** Authenticate using your ChatGPT login, eliminating the need for an API key.
*   **Flexible Deployment:** Run as a Python Flask server, or use the macOS GUI app or Docker for easy setup.
*   **Tool/Function Calling Support:** Leverage advanced OpenAI features.
*   **Vision/Image Understanding:** Utilize vision capabilities with ease.
*   **Customizable Reasoning:** Control the "thinking effort" and summarization of the model.
*   **Web Search Integration:** Enable web search capabilities for enhanced responses.
*   **Enhanced Context Size:** Benefit from a larger context window than the standard ChatGPT app.

## What is ChatMock?

ChatMock creates a local server that acts as an OpenAI-compatible API gateway, allowing you to use the power of your ChatGPT Plus/Pro account with other chat applications, coding tools, and custom projects. This project uses your authenticated ChatGPT login to make requests to OpenAI models without an API key.  This allows you to experience the full capabilities of models like GPT-5.

## Getting Started

### Mac Users

#### GUI Application

Download the macOS GUI app from the [GitHub releases](https://github.com/RayBytes/ChatMock/releases).

>   **Note:** You may need to run the following command in your terminal to open the app:
>
>   ```bash
>   xattr -dr com.apple.quarantine /Applications/ChatMock.app
>   ```
>
>   *[More info here.](https://github.com/deskflow/deskflow/wiki/Running-on-macOS)*

#### Command Line (Homebrew)

Install ChatMock using [Homebrew](https://brew.sh/):

```bash
brew tap RayBytes/chatmock
brew install chatmock
```

### Python

1.  Clone or download the repository.
2.  Navigate to the project directory.
3.  Login to your ChatGPT account:

```bash
python chatmock.py login
```

Verify with `python chatmock.py info`.

4.  Start the server:

```bash
python chatmock.py serve
```

Access the API at `http://127.0.0.1:8000/v1` (default).

### Docker

Refer to the [DOCKER.md](https://github.com/RayBytes/ChatMock/blob/main/DOCKER.md) file for Docker instructions.

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

## Configuration Options

*   **`--reasoning-effort`**:  (minimal, low, medium, high) - Adjust the model's "thinking effort".
*   **`--reasoning-summary`**: (auto, concise, detailed, none) - Customize reasoning summaries.
*   **`--enable-web-search`**: Enable OpenAI's web search tools.
*   **`--expose-reasoning-models`**: Expose different reasoning levels as individual models.

Run `python chatmock.py serve --h` for a complete list of parameters.

## Important Notes

*   Requires an active, paid ChatGPT account.
*   This project is not affiliated with OpenAI and is for educational purposes.
*   Context size is larger than that of the standard ChatGPT app.
*   For fastest responses, consider setting `--reasoning-effort` to minimal and `--reasoning-summary` to none.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=RayBytes/ChatMock&type=Timeline)](https://www.star-history.com/#RayBytes/ChatMock&Timeline)

[Original Repository](https://github.com/RayBytes/ChatMock)
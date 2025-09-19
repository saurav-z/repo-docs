<div align="center">
  <h1>ChatMock: Use Your ChatGPT Plus/Pro Account with OpenAI APIs</h1>
  <p>Access the power of GPT-5 and other OpenAI models through your existing ChatGPT subscription, without the need for an API key.</p>

  <div align="center">
    <a href="https://github.com/RayBytes/ChatMock/stargazers"><img src="https://img.shields.io/github/stars/RayBytes/ChatMock" alt="Stars Badge"/></a>
    <a href="https://github.com/RayBytes/ChatMock/network/members"><img src="https://img.shields.io/github/forks/RayBytes/ChatMock" alt="Forks Badge"/></a>
    <a href="https://github.com/RayBytes/ChatMock/pulls"><img src="https://img.shields.io/github/issues-pr/RayBytes/ChatMock" alt="Pull Requests Badge"/></a>
    <a href="https://github.com/RayBytes/ChatMock/issues"><img src="https://img.shields.io/github/issues/RayBytes/ChatMock" alt="Issues Badge"/></a>
    <a href="https://github.com/RayBytes/ChatMock/graphs/contributors"><img alt="GitHub contributors" src="https://img.shields.io/github/contributors/RayBytes/ChatMock?color=2b9348"></a>
    <a href="https://github.com/RayBytes/ChatMock/blob/master/LICENSE"><img src="https://img.shields.io/github/license/RayBytes/ChatMock?color=2b9348" alt="License Badge"/></a>
  </div>
  <br>
</div>

## Key Features

*   **ChatGPT Account Integration:** Leverage your existing ChatGPT Plus/Pro subscription.
*   **OpenAI API Compatibility:** Interact with models using standard OpenAI API calls.
*   **Model Support:** Access `gpt-5`, `gpt-5-codex`, and `codex-mini` models.
*   **Tool & Function Calling Support:** Utilize advanced OpenAI features.
*   **Vision/Image Understanding:** Enables image-based input.
*   **Thinking Summaries:** Enhanced model output with summaries and configurable effort.
*   **Web Search Integration:**  Utilize web search tools for more informed responses.
*   **Customizable Reasoning:** Fine-tune model behavior with effort and summary options.

## Getting Started

ChatMock provides multiple ways to get up and running:

### macOS GUI Application

For macOS users, a convenient GUI app is available:

1.  Download the GUI app from the [GitHub releases](https://github.com/RayBytes/ChatMock/releases).
2.  **Note:** Due to potential security restrictions, you may need to run the following command in your terminal to open the app:
    ```bash
    xattr -dr com.apple.quarantine /Applications/ChatMock.app
    ```
    For more information, see the [deskflow/deskflow/wiki/Running-on-macOS](https://github.com/deskflow/deskflow/wiki/Running-on-macOS) page.

### Command Line (Homebrew)

Alternatively, install ChatMock via Homebrew:

```bash
brew tap RayBytes/chatmock
brew install chatmock
```

### Python (Flask Server)

For flexibility, you can run ChatMock as a Python Flask server:

1.  Clone or download this repository.
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
    The default base URL is `http://127.0.0.1:8000`. Remember to include `/v1/` in your base URL for OpenAI compatibility (e.g., `http://127.0.0.1:8000/v1`).

### Docker

For containerized deployments, consult the [Docker instructions](https://github.com/RayBytes/ChatMock/blob/main/DOCKER.md).

## Example Usage

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

## Configuration Options

Customize ChatMock's behavior using these parameters:

### Reasoning Effort

*   `--reasoning-effort` (choices: `minimal`, `low`, `medium`, `high`): Adjust the model's "thinking effort" for potentially smarter responses. The default is `medium`.

### Reasoning Summaries

*   `--reasoning-summary` (choices: `auto`, `concise`, `detailed`, `none`): Control the format of thinking summaries.

### OpenAI Tools (Web Search)

*   `--enable-web-search`: Enables web search functionality.
    *   Use `responses_tools`: `[{"type":"web_search"}]` / `{ "type": "web_search_preview" }` in your API requests.
    *   Use `responses_tool_choice`: `"auto"` or `"none"` in your API requests.

#### Example: Web Search
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

*   `--expose-reasoning-models`: Exposes each reasoning level as a separate, queryable model, listing them under `/v1/models`.

## Important Notes

*   Requires a paid ChatGPT account.
*   Some context length is used by internal instructions.
*   This project is not affiliated with OpenAI and is an educational exercise.
*   For faster responses, consider `--reasoning-effort minimal` and `--reasoning-summary none`.
*   All available parameters are listed when using `python chatmock.py serve --h`.
*   The context size is larger than in the regular ChatGPT app.
*   Use `--reasoning-compat legacy` to set reasoning in the reasoning tag instead of the response text.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=RayBytes/ChatMock&type=Timeline)](https://www.star-history.com/#RayBytes/ChatMock&Timeline)

[Back to the original repository](https://github.com/RayBytes/ChatMock)
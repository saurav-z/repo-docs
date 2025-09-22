<div align="center">
  <h1>ChatMock</h1>
  <div align="center">
    <a href="https://github.com/RayBytes/ChatMock/stargazers"><img src="https://img.shields.io/github/stars/RayBytes/ChatMock" alt="Stars Badge"/></a>
    <a href="https://github.com/RayBytes/ChatMock/network/members"><img src="https://img.shields.io/github/forks/RayBytes/ChatMock" alt="Forks Badge"/></a>
    <a href="https://github.com/RayBytes/ChatMock/pulls"><img src="https://img.shields.io/github/issues-pr/RayBytes/ChatMock" alt="Pull Requests Badge"/></a>
    <a href="https://github.com/RayBytes/ChatMock/issues"><img src="https://img.shields.io/github/issues/RayBytes/ChatMock" alt="Issues Badge"/></a>
    <a href="https://github.com/RayBytes/ChatMock/graphs/contributors"><img alt="GitHub contributors" src="https://img.shields.io/github/contributors/RayBytes/ChatMock?color=2b9348"></a>
    <a href="https://github.com/RayBytes/ChatMock/blob/master/LICENSE"><img src="https://img.shields.io/github/license/RayBytes/ChatMock?color=2b9348" alt="License Badge"/></a>
  </div>
</div>

**Unlock the power of GPT-5 and other advanced OpenAI models with your existing ChatGPT Plus/Pro account using ChatMock!**  ([See the original repo](https://github.com/RayBytes/ChatMock))

## Key Features

*   **OpenAI & Ollama Compatibility:** Use your ChatGPT Plus/Pro account to interact with OpenAI models through an API, compatible with OpenAI and Ollama interfaces.
*   **No API Key Required:** Leverage your authenticated ChatGPT login for access, simplifying setup and usage.
*   **Model Support:** Access GPT-5, GPT-5-Codex, and codex-mini models.
*   **Tool/Function Calling Support:**  Enhanced functionality includes tool and function calls to extend model capabilities.
*   **Vision/Image Understanding:** ChatMock can handle models that support vision/image understanding.
*   **Thinking Summaries & Configuration:**  Gain insights with thinking summaries, including customisable "reasoning effort".
*   **Web Search Integration:** Use web search tools directly via the API.
*   **Customization Options:** Fine-tune responses with options like reasoning effort, summaries, and more.
*   **Easy Integration:**  Integrate with various chat apps and coding tools via a local server.

## Getting Started

### Installation

#### macOS

*   **GUI Application:** Download the GUI app from the [GitHub releases](https://github.com/RayBytes/ChatMock/releases).  
    *   **Note:**  If you encounter issues, try running `xattr -dr com.apple.quarantine /Applications/ChatMock.app` in your terminal.
*   **Command Line (Homebrew):**
    ```bash
    brew tap RayBytes/chatmock
    brew install chatmock
    ```

#### Python

1.  Clone or download the repository.
2.  `cd` into the project directory.
3.  Sign in with your ChatGPT account:
    ```bash
    python chatmock.py login
    ```
    Verify with `python chatmock.py info`.
4.  Start the local server:
    ```bash
    python chatmock.py serve
    ```
    Use the address and port (default: `http://127.0.0.1:8000`) as your `baseURL`.  Remember to add `/v1/` to the end of the URL when using it as an OpenAI compatible endpoint.

#### Docker

Refer to the [DOCKER.md](https://github.com/RayBytes/ChatMock/blob/main/DOCKER.md) file for detailed instructions.

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

## Customization & Configuration

*   **Thinking Effort:**  Control the model's reasoning effort (`minimal`, `low`, `medium`, `high`) with the `--reasoning-effort` parameter. Default: `medium`.
*   **Thinking Summaries:** Customize summary levels with `--reasoning-summary` (`auto`, `concise`, `detailed`, `none`).
*   **OpenAI Tools:** Enable web search with `--enable-web-search` or via the API:
    *   `responses_tools`: `[{"type":"web_search"}]` / `{ "type": "web_search_preview" }`
    *   `responses_tool_choice`: `"auto"` or `"none"`

    ```json
    {
      "model": "gpt-5",
      "messages": [{"role":"user","content":"Find current METAR rules"}],
      "stream": true,
      "responses_tools": [{"type": "web_search"}],
      "responses_tool_choice": "auto"
    }
    ```
*   **Expose Reasoning Models:** Expose each reasoning level as separate models using `--expose-reasoning-models` for easier use in apps.

## Notes & Considerations

*   Requires a paid ChatGPT account.
*   Context length considerations might apply.
*   Use responsibly. This project is not affiliated with OpenAI and is an educational exercise.
*   For fastest responses, consider `--reasoning-effort minimal` and `--reasoning-summary none`.
*   All parameters and options can be viewed with `python chatmock.py serve --h`.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=RayBytes/ChatMock&type=Timeline)](https://www.star-history.com/#RayBytes/ChatMock&Timeline)
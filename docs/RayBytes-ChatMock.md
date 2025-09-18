# ChatMock: Use Your ChatGPT Account with OpenAI-Compatible APIs

**Unlock the power of your ChatGPT Plus/Pro account and access advanced OpenAI models without an API key using ChatMock.**  ([View the original repository](https://github.com/RayBytes/ChatMock))

## Key Features

*   **OpenAI API Compatibility:** Interact with your ChatGPT account through a standard OpenAI API, allowing seamless integration with existing tools and applications.
*   **GPT-5 and More:** Access GPT-5, GPT-5-Codex, and other models directly through your ChatGPT subscription.
*   **No API Key Required:** Utilize your existing ChatGPT login, eliminating the need for a separate API key.
*   **GUI and Command-Line Options:** Choose your preferred method of interaction with a GUI application for macOS, or command-line tools using Homebrew or Python.
*   **Docker Support:** Easily deploy and run ChatMock with Docker for flexible and scalable usage.
*   **Tool/Function Calling & Vision Support:** Utilizes OpenAI Tools for web search capabilities.
*   **Customizable Reasoning:** Fine-tune the model's "thinking effort" and summarization with options for `minimal, low, medium, high` and `auto, concise, detailed, none` respectively.
*   **Enhanced Context Size:** Benefit from a larger context window compared to the standard ChatGPT app.
*   **Expose Reasoning Models:** Optionally expose each reasoning level as a separate, queryable model.

## Getting Started

Choose your preferred installation method:

### macOS

#### GUI Application

Download the GUI app from the [GitHub releases](https://github.com/RayBytes/ChatMock/releases).
> **Note:**  If the app doesn't open, you may need to run the following command in your terminal:
>
> ```bash
> xattr -dr com.apple.quarantine /Applications/ChatMock.app
> ```
>
> *[More info here.](https://github.com/deskflow/deskflow/wiki/Running-on-macOS)*

#### Command Line (Homebrew)

Install ChatMock using [Homebrew](https://brew.sh/):
```bash
brew tap RayBytes/chatmock
brew install chatmock
```

### Python

1.  **Clone or download** the repository.
2.  **Navigate** to the project directory.
3.  **Login to your ChatGPT account**:
    ```bash
    python chatmock.py login
    ```
    Verify successful login with `python chatmock.py info`.
4.  **Start the local server**:
    ```bash
    python chatmock.py serve
    ```
    Use the provided address and port (default: `http://127.0.0.1:8000`) as your `baseURL`, including `/v1/` for OpenAI compatibility.

### Docker

Refer to the [DOCKER.md](https://github.com/RayBytes/ChatMock/blob/main/DOCKER.md) for Docker instructions.

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

## Customization / Configuration

Configure the server's behavior using command-line arguments:

### Thinking Effort

*   `--reasoning-effort` (choices: `minimal`, `low`, `medium`, `high`): Adjust the model's thinking intensity. Defaults to `medium`.

### Thinking Summaries

*   `--reasoning-summary` (choices: `auto`, `concise`, `detailed`, `none`): Customize the format of the model's reasoning summaries.

### OpenAI Tools

*   `--enable-web-search`: Enable web search functionality. Use the following in your API requests:

    *   `responses_tools`: `[{"type":"web_search"}]` / `{ "type": "web_search_preview" }`
    *   `responses_tool_choice`: `"auto"` or `"none"`

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

*   `--expose-reasoning-models`: Allows you to select the reasoning effort.

## Notes

*   **Requires a paid ChatGPT account.**
*   Context length may be slightly reduced by internal instructions.
*   Use responsibly. This project is not affiliated with OpenAI and is an educational exercise.
*   For fastest responses, consider setting `--reasoning-effort` to `minimal` and `--reasoning-summary` to `none`.
*   See `python chatmock.py serve --h` for all parameters.
*   Consider setting `--reasoning-compat` to `legacy` for legacy reasoning compatibility.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=RayBytes/ChatMock&type=Timeline)](https://www.star-history.com/#RayBytes/ChatMock&Timeline)
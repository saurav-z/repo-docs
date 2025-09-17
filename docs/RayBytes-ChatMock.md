# ChatMock: Access ChatGPT Plus/Pro Models Without API Keys

**Unlock the power of your ChatGPT Plus/Pro subscription by using models like GPT-5 and others through an OpenAI-compatible API with ChatMock.** ([Original Repo](https://github.com/RayBytes/ChatMock))

## Key Features

*   **Bypass API Keys:** Utilize your existing ChatGPT Plus/Pro account, eliminating the need for an API key.
*   **OpenAI Compatibility:**  Interact with models using the familiar OpenAI API interface.
*   **GPT-5 and More:** Access advanced models like GPT-5, GPT-5-Codex, and codex-mini.
*   **GUI and CLI Options:** Choose your preferred method with a user-friendly GUI app for macOS and a command-line tool for all platforms.
*   **Flexible Deployment:** Run ChatMock as a Python Flask server or deploy it with Docker.
*   **Tool/Function Calling Support:** Use tools and functions with your ChatGPT models.
*   **Vision/Image Understanding:** Use image understanding
*   **Thinking Summaries and Effort Control:** Configure the level of reasoning effort and the style of thinking summaries.
*   **Web Search Integration:** Enable web search within your chat sessions.
*   **Larger Context Size:** Benefit from an increased context window compared to the standard ChatGPT app.

## Getting Started

### macOS Users

*   **GUI Application:** Download the macOS GUI app from the [GitHub releases](https://github.com/RayBytes/ChatMock/releases).  
    > **Note:** Since ChatMock isn't signed with an Apple Developer ID, you may need to run the following command in your terminal to open the app:
    >
    > ```bash
    > xattr -dr com.apple.quarantine /Applications/ChatMock.app
    > ```
    >
    > *[More info here.](https://github.com/deskflow/deskflow/wiki/Running-on-macOS)*
*   **Command Line (Homebrew):**
    ```bash
    brew tap RayBytes/chatmock
    brew install chatmock
    ```

### Python

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/RayBytes/ChatMock.git
    cd ChatMock
    ```

2.  **Login:**
    ```bash
    python chatmock.py login
    ```
    Verify login: `python chatmock.py info`

3.  **Start the server:**
    ```bash
    python chatmock.py serve
    ```
    Use `http://127.0.0.1:8000/v1` as your base URL in other applications.

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

## Configuration and Customization

### Thinking Effort

*   `--reasoning-effort`:  Configure the model's "thinking effort" with options like `minimal`, `low`, `medium` (default), and `high`.

### Thinking Summaries

*   `--reasoning-summary`: Choose how thinking summaries are returned:  `auto`, `concise`, `detailed`, or `none`.

### OpenAI Tools (Web Search)

*   `--enable-web-search`: Enable web search functionality.
*   API request parameters:
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

### Expose Reasoning Models

*   `--expose-reasoning-models`: Expose separate models for each reasoning level (e.g., `gpt-5-minimal`, `gpt-5-low`), enabling finer control in chat apps that support model selection.

## Notes & Considerations

*   Requires a paid ChatGPT account.
*   Context length is larger than the standard ChatGPT app.
*   Use responsibly and at your own risk.
*   For the fastest responses, set `--reasoning-effort` to minimal and `--reasoning-summary` to none.
*   All parameters and choices can be seen with `python chatmock.py serve --h`.

## Supported Models

*   `gpt-5`
*   `gpt-5-codex`
*   `codex-mini`

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=RayBytes/ChatMock&type=Timeline)](https://www.star-history.com/#RayBytes/ChatMock&Timeline)
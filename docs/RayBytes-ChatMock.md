# ChatMock: Access GPT-5 and More Through Your ChatGPT Account

**Unlock the power of GPT-5 and other OpenAI models using your existing ChatGPT Plus/Pro subscription without the need for API keys.**  [View the original repository](https://github.com/RayBytes/ChatMock).

## Key Features

*   **GPT-5 and Model Access:** Utilize GPT-5 and other supported OpenAI models directly through your ChatGPT account.
*   **OpenAI & Ollama Compatibility:** Emulates the OpenAI API, allowing you to use your account with existing OpenAI integrations and also provides compatibility with Ollama.
*   **No API Key Required:** Authenticate with your ChatGPT credentials instead of needing an API key.
*   **Tool Calling & Vision Support:** Leverages advanced functionalities like tool calling, and vision/image understanding.
*   **Customizable Reasoning:** Control the model's "thinking effort" and the level of thinking summaries for optimized performance.
*   **Local Server:** Runs a local server, making it easy to integrate with various applications and tools.

## Getting Started

### macOS

**GUI Application:**

*   Download the GUI app from the [GitHub releases](https://github.com/RayBytes/ChatMock/releases).
    *   **Note:** Due to security measures, you may need to run the following command in your terminal to open the app:
        ```bash
        xattr -dr com.apple.quarantine /Applications/ChatMock.app
        ```
        *[More info here.](https://github.com/deskflow/deskflow/wiki/Running-on-macOS)*

**Command Line (Homebrew):**

*   Install ChatMock using Homebrew:
    ```bash
    brew tap RayBytes/chatmock
    brew install chatmock
    ```

### Python

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/RayBytes/ChatMock.git
    cd ChatMock
    ```

2.  **Login:** Sign in with your ChatGPT account.
    ```bash
    python chatmock.py login
    ```
    Verify the login with:
    ```bash
    python chatmock.py info
    ```

3.  **Start the Server:** Launch the local server.
    ```bash
    python chatmock.py serve
    ```

4.  **Use the API:**  Use the server address and port (http://127.0.0.1:8000 by default) as the `baseURL` in your OpenAI integrations, adding `/v1/` for OpenAI compatibility.

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
*   `codex-mini`

## Customization & Configuration

*   **Reasoning Effort:** Control the model's thinking effort for potentially smarter answers:  `--reasoning-effort` (minimal, low, medium, high). Defaults to `medium`.
*   **Thinking Summaries:** Customize the level of thinking summaries: `--reasoning-summary` (auto, concise, detailed, none).

    For the fastest responses, consider `--reasoning-effort low` and `--reasoning-summary none`.

    View all parameters and options with `python chatmock.py serve --h`.

*   **Reasoning Compatibility:** `--reasoning-compat legacy` will put reasoning in the `reasoning` tag instead of in the response text.

## Important Notes

*   Requires an active, paid ChatGPT account.
*   Expect potentially lower rate limits compared to the official ChatGPT app.
*   The context size is larger than the regular ChatGPT app.
*   Use responsibly and at your own risk. This project is not affiliated with OpenAI and is for educational purposes.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=RayBytes/ChatMock&type=Timeline)](https://www.star-history.com/#RayBytes/ChatMock&Timeline)
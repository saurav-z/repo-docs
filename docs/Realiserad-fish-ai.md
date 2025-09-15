![fish-ai Badge](https://img.shields.io/badge/fish--ai-AI%20Powered%20Shell-blue?style=for-the-badge&logo=fishshell)

# Fish-AI: Supercharge Your Fish Shell with AI 🚀

**Tired of endless `man` pages and Stack Overflow searches?** Fish-AI brings the power of AI directly to your Fish shell, making command line interactions faster and more intuitive.  [Check out the GitHub Repository](https://github.com/Realiserad/fish-ai) for more details!

## Key Features:

*   **Comment to Command & Vice Versa:**  Effortlessly convert natural language comments into shell commands and vice versa, saving you time and effort.
*   **Intelligent Command Completion:**  Get smart, context-aware command suggestions with a built-in fuzzy finder, speeding up your workflow.
*   **Smart Command Correction:**  Automatically fix typos and common command errors, keeping your shell sessions smooth.
*   **Customizable & Flexible:**  Works with your preferred LLM (including self-hosted options) and offers extensive customization through keybindings, language settings, and more.
*   **Open Source & Auditable:**  Benefit from an open-source project (around 2000 lines of code) that allows you to inspect the code and understand how it works.
*   **Seamless Integration:**  Doesn't interfere with your existing Fish plugins like `fzf.fish` or `tide` and avoids unnecessary terminal wrapping or telemetry.

## 🎥 Demo

![Demo](https://github.com/user-attachments/assets/86b61223-e568-4152-9e5e-d572b2b1385b)

## Installation

### Prerequisites

Ensure you have `git` and either [`uv`](https://github.com/astral-sh/uv) or a supported version of Python (with `pip` and `venv`) installed.

### Installation with Fisher

Use [`fisher`](https://github.com/jorgebucaran/fisher) to install `fish-ai`:

```shell
fisher install realiserad/fish-ai
```

### Configuration

Create a configuration file (`$XDG_CONFIG_HOME/fish-ai.ini` or `~/.config/fish-ai.ini`) to specify your preferred LLM.  Example configurations are provided below for different LLM providers:

*   **GitHub Models**: Configure by creating a Personal Access Token (PAT) [here](https://github.com/settings/tokens)
    ```ini
    [fish-ai]
    configuration = github

    [github]
    provider = self-hosted
    server = https://models.inference.ai.azure.com
    api_key = <paste GitHub PAT here>
    model = gpt-4o-mini
    ```
*   **Self-Hosted LLM (OpenAI-compatible)**:
    ```ini
    [fish-ai]
    configuration = self-hosted

    [self-hosted]
    provider = self-hosted
    server = https://<your server>:<port>/v1
    model = <your model>
    api_key = <your API key>
    ```
*   **OpenRouter**:
    ```ini
    [fish-ai]
    configuration = openrouter

    [openrouter]
    provider = self-hosted
    server = https://openrouter.ai/api/v1
    model = google/gemini-2.0-flash-lite-001
    api_key = <your API key>
    ```

*   **OpenAI**:
    ```ini
    [fish-ai]
    configuration = openai

    [openai]
    provider = openai
    model = gpt-4o
    api_key = <your API key>
    organization = <your organization>
    ```
*   **Azure OpenAI**:
    ```ini
    [fish-ai]
    configuration = azure

    [azure]
    provider = azure
    server = https://<your instance>.openai.azure.com
    model = <your deployment name>
    api_key = <your API key>
    ```

*   **Mistral**, **Anthropic**, **Cohere**, **DeepSeek**, **Groq**, **Google** configurations are also supported.  See the original README for details.

### API Key Management

Use `fish_ai_put_api_key` to securely store your API keys in your keyring, improving security.

## 🙉 How to Use

*   **Comment to Command:** Type a comment (starting with `#`) and press **Ctrl + P** to generate a shell command.
*   **Command to Comment:**  Type a command and press **Ctrl + P** to get an explanation in a comment.
*   **Autocomplete:** Start typing a command/comment and press **Ctrl + Space** for intelligent completion suggestions via `fzf`.
*   **Suggest Fixes:** If a command fails, press **Ctrl + Space** at the prompt to receive suggested fixes.

## 🤸 Advanced Options

Customize `fish-ai` further within your `fish-ai.ini` file:

*   **Keybindings:** Change default keybindings (**Ctrl + P** and **Ctrl + Space**).
*   **Language:**  Specify a language for command explanations (e.g., `language = Swedish`).
*   **Temperature:** Control the LLM's creativity (default `0.2`).
*   **Completions:** Adjust the number of completion suggestions.
*   **Command History:** Personalize suggestions using your command line history.
*   **Preview Pipes:** Send the output of pipes to the LLM for better context.
*   **Progress Indicator:** Customize the visual indicator during LLM processing.

## 🐾 Data Privacy

`fish-ai` sends the OS name and commandline buffer to the LLM. It also may send the contents of files you mention, the output of `--help` and command history (optional).  Consider using a self-hosted LLM if you have data privacy concerns. Sensitive data is redacted from prompts.

## 🔨 Development

[See `ARCHITECTURE.md`](https://github.com/Realiserad/fish-ai/blob/main/ARCHITECTURE.md) for project architecture.  The project uses `devcontainer.json` for easy development with GitHub Codespaces or VS Code.  Contributions are welcome.

```shell
# Install from a local copy
fisher install .
```

## 🌟  Contribute & Support

Bug fixes are welcome! Feel free to open an issue to discuss feature requests first. Please add a ⭐ if you like this project!
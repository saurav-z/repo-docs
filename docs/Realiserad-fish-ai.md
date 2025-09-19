[![Time Spent](https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2FRealiserad%2Fd3ec7fdeecc35aeeb315b4efba493326%2Fraw%2Ffish-ai-git-estimate.json)](https://github.com/Realiserad/fish-ai)
[![Popularity](https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2FRealiserad%2Fd3ec7fdeecc35aeeb315b4efba493326%2Fraw%2Fpopularity.json)](https://github.com/Realiserad/fish-ai)
[![Donate XMR](https://img.shields.io/badge/Donate_XMR-grey?style=for-the-badge&logo=monero)](https://github.com/user-attachments/assets/07a2947f-6e5a-480f-990a-77204933411f)

# fish-ai: Supercharge Your Fish Shell with AI-Powered Commands

**Tired of wrestling with complex commands? fish-ai leverages the power of AI to simplify your command-line experience.** [Check out the original repo](https://github.com/Realiserad/fish-ai) for more details.

## Key Features

*   **Comment to Command & Vice Versa:** Transform comments into shell commands and commands into explanations, saving time and effort.
*   **Command Correction:** Automatically fix typos and errors in your commands, similar to `thefuck`.
*   **Intelligent Autocompletion:** Get context-aware command suggestions with built-in fuzzy finding.
*   **Keyboard-Driven:** Everything is accessible via customizable keyboard shortcuts, promoting efficiency.
*   **LLM Agnostic:** Integrate with your preferred LLM provider (OpenAI, self-hosted, etc.)
*   **Open Source & Auditable:** The code is readily available, making it transparent and easily customizable.
*   **Easy Installation & Updates:** Install and manage `fish-ai` using `fisher`.
*   **Cross-Platform Compatibility:** Works seamlessly on macOS and common Linux distributions.
*   **Non-Intrusive:** Doesn't interfere with other plugins or force you to use a specific terminal.
*   **Data Privacy Focused:** Offers options for redaction of sensitive information and self-hosting for maximum privacy.

## 🎥 Demo

![Demo](https://github.com/user-attachments/assets/86b61223-e568-4152-9e5e-d572b2b1385b)

## 🛠️ Installation

### Prerequisites

*   Ensure `git` is installed.
*   Install either [`uv`](https://github.com/astral-sh/uv) or a supported Python version along with `pip` and `venv`.

### Installation using Fisher

```shell
fisher install realiserad/fish-ai
```

### Configuration

1.  Create a configuration file at `$XDG_CONFIG_HOME/fish-ai.ini` (or `~/.config/fish-ai.ini` if `$XDG_CONFIG_HOME` is unset). This file specifies the LLM provider to use.

2.  Choose from the configuration options below:

    *   **GitHub Models:**

    ```ini
    [fish-ai]
    configuration = github

    [github]
    provider = self-hosted
    server = https://models.inference.ai.azure.com
    api_key = <your GitHub PAT>
    model = gpt-4o-mini
    ```

    *   **Self-Hosted (e.g., Ollama):**

    ```ini
    [fish-ai]
    configuration = self-hosted

    [self-hosted]
    provider = self-hosted
    server = https://<your server>:<port>/v1
    model = <your model>
    api_key = <your API key>
    ```
    *   **OpenRouter:**

    ```ini
    [fish-ai]
    configuration = openrouter

    [openrouter]
    provider = self-hosted
    server = https://openrouter.ai/api/v1
    model = google/gemini-2.0-flash-lite-001
    api_key = <your API key>
    ```

    *   **OpenAI:**

    ```ini
    [fish-ai]
    configuration = openai

    [openai]
    provider = openai
    model = gpt-4o
    api_key = <your API key>
    organization = <your organization>
    ```

    *   **Azure OpenAI:**

    ```ini
    [fish-ai]
    configuration = azure

    [azure]
    provider = azure
    server = https://<your instance>.openai.azure.com
    model = <your deployment name>
    api_key = <your API key>
    ```
    *   **Mistral:**

    ```ini
    [fish-ai]
    configuration = mistral

    [mistral]
    provider = mistral
    api_key = <your API key>
    ```
    *   **Anthropic:**

    ```ini
    [anthropic]
    provider = anthropic
    api_key = <your API key>
    ```

    *   **Cohere:**

    ```ini
    [cohere]
    provider = cohere
    api_key = <your API key>
    ```

    *   **DeepSeek:**

    ```ini
    [deepseek]
    provider = deepseek
    api_key = <your API key>
    model = deepseek-chat
    ```

    *   **Groq:**

    ```ini
    [groq]
    provider = groq
    api_key = <your API key>
    ```

    *   **Google Gemini:**

    ```ini
    [google]
    provider = google
    api_key = <your API key>
    ```

3.  Store API keys securely using `fish_ai_put_api_key`.

## ⌨️ Usage

### Transform Comments to Commands & Vice Versa

Type a comment (starting with `#`), then press **Ctrl + P** to convert it to a shell command. You can also reverse this process by typing a command and pressing **Ctrl + P** to get a comment explanation.

### Command Autocompletion

Start typing a command or comment and press **Ctrl + Space** to trigger AI-powered completions using `fzf`.

### Suggest Command Fixes

If a command fails, press **Ctrl + Space** at the prompt to receive suggested fixes.

## ⚙️ Customization Options

Modify the behavior of `fish-ai` by adding options to your `fish-ai.ini` configuration file:

*   **Change Key Bindings:** Customize the **Ctrl + P** and **Ctrl + Space** keybindings using `keymap_1` and `keymap_2`.
*   **Language:**  Set the `language` option to change the explanation language (e.g., `language = Swedish`).
*   **Temperature:** Adjust the randomness of output with the `temperature` option (e.g., `temperature = 0.5`).
*   **Completions:** Control the number of completions using the `completions` (default `5`) and `refined_completions` options.
*   **History:** Enable personalizing completions using command-line history with the `history_size` option.
*   **Preview Pipes:** Use the `preview_pipe = True` option to send pipe output to the LLM for completion.
*   **Progress Indicator:** Change the progress indicator with the `progress_indicator` option.

## 🛡️ Data Privacy & Security

*   `fish-ai` transmits OS information and the command-line buffer to the LLM.
*   It also sends file contents (if readable) when codifying or completing commands.
*   The contents of the `--help` flag will be provided to the LLM for explanation.
*   Command-line history can be sent based on the `history_size` setting.
*   Previous command and output are sent to correct the last command, along with the corresponding exit code.
*   Sensitive information is redacted (passwords, API keys, private keys, bearer tokens). Disable redaction with `redact = False` if you trust the LLM provider (e.g. self-hosting)

Use a self-hosted LLM for maximum data privacy.

## 👨‍💻 Development

For contributing, read [`ARCHITECTURE.md`](https://github.com/Realiserad/fish-ai/blob/main/ARCHITECTURE.md). Use the provided `devcontainer.json` for development.

*   Install from local copy using `fisher install .`.
*   Enable debug logging with `debug = True` in `fish-ai.ini`.
*   Run tests with `pytest`.
*   Create a release by pushing a new tag.
```shell
set tag (grep '^version =' pyproject.toml | \
    cut -d '=' -f2- | \
    string replace -ra '[ "]' '')
git tag -a "v$tag" -m "🚀 v$tag"
git push origin "v$tag"
```
[![Fish AI - AI-Powered Shell Plugin](https://img.shields.io/badge/Fish%20AI-AI--Powered%20Shell%20Plugin-blue)](https://github.com/Realiserad/fish-ai)
[![GitHub stars](https://img.shields.io/github/stars/Realiserad/fish-ai?style=social)](https://github.com/Realiserad/fish-ai)
![Badge with time spent](https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2FRealiserad%2Fd3ec7fdeecc35aeeb315b4efba493326%2Fraw%2Ffish-ai-git-estimate.json)
![Popularity badge](https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2FRealiserad%2Fd3ec7fdeecc35aeeb315b4efba493326%2Fraw%2Fpopularity.json)
[![Donate XMR](https://img.shields.io/badge/Donate_XMR-grey?style=for-the-badge&logo=monero)](https://github.com/user-attachments/assets/07a2947f-6e5a-480f-990a-77204933411f)

# Fish AI: Supercharge Your Fish Shell with AI

**Fish AI is a powerful plugin that brings the intelligence of Large Language Models (LLMs) to your Fish shell, making command-line tasks easier and faster.**

*   [View the original repo on GitHub](https://github.com/Realiserad/fish-ai)

## Key Features

*   **Comment to Command & Vice Versa:** Convert comments into shell commands and commands into explanations, eliminating the need to search man pages or Stack Overflow.
*   **Intelligent Autocompletion:** Get context-aware suggestions and generate entire commands with the help of a fuzzy finder.
*   **Command Correction:** Fix typos and errors in your commands, similar to `thefuck`.
*   **Keyboard Shortcut Driven:** Control everything with configurable keyboard shortcuts.
*   **Flexible LLM Integration:** Use your preferred LLM (GitHub Models, OpenAI, Azure, self-hosted, etc.).
*   **Open Source & Customizable:** Audit the code yourself, and tweak settings to your liking!
*   **Seamless Integration:** Works well with `fzf.fish`, `tide` and other plugins.
*   **No External Dependencies:** Does not interfere with your terminal setup.

## 🎥 Demo

![Demo](https://github.com/user-attachments/assets/86b61223-e568-4152-9e5e-d572b2b1385b)

## Installation

### Prerequisites

*   `git`
*   Either [`uv`](https://github.com/astral-sh/uv) or a supported Python version (with `pip` and `venv`)

### Installation Steps

1.  Install the plugin using `fisher`:

    ```shell
    fisher install realiserad/fish-ai
    ```

2.  **Configure Your LLM:** Create a configuration file, `$XDG_CONFIG_HOME/fish-ai.ini` (or `~/.config/fish-ai.ini` if `$XDG_CONFIG_HOME` is not set), to specify your LLM provider.  Examples are provided below to help you get started:

    *   **GitHub Models:**
    ```ini
    [fish-ai]
    configuration = github

    [github]
    provider = self-hosted
    server = https://models.inference.ai.azure.com
    api_key = <paste GitHub PAT here>
    model = gpt-4o-mini
    ```
    You can create a personal access token (PAT) [here](https://github.com/settings/tokens).
    The PAT does not require any permissions.

    *   **Self-hosted (OpenAI Compatible):**
    ```ini
    [fish-ai]
    configuration = self-hosted

    [self-hosted]
    provider = self-hosted
    server = https://<your server>:<port>/v1
    model = <your model>
    api_key = <your API key>
    ```
    Consider using [Ollama](https://github.com/ollama/ollama) with [Llama 3.3 70B](https://ollama.com/library/llama3.3).

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
    Available models are listed [here](https://openrouter.ai/models).

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
3.  **Store API Keys Securely:** Use `fish_ai_put_api_key` to store your API keys in your system's keyring.

## Usage

### Convert Comments to Commands / Commands to Comments

1.  Type a comment (starting with `#`) and press **Ctrl + P** to generate a command.
2.  Type a command and press **Ctrl + P** to get an explanation.

### Command Autocompletion

1.  Start typing your command or comment.
2.  Press **Ctrl + Space** to get suggestions via a fuzzy finder.

### Command Correction

1.  If a command fails, press **Ctrl + Space** immediately to get a suggested fix.

## Customization

Customize `fish-ai` by adding options to your `fish-ai.ini` file.

*   **Key Bindings:** Change default keybindings using `keymap_1` and `keymap_2`. Use `fish_key_reader` to get the correct key binding escape sequence.
*   **Language:** Set the `language` option to change the explanation language (e.g., `language = Swedish`).
*   **Temperature:** Adjust creativity with the `temperature` setting (e.g., `temperature = 0.5`).
*   **Completions:** Control the number of completion suggestions with `completions` and refined completions with `refined_completions`.
*   **Command History:** Personalize suggestions with command history using the `history_size` option.
*   **Preview Pipes:** Enable pipe output preview with `preview_pipe = True`.
*   **Progress Indicator:** Customize the progress indicator with the `progress_indicator` option.
*   **Context Switching:** Use the `fish_ai_switch_context` command to change your configurations.

## Data Privacy

*   `fish-ai` sends your OS name and command buffer to the LLM.
*   It can also send file contents, command help output, command history (if enabled), and the previous command with its output for correction.
*   Use a self-hosted LLM for maximum data privacy.
*   Sensitive information (passwords, API keys, etc.) is redacted, unless you disable redaction using the `redact = False` option.

## Development

If you want to contribute, read [`ARCHITECTURE.md`](https://github.com/Realiserad/fish-ai/blob/main/ARCHITECTURE.md) first.
This repository ships with a `devcontainer.json` which can be used with
GitHub Codespaces or Visual Studio Code with
[the Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers).

*   **Installation from Local Copy:** `fisher install .`
*   **Debug Logging:** Enable with `debug = True` in `fish-ai.ini`. Use `log = <path to file>` for file logging.
*   **Testing:**  Run installation tests (triggered by pushes and PRs) and Python tests with `pytest`.
*   **Releases:**  Created automatically when new tags are pushed.
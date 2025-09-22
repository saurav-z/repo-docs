![fish-ai Time Spent](https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2FRealiserad%2Fd3ec7fdeecc35aeeb315b4efba493326%2Fraw%2Ffish-ai-git-estimate.json)
![fish-ai Popularity](https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2FRealiserad%2Fd3ec7fdeecc35aeeb315b4efba493326%2Fraw%2Fpopularity.json)
[![Donate XMR](https://img.shields.io/badge/Donate_XMR-grey?style=for-the-badge&logo=monero)](https://github.com/user-attachments/assets/07a2947f-6e5a-480f-990a-77204933411f)

# fish-ai: Supercharge Your Fish Shell with AI-Powered Productivity 🐠🤖

**Tired of manual pages and endless Googling? fish-ai brings the power of AI directly into your Fish shell, making command-line tasks faster, easier, and more intuitive.** [Visit the fish-ai GitHub Repository](https://github.com/Realiserad/fish-ai)

## Key Features

*   **Comment to Command & Vice Versa:** Effortlessly convert descriptive comments into shell commands and clarify complex commands with human-readable explanations.
*   **Command Autocomplete:** Get intelligent command suggestions with a built-in fuzzy finder, saving you time and effort.
*   **Intelligent Error Correction:** Automatically fix common typos and command errors, similar to `thefuck`.
*   **Customizable Keybindings:** Control everything with two configurable keyboard shortcuts, eliminating the need for a mouse.
*   **Flexible LLM Integration:** Works with your preferred Large Language Model (LLM), including self-hosted options.
*   **Open Source & Auditable:** Review the code (approximately 2000 lines) to understand and customize the tool.
*   **Easy Installation & Updates:** Install and manage fish-ai effortlessly using `fisher`.
*   **Broad Compatibility:** Tested on macOS and common Linux distributions.
*   **Plugin Friendly:** Designed to integrate smoothly with existing Fish plugins like `fzf.fish` and `tide`.
*   **No System Interference:** Avoids shell wrapping, telemetry, or proprietary terminal requirements.

## How it Works - In Action!

![Demo](https://github.com/user-attachments/assets/86b61223-e568-4152-9e5e-d572b2b1385b)

## Installation

### Prerequisites

Ensure you have `git` installed, along with either [`uv`](https://github.com/astral-sh/uv) or a supported Python version (see [Python tests workflow](https://github.com/Realiserad/fish-ai/blob/main/.github/workflows/python-tests.yaml)), plus `pip` and `venv`.

### Install fish-ai with Fisher

```shell
fisher install realiserad/fish-ai
```

### Configuration

Create a configuration file at `$XDG_CONFIG_HOME/fish-ai.ini` (or `~/.config/fish-ai.ini` if `$XDG_CONFIG_HOME` is not set).  This file specifies which LLM to use.

#### Supported LLM Providers:

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
    Generate a personal access token (PAT) [here](https://github.com/settings/tokens) - no permissions needed.
*   **Self-Hosted (OpenAI Compatible):**
    ```ini
    [fish-ai]
    configuration = self-hosted

    [self-hosted]
    provider = self-hosted
    server = https://<your server>:<port>/v1
    model = <your model>
    api_key = <your API key>
    ```
    For self-hosting, [Ollama](https://github.com/ollama/ollama) with [Llama 3.3 70B](https://ollama.com/library/llama3.3) is recommended.
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
    See [available models](https://openrouter.ai/models).
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
*   **Google:**
    ```ini
    [google]
    provider = google
    api_key = <your API key>
    ```

### Store API Keys Securely

Use `fish_ai_put_api_key` to store and manage your API keys securely in your keyring.

## Usage

### Turn Comments into Commands & Vice Versa

1.  Type a comment (starting with `#`) and press **Ctrl + P** to generate a shell command.
2.  Type a command and press **Ctrl + P** to get a comment explaining it.

### Autocomplete Commands

1.  Start typing a command or comment.
2.  Press **Ctrl + Space** to trigger the autocomplete suggestions via [`fzf`](https://github.com/junegunn/fzf).

### Command Correction

If a command fails, press **Ctrl + Space** at the prompt to get a suggested fix.

## Advanced Configuration

Customize fish-ai to fit your workflow:

### Change Keybindings

Adjust default keybindings (**Ctrl + P** and **Ctrl + Space**) by modifying `keymap_1` and `keymap_2` with the desired key binding escape sequence. Use [`fish_key_reader`](https://fishshell.com/docs/current/cmds/fish_key_reader.html) to get the correct escape sequence.

### Language Preference

Set the `language` option in your configuration file to change the explanation language.

### Temperature Control

Adjust the randomness of the LLM's output using the `temperature` setting. Values range from 0 to 1 (default: 0.2).  Some models may not support temperature.

### Completion Options

*   `completions`: Control the number of completions suggested with **Ctrl + Space** (default: 5).
*   `refined_completions`: Sets the number of refined completions (default: 3).
*   `history_size`: (default 0) Enable personalizing completions using commandline history.  Consider using [`sponge`](https://github.com/meaningful-ooo/sponge) to remove broken commands from history.
*   `preview_pipe`: (default false) Optionally preview the output of pipes.

### Progress Indicator

Customize the progress indicator displayed when the LLM is processing (default: ⏳) with the `progress_indicator` option.

### Context Switching

Use the `fish_ai_switch_context` command to switch between different configurations.

## Data Privacy

*   `fish-ai` sends your OS name and commandline buffer to the LLM.
*   When codifying or completing commands, it sends the contents of any files you mention (if readable) and `<command> --help`.
*   You can send commandline history.
*   To fix the previous command, the previous commandline buffer, terminal output, and exit code is sent to the LLM.

Consider using a self-hosted LLM if you have data privacy concerns.

### Redaction

`fish-ai` attempts to redact sensitive data from prompts using `<REDACTED>`.  Disable this using the `redact = False` option if you trust your LLM provider.

## Development

See [`ARCHITECTURE.md`](https://github.com/Realiserad/fish-ai/blob/main/ARCHITECTURE.md) for an overview.

### Development Setup

Use the included `devcontainer.json` with GitHub Codespaces or VS Code's Dev Containers extension.

### Local Installation

Install from a local copy using: `fisher install .`

### Enable Debug Logging

Add `debug = True` to your `fish-ai.ini` file.  Set `log = <path to file>` to enable file logging.

### Testing

Automated tests run on macOS, Fedora, Ubuntu, and Arch Linux via GitHub Actions.  Test the Python modules with `pytest`.

### Releasing

Releases are automated via GitHub Actions upon pushing a new tag.
```shell
set tag (grep '^version =' pyproject.toml | \
    cut -d '=' -f2- | \
    string replace -ra '[ "]' '')
git tag -a "v$tag" -m "🚀 v$tag"
git push origin "v$tag"
```
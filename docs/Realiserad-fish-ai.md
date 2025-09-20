![Badge with time spent](https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2FRealiserad%2Fd3ec7fdeecc35aeeb315b4efba493326%2Fraw%2Ffish-ai-git-estimate.json)
![Popularity badge](https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2FRealiserad%2Fd3ec7fdeecc35aeeb315b4efba493326%2Fraw%2Fpopularity.json)
[![Donate XMR](https://img.shields.io/badge/Donate_XMR-grey?style=for-the-badge&logo=monero)](https://github.com/user-attachments/assets/07a2947f-6e5a-480f-990a-77204933411f)

# Fish-AI: Supercharge Your Fish Shell with AI-Powered Productivity

**Tired of endless man pages and Stack Overflow searches? Fish-AI brings the power of AI to your Fish shell, making command-line tasks faster and easier.** Check out the original repo [here](https://github.com/Realiserad/fish-ai)!

## Key Features:

*   **Comment to Command & Vice Versa:**  Effortlessly turn comments into executable commands and commands into clear explanations.
*   **Command Autocorrection:**  Automatically fix typos in your commands, saving you time and frustration.
*   **Intelligent Autocompletion:** Get smart command suggestions with a built-in fuzzy finder.
*   **Keyboard-Driven Efficiency:** Control everything with simple, configurable keyboard shortcuts – no mouse needed.
*   **LLM Flexibility:** Works with a variety of LLMs, including self-hosted options, GitHub Models, OpenAI, Azure OpenAI, Mistral, Anthropic, Cohere, DeepSeek, Groq and Google.
*   **Open Source & Customizable:**  Audit the code yourself and tailor `fish-ai` to your specific needs.
*   **Seamless Integration:** Works with your favorite Fish plugins like `fzf.fish` and `tide` without interference.
*   **Easy Installation & Updates:** Install and manage with ease using `fisher`.
*   **Privacy-Focused Design:** Doesn't track your usage or force you to use a proprietary terminal.

## Installation

### Prerequisites:

Ensure you have `git` installed, along with either `uv` or a supported Python version with `pip` and `venv`.

### Installation Steps:

1.  **Install fish-ai:**

    ```shell
    fisher install realiserad/fish-ai
    ```

2.  **Configure:**

    *   Create a configuration file:  `$XDG_CONFIG_HOME/fish-ai.ini` or `~/.config/fish-ai.ini`.
    *   Configure your preferred LLM provider (see the detailed examples below for GitHub Models, self-hosted LLMs, OpenRouter, OpenAI, Azure OpenAI, Mistral, Anthropic, Cohere, DeepSeek, Groq, and Google).

### Detailed Configuration Examples:

Choose one of the sections below depending on the LLM you want to use.
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
    (Get your PAT [here](https://github.com/settings/tokens).)
*   **Self-hosted:**
    ```ini
    [fish-ai]
    configuration = self-hosted

    [self-hosted]
    provider = self-hosted
    server = https://<your server>:<port>/v1
    model = <your model>
    api_key = <your API key>
    ```
    (Example: Ollama)
    ```ini
    [fish-ai]
    configuration = local-llama

    [local-llama]
    provider = self-hosted
    model = llama3.3
    server = http://localhost:11434/v1
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
    (View models [here](https://openrouter.ai/models).)
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

### API Key Management

*   To securely store your API key in your keyring, run `fish_ai_put_api_key`.

## How to Use:

### Transform Comments/Commands

*   Type a comment (starting with `#`), then press **Ctrl + P** to generate a command.
*   Type a command, press **Ctrl + P** to get an explanation.

### Autocomplete

*   Start typing a command and press **Ctrl + Space** for suggestions from `fzf`.
*   Refine completions by typing additional instructions and pressing **Ctrl + P** within `fzf`.

### Suggest Fixes

*   If a command fails, press **Ctrl + Space** immediately for suggested fixes.

## Customization Options:

### Key Bindings

*   Modify key bindings via `keymap_1` (default: **Ctrl + P**) and `keymap_2` (default: **Ctrl + Space**) in your `fish-ai.ini` using [fish_key_reader](https://fishshell.com/docs/current/cmds/fish_key_reader.html).

    ```ini
    [fish-ai]
    keymap_1 = \cP
    keymap_2 = '-k nul'
    ```

### Language

*   Set `language` in your `fish-ai.ini` to translate command explanations.

    ```ini
    [fish-ai]
    language = Swedish
    ```

### Temperature

*   Adjust the randomness of output with `temperature` (between 0 and 1). Set to `None` to disable.

    ```ini
    [fish-ai]
    temperature = 0.5
    ```

### Completions

*   Control the number of suggestions:  `completions` and `refined_completions`.

    ```ini
    [fish-ai]
    completions = 10
    ```

### History

*   Personalize completions using commandline history: `history_size`.

    ```ini
    [fish-ai]
    history_size = 5
    ```

### Pipe Preview

*   Enable previewing pipe output: `preview_pipe`.

    ```ini
    [fish-ai]
    preview_pipe = True
    ```

### Progress Indicator

*   Customize the progress indicator with `progress_indicator`.

    ```ini
    [fish-ai]
    progress_indicator = wait...
    ```

### Context Switching

*   Switch between configuration sections using the `fish_ai_switch_context` command.

## Data Privacy

*   `fish-ai` sends the OS name, commandline buffer, and potentially file contents and command help output to the LLM provider.
*   Command history can optionally be sent.
*   Failed commands, output, and exit codes are sent for fixes.
*   Use a self-hosted LLM for maximum privacy.
*   Sensitive information (API keys, private keys, etc.) is redacted.

## Development

*   Refer to [`ARCHITECTURE.md`](https://github.com/Realiserad/fish-ai/blob/main/ARCHITECTURE.md) for development details.
*   Use the provided `devcontainer.json` for development with GitHub Codespaces or VS Code.
*   Install from local copy using `fisher install .`.
*   Enable debug logging with `debug = True` in `fish-ai.ini` (logs to syslog or a specified file).
*   Run tests with `pytest`.
*   Releases are automated via GitHub Actions on tag pushes.
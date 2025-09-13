![Fish AI - AI-Powered Shell for Fish Shell](https://github.com/user-attachments/assets/86b61223-e568-4152-9e5e-d572b2b1385b)
[![Badge with time spent](https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2FRealiserad%2Fd3ec7fdeecc35aeeb315b4efba493326%2Fraw%2Ffish-ai-git-estimate.json)]()
[![Popularity badge](https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2FRealiserad%2Fd3ec7fdeecc35aeeb315b4efba493326%2Fraw%2Fpopularity.json)]()
[![Donate XMR](https://img.shields.io/badge/Donate_XMR-grey?style=for-the-badge&logo=monero)](https://github.com/user-attachments/assets/07a2947f-6e5a-480f-990a-77204933411f)

# Fish AI: Supercharge Your Fish Shell with AI 

**Fish AI enhances your Fish shell experience with AI, making it easier than ever to write, understand, and fix shell commands, [check out the original repo](https://github.com/Realiserad/fish-ai).**

## Key Features

*   **Command Generation from Comments:** Convert natural language comments into executable shell commands.
*   **Comment Generation from Commands:**  Translate complex commands into easy-to-understand explanations.
*   **Intelligent Autocompletion:** Get context-aware suggestions for your commands using LLMs and fuzzy-finder.
*   **Command Correction:** Automatically fix typos and errors in your commands.
*   **Keyboard-Driven Workflow:**  Access all features with configurable keyboard shortcuts, eliminating the need for mouse interaction.
*   **LLM Flexibility:** Compatible with a wide array of LLMs, including self-hosted options, OpenAI, and more.
*   **Open Source:** Fully open source, allowing for easy auditing and community contributions.
*   **Seamless Integration:** Works alongside other plugins like `fzf.fish` and `tide`.
*   **Lightweight and Non-Intrusive:** Doesn't wrap your shell or require a proprietary terminal emulator.
*   **Easy Installation:** Install and update effortlessly with `fisher`.

## How to Get Started

### 1. Installation

Ensure you have `git` and either [`uv`](https://github.com/astral-sh/uv) or a supported Python version with `pip` and `venv` installed. Then, install `fish-ai` using `fisher`:

```bash
fisher install realiserad/fish-ai
```

### 2. Configuration

Create a configuration file at `$XDG_CONFIG_HOME/fish-ai.ini` (or `~/.config/fish-ai.ini`) to specify your preferred LLM.

#### Example Configurations

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

    Generate a GitHub PAT [here](https://github.com/settings/tokens).

*   **Self-Hosted (e.g., Ollama):**
    ```ini
    [fish-ai]
    configuration = local-llama

    [local-llama]
    provider = self-hosted
    model = llama3.3
    server = http://localhost:11434/v1
    ```

*   **Other Providers:** Configuration examples are provided for OpenRouter, OpenAI, Azure OpenAI, Mistral, Anthropic, Cohere, DeepSeek, Groq, and Google in the original README.

### 3. API Key Management

Store your API key using `fish_ai_put_api_key` for secure storage:

```bash
fish_ai_put_api_key
```

## How to Use Fish AI

### Convert Comments to Commands (and vice versa)

*   Type a comment (starting with `#`) and press **Ctrl + P** to generate a shell command.
*   Type a command and press **Ctrl + P** to generate an explanation.

### Autocomplete Commands

*   Start typing your command and press **Ctrl + Space** to view LLM-powered autocompletions in `fzf`.

### Command Correction

*   If a command fails, press **Ctrl + Space** to get suggestions for a fix.

## Customization Options

*   **Keybindings:** Modify the default **Ctrl + P** and **Ctrl + Space** keybindings in your `fish-ai.ini` configuration.
*   **Language:** Explain commands in a different language with the `language` option.
*   **Temperature:** Adjust the randomness of LLM responses with the `temperature` setting (0.0-1.0).
*   **Completions:**  Control the number of suggested completions with the `completions` and `refined_completions` options.
*   **History Integration:** Personalize completions using command history with the `history_size` option.
*   **Preview Pipes:** Send pipe output to the LLM using the `preview_pipe` option.
*   **Progress Indicator:** Customize the visual indicator during LLM processing with `progress_indicator`.

## Data Privacy

*   `fish-ai` transmits your OS, command buffer, and may send file contents or command output to the LLM.
*   Sensitive information is redacted by default. Disable redaction with the `redact = False` option.
*   Use a self-hosted LLM for maximum data privacy.

## Development

See the original README for instructions on contributing, debugging, running tests, and creating releases.
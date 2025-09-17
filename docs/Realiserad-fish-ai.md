[![Time Spent](https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2FRealiserad%2Fd3ec7fdeecc35aeeb315b4efba493326%2Fraw%2Ffish-ai-git-estimate.json)](https://github.com/Realiserad/fish-ai)
[![Popularity](https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2FRealiserad%2Fd3ec7fdeecc35aeeb315b4efba493326%2Fraw%2Fpopularity.json)](https://github.com/Realiserad/fish-ai)
[![Donate XMR](https://img.shields.io/badge/Donate_XMR-grey?style=for-the-badge&logo=monero)](https://github.com/user-attachments/assets/07a2947f-6e5a-480f-990a-77204933411f)

# Fish-AI: Supercharge Your Fish Shell with AI 🚀

**Fish-AI is a powerful plugin that leverages the power of AI to make your Fish shell experience more efficient, intuitive, and fun.** Enhance your command-line productivity by allowing AI to transform, fix, and complete commands. [Check it out on GitHub!](https://github.com/Realiserad/fish-ai)

## Key Features:

*   **Comment-to-Command & Command-to-Comment Conversion:** Effortlessly turn comments into executable commands and vice versa with a simple keystroke.
*   **Intelligent Command Correction:** Automatically fix typos and errors in your commands, similar to `thefuck`.
*   **AI-Powered Autocompletion:** Get context-aware command suggestions as you type with a built-in fuzzy finder.
*   **Customizable Keybindings:** Control the AI features with two configurable keyboard shortcuts.
*   **LLM Flexibility:** Connect to your preferred LLM, including self-hosted options, for ultimate control.
*   **Open Source & Auditable:** Enjoy the transparency and flexibility of open-source code (approx. 2000 lines of code).
*   **Easy Installation & Updates:** Install and manage `fish-ai` seamlessly using `fisher`.
*   **Compatibility:** Works well with other Fish plugins like `fzf.fish` and `tide`.
*   **Non-Invasive:** Doesn't wrap your shell, install telemetry, or require a specific terminal emulator.

## 🎥 Demo

![Demo](https://github.com/user-attachments/assets/86b61223-e568-4152-9e5e-d572b2b1385b)

## 🛠️ Installation

### Prerequisites

*   Ensure you have `git` installed.
*   Install either [`uv`](https://github.com/astral-sh/uv) or a supported version of Python with `pip` and `venv`.

### Install `fish-ai`

```bash
fisher install realiserad/fish-ai
```

### Configuration

Create a configuration file (`$XDG_CONFIG_HOME/fish-ai.ini` or `~/.config/fish-ai.ini`) to specify your LLM.

#### GitHub Models

```ini
[fish-ai]
configuration = github

[github]
provider = self-hosted
server = https://models.inference.ai.azure.com
api_key = <paste GitHub PAT here>
model = gpt-4o-mini
```

*   Generate a GitHub Personal Access Token (PAT) [here](https://github.com/settings/tokens) (no permissions required).

#### Self-hosted

```ini
[fish-ai]
configuration = self-hosted

[self-hosted]
provider = self-hosted
server = https://<your server>:<port>/v1
model = <your model>
api_key = <your API key>
```

*   Consider [Ollama](https://github.com/ollama/ollama) with [Llama 3.3 70B](https://ollama.com/library/llama3.3).

#### OpenRouter

```ini
[fish-ai]
configuration = openrouter

[openrouter]
provider = self-hosted
server = https://openrouter.ai/api/v1
model = google/gemini-2.0-flash-lite-001
api_key = <your API key>
```

*   View available models on [OpenRouter](https://openrouter.ai/models).

#### OpenAI

```ini
[fish-ai]
configuration = openai

[openai]
provider = openai
model = gpt-4o
api_key = <your API key>
organization = <your organization>
```

#### Azure OpenAI

```ini
[fish-ai]
configuration = azure

[azure]
provider = azure
server = https://<your instance>.openai.azure.com
model = <your deployment name>
api_key = <your API key>
```

#### Mistral

```ini
[fish-ai]
configuration = mistral

[mistral]
provider = mistral
api_key = <your API key>
```

#### Anthropic

```ini
[anthropic]
provider = anthropic
api_key = <your API key>
```

#### Cohere

```ini
[cohere]
provider = cohere
api_key = <your API key>
```

#### DeepSeek

```ini
[deepseek]
provider = deepseek
api_key = <your API key>
model = deepseek-chat
```

#### Groq

```ini
[groq]
provider = groq
api_key = <your API key>
```

#### Google

```ini
[google]
provider = google
api_key = <your API key>
```

### Store API Keys Securely

Use the `fish_ai_put_api_key` command to store API keys in your keyring.

## 🕹️ How to Use

### Comment/Command Conversion

*   Type a comment (starting with `#`) and press **Ctrl + P** to get a command.
*   Type a command and press **Ctrl + P** to get a comment explaining it.

### Autocompletion

*   Start typing a command/comment, then press **Ctrl + Space** to see completions in `fzf`.

### Suggest Fixes

*   If a command fails, press **Ctrl + Space** to get suggested fixes.

## ⚙️ Advanced Configuration

Modify `fish-ai.ini` for advanced settings:

### Keybindings

*   Customize with `keymap_1` (default: Ctrl + P) and `keymap_2` (default: Ctrl + Space). Use `fish_key_reader` to get escape sequences.

### Language

*   Set `language = <language>` to get command explanations in another language.

### Temperature

*   Adjust the randomness of responses with `temperature = <value>` (0.0 - 1.0).  Set to `None` to disable for models that do not support temperature.

### Completions

*   Control the number of suggestions with `completions = <number>` (default: 5) and `refined_completions = <number>` (default: 3).

### History Personalization

*   Enable commandline history with `history_size = <number>` (default: 0).

### Preview Pipes

*   Preview pipe output with `preview_pipe = True`.  (Disabled by default)

### Progress Indicator

*   Customize the loading indicator with `progress_indicator = <characters>`.

### Switching Contexts

*   Use the `fish_ai_switch_context` command to switch between configuration sections.

## 🔒 Data Privacy

`fish-ai` sends the OS name and command-line buffer to the LLM.  It also sends the contents of any files mentioned, and `<command> --help` when explaining commands. An excerpt of command history can also be sent.

If you're concerned about data privacy, use a self-hosted LLM.

### Redaction

*   Sensitive info (passwords, API keys, private keys, bearer tokens) is redacted before submission.
*   Disable redaction with `redact = False` (only if you trust the LLM provider).

## 👨‍💻 Development

*   Review `ARCHITECTURE.md` for development information.
*   Use `devcontainer.json` with Codespaces or VS Code's Dev Containers extension for development.
*   Install from a local copy using `fisher install .`.
*   Enable debug logging with `debug = True` and optionally `log = /path/to/file`.
*   Installation tests run automatically.
*   Python modules are tested using `pytest`.
*   Releases are automated via GitHub Actions.
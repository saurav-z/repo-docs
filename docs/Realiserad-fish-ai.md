[![Time Spent](https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2FRealiserad%2Fd3ec7fdeecc35aeeb315b4efba493326%2Fraw%2Ffish-ai-git-estimate.json)](https://github.com/Realiserad/fish-ai)
[![Popularity](https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2FRealiserad%2Fd3ec7fdeecc35aeeb315b4efba493326%2Fraw%2Fpopularity.json)](https://github.com/Realiserad/fish-ai)
[![Donate XMR](https://img.shields.io/badge/Donate_XMR-grey?style=for-the-badge&logo=monero)](https://github.com/user-attachments/assets/07a2947f-6e5a-480f-990a-77204933411f)

# Fish-AI: Supercharge Your Fish Shell with AI-Powered Commands

**Tired of endless `man` pages and complex commands?** Fish-AI integrates AI directly into your Fish shell, offering intelligent command generation, correction, and completion for a more efficient and enjoyable terminal experience. Explore the [original repository](https://github.com/Realiserad/fish-ai) for more details.

## Key Features

*   **Comment to Command & Vice Versa:**  Transform natural language comments into executable shell commands and explain existing commands with ease.
*   **Command Correction:** Automatically fix typos and errors in your commands, similar to `thefuck`.
*   **AI-Powered Autocompletion:**  Get smart suggestions for commands with a built-in fuzzy finder, saving time and effort.
*   **Customizable Shortcuts:** Control all the magic with configurable, keyboard-driven shortcuts.
*   **Flexible LLM Integration:** Use your preferred LLM provider, including self-hosted options for complete control.
*   **Open Source & Auditable:** The entire plugin is open source and designed for easy understanding and modification.
*   **Easy Installation & Updates:** Install and manage `fish-ai` effortlessly with `fisher`.
*   **Compatibility:** Works seamlessly with existing Fish plugins and configurations, including `fzf.fish` and `tide`.
*   **No External Dependencies:** Doesn't require wrapping your shell or using proprietary terminal emulators.

## 🎥 Demo

![Demo](https://github.com/user-attachments/assets/86b61223-e568-4152-9e5e-d572b2b1385b)

## Getting Started

### 1. Installation

Ensure you have `git` and either `uv` or a supported Python version with `pip` and `venv` installed. Then, install `fish-ai` using `fisher`:

```shell
fisher install realiserad/fish-ai
```

### 2. Configuration

Create a configuration file at `$XDG_CONFIG_HOME/fish-ai.ini` (or `~/.config/fish-ai.ini` if `$XDG_CONFIG_HOME` is not set) to specify your preferred LLM.  Here are several configuration options:

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

Get a personal access token (PAT) [here](https://github.com/settings/tokens).

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

Consider using [Ollama](https://github.com/ollama/ollama) with [Llama 3.3 70B](https://ollama.com/library/llama3.3).

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

See available models [here](https://openrouter.ai/models).

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

### 3. Secure API Key Storage

Use `fish_ai_put_api_key` to securely store your API key in your system's keyring, rather than in the configuration file.

## How to Use

### 1. Convert Comments to Commands & Vice Versa

Type a comment (starting with `#`) and press **Ctrl + P** to generate a shell command. Use the same shortcut to explain existing commands.

### 2. Autocomplete Commands

Start typing and press **Ctrl + Space** for intelligent autocompletions via `fzf`. Refine results further by typing instructions and pressing **Ctrl + P** within `fzf`.

### 3. Suggest Command Fixes

If a command fails, press **Ctrl + Space** immediately to get suggested fixes.

## Customization Options

Customize `fish-ai`'s behavior in your `fish-ai.ini` file:

*   **Key Bindings:**  Modify the default shortcuts (Ctrl+P, Ctrl+Space) using `keymap_1` and `keymap_2`. Use `fish_key_reader` to get the correct escape sequences.
*   **Language:** Set `language` to translate command explanations (requires LLM support).
*   **Temperature:** Control randomness with the `temperature` setting (0.0 - 1.0).
*   **Number of Completions:** Adjust the number of suggestions with the `completions` and `refined_completions` options.
*   **Commandline History:** Personalize suggestions with `history_size`.
*   **Preview Pipes:** Use `preview_pipe` to send pipe outputs to the LLM (disabled by default).
*   **Progress Indicator:** Customize the indicator with the `progress_indicator` option.
*   **Context Switching:** Use the `fish_ai_switch_context` command.

## Data Privacy

`fish-ai` sends the OS name, command line buffer, and potentially file contents, help output, and command history to the LLM. Use self-hosting for maximum data privacy.

### Redaction

`fish-ai` attempts to redact sensitive data (passwords, API keys, private keys, and bearer tokens) before sending prompts to the LLM. Disable redaction with `redact = False` if you trust the LLM provider.

## Development

Contribute to `fish-ai`! Review `ARCHITECTURE.md` before you start and use the provided `devcontainer.json` for easy development.

*   **Installation from Local Copy:** `fisher install .`
*   **Debug Logging:** Enable logging with `debug = True` and/or `log = /tmp/fish-ai.log`.
*   **Testing:** Installation tests are available in the GitHub Actions.  Python module tests can be executed via `pytest`.
*   **Releasing:** Create releases by pushing a new tag with the version number.
```shell
set tag (grep '^version =' pyproject.toml | \
    cut -d '=' -f2- | \
    string replace -ra '[ "]' '')
git tag -a "v$tag" -m "🚀 v$tag"
git push origin "v$tag"
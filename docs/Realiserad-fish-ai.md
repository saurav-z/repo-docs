[![Badge with time spent](https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2FRealiserad%2Fd3ec7fdeecc35aeeb315b4efba493326%2Fraw%2Ffish-ai-git-estimate.json)](https://github.com/Realiserad/fish-ai)
[![Popularity badge](https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2FRealiserad%2Fd3ec7fdeecc35aeeb315b4efba493326%2Fraw%2Fpopularity.json)](https://github.com/Realiserad/fish-ai)
[![Donate XMR](https://img.shields.io/badge/Donate_XMR-grey?style=for-the-badge&logo=monero)](https://github.com/Realiserad/fish-ai)

# Fish-AI: Supercharge Your Fish Shell with AI-Powered Productivity

**Tired of wrestling with complex commands?** Fish-AI brings the power of AI directly into your Fish shell, making command-line interactions easier, faster, and more intuitive.  [Explore the original repository](https://github.com/Realiserad/fish-ai) to learn more!

## Key Features

*   **Comment-to-Command & Vice Versa:** Effortlessly convert comments into executable commands and explain what commands do with a simple shortcut.
*   **Intelligent Command Autocompletion:** Get context-aware command suggestions with built-in fuzzy finder, saving time and frustration.
*   **Smart Command Correction:**  Automatically fix typos and errors in your commands, keeping you productive.
*   **Keyboard-Driven Interface:** Navigate and utilize all features with just two convenient keyboard shortcuts.
*   **Flexible LLM Integration:** Connect to your preferred LLM, including self-hosted options, OpenAI, and others.
*   **Open Source & Auditable:** Inspect the code with confidence; it is open-source with approximately 2000 lines of code.
*   **Seamless Integration:** Works with existing Fish plugins like `fzf.fish` and `tide` without interference.
*   **Cross-Platform Compatibility:** Tested on macOS and common Linux distributions.

## 🎥 Demo

![Demo](https://github.com/user-attachments/assets/86b61223-e568-4152-9e5e-d572b2b1385b)

## Installation

### Prerequisites

Ensure `git` is installed, along with either [`uv`](https://github.com/astral-sh/uv) or a supported version of Python (with `pip` and `venv`).

### Installation using `fisher`

```shell
fisher install realiserad/fish-ai
```

### Configuration

Create a configuration file at `$XDG_CONFIG_HOME/fish-ai.ini` (or `~/.config/fish-ai.ini` if `$XDG_CONFIG_HOME` is not set).  Specify your preferred LLM provider within this file.  Examples for different providers are below:

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

Get a personal access token (PAT) from [here](https://github.com/settings/tokens) (no permissions needed).

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

A recommended self-hosted setup uses [Ollama](https://github.com/ollama/ollama) with [Llama 3.3 70B](https://ollama.com/library/llama3.3).

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

Find available models at [OpenRouter's website](https://openrouter.ai/models).

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

### Keyring for API Keys

For enhanced security, use `fish_ai_put_api_key` to store your API key in your system's keyring instead of the configuration file.

## Usage

### Transform Comments/Commands

*   Type a comment (starting with `#`) and press **Ctrl + P** to convert it to a shell command.
*   Type a command and press **Ctrl + P** to generate an explanation comment.

### Autocomplete Commands

*   Start typing your command and press **Ctrl + Space** to see completion suggestions in `fzf`.

### Suggest Fixes

*   If a command fails, press **Ctrl + Space** at the prompt to receive AI-powered suggestions for correction.

## Configuration Options

Customize `fish-ai`'s behavior by adjusting settings in your `fish-ai.ini` configuration file or using environment variables.

### Key Bindings

Change the default shortcuts (Ctrl + P and Ctrl + Space) by setting the environment variables `FISH_AI_KEYMAP_1` and `FISH_AI_KEYMAP_2`.  Use `fish_key_reader` to find the correct key binding escape sequence.

### Language

Set the `language` option to explain commands in another language (requires the LLM to be trained in that language).

### Temperature

Control the randomness of the output with the `temperature` option (0.0 to 1.0).

### Number of Completions

Adjust the number of command completion suggestions with the `completions` option, and refined completion results using `refined_completions`.

### Commandline History

Personalize command completions by sending a portion of your command history, using the `history_size` option.

### Preview Pipes

Enable the `preview_pipe` option to send the output of a pipe to the LLM.

### Progress Indicator

Customize the progress indicator displayed while waiting for LLM responses with the `progress_indicator` option.

## Context Switching

Use the `fish_ai_switch_context` command to switch between different configuration sections.

## Data Privacy

`fish-ai` transmits your OS name and command-line input to the LLM.  It may also send file contents (if readable), the output of `<command> --help`, and a portion of your command history (if enabled). Consider using a self-hosted LLM for maximum data privacy.

### Redaction

The plugin attempts to redact sensitive information like passwords, API keys, and private keys from prompts before submitting them to the LLM. This can be disabled via `redact = False`.

## Development

See [`ARCHITECTURE.md`](https://github.com/Realiserad/fish-ai/blob/main/ARCHITECTURE.md) for architectural details.

Use the provided `devcontainer.json` with GitHub Codespaces or VS Code's Dev Containers extension for a ready-to-go development environment.

### Local Installation for Development

```shell
fisher install .
```

### Debugging

Enable debug logging by setting `debug = True` in `fish-ai.ini` (logging defaults to syslog).

### Testing

Installation tests run automatically on push.  Python module tests are available with `pytest`.

### Releases

Releases are automatically created upon pushing a new tag (e.g., `git tag -a "v1.2.3" -m "Release v1.2.3"` followed by `git push origin "v1.2.3"`).
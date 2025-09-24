[![Time Spent](https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2FRealiserad%2Fd3ec7fdeecc35aeeb315b4efba493326%2Fraw%2Ffish-ai-git-estimate.json)](https://github.com/Realiserad/fish-ai)
[![Popularity](https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2FRealiserad%2Fd3ec7fdeecc35aeeb315b4efba493326%2Fraw%2Fpopularity.json)](https://github.com/Realiserad/fish-ai)
[![Donate XMR](https://img.shields.io/badge/Donate_XMR-grey?style=for-the-badge&logo=monero)](https://github.com/user-attachments/assets/07a2947f-6e5a-480f-990a-77204933411f)

# fish-ai: Supercharge Your Fish Shell with AI 🐟🤖

**Enhance your Fish shell experience with `fish-ai`, an AI-powered plugin that simplifies command creation, correction, and completion.** ([Back to Original Repo](https://github.com/Realiserad/fish-ai))

## Key Features

*   **Comment-to-Command & Command-to-Comment Conversion:** Effortlessly translate natural language comments into shell commands and vice-versa, eliminating the need to memorize complex syntax or search for answers.
*   **Intelligent Command Correction:** Automatically fix typos and errors in your commands, similar to `thefuck`.
*   **AI-Powered Command Autocompletion:** Get smart suggestions for your next command with a built-in fuzzy finder, saving you time and effort.
*   **Customizable Keybindings:** Control functionality using two configurable keyboard shortcuts for a seamless workflow.
*   **Flexible LLM Integration:** Works with your preferred Large Language Model, including self-hosted options, GitHub Models, OpenAI, and more.
*   **Open Source & Auditable:** The code is open-source and easily readable, allowing you to review and understand the plugin's functionality.
*   **Easy Installation & Updates:** Install and manage the plugin effortlessly using `fisher`.
*   **Compatibility:** Designed to work harmoniously with other popular Fish plugins like `fzf.fish` and `tide`.
*   **No Restrictions:**  `fish-ai` doesn't interfere with your terminal or track your activity.

## 🎥 Demo

[Demo Video](https://github.com/user-attachments/assets/86b61223-e568-4152-9e5e-d572b2b1385b)

## 👨‍🔧 Installation

### Prerequisites

Ensure you have `git` installed, along with either [`uv`](https://github.com/astral-sh/uv) or a supported Python version and `pip`/`venv`.

### Install `fish-ai`

```bash
fisher install realiserad/fish-ai
```

### Configure

Create a configuration file at `$XDG_CONFIG_HOME/fish-ai.ini` (or `~/.config/fish-ai.ini` if `$XDG_CONFIG_HOME` is not set) to specify your preferred LLM.

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

Get a personal access token (PAT) [here](https://github.com/settings/tokens). No permissions are required.

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

Available models are listed [here](https://openrouter.ai/models).

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

### Store API Key

Use `fish_ai_put_api_key` to securely store your API key in your keyring, instead of in the configuration file.

## 🙉 How to Use

### Comment-to-Command & Command-to-Comment

*   Type a comment (starting with `#`) and press **Ctrl + P** to convert it into a shell command.
*   Type a command and press **Ctrl + P** to generate a comment explaining the command's function.

### Autocompletion

*   Start typing a command or comment and press **Ctrl + Space** to see completion suggestions via [`fzf`](https://github.com/junegunn/fzf).
*   Refine the results by providing additional instructions and pressing **Ctrl + P** inside `fzf`.

### Command Fixes

*   If a command fails, press **Ctrl + Space** at the command prompt to get a suggestion to fix the error.

## 🤸 Additional Options

Customize `fish-ai` behavior within your `fish-ai.ini` configuration:

*   **Change Keybindings:** Modify `keymap_1` (Ctrl + P) and `keymap_2` (Ctrl + Space) with desired key sequences. Use [`fish_key_reader`](https://fishshell.com/docs/current/cmds/fish_key_reader.html) to identify key codes.
*   **Language:** Set the `language` option for command explanations in a different language.
*   **Temperature:** Adjust the `temperature` (0-1) for LLM creativity.  Disable with `temperature = None`.
*   **Completions:** Control the number of suggested completions with `completions` and `refined_completions` options.
*   **History:** Personalize completions using your command-line history with the `history_size` option.
*   **Preview Pipe:** Enable pipe output with `preview_pipe = True`.
*   **Progress Indicator:** Customize the progress indicator using `progress_indicator`.

## 🎭 Context Switching

Switch between configuration sections using the `fish_ai_switch_context` command.

## 🐾 Data Privacy

`fish-ai` sends your OS name and command buffer to the LLM.  It also sends file contents, command help outputs, previous command buffers and terminal output.  Use a self-hosted LLM for maximum privacy.

### Redaction of Sensitive Information

`fish-ai` attempts to redact sensitive information, such as passwords and API keys. Disable redaction with `redact = False`.

## 🔨 Development

See [`ARCHITECTURE.md`](https://github.com/Realiserad/fish-ai/blob/main/ARCHITECTURE.md) for development insights.

Use the provided `devcontainer.json` for development with GitHub Codespaces or VS Code's Dev Containers extension.

Install from a local copy using: `fisher install .`

### Enable Debug Logging

Enable logging by setting `debug = True` in `fish-ai.ini`. Customize the log file location with `log = <path>`.

### Testing

Installation tests are run on GitHub Actions.  Python modules can be tested with `pytest`.

### Release Creation

Releases are automatically created by GitHub Actions upon pushing a new tag.
```bash
set tag (grep '^version =' pyproject.toml | \
    cut -d '=' -f2- | \
    string replace -ra '[ "]' '')
git tag -a "v$tag" -m "🚀 v$tag"
git push origin "v$tag"
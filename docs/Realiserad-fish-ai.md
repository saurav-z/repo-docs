[![Time Spent](https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2FRealiserad%2Fd3ec7fdeecc35aeeb315b4efba493326%2Fraw%2Ffish-ai-git-estimate.json)](https://github.com/Realiserad/fish-ai)
[![Popularity](https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2FRealiserad%2Fd3ec7fdeecc35aeeb315b4efba493326%2Fraw%2Fpopularity.json)](https://github.com/Realiserad/fish-ai)
[![Donate XMR](https://img.shields.io/badge/Donate_XMR-grey?style=for-the-badge&logo=monero)](https://github.com/user-attachments/assets/07a2947f-6e5a-480f-990a-77204933411f)

# Fish-AI: Supercharge Your Fish Shell with AI!

**Effortlessly translate comments to commands, autocomplete with intelligent suggestions, and fix typos – all powered by AI within your Fish shell.**  [Explore the fish-ai repository on GitHub](https://github.com/Realiserad/fish-ai).

## Key Features

*   **Comment to Command & Vice Versa:** Convert natural language comments into executable shell commands and explain complex commands with AI-generated descriptions.
*   **Intelligent Autocompletion:** Get context-aware command suggestions with a built-in fuzzy finder.
*   **Smart Error Correction:** Automatically fix typos and broken commands, saving you time and frustration.
*   **Customizable & Configurable:**  Choose your preferred Large Language Model (LLM), including self-hosted, open-source, and popular cloud providers.
*   **Keyboard Shortcut Driven:** Access all features with simple, configurable keyboard shortcuts (Ctrl+P & Ctrl+Space by default).
*   **Open Source & Auditable:** Inspect the code yourself – it's open source, easy to read, and under 2000 lines.
*   **Fisher Integration:**  Seamlessly install and update the plugin using Fisher.
*   **Non-Intrusive:**  Doesn't interfere with existing plugins or require a proprietary terminal.

## 🚀 Demo

![Demo](https://github.com/user-attachments/assets/86b61223-e568-4152-9e5e-d572b2b1385b)

## ⚙️ Installation

### Prerequisites

Ensure you have `git` installed, along with either [`uv`](https://github.com/astral-sh/uv), or a supported version of Python with `pip` and `venv`.

### Install `fish-ai` using `fisher`

```bash
fisher install realiserad/fish-ai
```

### Configure the LLM

Create a configuration file at `$XDG_CONFIG_HOME/fish-ai.ini` (or `~/.config/fish-ai.ini` if `$XDG_CONFIG_HOME` isn't set) to specify your preferred LLM.  Below are example configurations for different providers. **Remember to replace the placeholder values (e.g., `<YOUR API KEY>`) with your actual credentials.**

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

#### Self-hosted (Example: Ollama)

```ini
[fish-ai]
configuration = local-llama

[local-llama]
provider = self-hosted
model = llama3.3
server = http://localhost:11434/v1
```

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

#### Google Gemini

```ini
[google]
provider = google
api_key = <your API key>
```

### Securely Store API Keys (Optional)

Use the `fish_ai_put_api_key` command to securely store your API keys in your system's keyring.

## 🕹️ How to Use

*   **Convert Comments to Commands:** Type a comment (starting with `#`) and press **Ctrl + P** to generate a shell command. Press Ctrl+P again to refine.
*   **Convert Commands to Comments:** Type a command and press **Ctrl + P** to explain it.
*   **Autocomplete Commands:** Start typing a command and press **Ctrl + Space** to see suggestions using `fzf`.
*   **Fix Broken Commands:** Press **Ctrl + Space** after a failed command for suggested fixes.

## ⚙️ Additional Options & Customization

Modify the `fish-ai.ini` file to customize the plugin's behavior.

*   **Change Keybindings:** Customize the  `keymap_1` and `keymap_2` options in your config.
*   **Language:**  Set the `language` option to get explanations in a different language.
*   **Temperature:** Adjust the randomness of responses with the `temperature` option (0.0 to 1.0).
*   **Number of Completions:** Configure the number of suggestions displayed with `completions` and `refined_completions`.
*   **Personalized Completions:** Enable `history_size` to include command history in LLM prompts.
*   **Preview Pipes:** Use the `preview_pipe` option to send the output of a pipe to the LLM.
*   **Progress Indicator:** Customize the visual indicator shown during LLM processing using `progress_indicator`.

## 🛡️ Data Privacy

`fish-ai` transmits your OS name and command line buffer to the LLM provider.  It also sends the contents of files you mention and the output of  `<command> --help`. Additionally, an excerpt of your command history may be sent if enabled.

**For maximum data privacy, use a self-hosted LLM.**

### Data Redaction

The plugin attempts to redact sensitive information (passwords, API keys, private keys, bearer tokens) from prompts before sending them to the LLM. You can disable redaction with `redact = False`.

## 🛠️ Development

See [`ARCHITECTURE.md`](https://github.com/Realiserad/fish-ai/blob/main/ARCHITECTURE.md) for details.

### Install Locally

```bash
fisher install .
```

### Debugging

Enable debug logging with `debug = True` in `fish-ai.ini`. Log to a file with `log = /path/to/file`.

### Testing

Installation tests run automatically on pushes and PRs.  Python module tests can be run with `pytest`.

### Release

Releases are automated via GitHub Actions on tag pushes.
```bash
set tag (grep '^version =' pyproject.toml | \
    cut -d '=' -f2- | \
    string replace -ra '[ "]' '')
git tag -a "v$tag" -m "🚀 v$tag"
git push origin "v$tag"
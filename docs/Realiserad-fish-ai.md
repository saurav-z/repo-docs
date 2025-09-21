![Time Spent Badge](https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2FRealiserad%2Fd3ec7fdeecc35aeeb315b4efba493326%2Fraw%2Ffish-ai-git-estimate.json)
![Popularity Badge](https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2FRealiserad%2Fd3ec7fdeecc35aeeb315b4efba493326%2Fraw%2Fpopularity.json)
[![Donate XMR](https://img.shields.io/badge/Donate_XMR-grey?style=for-the-badge&logo=monero)](https://github.com/user-attachments/assets/07a2947f-6e5a-480f-990a-77204933411f)

# fish-ai: Supercharge Your Fish Shell with AI 🐠

**Effortlessly enhance your Fish shell experience with fish-ai, an AI-powered plugin that boosts productivity by translating natural language into commands, correcting errors, and providing intelligent autocompletions.**  [Explore the original repository](https://github.com/Realiserad/fish-ai)!

## Key Features

*   **Comment-to-Command & Vice Versa:** Convert comments into shell commands and commands into explanatory comments.
*   **Command Correction:** Automatically fix typos and errors in your commands.
*   **Intelligent Autocompletion:** Get smart suggestions for your next command with a built-in fuzzy finder.
*   **Keyboard-Driven:** Control everything with two configurable keyboard shortcuts, eliminating the need for the mouse.
*   **LLM Flexibility:** Integrate with your preferred LLM, including self-hosted options.
*   **Open Source & Auditable:** The code is open source, easy to read, and auditable.
*   **Easy Installation:** Install and update seamlessly using `fisher`.
*   **Broad Compatibility:** Tested on macOS and popular Linux distributions.
*   **Seamless Integration:** Works alongside other plugins without interference.
*   **Privacy-Focused:** Does not use a wrapper, install telemetry, or require proprietary terminal emulators.

## 🎥 Demo

![Demo](https://github.com/user-attachments/assets/86b61223-e568-4152-9e5e-d572b2b1385b)

## Installation

### Prerequisites
Ensure you have `git` and either [`uv`](https://github.com/astral-sh/uv), or a supported version of Python along with `pip` and `venv` installed.

### Install fish-ai

Use `fisher` to install the plugin:

```shell
fisher install realiserad/fish-ai
```

### Configuration

Create a configuration file (`$XDG_CONFIG_HOME/fish-ai.ini` or `~/.config/fish-ai.ini`) and specify your LLM settings:

**Choose Your LLM Provider:**  Configuration examples are provided for:

*   GitHub Models
*   Self-hosted (OpenAI-compatible)
*   OpenRouter
*   OpenAI
*   Azure OpenAI
*   Mistral
*   Anthropic
*   Cohere
*   DeepSeek
*   Groq
*   Google

**Example: GitHub Models**

```ini
[fish-ai]
configuration = github

[github]
provider = self-hosted
server = https://models.inference.ai.azure.com
api_key = <paste GitHub PAT here>
model = gpt-4o-mini
```
*Note: You'll need a GitHub Personal Access Token (PAT) which doesn't require any permissions.  Get it [here](https://github.com/settings/tokens).*

**API Key Security:** Securely store your API keys using `fish_ai_put_api_key` to add them to your keyring.

## 🙉 How to Use

### Comment/Command Conversion (Ctrl + P)

Type a comment (starting with `#`) and press **Ctrl + P** to convert it to a shell command.  Use **Ctrl + P** again to refine the command if needed. Do the reverse to convert command into a comment.

### Command Autocomplete (Ctrl + Space)

Start typing a command or comment and press **Ctrl + Space** to see a list of suggestions in [`fzf`](https://github.com/junegunn/fzf). Refine results by typing instructions and pressing **Ctrl + P** within `fzf`.

### Command Fixes (Ctrl + Space)

If a command fails, press **Ctrl + Space** at the prompt for `fish-ai` to suggest a fix.

## 🤸 Additional Options (Customization)

Customize `fish-ai` behavior in your `fish-ai.ini` file:

### Key Bindings
Change the default keyboard shortcuts (`Ctrl + P` and `Ctrl + Space`) by setting `keymap_1` and `keymap_2` to the key binding escape sequences from [`fish_key_reader`](https://fishshell.com/docs/current/cmds/fish_key_reader.html).

### Language
Set the `language` option to translate explanations into different languages (LLM dependent).

### Temperature
Control the randomness of responses using the `temperature` option (0.0 - 1.0). Set to `None` if the model doesn't support temperature.

### Number of Completions
Adjust the number of suggested completions:
*   `completions` : Sets the number of suggested completions.
*   `refined_completions` : Sets the number of refined completions in fzf.

### History
Enable command line history integration using the `history_size` option. Consider using [`sponge`](https://github.com/meaningful-ooo/sponge) with this feature.

### Preview Pipes
Enable previewing pipes with the `preview_pipe` option. This setting may affect performance.

### Progress Indicator
Customize the progress indicator with the `progress_indicator` option.

## 🎭 Context Switching
Use the `fish_ai_switch_context` command to switch between different configuration sections.

## 🐾 Data Privacy
`fish-ai` sends the OS name and commandline buffer to the LLM.  It also sends the contents of files you mention when codifying, and the output of `<command> --help` when explaining.  You can optionally send command history.
To fix a command, it sends the command, its output, and exit code.
Use a self-hosted LLM for maximum privacy.
Sensitive information is redacted by default (passwords, API keys, private keys, bearer tokens). Disable redaction with `redact = False`.

## 🔨 Development

Explore the [`ARCHITECTURE.md`](https://github.com/Realiserad/fish-ai/blob/main/ARCHITECTURE.md) for contribution guidelines. Use the `devcontainer.json` file for development with GitHub Codespaces or VS Code.

Install from a local copy:
```shell
fisher install .
```

### Debugging
Enable debug logging with `debug = True` and `log = <path to file>` in `fish-ai.ini`.

### Testing
The installation tests are run on the CI on push. The Python modules can be tested using `pytest`.

### Release
Releases are created automatically upon pushing a new tag.
```shell
set tag (grep '^version =' pyproject.toml | \
    cut -d '=' -f2- | \
    string replace -ra '[ "]' '')
git tag -a "v$tag" -m "🚀 v$tag"
git push origin "v$tag"
```
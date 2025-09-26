[![Time Spent](https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2FRealiserad%2Fd3ec7fdeecc35aeeb315b4efba493326%2Fraw%2Ffish-ai-git-estimate.json)](https://github.com/Realiserad/fish-ai)
[![Popularity](https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2FRealiserad%2Fd3ec7fdeecc35aeeb315b4efba493326%2Fraw%2Fpopularity.json)](https://github.com/Realiserad/fish-ai)
[![Donate XMR](https://img.shields.io/badge/Donate_XMR-grey?style=for-the-badge&logo=monero)](https://github.com/user-attachments/assets/07a2947f-6e5a-480f-990a-77204933411f)

# fish-ai: Supercharge Your Fish Shell with AI 🐠

**Unlock the power of AI within your Fish shell and streamline your workflow with intelligent command generation, completion, and error correction.**  [Get Started](https://github.com/Realiserad/fish-ai)

## Key Features

*   **Comment-to-Command Conversion:**  Effortlessly translate natural language comments into executable shell commands and vice-versa, saving time and reducing the need for manual research.
*   **Intelligent Command Completion:**  Get context-aware autocompletions with a built-in fuzzy finder for quicker and more accurate command input.
*   **Smart Error Correction:**  Automatically suggest and fix typos and syntax errors in your commands, eliminating frustrating debugging sessions.
*   **Customizable & Flexible:** Configure your preferred Large Language Model (LLM) provider, including self-hosted options for enhanced privacy and control.
*   **Keyboard-Driven:**  Utilize two configurable keyboard shortcuts for all core functionalities, allowing a seamless, mouse-free experience.
*   **Open Source & Auditable:** Review and understand the underlying code with ease. The project is open-source, and around 2000 lines of code, meaning it's easier to audit.
*   **Easy Installation & Updates:** Install and manage `fish-ai` with ease via `fisher`.
*   **Unobtrusive Integration:** Compatible with popular Fish shell plugins like `fzf.fish` and `tide`, and does not interfere with your existing setup.
*   **Privacy-Focused:** No telemetry, no forced terminal changes, and built-in redaction of sensitive information.

## 🎥 Demo

![Demo](https://github.com/user-attachments/assets/86b61223-e568-4152-9e5e-d572b2b1385b)

## 👨‍🔧 Installation

### Prerequisites

Ensure that you have `git` installed. If you want to use `fish-ai`, then install either [`uv`](https://github.com/astral-sh/uv), or  [a supported version of Python](https://github.com/Realiserad/fish-ai/blob/main/.github/workflows/python-tests.yaml) along with `pip` and `venv` is installed.

### Install fish-ai

Use [`fisher`](https://github.com/jorgebucaran/fisher) to install the plugin:

```shell
fisher install realiserad/fish-ai
```

### Configuration

Create a configuration file at `$XDG_CONFIG_HOME/fish-ai.ini` (or `~/.config/fish-ai.ini` if `$XDG_CONFIG_HOME` is not set) to specify your chosen LLM. Example configurations are available for the following providers:

*   [GitHub Models](#github-models)
*   [Self-hosted](#self-hosted)
*   [OpenRouter](#openrouter)
*   [OpenAI](#openai)
*   [Azure OpenAI](#azure-openai)
*   [Mistral](#mistral)
*   [Anthropic](#anthropic)
*   [Cohere](#cohere)
*   [DeepSeek](#deepseek)
*   [Groq](#groq)
*   [Google](#google)

**See the original README for detailed configuration instructions and examples for each provider.**

## 🙉 How to Use

### Comment to Command

Type a comment (starting with `#`) and press **Ctrl + P** to convert it into a shell command. Press **Ctrl + P** again to refine the generated command.

Reverse the process:  Type a command and press **Ctrl + P** to get an explanation.

### Autocomplete

Press **Ctrl + Space** to display command completions and let the LLM suggest the right command for you. Press **Ctrl + P** inside `fzf` to refine the results.

### Fix Errors

If a command fails, press **Ctrl + Space** immediately to receive suggestions for fixing it.

## 🤸 Additional Options

Customize your `fish-ai` experience with a range of configuration options within your `fish-ai.ini` file.

*   **Change Keybindings:** Modify the default keyboard shortcuts (Ctrl + P and Ctrl + Space) to prevent conflicts.
*   **Change the language for the LLM:** Configure the language of explanations with the `language` option.
*   **Adjust Temperature:** Control the randomness of the LLM's output using the `temperature` setting (between 0 and 1).
*   **Number of Completions:** Fine-tune the number of suggested completions with the `completions` and `refined_completions` options.
*   **Personalise with Commandline History:** Enable the `history_size` option to personalize completions using your command history.
*   **Preview Pipes:**  Use the `preview_pipe` option to send the output of a pipe to the LLM for completion.
*   **Configure the Progress Indicator:** Customize the visual indicator while waiting for an LLM response with the `progress_indicator` option.

## 🎭 Switch Between Contexts

Use the `fish_ai_switch_context` command to switch between different configuration sections.

## 🐾 Data Privacy

When using `fish-ai`, data privacy is a priority. To learn more about what kind of information is sent, what is redacted and what settings you can tweak to safeguard data privacy, read the "Data Privacy" section in the [original repository](https://github.com/Realiserad/fish-ai).

## 🔨 Development

For development information, testing, and release procedures, see the "Development" section of the [original repository](https://github.com/Realiserad/fish-ai).

[**View the fish-ai Repository on GitHub**](https://github.com/Realiserad/fish-ai)
[![fish-ai](https://img.shields.io/github/stars/Realiserad/fish-ai?style=social)](https://github.com/Realiserad/fish-ai)
[![Time Spent](https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2FRealiserad%2Fd3ec7fdeecc35aeeb315b4efba493326%2Fraw%2Ffish-ai-git-estimate.json)](https://github.com/Realiserad/fish-ai)
[![Popularity](https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2FRealiserad%2Fd3ec7fdeecc35aeeb315b4efba493326%2Fraw%2Fpopularity.json)](https://github.com/Realiserad/fish-ai)
[![Donate Monero](https://img.shields.io/badge/Donate_XMR-grey?style=for-the-badge&logo=monero)](https://github.com/user-attachments/assets/07a2947f-6e5a-480f-990a-77204933411f)

# fish-ai: Supercharge Your Fish Shell with AI-Powered Productivity

**Tired of wrestling with the command line? `fish-ai` uses AI to make you a more productive shell user.** [Explore the fish-ai repository on GitHub](https://github.com/Realiserad/fish-ai)

## Key Features:

*   **Comment to Command & Vice Versa:** Effortlessly convert comments into shell commands and commands into clear explanations, eliminating the need for extensive man page searches.
*   **Intelligent Command Correction:**  Automatically fix typos and syntax errors in your commands, saving you time and frustration.
*   **AI-Powered Autocompletion:** Get smart suggestions for commands and options with a built-in fuzzy finder, boosting your efficiency.
*   **Keyboard-Centric Workflow:** Utilize configurable keyboard shortcuts for all functionality, ensuring a seamless, mouse-free experience.
*   **LLM Agnostic:** Connect to your preferred Large Language Model, including self-hosted options for maximum flexibility.
*   **Open Source & Customizable:**  Benefit from the transparency of open-source code and tailor `fish-ai` to your exact needs.
*   **Easy Installation & Updates:**  Install and manage with ease using [`fisher`](https://github.com/jorgebucaran/fisher).
*   **Cross-Platform Compatibility:**  Works seamlessly on macOS and popular Linux distributions.
*   **Non-Invasive Design:** Plays well with other plugins without telemetry or proprietary terminal requirements.

## How to Install:

1.  **Prerequisites:** Ensure you have `git` and either [`uv`](https://github.com/astral-sh/uv), or a supported Python version with `pip` and `venv` installed.
2.  **Install `fish-ai`:**
    ```shell
    fisher install realiserad/fish-ai
    ```
3.  **Configure your LLM:** Create a configuration file at `$XDG_CONFIG_HOME/fish-ai.ini` (or `~/.config/fish-ai.ini` if `$XDG_CONFIG_HOME` is not set) to specify your LLM provider.  Examples are provided below:

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

        (Generate a GitHub PAT [here](https://github.com/settings/tokens) - no specific permissions needed).
    *   **Self-hosted (e.g., Ollama):**
        ```ini
        [fish-ai]
        configuration = self-hosted

        [self-hosted]
        provider = self-hosted
        server = http://localhost:11434/v1  # Adjust port as needed
        model = llama3.3                # Example: llama3.3
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
    *   **Azure OpenAI, Mistral, Anthropic, Cohere, DeepSeek, Groq, Google**:  See the original README for details on setting up these providers.
4.  **Keyring:** Use `fish_ai_put_api_key` to securely store your API keys in your system's keyring.

## How to Use:

*   **Comment to Command:** Type a comment (starting with `#`) and press **Ctrl + P** to generate a command.
*   **Command to Comment:** Type a command and press **Ctrl + P** to get an explanation.
*   **Autocomplete:** Start typing a command and press **Ctrl + Space** for suggestions using [`fzf`](https://github.com/junegunn/fzf).  Refine the suggestions with **Ctrl + P** within `fzf`.
*   **Fix Command:** If a command fails, press **Ctrl + Space** to receive potential fixes.

## Additional Options:

*   **Customize Keybindings:** Modify the default **Ctrl + P** and **Ctrl + Space** shortcuts by setting `keymap_1` and `keymap_2` in your `fish-ai.ini` configuration using the output from `fish_key_reader`.
*   **Language:** Change the output language (e.g., `language = Swedish`).
*   **Temperature:** Adjust the randomness of the AI output (`temperature = 0.5`).
*   **Completions:** Change the number of suggested completions from the LLM (`completions = 10` and `refined_completions = 5`).
*   **History Integration:**  Enable history integration to personalize completions with `history_size = 5`. Consider using [`sponge`](https://github.com/meaningful-ooo/sponge) to remove bad commands from your history.
*   **Pipe Preview:** Send the output of a pipe to the LLM for command completion with `preview_pipe = True`.
*   **Progress Indicator:** Customize the progress indicator with `progress_indicator = wait...`.

## Data Privacy:

`fish-ai` transmits your OS name and the command-line buffer to the LLM. For codifying or completing commands, it sends file contents and the output of `<command> --help`.  You can control the amount of your command line history sent.  For enhanced privacy, consider a self-hosted LLM.  Sensitive information is redacted before being sent to the LLM. Redaction can be disabled with `redact = False`.

## Development:

Contribute to `fish-ai` by reading `ARCHITECTURE.md`. Use the `devcontainer.json` with GitHub Codespaces or Visual Studio Code.  Install a local copy of `fish-ai` using `fisher install .`. Enable debug logging by setting `debug = True` in your `fish-ai.ini`.  Run tests with `pytest`. Create releases by pushing a new tag.
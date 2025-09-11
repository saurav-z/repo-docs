[![Badge with time spent](https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2FRealiserad%2Fd3ec7fdeecc35aeeb315b4efba493326%2Fraw%2Ffish-ai-git-estimate.json)](https://github.com/Realiserad/fish-ai)
[![Popularity badge](https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2FRealiserad%2Fd3ec7fdeecc35aeeb315b4efba493326%2Fraw%2Fpopularity.json)
[![Donate XMR](https://img.shields.io/badge/Donate_XMR-grey?style=for-the-badge&logo=monero)](https://github.com/user-attachments/assets/07a2947f-6e5a-480f-990a-77204933411f)

# **fish-ai: Supercharge Your Fish Shell with AI-Powered Command Assistance**

**Tired of wrestling with command-line syntax?** `fish-ai` is the ultimate AI companion for your Fish shell, transforming how you interact with your terminal.  [Check out the original repo](https://github.com/Realiserad/fish-ai).

**Key Features:**

*   **Comment-to-Command & Vice-Versa:** Effortlessly translate comments into executable shell commands and vice versa.  Stop manually searching and copy/pasting and get the answers you need instantly.
*   **Intelligent Command Correction:**  Fixes typos and provides intelligent suggestions for broken commands, just like magic!
*   **Smart Autocompletion with Fuzzy Finder:** Get suggestions for commands, with a built-in fuzzy finder.
*   **Keyboard-Driven Workflow:**  Interact with `fish-ai` entirely via keyboard shortcuts (Ctrl+P and Ctrl+Space) for seamless integration.
*   **LLM Flexibility:** Connect to your preferred LLM, including self-hosted options and popular services.
*   **Open Source & Auditable:**  The code is available for review.
*   **Easy Installation with Fisher:** Get up and running fast.
*   **Broad Compatibility:** Works on macOS and major Linux distributions.
*   **Non-Intrusive:** Doesn't interfere with other plugins.
*   **Privacy Focused:** No telemetry, no forced terminal changes.

## **🎥 Demo**

![Demo](https://github.com/user-attachments/assets/86b61223-e568-4152-9e5e-d572b2b1385b)

## **🚀 Installation**

### **Prerequisites**

Ensure `git` and either [`uv`](https://github.com/astral-sh/uv) or a Python environment (with `pip` and `venv`) is installed.

### **Installation Steps**

1.  **Install `fish-ai` using Fisher:**
    ```shell
    fisher install realiserad/fish-ai
    ```
2.  **Configure `fish-ai.ini`:** Create a configuration file at `$XDG_CONFIG_HOME/fish-ai.ini` (or `~/.config/fish-ai.ini`). Choose your LLM provider:

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

        *   Generate a GitHub Personal Access Token (PAT) [here](https://github.com/settings/tokens) (no permissions needed).

    *   **Self-hosted (e.g., Ollama):**

        ```ini
        [fish-ai]
        configuration = local-llama

        [local-llama]
        provider = self-hosted
        model = llama3.3
        server = http://localhost:11434/v1
        ```

        *   Example using [Ollama](https://github.com/ollama/ollama) with [Llama 3.3 70B](https://ollama.com/library/llama3.3).
    *   **Other Providers:** Configuration examples are available for:
        *   [OpenRouter](https://openrouter.ai)
        *   [OpenAI](https://platform.openai.com)
        *   [Azure OpenAI](https://azure.microsoft.com/en-us/products/ai-services/openai-service)
        *   [Mistral](https://mistral.ai)
        *   [Anthropic](https://www.anthropic.com)
        *   [Cohere](https://cohere.com)
        *   [DeepSeek](https://www.deepseek.com)
        *   [Groq](https://groq.com)
        *   [Google Gemini](https://ai.google.com)

3.  **API Key Management:** Securely store API keys using `fish_ai_put_api_key`.

## **⌨️ How to Use**

*   **Comment to Command:** Type a comment (starting with `#`) and press **Ctrl + P** to generate a shell command. Press **Ctrl + P** again to refine.
*   **Command to Comment:** Type a command and press **Ctrl + P** to get an explanation.
*   **Command Autocompletion:** Begin typing and press **Ctrl + Space** to see suggestions via `fzf`.  Refine results by typing instructions and pressing **Ctrl + P** inside `fzf`.
*   **Fixing Commands:** If a command fails, press **Ctrl + Space** to get suggestions.

## **⚙️ Configuration Options**

*   **Change Keybindings:** Modify default shortcuts via `FISH_AI_KEYMAP_1` and `FISH_AI_KEYMAP_2` environment variables and `fish_key_reader`.
*   **Language:** Set `language = <language>` in `fish-ai.ini` for explanations in a different language.
*   **Temperature:** Adjust LLM randomness with `temperature = <value>` (0.0 to 1.0, or `None` for certain models).
*   **Completions:** Control suggestion counts via `completions = <number>` and `refined_completions = <number>`.
*   **Personalized Completions:**  Use `history_size = <number>` to include command history in prompts.
*   **Preview Pipes:** Enable with `preview_pipe = True` to use pipe output in completions.
*   **Progress Indicator:** Customize the progress indicator using `progress_indicator = <text>`.
*   **Context Switching:** Use `fish_ai_switch_context` to change LLM configurations.

## **🔒 Data Privacy**

`fish-ai` sends the OS name and command buffer to the LLM. It can also submit file contents, `help` output, command history (optional) and the previous command/output to the LLM. Using a self-hosted LLM is recommended for the highest privacy.  Sensitive information is redacted.  Redaction can be disabled with `redact = False`.

## **🛠️ Development**

*   Read [`ARCHITECTURE.md`](https://github.com/Realiserad/fish-ai/blob/main/ARCHITECTURE.md) before contributing.
*   Develop with GitHub Codespaces or VS Code's Dev Containers extension.
*   Install from a local copy: `fisher install .`
*   Enable debug logging: `debug = True` in `fish-ai.ini`.  Logging to a file is available with `log = <file path>`.
*   Run tests.
*   Create releases.
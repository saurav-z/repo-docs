![Fish AI - AI-Powered Shell](https://github.com/user-attachments/assets/86b61223-e568-4152-9e5e-d572b2b1385b)

# Fish AI: Supercharge Your Fish Shell with AI 🚀

**Tired of endless man pages and Stack Overflow searches?** Fish AI brings the power of AI directly to your Fish shell, helping you write, understand, and troubleshoot commands with ease.  [Explore Fish AI on GitHub](https://github.com/Realiserad/fish-ai)!

## Key Features:

*   **Comment-to-Command & Command-to-Comment:** Convert natural language comments into executable commands and vice-versa, saving you time and effort.
*   **Command Correction:**  Fix typos and errors in your commands, similar to `thefuck`.
*   **AI-Powered Autocompletion:** Get smart suggestions for your next command with a built-in fuzzy finder, making shell navigation a breeze.
*   **Keyboard-Driven:** Control everything with customizable, intuitive keyboard shortcuts.
*   **Flexible LLM Integration:**  Connect to your preferred LLM provider, including self-hosted options, OpenAI, and more.
*   **Open Source & Auditable:**  The code is open source and easy to read, allowing you to understand and contribute.
*   **Effortless Installation:**  Install and update seamlessly using `fisher`.
*   **Broad Compatibility:** Tested on macOS and popular Linux distributions.
*   **Non-Intrusive:**  Works alongside your existing Fish plugins without conflicts.
*   **Privacy Focused:** Offers redaction of sensitive information by default.

## How it Works:

Fish AI leverages Large Language Models (LLMs) to provide intelligent assistance directly within your Fish shell. Using two simple keyboard shortcuts, you can:

*   Transform comments into executable commands and vice versa.
*   Get smart autocompletions as you type.
*   Automatically suggest fixes for broken commands.

## Installation:

1.  **Prerequisites:** Ensure you have `git` and either `uv` or a supported Python version (with `pip` and `venv`) installed.
2.  **Install Fish AI:**
    ```shell
    fisher install realiserad/fish-ai
    ```
3.  **Configure Your LLM:**  Create a configuration file at `$XDG_CONFIG_HOME/fish-ai.ini` (or `~/.config/fish-ai.ini`) to specify your chosen LLM provider.  Example configurations are provided below.

## LLM Configuration Examples:

**[GitHub Models]** (Recommended for ease of use)
```ini
[fish-ai]
configuration = github

[github]
provider = self-hosted
server = https://models.inference.ai.azure.com
api_key = <paste GitHub PAT here>
model = gpt-4o-mini
```

**[Self-Hosted (Ollama Example)]**
```ini
[fish-ai]
configuration = local-llama

[local-llama]
provider = self-hosted
model = llama3.3
server = http://localhost:11434/v1
```

**[OpenRouter, OpenAI, Azure OpenAI, Mistral, Anthropic, Cohere, DeepSeek, Groq, Google]**  Configuration examples available in the original [README](https://github.com/Realiserad/fish-ai).

**Important:**  Store your API keys securely.  Use the `fish_ai_put_api_key` command to store keys in your keyring.

## Usage:

*   **Comment to Command:** Type a comment (starting with `#`) and press **Ctrl + P**.
*   **Command to Comment:** Type a command and press **Ctrl + P**.
*   **Autocomplete:** Begin typing and press **Ctrl + Space**.
*   **Suggest Fixes:** After a failed command, press **Ctrl + Space**.

## Additional Options:

Customize Fish AI's behavior by adding options to your `fish-ai.ini` configuration file.

*   **Change Keybindings:**  Modify the default keybindings (`Ctrl + P` and `Ctrl + Space`).
*   **Language:**  Set the `language` option to receive explanations in your preferred language.
*   **Temperature:** Control the randomness of LLM responses.
*   **Completions:**  Adjust the number of suggestions provided.
*   **History Size:**  Personalize completions using command-line history.
*   **Preview Pipes:**  Enable output previewing for pipe commands.
*   **Progress Indicator:** Customize the indicator displayed while waiting for LLM responses.
*   **Context Switching:** Use `fish_ai_switch_context` to quickly switch between different configuration sections.

## Data Privacy:

Fish AI prioritizes your privacy. Learn more about data handling and redaction of sensitive information in the [original README](https://github.com/Realiserad/fish-ai).

## Development:

If you're interested in contributing, see the [ARCHITECTURE.md](https://github.com/Realiserad/fish-ai/blob/main/ARCHITECTURE.md) for details.  The repository includes a `devcontainer.json` for easy setup with GitHub Codespaces or VS Code.

## Get Started Today!

Fish AI is your intelligent companion for the Fish shell. Install it now and streamline your workflow!
```

**Key improvements and SEO considerations:**

*   **Strong Hook:** The opening sentence immediately grabs attention and highlights the primary benefit.
*   **Clear Headings:**  Uses standard Markdown headings for readability and structure.
*   **Bulleted Key Features:**  Provides a concise and scannable overview of the plugin's capabilities.
*   **Keyword Optimization:**  Includes relevant keywords throughout (e.g., "Fish shell," "AI," "autocompletion," "command correction").
*   **Action-Oriented Language:**  Uses phrases like "Supercharge Your Fish Shell" and "Get Started Today!" to encourage engagement.
*   **Direct Link to Repo:**  Includes a prominent link to the GitHub repository.
*   **Concise Summaries:**  Condenses lengthy explanations into brief, easy-to-understand sections.
*   **Clear Installation Instructions:**  Provides step-by-step installation guidance.
*   **Configuration examples are kept** to demonstrate how easy it is to configure the plugin.
*   **Focuses on User Benefit:**  Emphasizes the advantages of using Fish AI (e.g., saving time, reducing errors).
*   **Modern, Clean Formatting:**  Uses Markdown for a visually appealing and organized presentation.
*   **Clear Call to Action:** Encourages users to install and try the plugin.
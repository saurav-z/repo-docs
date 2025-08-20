# WilmerAI: Expertly Routing Your Language Model Inferences

**Unlock the power of complex, multi-LLM workflows with WilmerAI, a flexible and customizable application for connecting your front-end to the latest AI models.** [Visit the original repository](https://github.com/SomeOddCodeGuy/WilmerAI) to get started!

---

## Key Features:

*   **Advanced Prompt Routing:** Direct prompts to specific categories or personas, ensuring optimal results.
*   **Customizable Workflows:** Tailor your AI interactions with flexible, multi-step processes.
*   **Multi-LLM Orchestration:** Leverage multiple LLMs within a single prompt for superior results.
*   **Offline Wikipedia Integration:** Enhance responses with factual data using the Offline Wikipedia API.
*   **Conversation Memory:** Maintain context over long interactions with continually generated chat summaries.
*   **Model Hotswapping:** Maximize VRAM usage by hotswapping LLMs using Ollama.
*   **Custom Presets:** Easily configure and customize parameters for different LLM types.
*   **Vision Multi-Modal Support:** Process images with Ollama, even if your LLM doesn't support images directly.
*   **Mid-Workflow Conditional Logic:** Create dynamic workflows with branching based on LLM responses.
*   **Tool Integration:** Use tool integration using MCPO.

---

## Getting Started

### Disclaimer

This is an actively developed personal project and may contain bugs. The software is provided "as-is" without warranty. Views and methodologies within are those of the maintainer and contributors.

### What is WilmerAI?

WilmerAI acts as an intermediary between your front-end application and your chosen Large Language Models (LLMs). It exposes API endpoints compatible with OpenAI and Ollama, while seamlessly connecting to various LLM APIs like OpenAI, KoboldCpp, and Ollama on the backend.

Think of it as a sophisticated workflow manager. You send a prompt to WilmerAI, which then processes it through a series of customizable workflows. Each workflow may involve multiple LLMs and tools, ultimately returning a comprehensive response to you.

### Installation

**Prerequisites:** Python 3.10 or 3.12

**Option 1: Using Provided Scripts (Recommended)**
   *   **Windows:** Run the `.bat` file.
   *   **macOS:** Run the `.sh` file.

**Option 2: Manual Installation**

1.  Install dependencies: `pip install -r requirements.txt`
2.  Run the server: `python server.py`

### Configuration

WilmerAI is configured using JSON files located in the `Public` folder. Copy your `Public` folder to preserve your settings when updating.

*   **Endpoints:** Define your LLM API connections in the `Public/Configs/Endpoints` folder.
*   **Users:** Create a user configuration file in `Public/Configs/Users` and specify it in the `_current-user.json` file or use the --User argument.
*   **Workflows:** Customize your workflows in the `Public/Workflows` folder within the user's specific directory.

**For detailed setup instructions, please consult the original project's [README](https://github.com/SomeOddCodeGuy/WilmerAI).**

---

## Key Concepts

*   **Workflows:** Chains of nodes that orchestrate LLM calls.
*   **Nodes:** Individual steps within a workflow, each with specific configurations.
*   **Variables:** Dynamic placeholders for data within prompts, supporting inter-node communication.
*   **Memories:** Long-term, rolling summary, or vector-based storage of conversation history for context.
*   **Presets:** Customizable configurations for LLM parameters.

---

## Advanced Features

*   **Workflow Locks:** Prevent race conditions during concurrent operations with workflow locks.
*   **Image Processing:** Process images through Ollama, providing rich context to your LLMs.
*   **Conditional Workflows:** Dynamically route prompts based on LLM responses.
*   **Memory System:**  The new memory system is designed to store both file-based memories and the more powerful, searchable Vector Memories (via SQLite). 
*   **Parallel Processing**:  Allows for using more than one LLM simultaneously when generating and processing summaries and memories.
*   **Custom Node Functionality:** Add custom logic by using the Python Module Caller.

---

## Community and Support

*   **Contact:** WilmerAI.Project@gmail.com
*   **License:**  GNU General Public License (GPLv3)
*   **Third Party Libraries:**  Flask, requests, scikit-learn, urllib3, jinja2, pillow.  See the README in the ThirdParty-Licenses folder for more information.
---
# NotebookLlama: Your Open-Source AI-Powered Notebook Alternative 🦙

**NotebookLlama** offers a compelling, open-source alternative to NotebookLM, empowering you to harness the power of AI for your notes and research, all backed by [LlamaCloud](https://cloud.llamaindex.ai?utm_source=demo&utm_medium=notebookLM). [Check out the original repository](https://github.com/run-llama/notebookllama) to get started!

[![License](https://img.shields.io/github/license/run-llama/notebookllama?color=blue)](https://github.com/run-llama/notebookllama/blob/main/LICENSE)
[![Stars](https://img.shields.io/github/stars/run-llama/notebookllama?color=yellow)](https://github.com/run-llama/notebookllama/stargazers)
[![Issues](https://img.shields.io/github/issues/run-llama/notebookllama?color=orange)](https://github.com/run-llama/notebookllama/issues)
[![MseeP.ai Security Assessment Badge](https://mseep.net/pr/run-llama-notebookllama-badge.png)](https://mseep.ai/app/run-llama-notebookllama)

## Key Features

*   **Open-Source:** Fully open-source, providing transparency and community-driven development.
*   **AI-Powered:** Leverages AI to enhance your note-taking and research capabilities.
*   **LlamaCloud Integration:** Backed by LlamaCloud for robust performance and scalability.
*   **Customizable:** Allows for custom embedding models and configurations.

## Getting Started

### Prerequisites

This project utilizes `uv` for dependency management. Ensure you have `uv` installed before proceeding.

**Install `uv`:**

*   **macOS and Linux:**
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
*   **Windows:**
    ```powershell
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

For detailed installation instructions, refer to the official `uv` [documentation](https://docs.astral.sh/uv/getting-started/installation/).

### Installation & Setup

Follow these steps to get NotebookLlama up and running:

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/run-llama/notebookllama
    cd notebookllama/
    ```

2.  **Install Dependencies:**
    ```bash
    uv sync
    ```

3.  **Configure API Keys:**

    *   Create a `.env` file:
        ```bash
        mv .env.example .env
        ```
    *   Add your API keys to the `.env` file:
        *   `OPENAI_API_KEY`: [OpenAI Platform](https://platform.openai.com/api-keys)
        *   `ELEVENLABS_API_KEY`: [ElevenLabs Settings](https://elevenlabs.io/app/settings/api-keys)
        *   `LLAMACLOUD_API_KEY`: [LlamaCloud Dashboard](https://cloud.llamaindex.ai?utm_source=demo&utm_medium=notebookLM)

4.  **Activate the Virtual Environment:**

    *   **(mac/unix):**
        ```bash
        source .venv/bin/activate
        ```
    *   **(Windows):**
        ```bash
        .\.venv\Scripts\activate
        ```

5.  **Create LlamaCloud Agent & Pipeline:**

    *   Create the data extraction agent:
        ```bash
        uv run tools/create_llama_extract_agent.py
        ```
    *   Run the interactive setup wizard:
        *   **Quick Start (Default OpenAI):** Select "With Default Settings" for the fastest setup.
        *   **Advanced (Custom Embedding Models):** Select "With Custom Settings" and follow on-screen instructions.
        ```bash
        uv run tools/create_llama_cloud_index.py
        ```

6.  **Launch Backend Services:**
    ```bash
    docker compose up -d
    ```

7.  **Run the Application:**

    *   Run the MCP server:
        ```bash
        uv run src/notebookllama/server.py
        ```
    *   In a **new terminal window**, launch the Streamlit app:
        ```bash
        streamlit run src/notebookllama/Home.py
        ```
    *   Access the app at `http://localhost:8501/`.

    > [!IMPORTANT]
    >
    > _You might need to install `ffmpeg` if you do not have it installed already_

---

### Contributing

Contribute to the project by following the [guidelines](./CONTRIBUTING.md).

### License

This project is licensed under the [MIT License](./LICENSE).
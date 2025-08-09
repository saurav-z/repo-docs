# NotebookLlaMa: Open-Source AI-Powered Knowledge Assistant 🦙

**NotebookLlaMa** is a fully open-source alternative to NotebookLM, providing an AI-powered knowledge assistant experience.  Explore the power of NotebookLlaMa on [GitHub](https://github.com/run-llama/notebookllama).

[![License](https://img.shields.io/github/license/run-llama/notebookllama?color=blue)](https://github.com/run-llama/notebookllama/blob/main/LICENSE)
[![Stars](https://img.shields.io/github/stars/run-llama/notebookllama?color=yellow)](https://github.com/run-llama/notebookllama/stargazers)
[![Issues](https://img.shields.io/github/issues/run-llama/notebookllama?color=orange)](https://github.com/run-llama/notebookllama/issues)
[![MseeP.ai Security Assessment Badge](https://mseep.net/pr/run-llama-notebookllama-badge.png)](https://mseep.ai/app/run-llama-notebookllama)

## Key Features

*   **Open Source:**  Freely available and customizable.
*   **AI-Powered:** Leverages the power of language models to assist with knowledge retrieval and summarization.
*   **LlamaCloud Integration:**  Backed by [LlamaCloud](https://cloud.llamaindex.ai?utm_source=demo&utm_medium=notebookLM) for enhanced performance and capabilities.
*   **Easy Setup:** Straightforward installation with `uv` for dependency management.
*   **Customization:**  Supports custom embedding models for advanced users.

## Prerequisites

This project uses `uv` for dependency management.  Ensure you have `uv` installed before proceeding.

**Installation:**

*   **macOS and Linux:**

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
*   **Windows:**

    ```powershell
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

For additional installation options, consult the official `uv` [documentation](https://docs.astral.sh/uv/getting-started/installation/).

## Getting Started

Follow these steps to set up and run NotebookLlaMa:

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

    *   Add your API keys to `.env`:

        *   `OPENAI_API_KEY`: [OpenAI Platform](https://platform.openai.com/api-keys)
        *   `ELEVENLABS_API_KEY`: [ElevenLabs Settings](https://elevenlabs.io/app/settings/api-keys)
        *   `LLAMACLOUD_API_KEY`: [LlamaCloud Dashboard](https://cloud.llamaindex.ai?utm_source=demo&utm_medium=notebookLM)

4.  **Activate the Virtual Environment:**

    *   **mac/unix:**

        ```bash
        source .venv/bin/activate
        ```

    *   **Windows:**

        ```bash
        .\.venv\Scripts\activate
        ```

5.  **Create LlamaCloud Agent & Pipeline:**

    *   Create the data extraction agent:

        ```bash
        uv run tools/create_llama_extract_agent.py
        ```

    *   Configure the index pipeline with the interactive setup wizard:

        *   **Quick Start (Default OpenAI):** Select  **"With Default Settings"** for rapid setup using OpenAI's `text-embedding-3-small`.
        *   **Advanced (Custom Embedding Models):** Select  **"With Custom Settings"** and follow on-screen instructions.

        Run the wizard:

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

        > [!IMPORTANT]
        >
        > _You might need to install `ffmpeg` if you do not have it installed already_

    *   Access the app at `http://localhost:8501/`.

## Contributing

Contribute to NotebookLlaMa following the [contribution guidelines](./CONTRIBUTING.md).

## License

NotebookLlaMa is licensed under the [MIT License](./LICENSE).
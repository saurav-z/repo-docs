# NotebookLlama: An Open-Source Alternative to NotebookLM 🦙

Tired of closed-source AI notebooks? **NotebookLlama is a fully open-source alternative to Google's NotebookLM, empowering you with the ability to build and manage your knowledge bases with the power of open-source and LlamaIndex.**

[![License](https://img.shields.io/github/license/run-llama/notebookllama?color=blue)](https://github.com/run-llama/notebookllama/blob/main/LICENSE)
[![Stars](https://img.shields.io/github/stars/run-llama/notebookllama?color=yellow)](https://github.com/run-llama/notebookllama/stargazers)
[![Issues](https://img.shields.io/github/issues/run-llama/notebookllama?color=orange)](https://github.com/run-llama/notebookllama/issues)
[![MseeP.ai Security Assessment Badge](https://mseep.net/pr/run-llama-notebookllama-badge.png)](https://mseep.ai/app/run-llama-notebookllama)

**[View the original repository on GitHub](https://github.com/run-llama/notebookllama)**

## Key Features

*   **Open-Source:** Build and customize your AI notebook experience.
*   **Powered by LlamaCloud:** Leverages the power and flexibility of LlamaIndex for robust knowledge base management.
*   **Easy Setup:** Simplified installation and configuration process.
*   **Customizable:** Configure your AI notebook with different models and settings.
*   **Integration with OpenAI and ElevenLabs:** Support for popular API integrations.

## Getting Started

### Prerequisites

This project uses `uv` to manage dependencies. Ensure you have `uv` installed before proceeding.

**Installation Instructions:**

*   **macOS and Linux:**

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

*   **Windows:**

    ```powershell
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

For more installation options, see the `uv` [official documentation](https://docs.astral.sh/uv/getting-started/installation/).

### Setup

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
    *   Create a `.env` file: `mv .env.example .env`
    *   Populate the `.env` file with your API keys:
        *   `OPENAI_API_KEY`: Find it on the [OpenAI Platform](https://platform.openai.com/api-keys)
        *   `ELEVENLABS_API_KEY`: Find it on [ElevenLabs Settings](https://elevenlabs.io/app/settings/api-keys)
        *   `LLAMACLOUD_API_KEY`: Find it on the [LlamaCloud Dashboard](https://cloud.llamaindex.ai?utm_source=demo&utm_medium=notebookLM)

4.  **Activate the Virtual Environment:**
    *   **(mac/unix):**  `source .venv/bin/activate`
    *   **(Windows):**  `.\.venv\Scripts\activate`

5.  **Create LlamaCloud Agent & Pipeline:**
    *   Create the data extraction agent:
        ```bash
        uv run tools/create_llama_extract_agent.py
        ```
    *   Configure your index pipeline using the interactive setup wizard:
        *   **Quick Start (Default OpenAI):** For the fastest setup, select "With Default Settings" when prompted. This will use OpenAI's `text-embedding-3-small`.
        *   **Advanced (Custom Embedding Models):** Select "With Custom Settings" and follow the on-screen instructions.

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

    *   In a new terminal, launch the Streamlit app:

        ```bash
        streamlit run src/notebookllama/Home.py
        ```

    *   Access the app at `http://localhost:8501/`.

    >   [!IMPORTANT]
    >   _You might need to install `ffmpeg` if you do not have it installed already._

## Contributing

Contributions are welcome! Please review the [contributing guidelines](./CONTRIBUTING.md) for more information.

## License

NotebookLlama is licensed under the [MIT License](./LICENSE).
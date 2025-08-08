# NotebookLlaMa: Your Open-Source AI Notebook Assistant 🦙

**Unlock the power of AI-assisted note-taking with NotebookLlaMa, a fully open-source alternative to NotebookLM!**  [View the project on GitHub](https://github.com/run-llama/notebookllama).

[![License](https://img.shields.io/github/license/run-llama/notebookllama?color=blue)](https://github.com/run-llama/notebookllama/blob/main/LICENSE)
[![Stars](https://img.shields.io/github/stars/run-llama/notebookllama?color=yellow)](https://github.com/run-llama/notebookllama/stargazers)
[![Issues](https://img.shields.io/github/issues/run-llama/notebookllama?color=orange)](https://github.com/run-llama/notebookllama/issues)
<br>
[![MseeP.ai Security Assessment Badge](https://mseep.net/pr/run-llama-notebookllama-badge.png)](https://mseep.ai/app/run-llama-notebookllama)

## Key Features

*   **Open-Source Alternative:** A fully open-source option for AI-powered note-taking, providing flexibility and control.
*   **Backed by LlamaCloud:** Integrates seamlessly with LlamaCloud for enhanced functionality and performance.
*   **Customizable:** Supports various embedding models, giving you the freedom to tailor the experience to your needs.
*   **Easy Setup:** Simple installation process leveraging `uv` for dependency management.
*   **Interactive Setup Wizard:** Streamlined configuration using a wizard for backend agent and pipeline setup.
*   **Dockerized Backend:**  Uses Docker Compose for simplified deployment of essential services (Postgres, Jaeger).
*   **Streamlit Frontend:**  Provides a user-friendly interface built with Streamlit.

## Prerequisites

This project utilizes `uv` for dependency management. Ensure you have `uv` installed before proceeding.

**Installation Instructions:**

*   **macOS and Linux:**
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
*   **Windows:**
    ```powershell
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

For comprehensive installation details, refer to the official `uv` [documentation](https://docs.astral.sh/uv/getting-started/installation/).

## Getting Started

Follow these steps to get NotebookLlaMa up and running:

**1. Clone the Repository:**

```bash
git clone https://github.com/run-llama/notebookllama
cd notebookllama/
```

**2. Install Dependencies:**

```bash
uv sync
```

**3. Configure API Keys:**

*   Create a `.env` file:

    ```bash
    mv .env.example .env
    ```

*   Add your API keys to the `.env` file:

    *   `OPENAI_API_KEY`: [OpenAI Platform](https://platform.openai.com/api-keys)
    *   `ELEVENLABS_API_KEY`: [ElevenLabs Settings](https://elevenlabs.io/app/settings/api-keys)
    *   `LLAMACLOUD_API_KEY`: [LlamaCloud Dashboard](https://cloud.llamaindex.ai?utm_source=demo&utm_medium=notebookLM)

**4. Activate the Virtual Environment:**

*   **macOS/Linux:**

    ```bash
    source .venv/bin/activate
    ```

*   **Windows:**

    ```bash
    .\.venv\Scripts\activate
    ```

**5. Create LlamaCloud Agent & Pipeline:**

*   Create the data extraction agent:

    ```bash
    uv run tools/create_llama_extract_agent.py
    ```

*   Run the interactive setup wizard:

    ```bash
    uv run tools/create_llama_cloud_index.py
    ```

    *   **Quick Start (Default OpenAI):** Select **"With Default Settings"** for the fastest setup using OpenAI's `text-embedding-3-small`.
    *   **Advanced (Custom Embedding Models):** Select **"With Custom Settings"** for different embedding models.

**6. Launch Backend Services:**

```bash
docker compose up -d
```

**7. Run the Application:**

*   Run the MCP server:

    ```bash
    uv run src/notebookllama/server.py
    ```

*   In a *new terminal window*, launch the Streamlit app:

    ```bash
    streamlit run src/notebookllama/Home.py
    ```

    Ensure you have `ffmpeg` installed if you don't already.

*   Access the app at `http://localhost:8501/`.

## Contributing

Contributions are welcome! Please review the [contribution guidelines](./CONTRIBUTING.md).

## License

NotebookLlaMa is provided under the [MIT License](./LICENSE).
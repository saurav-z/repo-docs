# NotebookLlama: Open-Source NotebookLM Alternative

**Unleash the power of your documents with NotebookLlama, a fully open-source alternative to NotebookLM!** 🦙 Explore the world of smart document analysis and interaction, powered by LlamaIndex and the innovative LlamaCloud platform.  [Check out the original repository on GitHub](https://github.com/run-llama/notebookllama).

[![License](https://img.shields.io/github/license/run-llama/notebookllama?color=blue)](https://github.com/run-llama/notebookllama/blob/main/LICENSE)
[![Stars](https://img.shields.io/github/stars/run-llama/notebookllama?color=yellow)](https://github.com/run-llama/notebookllama/stargazers)
[![Issues](https://img.shields.io/github/issues/run-llama/notebookllama?color=orange)](https://github.com/run-llama/notebookllama/issues)
[![MseeP.ai Security Assessment Badge](https://mseep.net/pr/run-llama-notebookllama-badge.png)](https://mseep.ai/app/run-llama-notebookllama)

## Key Features

*   **Open-Source & Free:** Leverage a powerful, open-source alternative to NotebookLM without any licensing costs.
*   **LlamaCloud Integration:** Seamlessly integrates with LlamaCloud for enhanced performance and capabilities.
*   **Easy Setup:** Simple installation and configuration using `uv` for dependency management.
*   **Customizable:** Configure your own API keys for OpenAI, ElevenLabs, and LlamaCloud.
*   **Flexible Indexing:** Create and customize indexing pipelines to fit your specific needs using the tools provided.
*   **Local Development:** Run the application locally with Docker Compose for a fully functional development environment.

## Prerequisites

Before you get started, ensure you have `uv` installed for dependency management.

**macOS and Linux:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

For more installation options, please see `uv`'s [official documentation](https://docs.astral.sh/uv/getting-started/installation/).

## Getting Started

Follow these steps to get NotebookLlama up and running:

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

*   Create a `.env` file by renaming the example:

    ```bash
    mv .env.example .env
    ```

*   Add your API keys to the `.env` file:

    *   `OPENAI_API_KEY`: Find it [on OpenAI Platform](https://platform.openai.com/api-keys)
    *   `ELEVENLABS_API_KEY`: Find it [on ElevenLabs Settings](https://elevenlabs.io/app/settings/api-keys)
    *   `LLAMACLOUD_API_KEY`: Find it [on LlamaCloud Dashboard](https://cloud.llamaindex.ai?utm_source=demo&utm_medium=notebookLM)

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

*   Run the interactive setup wizard to configure your index pipeline:

    *   **Quick Start (Default OpenAI):** Select "With Default Settings" for the quickest setup.
    *   **Advanced (Custom Embedding Models):** Select "With Custom Settings" for model customization.

    Run the wizard:

    ```bash
    uv run tools/create_llama_cloud_index.py
    ```

**6. Launch Backend Services:**

This command will start the required Postgres and Jaeger containers.

```bash
docker compose up -d
```

**7. Run the Application:**

*   Run the MCP server:

    ```bash
    uv run src/notebookllama/server.py
    ```

*   In a new terminal, launch the Streamlit app:

    ```bash
    streamlit run src/notebookllama/Home.py
    ```

    > [!IMPORTANT]
    >
    > _You might need to install `ffmpeg` if you do not have it installed already_

*   Access the app at `http://localhost:8501/`.

## Contributing

Contribute to the project by following the [guidelines](./CONTRIBUTING.md).

## License

This project is provided under an [MIT License](./LICENSE).
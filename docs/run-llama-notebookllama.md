# NotebookLlaMa: Your Open-Source AI-Powered Notebook Companion

**NotebookLlaMa** is a fully open-source alternative to NotebookLM, empowering you with AI-driven insights for your documents, and is backed by [LlamaCloud](https://cloud.llamaindex.ai?utm_source=demo&utm_medium=notebookLM). ([Original Repository](https://github.com/run-llama/notebookllama))

[![License](https://img.shields.io/github/license/run-llama/notebookllama?color=blue)](https://github.com/run-llama/notebookllama/blob/main/LICENSE)
[![Stars](https://img.shields.io/github/stars/run-llama/notebookllama?color=yellow)](https://github.com/run-llama/notebookllama/stargazers)
[![Issues](https://img.shields.io/github/issues/run-llama/notebookllama?color=orange)](https://github.com/run-llama/notebookllama/issues)
[![MseeP.ai Security Assessment Badge](https://mseep.net/pr/run-llama-notebookllama-badge.png)](https://mseep.ai/app/run-llama-notebookllama)

## Key Features

*   **Open-Source:** Fully accessible and customizable.
*   **AI-Powered:** Leverages the power of AI for document analysis and insights.
*   **LlamaCloud Integration:** Backed by LlamaCloud for enhanced performance and features.
*   **Easy Setup:** Simplified installation process with clear instructions.
*   **Flexible Configuration:** Supports OpenAI and custom embedding models.

## Prerequisites

This project utilizes `uv` for dependency management. Ensure you have `uv` installed before proceeding.

**Installation (macOS and Linux):**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Installation (Windows):**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

For more installation options, refer to the [official `uv` documentation](https://docs.astral.sh/uv/getting-started/installation/).

## Getting Started

Follow these steps to set up and run NotebookLlaMa:

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
    *   `OPENAI_API_KEY`: Obtain from [OpenAI Platform](https://platform.openai.com/api-keys)
    *   `ELEVENLABS_API_KEY`: Obtain from [ElevenLabs Settings](https://elevenlabs.io/app/settings/api-keys)
    *   `LLAMACLOUD_API_KEY`: Obtain from [LlamaCloud Dashboard](https://cloud.llamaindex.ai?utm_source=demo&utm_medium=notebookLM)

**4. Activate the Virtual Environment:**

*   **(mac/unix)**
    ```bash
    source .venv/bin/activate
    ```
*   **(Windows)**
    ```bash
    .\.venv\Scripts\activate
    ```

**5. Create LlamaCloud Agent & Pipeline:**

*   Create the data extraction agent:

    ```bash
    uv run tools/create_llama_extract_agent.py
    ```

*   Run the index pipeline setup wizard:

    *   **⚡ Quick Start (Default OpenAI):** Select "With Default Settings" for the fastest setup.
    *   **🧠 Advanced (Custom Embedding Models):** Choose "With Custom Settings" and follow the instructions.

    ```bash
    uv run tools/create_llama_cloud_index.py
    ```

**6. Launch Backend Services:**

```bash
docker compose up -d
```

**7. Run the Application:**

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
    > *You may need to install `ffmpeg` if it's not already installed.*

*   Access the app at `http://localhost:8501/`.

## Contributing

Contribute to the project by following the guidelines in the [CONTRIBUTING.md](./CONTRIBUTING.md) file.

## License

This project is licensed under the [MIT License](./LICENSE).
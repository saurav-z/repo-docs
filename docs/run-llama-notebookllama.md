# NotebookLlaMa: Your Open-Source AI Notebook Assistant 🦙

**Unlock the power of AI in your notes with NotebookLlaMa, a fully open-source alternative to NotebookLM, built with the flexibility of [LlamaIndex](https://github.com/run-llama/notebookllama).**

[![License](https://img.shields.io/github/license/run-llama/notebookllama?color=blue)](https://github.com/run-llama/notebookllama/blob/main/LICENSE)
[![Stars](https://img.shields.io/github/stars/run-llama/notebookllama?color=yellow)](https://github.com/run-llama/notebookllama/stargazers)
[![Issues](https://img.shields.io/github/issues/run-llama/notebookllama?color=orange)](https://github.com/run-llama/notebookllama/issues)
[![MseeP.ai Security Assessment Badge](https://mseep.net/pr/run-llama-notebookllama-badge.png)](https://mseep.ai/app/run-llama-notebookllama)

## Key Features

*   **Open-Source:** Enjoy the freedom of a fully open-source project.
*   **Alternative to NotebookLM:** Provides similar functionality and aims to be a powerful alternative.
*   **Powered by LlamaIndex and LlamaCloud:** Leverages robust AI frameworks for advanced capabilities.
*   **Customizable:** Adapt the AI model to your use case through custom embedding models.

## Prerequisites

This project uses `uv` to manage dependencies. Make sure you have `uv` installed.

**Install `uv`:**

*   **macOS and Linux:**

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

*   **Windows:**

    ```powershell
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

For more install options, see `uv`'s [official documentation](https://docs.astral.sh/uv/getting-started/installation/).

## Getting Started

Follow these steps to get NotebookLlaMa up and running:

**1. Clone the Repository**

```bash
git clone https://github.com/run-llama/notebookllama
cd notebookllama/
```

**2. Install Dependencies**

```bash
uv sync
```

**3. Configure API Keys**

*   Create a `.env` file:

    ```bash
    mv .env.example .env
    ```

*   Add your API keys to the `.env` file:
    *   `OPENAI_API_KEY`: Obtain from [OpenAI Platform](https://platform.openai.com/api-keys)
    *   `ELEVENLABS_API_KEY`: Obtain from [ElevenLabs Settings](https://elevenlabs.io/app/settings/api-keys)
    *   `LLAMACLOUD_API_KEY`: Obtain from [LlamaCloud Dashboard](https://cloud.llamaindex.ai?utm_source=demo&utm_medium=notebookLM)

**4. Activate the Virtual Environment**

*   **mac/unix:**

    ```bash
    source .venv/bin/activate
    ```

*   **Windows:**

    ```bash
    .\.venv\Scripts\activate
    ```

**5. Create LlamaCloud Agent & Pipeline**

*   Create the data extraction agent:

    ```bash
    uv run tools/create_llama_extract_agent.py
    ```

*   Run the interactive setup wizard to configure your index pipeline:

    ```bash
    uv run tools/create_llama_cloud_index.py
    ```

    *   **Quick Start (Default OpenAI):** Select **"With Default Settings"** for a quick setup using OpenAI's `text-embedding-3-small`.
    *   **Advanced (Custom Embedding Models):** Select **"With Custom Settings"** to use different embedding models.

**6. Launch Backend Services**

```bash
docker compose up -d
```

**7. Run the Application**

*   Run the **MCP** server:

    ```bash
    uv run src/notebookllama/server.py
    ```

*   In a **new terminal window**, launch the Streamlit app:

    ```bash
    streamlit run src/notebookllama/Home.py
    ```

    *   You may need to install `ffmpeg` if you don't have it installed already.
    *   Access the app at `http://localhost:8501/`.

## Contributing

Contributions are welcome! Please refer to the [CONTRIBUTING.md](./CONTRIBUTING.md) guidelines.

## License

This project is licensed under the [MIT License](./LICENSE).

**[Back to the project repository](https://github.com/run-llama/notebookllama)**
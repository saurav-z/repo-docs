# NotebookLlaMa: Open-Source Knowledge Base for Your Documents

**NotebookLlaMa** is a fully open-source alternative to NotebookLM, empowering you to build a powerful knowledge base from your documents.

[Visit the original repository on GitHub](https://github.com/run-llama/notebookllama)

<p align="center">
  <img src="https://github.com/user-attachments/assets/7e9cca45-8a4c-4dfa-98d2-2cef147422f2" alt="NotebookLlaMa Logo" width="200"/>
</p>

<p align="center">
  <a href="https://github.com/run-llama/notebookllama/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/run-llama/notebookllama?color=blue"></a>
  <a href="https://github.com/run-llama/notebookllama/stargazers"><img alt="Stars" src="https://img.shields.io/github/stars/run-llama/notebookllama?color=yellow"></a>
  <a href="https://github.com/run-llama/notebookllama/issues"><img alt="Issues" src="https://img.shields.io/github/issues/run-llama/notebookllama?color=orange"></a>
  <br>
  <a href="https://mseep.ai/app/run-llama-notebookllama"><img alt="MseeP.ai Security Assessment Badge" src="https://mseep.net/pr/run-llama-notebookllama-badge.png"></a>
</p>

## Key Features

*   **Open-Source:** Freedom to use, modify, and distribute the code.
*   **Knowledge Base Creation:** Build a searchable knowledge base from your documents.
*   **Powered by LlamaCloud:** Leverages the power of LlamaCloud for efficient indexing and retrieval.
*   **Customizable:** Configure with your preferred embedding models for optimal performance.
*   **Easy Setup:** Streamlined setup process to get you up and running quickly.

## Prerequisites

This project uses `uv` to manage dependencies. Ensure you have `uv` installed before proceeding.

**Installation:**

*   **macOS and Linux:**

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

*   **Windows:**

    ```powershell
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

For more installation options, consult `uv`'s [official documentation](https://docs.astral.sh/uv/getting-started/installation/).

## Getting Started

Follow these steps to set up and run NotebookLlaMa:

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

*   Create a `.env` file by renaming the example file:

    ```bash
    mv .env.example .env
    ```

*   Open the `.env` file and add your API keys:
    *   `OPENAI_API_KEY`: Find it [on OpenAI Platform](https://platform.openai.com/api-keys)
    *   `ELEVENLABS_API_KEY`: Find it [on ElevenLabs Settings](https://elevenlabs.io/app/settings/api-keys)
    *   `LLAMACLOUD_API_KEY`: Find it [on LlamaCloud Dashboard](https://cloud.llamaindex.ai?utm_source=demo&utm_medium=notebookLM)

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

    > **⚡ Quick Start (Default OpenAI):** For the fastest setup, select **"With Default Settings"** when prompted. This will automatically create a pipeline using OpenAI's `text-embedding-3-small` embedding model.
    >
    > **🧠 Advanced (Custom Embedding Models):** To use a different embedding model, select **"With Custom Settings"** and follow the on-screen instructions.

    Run the wizard:

    ```bash
    uv run tools/create_llama_cloud_index.py
    ```

**6. Launch Backend Services**

```bash
docker compose up -d
```

**7. Run the Application**

*   First, run the **MCP** server:

    ```bash
    uv run src/notebookllama/server.py
    ```

*   Then, in a **new terminal window**, launch the Streamlit app:

    ```bash
    streamlit run src/notebookllama/Home.py
    ```

> [!IMPORTANT]
>
> _You might need to install `ffmpeg` if you do not have it installed already._

Explore the app at `http://localhost:8501/`.

## Contributing

Contribute to this project following the [guidelines](./CONTRIBUTING.md).

## License

This project is provided under an [MIT License](./LICENSE).
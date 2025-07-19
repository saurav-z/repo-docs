# NotebookLlaMa: Your Open-Source AI Notebook Assistant 🦙

**Unlock the power of AI-powered note-taking with NotebookLlaMa, a fully open-source alternative to NotebookLM.**  ([View the original repository](https://github.com/run-llama/notebookllama))

<p align="center">
  <img src="https://github.com/user-attachments/assets/7e9cca45-8a4c-4dfa-98d2-2cef147422f2" alt="NotebookLlaMa Screenshot" width="600">
</p>

[![License](https://img.shields.io/github/license/run-llama/notebookllama?color=blue)](https://github.com/run-llama/notebookllama/blob/main/LICENSE)
[![Stars](https://img.shields.io/github/stars/run-llama/notebookllama?color=yellow)](https://github.com/run-llama/notebookllama/stargazers)
[![Issues](https://img.shields.io/github/issues/run-llama/notebookllama?color=orange)](https://github.com/run-llama/notebookllama/issues)
<br>
[![MseeP.ai Security Assessment Badge](https://mseep.net/pr/run-llama-notebookllama-badge.png)](https://mseep.ai/app/run-llama-notebookllama)

## Key Features

*   **Open-Source:**  Benefit from a community-driven project with full access to the code.
*   **AI-Powered:** Leverage the power of language models for intelligent note organization and generation.
*   **LlamaCloud Integration:**  Seamlessly integrates with [LlamaCloud](https://cloud.llamaindex.ai?utm_source=demo&utm_medium=notebookLM) for enhanced functionality.
*   **Customization:** Easily configure the application with your preferred API keys for OpenAI, ElevenLabs, and LlamaCloud.
*   **Flexible Setup:** Supports both default and custom settings for embedding models, providing flexibility for your needs.

## Getting Started

Follow these steps to get NotebookLlaMa up and running:

### Prerequisites

This project uses `uv` to manage dependencies. Make sure you have `uv` installed.

**On macOS and Linux:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**On Windows:**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

For more install options, see `uv`'s [official documentation](https://docs.astral.sh/uv/getting-started/installation/).

### Installation and Configuration

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
        *   `OPENAI_API_KEY`:  Find it [on OpenAI Platform](https://platform.openai.com/api-keys)
        *   `ELEVENLABS_API_KEY`:  Find it [on ElevenLabs Settings](https://elevenlabs.io/app/settings/api-keys)
        *   `LLAMACLOUD_API_KEY`: Find it [on LlamaCloud Dashboard](https://cloud.llamaindex.ai?utm_source=demo&utm_medium=notebookLM)

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

    *   Run the interactive setup wizard to configure your index pipeline.
        *   **⚡ Quick Start (Default OpenAI):**  Select **"With Default Settings"** for the fastest setup using OpenAI's `text-embedding-3-small`.
        *   **🧠 Advanced (Custom Embedding Models):** Select **"With Custom Settings"** and follow the on-screen instructions.

        ```bash
        uv run tools/create_llama_cloud_index.py
        ```

6.  **Launch Backend Services:**

    ```bash
    docker compose up -d
    ```

7.  **Run the Application:**

    *   Run the **MCP** server:

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

## Contributing

Contribute to NotebookLlaMa by following the [guidelines](./CONTRIBUTING.md).

## License

This project is licensed under the [MIT License](./LICENSE).
# NotebookLlama: Your Open-Source AI-Powered Note-Taking Companion

**NotebookLlama** offers a free and open-source alternative to NotebookLM, empowering you to create, analyze, and organize your notes with the power of AI. ([View on GitHub](https://github.com/run-llama/notebookllama))

<p align="center">
  <img src="https://github.com/user-attachments/assets/7e9cca45-8a4c-4dfa-98d2-2cef147422f2" alt="NotebookLlama Screenshot">
</p>

<p align="center">
  A fully open-source alternative to NotebookLM, backed by <a href="https://cloud.llamaindex.ai?utm_source=demo&utm_medium=notebookLM"><strong>LlamaCloud</strong></a>.
</p>

<p align="center">
    <a href="https://github.com/run-llama/notebookllama/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/run-llama/notebookllama?color=blue"></a>
    <a href="https://github.com/run-llama/notebookllama/stargazers"><img alt="Stars" src="https://img.shields.io/github/stars/run-llama/notebookllama?color=yellow"></a>
    <a href="https://github.com/run-llama/notebookllama/issues"><img alt="Issues" src="https://img.shields.io/github/issues/run-llama/notebookllama?color=orange"></a>
    <br>
    <a href="https://mseep.ai/app/run-llama-notebookllama"><img alt="MseeP.ai Security Assessment Badge" src="https://mseep.net/pr/run-llama-notebookllama-badge.png"></a>
</p>

## Key Features:

*   **Open-Source:** Benefit from community contributions and transparency.
*   **AI-Powered:** Leverage the capabilities of LlamaCloud for intelligent note analysis and organization.
*   **Note Management:**  Organize and access your notes with ease.
*   **Customizable:** Configure with custom embedding models.

## Getting Started

### Prerequisites

This project utilizes `uv` for dependency management. Ensure you have `uv` installed before proceeding.

*   **macOS and Linux:**

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

*   **Windows:**

    ```powershell
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

    For comprehensive installation options, consult the official `uv` documentation: [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)

### Setup

Follow these steps to get NotebookLlama up and running:

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

Create a `.env` file by copying the example:

```bash
mv .env.example .env
```

Add your API keys to the `.env` file:

*   `OPENAI_API_KEY`:  Obtain it from the [OpenAI Platform](https://platform.openai.com/api-keys)
*   `ELEVENLABS_API_KEY`: Found in [ElevenLabs Settings](https://elevenlabs.io/app/settings/api-keys)
*   `LLAMACLOUD_API_KEY`: Access it via the [LlamaCloud Dashboard](https://cloud.llamaindex.ai?utm_source=demo&utm_medium=notebookLM)

**4. Activate the Virtual Environment**

*   **(macOS/Unix):**

    ```bash
    source .venv/bin/activate
    ```

*   **(Windows):**

    ```bash
    .\.venv\Scripts\activate
    ```

**5. Create LlamaCloud Agent & Pipeline**

Execute these scripts to set up your backend agents and pipelines.

Create the data extraction agent:

```bash
uv run tools/create_llama_extract_agent.py
```

Run the interactive setup for your index pipeline.

> **⚡ Quick Start (Default OpenAI):**  Select **"With Default Settings"** for the fastest setup, which uses OpenAI's `text-embedding-3-small` embedding model.

> **🧠 Advanced (Custom Embedding Models):** Choose **"With Custom Settings"** to use a different embedding model and follow the on-screen instructions.

```bash
uv run tools/create_llama_cloud_index.py
```

**6. Launch Backend Services**

Start the required Postgres and Jaeger containers:

```bash
docker compose up -d
```

**7. Run the Application**

First, run the **MCP** server:

```bash
uv run src/notebookllama/server.py
```

Then, in a **new terminal window**, launch the Streamlit app:

```bash
streamlit run src/notebookllama/Home.py
```

> [!IMPORTANT]
>
> _You might need to install `ffmpeg` if you do not have it installed already_

Access the app at `http://localhost:8501/`.

---

## Contributing

Contribute to NotebookLlama by following the [guidelines](./CONTRIBUTING.md).

## License

NotebookLlama is released under the [MIT License](./LICENSE).
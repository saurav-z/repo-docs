# NotebookLlaMa: Your Open-Source AI-Powered Notebook Companion

**NotebookLlaMa** is a fully open-source alternative to NotebookLM, empowering you to interact with and analyze your documents with AI.  Check out the original repository [here](https://github.com/run-llama/notebookllama)!

[![License](https://img.shields.io/github/license/run-llama/notebookllama?color=blue)](https://github.com/run-llama/notebookllama/blob/main/LICENSE)
[![Stars](https://img.shields.io/github/stars/run-llama/notebookllama?color=yellow)](https://github.com/run-llama/notebookllama/stargazers)
[![Issues](https://img.shields.io/github/issues/run-llama/notebookllama?color=orange)](https://github.com/run-llama/notebookllama/issues)
[![MseeP.ai Security Assessment Badge](https://mseep.net/pr/run-llama-notebookllama-badge.png)](https://mseep.ai/app/run-llama-notebookllama)

## Key Features

*   **Open-Source:**  Leverage the power of AI with a transparent and community-driven platform.
*   **Document Interaction:**  Effortlessly analyze and interact with your documents.
*   **Backed by LlamaCloud:** Powered by the robust infrastructure of [LlamaCloud](https://cloud.llamaindex.ai?utm_source=demo&utm_medium=notebookLM).
*   **Customizable:** Configure your experience with custom settings for embedding models.

## Prerequisites

This project utilizes `uv` for dependency management. Ensure `uv` is installed on your system before proceeding.

**Install `uv`:**

**macOS and Linux:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

For additional installation options, consult the official `uv` [documentation](https://docs.astral.sh/uv/getting-started/installation/).

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

Create a `.env` file and add your API keys:

```bash
mv .env.example .env
```

Edit `.env` with your API keys:

*   `OPENAI_API_KEY`: [OpenAI Platform](https://platform.openai.com/api-keys)
*   `ELEVENLABS_API_KEY`: [ElevenLabs Settings](https://elevenlabs.io/app/settings/api-keys)
*   `LLAMACLOUD_API_KEY`: [LlamaCloud Dashboard](https://cloud.llamaindex.ai?utm_source=demo&utm_medium=notebookLM)

**4. Activate the Virtual Environment**

**(mac/unix):**

```bash
source .venv/bin/activate
```

**(Windows):**

```bash
.\.venv\Scripts\activate
```

**5. Create LlamaCloud Agent & Pipeline**

Create the data extraction agent:

```bash
uv run tools/create_llama_extract_agent.py
```

Configure your index pipeline using the interactive setup wizard:

```bash
uv run tools/create_llama_cloud_index.py
```

**Quick Start (Default OpenAI):** Choose **"With Default Settings"** for a quick setup using OpenAI's `text-embedding-3-small`.

**Advanced (Custom Embedding Models):** Select **"With Custom Settings"** and follow the on-screen instructions to utilize a different embedding model.

**6. Launch Backend Services**

Start the Postgres and Jaeger containers:

```bash
docker compose up -d
```

**7. Run the Application**

Run the MCP server:

```bash
uv run src/notebookllama/server.py
```

In a **new terminal window**, run the Streamlit app:

```bash
streamlit run src/notebookllama/Home.py
```

Access the app at `http://localhost:8501/`.

> [!IMPORTANT]
>
> _You might need to install `ffmpeg` if you do not have it installed already_

## Contributing

Contribute to the project by following the [guidelines](./CONTRIBUTING.md).

## License

This project is licensed under the [MIT License](./LICENSE).
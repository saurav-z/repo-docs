# NotebookLlaMa: An Open-Source Alternative to NotebookLM

Unlock the power of your documents with **NotebookLlaMa**, a fully open-source and customizable alternative to NotebookLM, built with the backing of [LlamaCloud](https://cloud.llamaindex.ai?utm_source=demo&utm_medium=notebookLM).  [View the project on GitHub](https://github.com/run-llama/notebookllama).

[![License](https://img.shields.io/github/license/run-llama/notebookllama?color=blue)](https://github.com/run-llama/notebookllama/blob/main/LICENSE)
[![Stars](https://img.shields.io/github/stars/run-llama/notebookllama?color=yellow)](https://github.com/run-llama/notebookllama/stargazers)
[![Issues](https://img.shields.io/github/issues/run-llama/notebookllama?color=orange)](https://github.com/run-llama/notebookllama/issues)
<br>
[![MseeP.ai Security Assessment Badge](https://mseep.net/pr/run-llama-notebookllama-badge.png)](https://mseep.ai/app/run-llama-notebookllama)

## Key Features

*   **Open Source:** Benefit from a community-driven project with transparent code.
*   **Powered by LlamaCloud:**  Leverage the robust infrastructure of LlamaCloud for enhanced performance.
*   **Customizable:** Tailor the application to your specific needs with flexible configuration options.
*   **Easy Setup:** Get started quickly with straightforward installation and configuration steps.

## Prerequisites

This project uses `uv` for dependency management. Install `uv` using the instructions below:

**macOS and Linux:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

For more installation options, see `uv`'s [official documentation](https://docs.astral.sh/uv/getting-started/installation/).

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

Edit `.env` with the following keys:

*   `OPENAI_API_KEY`:  [Get from OpenAI Platform](https://platform.openai.com/api-keys)
*   `ELEVENLABS_API_KEY`: [Get from ElevenLabs Settings](https://elevenlabs.io/app/settings/api-keys)
*   `LLAMACLOUD_API_KEY`: [Get from LlamaCloud Dashboard](https://cloud.llamaindex.ai?utm_source=demo&utm_medium=notebookLM)

**4. Activate the Virtual Environment**

**(mac/unix)**

```bash
source .venv/bin/activate
```

**(Windows)**

```bash
.\.venv\Scripts\activate
```

**5. Create LlamaCloud Agent & Pipeline**

First, create the data extraction agent:

```bash
uv run tools/create_llama_extract_agent.py
```

Next, run the interactive setup wizard to configure your index pipeline.

> **⚡ Quick Start (Default OpenAI):**
> For the fastest setup, select **"With Default Settings"** when prompted. This will automatically create a pipeline using OpenAI's `text-embedding-3-small` embedding model.

> **🧠 Advanced (Custom Embedding Models):**
> To use a different embedding model, select **"With Custom Settings"** and follow the on-screen instructions.

Run the wizard with the following command:

```bash
uv run tools/create_llama_cloud_index.py
```

**6. Launch Backend Services**

Start Postgres and Jaeger containers:

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

## Contributing

Contribute to this project following the [guidelines](./CONTRIBUTING.md).

## License

NotebookLlaMa is provided under the [MIT License](./LICENSE).
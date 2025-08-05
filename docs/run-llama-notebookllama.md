# NotebookLlama: Your Open-Source NotebookLM Alternative 🦙

**NotebookLlama offers a fully open-source and customizable alternative to NotebookLM, empowering you to harness the power of language models for your notes and documents.**  (Read the [original repo](https://github.com/run-llama/notebookllama) for more details.)

[![License](https://img.shields.io/github/license/run-llama/notebookllama?color=blue)](https://github.com/run-llama/notebookllama/blob/main/LICENSE)
[![Stars](https://img.shields.io/github/stars/run-llama/notebookllama?color=yellow)](https://github.com/run-llama/notebookllama/stargazers)
[![Issues](https://img.shields.io/github/issues/run-llama/notebookllama?color=orange)](https://github.com/run-llama/notebookllama/issues)
<br>
[![MseeP.ai Security Assessment Badge](https://mseep.net/pr/run-llama-notebookllama-badge.png)](https://mseep.ai/app/run-llama-notebookllama)

## Key Features

*   **Open-Source:** Benefit from the flexibility and transparency of a fully open-source project.
*   **LlamaCloud Integration:** Powered by [LlamaCloud](https://cloud.llamaindex.ai?utm_source=demo&utm_medium=notebookLM), ensuring robust and scalable performance.
*   **Customizable:** Tailor the application to your specific needs, including the ability to use different embedding models.
*   **Easy Setup:** Simple installation and configuration with clear instructions.

## Prerequisites

This project uses `uv` to manage dependencies. Ensure `uv` is installed before proceeding.

**Installation Instructions:**

**macOS and Linux:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

For additional installation options, consult `uv`'s [official documentation](https://docs.astral.sh/uv/getting-started/installation/).

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

Create a `.env` file by renaming the example file:

```bash
mv .env.example .env
```

Populate your `.env` file with the following API keys:

*   `OPENAI_API_KEY`:  Obtain from [OpenAI Platform](https://platform.openai.com/api-keys)
*   `ELEVENLABS_API_KEY`: Obtain from [ElevenLabs Settings](https://elevenlabs.io/app/settings/api-keys)
*   `LLAMACLOUD_API_KEY`: Obtain from [LlamaCloud Dashboard](https://cloud.llamaindex.ai?utm_source=demo&utm_medium=notebookLM)

**4. Activate the Virtual Environment:**

**(mac/unix):**

```bash
source .venv/bin/activate
```

**(Windows):**

```bash
.\.venv\Scripts\activate
```

**5. Create LlamaCloud Agent & Pipeline:**

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

**6. Launch Backend Services:**

This command will start the required Postgres and Jaeger containers.

```bash
docker compose up -d
```

**7. Run the Application:**

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

Access the application at `http://localhost:8501/`.

## Contributing

Contribute to the project following the [guidelines](./CONTRIBUTING.md).

## License

NotebookLlama is released under the [MIT License](./LICENSE).
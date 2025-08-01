# NotebookLlama: Your Open-Source AI-Powered Notebook Companion

**NotebookLlama** is a fully open-source alternative to NotebookLM, offering a powerful and customizable AI experience for your notes and documents. 🦙  [View the project on GitHub](https://github.com/run-llama/notebookllama)

<p align="center">
  <a href="https://github.com/run-llama/notebookllama/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/run-llama/notebookllama?color=blue"></a>
  <a href="https://github.com/run-llama/notebookllama/stargazers"><img alt="Stars" src="https://img.shields.io/github/stars/run-llama/notebookllama?color=yellow"></a>
  <a href="https://github.com/run-llama/notebookllama/issues"><img alt="Issues" src="https://img.shields.io/github/issues/run-llama/notebookllama?color=orange"></a>
  <br>
  <a href="https://mseep.ai/app/run-llama-notebookllama"><img alt="MseeP.ai Security Assessment Badge" src="https://mseep.net/pr/run-llama-notebookllama-badge.png"></a>
</p>

## Key Features

*   **Open-Source Alternative:**  Enjoy the benefits of a powerful AI notebook companion without vendor lock-in.
*   **Powered by LlamaCloud:** Seamlessly integrated with LlamaCloud for enhanced performance and features.
*   **Customizable:**  Configure your own embedding models and settings.
*   **Easy Setup:**  Simple installation process using `uv` and clear instructions.
*   **Backend Services:** Includes required Postgres and Jaeger containers via Docker Compose.

## Prerequisites

This project utilizes `uv` for dependency management. Ensure you have `uv` installed before proceeding.

**macOS and Linux:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

For further installation options, consult the [official `uv` documentation](https://docs.astral.sh/uv/getting-started/installation/).

## Getting Started

Follow these steps to set up and run NotebookLlama:

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

Create a `.env` file from the example:

```bash
mv .env.example .env
```

Populate the `.env` file with your API keys:

*   `OPENAI_API_KEY`:  [OpenAI Platform](https://platform.openai.com/api-keys)
*   `ELEVENLABS_API_KEY`: [ElevenLabs Settings](https://elevenlabs.io/app/settings/api-keys)
*   `LLAMACLOUD_API_KEY`:  [LlamaCloud Dashboard](https://cloud.llamaindex.ai?utm_source=demo&utm_medium=notebookLM)

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

Execute the following scripts to configure your backend agents and pipelines.

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

This command starts the required Postgres and Jaeger containers.

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

Access the application at `http://localhost:8501/`.

## Contributing

Contributions are welcome! Please refer to the [Contribution Guidelines](./CONTRIBUTING.md).

## License

This project is licensed under the [MIT License](./LICENSE).
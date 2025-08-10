# NotebookLlaMa: Your Open-Source AI-Powered Notebook Companion 🦙

**Unlock the power of AI for your notes with NotebookLlaMa, a fully open-source alternative to NotebookLM!** Developed by [Run Llama](https://github.com/run-llama/notebookllama), NotebookLlaMa empowers you to summarize, analyze, and interact with your notes using the power of large language models, all while maintaining complete control over your data.

[View the original repository on GitHub](https://github.com/run-llama/notebookllama)

<p align="center">
  <img src="https://github.com/user-attachments/assets/7e9cca45-8a4c-4dfa-98d2-2cef147422f2" alt="NotebookLlaMa Screenshot" width="600">
</p>

<p align="center">
    <a href="https://github.com/run-llama/notebookllama/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/run-llama/notebookllama?color=blue"></a>
    <a href="https://github.com/run-llama/notebookllama/stargazers"><img alt="Stars" src="https://img.shields.io/github/stars/run-llama/notebookllama?color=yellow"></a>
    <a href="https://github.com/run-llama/notebookllama/issues"><img alt="Issues" src="https://img.shields.io/github/issues/run-llama/notebookllama?color=orange"></a>
    <br>
    <a href="https://mseep.ai/app/run-llama-notebookllama"><img alt="MseeP.ai Security Assessment Badge" src="https://mseep.net/pr/run-llama-notebookllama-badge.png"></a>
</p>

## Key Features

*   **Open-Source:** Enjoy the freedom of open-source, with full control and transparency over your data.
*   **AI-Powered Note Interaction:** Summarize, analyze, and extract insights from your notes using powerful language models.
*   **LlamaCloud Integration:** Backed by [LlamaCloud](https://cloud.llamaindex.ai?utm_source=demo&utm_medium=notebookLM) for enhanced performance and features.
*   **Customizable Setup:** Configure your backend with different embedding models, giving you flexibility and control.
*   **Easy Deployment:** Get up and running quickly with clear instructions and streamlined setup processes.

## Prerequisites

This project uses `uv` to manage dependencies. Ensure you have `uv` installed before proceeding.

**Installation:**

**macOS and Linux:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

For more installation options, see the official `uv` [documentation](https://docs.astral.sh/uv/getting-started/installation/).

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

Create your `.env` file:

```bash
mv .env.example .env
```

Add your API keys to the `.env` file:

*   `OPENAI_API_KEY`: find it [on OpenAI Platform](https://platform.openai.com/api-keys)
*   `ELEVENLABS_API_KEY`: find it [on ElevenLabs Settings](https://elevenlabs.io/app/settings/api-keys)
*   `LLAMACLOUD_API_KEY`: find it [on LlamaCloud Dashboard](https://cloud.llamaindex.ai?utm_source=demo&utm_medium=notebookLM)

**4. Activate the Virtual Environment**

**(macOS/Linux):**

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

Configure your index pipeline (choose "With Default Settings" for the fastest setup using OpenAI's embedding model, or "With Custom Settings" for advanced options):

```bash
uv run tools/create_llama_cloud_index.py
```

**6. Launch Backend Services**

Start the required Postgres and Jaeger containers:

```bash
docker compose up -d
```

**7. Run the Application**

Run the MCP server:

```bash
uv run src/notebookllama/server.py
```

In a new terminal window, launch the Streamlit app:

```bash
streamlit run src/notebookllama/Home.py
```

Access the app at `http://localhost:8501/`.

> [!IMPORTANT]
>
> _You might need to install `ffmpeg` if you do not have it installed already_

## Contributing

Contribute to NotebookLlaMa by following the [guidelines](./CONTRIBUTING.md).

## License

NotebookLlaMa is provided under an [MIT License](./LICENSE).
# NotebookLlama: Your Open-Source AI-Powered Notebook Companion 🦙

**Unlock the power of AI-assisted note-taking and knowledge management with NotebookLlama, an open-source alternative to NotebookLM.** ([View the original repo](https://github.com/run-llama/notebookllama))

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

## Key Features

*   **Open-Source:**  Benefit from the transparency and community-driven development of an open-source project.
*   **AI-Powered Assistance:** Leverage the power of AI for note-taking, summarization, and knowledge discovery.
*   **LlamaCloud Integration:** Powered by LlamaCloud, ensuring robust performance and scalability.
*   **Customizable:** Tailor the application to your specific needs and preferences.
*   **Easy Setup:**  Quickly get started with straightforward installation and configuration steps.

## Prerequisites

This project uses `uv` to manage dependencies. Make sure you have `uv` installed before proceeding.

**Installation (macOS and Linux):**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Installation (Windows):**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

For more install options, see `uv`'s [official documentation](https://docs.astral.sh/uv/getting-started/installation/).

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

*   Create a `.env` file:

```bash
mv .env.example .env
```

*   Add your API keys to the `.env` file:

    *   `OPENAI_API_KEY`:  Get it from [OpenAI Platform](https://platform.openai.com/api-keys)
    *   `ELEVENLABS_API_KEY`: Find it on [ElevenLabs Settings](https://elevenlabs.io/app/settings/api-keys)
    *   `LLAMACLOUD_API_KEY`: Find it on [LlamaCloud Dashboard](https://cloud.llamaindex.ai?utm_source=demo&utm_medium=notebookLM)

**4. Activate the Virtual Environment**

*(mac/unix)*

```bash
source .venv/bin/activate
```

*(Windows)*

```bash
.\.venv\Scripts\activate
```

**5. Create LlamaCloud Agent & Pipeline**

*   Create the data extraction agent:

```bash
uv run tools/create_llama_extract_agent.py
```

*   Configure your index pipeline using the interactive setup wizard:

    *   **Quick Start (Default OpenAI):** Select "**With Default Settings**" for a fast setup using OpenAI.
    *   **Advanced (Custom Embedding Models):** Select "**With Custom Settings**" to use a different embedding model.

```bash
uv run tools/create_llama_cloud_index.py
```

**6. Launch Backend Services**

*   Start the required Postgres and Jaeger containers:

```bash
docker compose up -d
```

**7. Run the Application**

*   Run the MCP server:

```bash
uv run src/notebookllama/server.py
```

*   In a new terminal window, launch the Streamlit app:

```bash
streamlit run src/notebookllama/Home.py
```

*   Access the app at `http://localhost:8501/`.

> [!IMPORTANT]
>
> _You might need to install `ffmpeg` if you do not have it installed already_

## Contributing

Contribute to this project following the [guidelines](./CONTRIBUTING.md).

## License

This project is provided under an [MIT License](./LICENSE).
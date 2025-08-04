# NotebookLlama: Your Open-Source AI-Powered Notebook Alternative 🦙

**Unleash the power of AI in your notes with NotebookLlama, a fully open-source alternative to NotebookLM!**  [View the original repository on GitHub](https://github.com/run-llama/notebookllama).

<p align="center">
  <img src="https://github.com/user-attachments/assets/7e9cca45-8a4c-4dfa-98d2-2cef147422f2" alt="NotebookLlama Logo">
</p>

<p align="center">
  Backed by <a href="https://cloud.llamaindex.ai?utm_source=demo&utm_medium=notebookLM"><strong>LlamaCloud</strong></a>.
</p>

<p align="center">
    <a href="https://github.com/run-llama/notebookllama/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/run-llama/notebookllama?color=blue"></a>
    <a href="https://github.com/run-llama/notebookllama/stargazers"><img alt="Stars" src="https://img.shields.io/github/stars/run-llama/notebookllama?color=yellow"></a>
    <a href="https://github.com/run-llama/notebookllama/issues"><img alt="Issues" src="https://img.shields.io/github/issues/run-llama/notebookllama?color=orange"></a>
    <br>
    <a href="https://mseep.ai/app/run-llama-notebookllama"><img alt="MseeP.ai Security Assessment Badge" src="https://mseep.net/pr/run-llama-notebookllama-badge.png"></a>
</p>

## Key Features

*   **Open Source:** Fully accessible and customizable.
*   **AI-Powered:**  Leverages cutting-edge AI models for enhanced note-taking.
*   **LlamaCloud Integration:** Seamlessly integrated with LlamaCloud for powerful backend services.
*   **Flexible Configuration:** Easily configure with your preferred API keys.
*   **Quick Setup:** Get up and running with simple installation steps.

## Getting Started

### Prerequisites

This project uses `uv` for dependency management.  Ensure `uv` is installed before proceeding.

**Installation (macOS and Linux):**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Installation (Windows):**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

For more installation options, consult the [official `uv` documentation](https://docs.astral.sh/uv/getting-started/installation/).

---

### Step-by-Step Installation and Setup

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

Create a `.env` file:

```bash
mv .env.example .env
```

Add your API keys to the `.env` file:

*   `OPENAI_API_KEY`: Get it from [OpenAI Platform](https://platform.openai.com/api-keys)
*   `ELEVENLABS_API_KEY`: Find it in [ElevenLabs Settings](https://elevenlabs.io/app/settings/api-keys)
*   `LLAMACLOUD_API_KEY`: Access it on the [LlamaCloud Dashboard](https://cloud.llamaindex.ai?utm_source=demo&utm_medium=notebookLM)

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

Create the data extraction agent:

```bash
uv run tools/create_llama_extract_agent.py
```

Run the interactive setup wizard:

> **⚡ Quick Start (Default OpenAI):** Select **"With Default Settings"** for a fast setup using OpenAI's `text-embedding-3-small` embedding model.

> **🧠 Advanced (Custom Embedding Models):** Choose **"With Custom Settings"** for custom embedding models.

Run the wizard:

```bash
uv run tools/create_llama_cloud_index.py
```

**6. Launch Backend Services:**

Starts Postgres and Jaeger containers:

```bash
docker compose up -d
```

**7. Run the Application:**

Run the MCP server:

```bash
uv run src/notebookllama/server.py
```

In a **new terminal window**, launch the Streamlit app:

```bash
streamlit run src/notebookllama/Home.py
```

> [!IMPORTANT]
>
> _Install `ffmpeg` if you do not already have it installed_

Access the app at `http://localhost:8501/`.

---

### Contributing

Contribute to the project by following the [contribution guidelines](./CONTRIBUTING.md).

### License

This project is released under the [MIT License](./LICENSE).
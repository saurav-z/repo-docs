# NotebookLlama: Your Open-Source AI Notebook Assistant 🦙

**Unleash the power of AI with NotebookLlama, a free and open-source alternative to NotebookLM, built with LlamaIndex and backed by LlamaCloud!** ([Original Repo](https://github.com/run-llama/notebookllama))

<p align="center">
  <img src="https://github.com/user-attachments/assets/7e9cca45-8a4c-4dfa-98d2-2cef147422f2" alt="NotebookLlama Logo" width="200">
</p>

<p align="center">
    <a href="https://github.com/run-llama/notebookllama/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/run-llama/notebookllama?color=blue"></a>
    <a href="https://github.com/run-llama/notebookllama/stargazers"><img alt="Stars" src="https://img.shields.io/github/stars/run-llama/notebookllama?color=yellow"></a>
    <a href="https://github.com/run-llama/notebookllama/issues"><img alt="Issues" src="https://img.shields.io/github/issues/run-llama/notebookllama?color=orange"></a>
    <br>
    <a href="https://mseep.ai/app/run-llama-notebookllama"><img alt="MseeP.ai Security Assessment Badge" src="https://mseep.net/pr/run-llama-notebookllama-badge.png"></a>
</p>

## Key Features

*   **Open Source & Free:** Enjoy a fully open-source and cost-free alternative to NotebookLM.
*   **Powered by LlamaCloud:** Leverages the robust capabilities of LlamaCloud for enhanced performance and features.
*   **Easy Setup:** Simple installation and configuration with clear instructions.
*   **Customizable:** Supports custom embedding models for advanced users.
*   **Streamlit Interface:**  Built with Streamlit for an intuitive and user-friendly experience.
*   **Extensible:**  Contribute to the project and tailor it to your specific needs.

## Getting Started

### Prerequisites

This project uses `uv` for dependency management.  Install it using the following instructions.

**macOS and Linux:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

For more install options, see `uv`'s [official documentation](https://docs.astral.sh/uv/getting-started/installation/).

### Installation and Setup

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

Create a `.env` file by renaming the example file:

```bash
mv .env.example .env
```

Add your API keys to the `.env` file:

*   `OPENAI_API_KEY`: Get it from the [OpenAI Platform](https://platform.openai.com/api-keys).
*   `ELEVENLABS_API_KEY`: Find it in [ElevenLabs Settings](https://elevenlabs.io/app/settings/api-keys).
*   `LLAMACLOUD_API_KEY`: Obtain it from the [LlamaCloud Dashboard](https://cloud.llamaindex.ai?utm_source=demo&utm_medium=notebookLM).

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

First, create the data extraction agent:

```bash
uv run tools/create_llama_extract_agent.py
```

Next, run the interactive setup wizard to configure your index pipeline:

> **⚡ Quick Start (Default OpenAI):**
> For the fastest setup, select **"With Default Settings"** when prompted. This will automatically create a pipeline using OpenAI's `text-embedding-3-small` embedding model.

> **🧠 Advanced (Custom Embedding Models):**
> To use a different embedding model, select **"With Custom Settings"** and follow the on-screen instructions.

Run the wizard with the following command:

```bash
uv run tools/create_llama_cloud_index.py
```

**6. Launch Backend Services**

This command starts the required Postgres and Jaeger containers:

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

Contribute to NotebookLlama following the [contributing guidelines](./CONTRIBUTING.md).

## License

This project is licensed under the [MIT License](./LICENSE).
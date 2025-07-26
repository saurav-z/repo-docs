# NotebookLlaMa: Your Open-Source NotebookLM Alternative 🦙

**Unlock the power of knowledge with NotebookLlaMa, a fully open-source alternative to NotebookLM, powered by LlamaCloud.** Explore the [original repository](https://github.com/run-llama/notebookllama) for more information and contributions.

<p align="center">
  <img src="https://github.com/user-attachments/assets/7e9cca45-8a4c-4dfa-98d2-2cef147422f2" alt="NotebookLlaMa Logo" width="200">
</p>

<p align="center">
    <a href="https://github.com/run-llama/notebookllama/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/run-llama/notebookllama?color=blue"></a>
    <a href="https://github.com/run-llama/notebookllama/stargazers"><img alt="Stars" src="https://img.shields.io/github/stars/run-llama/notebookllama?color=yellow"></a>
    <a href="https://github.com/run-llama/notebookllama/issues"><img alt="Issues" src="https://img.shields.io/github/issues/run-llama/notebookllama?color=orange"></a>
    <br>
    <a href="https://mseep.ai/app/run-llama-notebookllama"><img alt="MseeP.ai Security Assessment Badge" src="https://mseep.net/pr/run-llama-notebookllama-badge.png"></a>
</p>

## Key Features

*   **Open-Source:**  Freely accessible and customizable.
*   **Powered by LlamaCloud:** Leverages the power of LlamaCloud for enhanced performance.
*   **NotebookLM Alternative:** Provides similar functionality to NotebookLM.
*   **Flexible Configuration:** Supports custom embedding models.
*   **Easy Setup:** Includes clear, step-by-step instructions for getting started.

## Prerequisites

This project utilizes `uv` for dependency management.  Install `uv` based on your operating system:

**macOS and Linux:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

For advanced installation options, consult the [uv official documentation](https://docs.astral.sh/uv/getting-started/installation/).

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

Create a `.env` file:

```bash
mv .env.example .env
```

Add your API keys to `.env`:

*   `OPENAI_API_KEY`: Get it from [OpenAI Platform](https://platform.openai.com/api-keys).
*   `ELEVENLABS_API_KEY`: Get it from [ElevenLabs Settings](https://elevenlabs.io/app/settings/api-keys).
*   `LLAMACLOUD_API_KEY`: Get it from [LlamaCloud Dashboard](https://cloud.llamaindex.ai?utm_source=demo&utm_medium=notebookLM).

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

Then, run the interactive setup wizard:

*   **Quick Start (Default OpenAI):** Choose "With Default Settings" for a quick setup using OpenAI's `text-embedding-3-small`.
*   **Advanced (Custom Embedding Models):** Select "With Custom Settings" and follow the instructions.

Run the setup wizard:

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

In a new terminal, launch the Streamlit app:

```bash
streamlit run src/notebookllama/Home.py
```

You may need to install `ffmpeg`.

Access the app at `http://localhost:8501/`.

---

## Contributing

Contribute to the project following the [guidelines](./CONTRIBUTING.md).

## License

This project is licensed under the [MIT License](./LICENSE).
# NotebookLlaMa: Your Open-Source AI-Powered Notebook Companion

**NotebookLlaMa** is a powerful, open-source alternative to NotebookLM, offering a secure and customizable way to explore and interact with your documents. ([Original Repository](https://github.com/run-llama/notebookllama))

<p align="center">
  <img src="https://github.com/user-attachments/assets/7e9cca45-8a4c-4dfa-98d2-2cef147422f2" alt="NotebookLlaMa Logo" width="200"/>
</p>

<p align="center">
  Built with the support of <a href="https://cloud.llamaindex.ai?utm_source=demo&utm_medium=notebookLM"><strong>LlamaCloud</strong></a>.
</p>

<p align="center">
    <a href="https://github.com/run-llama/notebookllama/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/run-llama/notebookllama?color=blue"></a>
    <a href="https://github.com/run-llama/notebookllama/stargazers"><img alt="Stars" src="https://img.shields.io/github/stars/run-llama/notebookllama?color=yellow"></a>
    <a href="https://github.com/run-llama/notebookllama/issues"><img alt="Issues" src="https://img.shields.io/github/issues/run-llama/notebookllama?color=orange"></a>
    <br>
    <a href="https://mseep.ai/app/run-llama-notebookllama"><img alt="MseeP.ai Security Assessment Badge" src="https://mseep.net/pr/run-llama-notebookllama-badge.png"></a>
</p>

## Key Features

*   **Open-Source:** Fully open-source and community-driven.
*   **Document Interaction:** Designed to help you interact with your documents.
*   **Customization:** Built with LlamaCloud, allowing for advanced customization options.

## Getting Started

### Prerequisites

This project uses `uv` for dependency management. Ensure you have `uv` installed before proceeding.

**Install `uv`:**

*   **macOS/Linux:**

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

*   **Windows:**

    ```powershell
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

For more installation options, please refer to `uv`'s [official documentation](https://docs.astral.sh/uv/getting-started/installation/).

### Installation and Setup

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

Create a `.env` file from the example file:

```bash
mv .env.example .env
```

Add your API keys to the `.env` file:

*   `OPENAI_API_KEY`: [OpenAI Platform](https://platform.openai.com/api-keys)
*   `ELEVENLABS_API_KEY`: [ElevenLabs Settings](https://elevenlabs.io/app/settings/api-keys)
*   `LLAMACLOUD_API_KEY`: [LlamaCloud Dashboard](https://cloud.llamaindex.ai?utm_source=demo&utm_medium=notebookLM)

**4. Activate the Virtual Environment**

*   **macOS/Linux:**

    ```bash
    source .venv/bin/activate
    ```

*   **Windows:**

    ```bash
    .\.venv\Scripts\activate
    ```

**5. Create LlamaCloud Agent & Pipeline**

First, create the data extraction agent:

```bash
uv run tools/create_llama_extract_agent.py
```

Next, configure your index pipeline using the interactive setup wizard:

>   **⚡ Quick Start (Default OpenAI):** For rapid setup, select "**With Default Settings**". This will configure a pipeline using OpenAI's `text-embedding-3-small` embedding model.

>   **🧠 Advanced (Custom Embedding Models):** Choose "**With Custom Settings**" to use a different embedding model and follow the on-screen instructions.

Run the wizard:

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

>   [!IMPORTANT]
>
>   _You might need to install `ffmpeg` if you do not have it installed already_

Access the app at `http://localhost:8501/`.

---

### Contributing

Contribute to this project following the [guidelines](./CONTRIBUTING.md).

### License

This project is provided under an [MIT License](./LICENSE).
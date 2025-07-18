# NotebookLlama: Your Open-Source AI-Powered Notebook Companion 🦙

**Unleash the power of AI with NotebookLlama, a fully open-source alternative to NotebookLM, empowering you to explore, create, and learn.**  ([View the Original Repo](https://github.com/run-llama/notebookllama))

<p align="center">
  <img src="https://github.com/user-attachments/assets/7e9cca45-8a4c-4dfa-98d2-2cef147422f2" alt="NotebookLlama Screenshot" width="80%">
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

*   **Open-Source Alternative:**  A free and open-source solution mirroring the functionality of NotebookLM.
*   **LlamaCloud Integration:** Leverages the power of LlamaCloud for enhanced performance and capabilities.
*   **Customizable Setup:** Easily configure your setup with various embedding models.
*   **Local Development:** Run NotebookLlama locally with Docker Compose and Streamlit.
*   **AI-Powered Assistance:** Explore your knowledge with AI-driven features and interact with your data in a new way.

## Getting Started

### Prerequisites

*   **uv:** This project uses `uv` for dependency management. Install it using the instructions below.

    *   **macOS and Linux:**
        ```bash
        curl -LsSf https://astral.sh/uv/install.sh | sh
        ```
    *   **Windows:**
        ```powershell
        powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
        ```
    *   For more installation options, see the `uv` [official documentation](https://docs.astral.sh/uv/getting-started/installation/).

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

*   Create a `.env` file:

    ```bash
    mv .env.example .env
    ```

*   Add your API keys to `.env`:
    *   `OPENAI_API_KEY`: Find it [on OpenAI Platform](https://platform.openai.com/api-keys)
    *   `ELEVENLABS_API_KEY`: Find it [on ElevenLabs Settings](https://elevenlabs.io/app/settings/api-keys)
    *   `LLAMACLOUD_API_KEY`: Find it [on LlamaCloud Dashboard](https://cloud.llamaindex.ai?utm_source=demo&utm_medium=notebookLM)

**4. Activate the Virtual Environment**

*   **mac/unix:**

    ```bash
    source .venv/bin/activate
    ```
*   **Windows:**

    ```bash
    .\.venv\Scripts\activate
    ```

**5. Create LlamaCloud Agent & Pipeline**

*   Create the data extraction agent:

    ```bash
    uv run tools/create_llama_extract_agent.py
    ```

*   Run the interactive setup wizard to configure your index pipeline.

    *   **⚡ Quick Start (Default OpenAI):**  Select **"With Default Settings"** for a fast setup using OpenAI's `text-embedding-3-small` embedding model.

    *   **🧠 Advanced (Custom Embedding Models):** Select **"With Custom Settings"** and follow the instructions.

    ```bash
    uv run tools/create_llama_cloud_index.py
    ```

**6. Launch Backend Services**

```bash
docker compose up -d
```

**7. Run the Application**

*   Run the **MCP** server:

    ```bash
    uv run src/notebookllama/server.py
    ```
*   In a **new terminal window**, launch the Streamlit app:

    ```bash
    streamlit run src/notebookllama/Home.py
    ```

> [!IMPORTANT]
>
> _You might need to install `ffmpeg` if you do not have it installed already._

Access the app at `http://localhost:8501/`.

## Contributing

Contribute to this project following the [guidelines](./CONTRIBUTING.md).

## License

This project is provided under an [MIT License](./LICENSE).
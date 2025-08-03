# NotebookLlaMa: Your Open-Source AI-Powered Notebook Companion

**Unlock the power of AI for your notes with NotebookLlaMa, a fully open-source alternative to NotebookLM!**  Find the original repo [here](https://github.com/run-llama/notebookllama).

<p align="center">
  <img src="https://github.com/user-attachments/assets/7e9cca45-8a4c-4dfa-98d2-2cef147422f2" alt="NotebookLlaMa Screenshot">
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

*   **Open-Source:** Fully accessible and customizable.
*   **AI-Powered:** Leverages advanced AI models for note-taking and knowledge management.
*   **LlamaCloud Integration:** Powered by LlamaCloud for robust backend infrastructure.
*   **Easy Setup:** Simple installation process using `uv`.

## Getting Started

### Prerequisites

Ensure you have `uv` installed to manage project dependencies.

**Installation:**

*   **macOS and Linux:**

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

*   **Windows:**

    ```powershell
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

    For additional installation methods, refer to the official `uv` [documentation](https://docs.astral.sh/uv/getting-started/installation/).

### Installation and Configuration

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

*   Create a `.env` file:

    ```bash
    mv .env.example .env
    ```

*   Add your API keys to `.env`:

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

*   Create the data extraction agent:

    ```bash
    uv run tools/create_llama_extract_agent.py
    ```

*   Run the index pipeline setup wizard:

    ```bash
    uv run tools/create_llama_cloud_index.py
    ```

    *   **Quick Start (Default OpenAI):**  Choose "With Default Settings".
    *   **Advanced (Custom Embedding Models):** Choose "With Custom Settings".

**6. Launch Backend Services**

```bash
docker compose up -d
```

**7. Run the Application**

*   Run the MCP server:

    ```bash
    uv run src/notebookllama/server.py
    ```

*   In a **new terminal window**, launch the Streamlit app:

    ```bash
    streamlit run src/notebookllama/Home.py
    ```

    > [!IMPORTANT]
    >
    > _You might need to install `ffmpeg` if you do not have it installed already_

Access the app at `http://localhost:8501/`.

## Contributing

Contribute to the project by following the [guidelines](./CONTRIBUTING.md).

## License

NotebookLlaMa is licensed under the [MIT License](./LICENSE).
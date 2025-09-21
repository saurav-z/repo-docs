# 🤖 AI Agent Service Toolkit: Build, Deploy, and Scale Your AI Agents with Ease

**Get started building AI agents with LangGraph, FastAPI, and Streamlit using this comprehensive toolkit!** Explore the original repo on [GitHub](https://github.com/JoshuaC215/agent-service-toolkit).

## Key Features

*   **Advanced LangGraph Agents:** Leverage the power of LangGraph, including the latest features like human-in-the-loop, flow control, long-term memory, and `langgraph-supervisor`, for sophisticated agent design.
*   **FastAPI-Powered Service:** Deploy your agents with a robust FastAPI service, offering both streaming and non-streaming endpoints for efficient interaction.
*   **Enhanced Streaming Capabilities:** Benefit from a novel streaming approach that supports both token-based and message-based streaming, providing a superior user experience.
*   **User-Friendly Streamlit Interface:** Create intuitive chat interfaces with the integrated Streamlit app, making it easy for users to interact with your agents.
*   **Multi-Agent Support:** Run and manage multiple agents within your service, accessible via URL paths, with agent and model details available via `/info`.
*   **Asynchronous Design:** Experience efficient handling of concurrent requests with an asynchronous architecture.
*   **Content Moderation:** Ensure safe and responsible use with LlamaGuard integration (requires Groq API key).
*   **RAG Agent Implementation:** Get started with a Retrieval-Augmented Generation (RAG) agent powered by ChromaDB (see [docs](docs/RAG_Assistant.md)).
*   **Feedback and Tracing:** Implement a star-based feedback system integrated with LangSmith for continuous improvement.
*   **Docker and Production Ready:** Streamline development and deployment with Docker support, including Dockerfiles and a compose file.
*   **Extensive Testing:** Ensure reliability with comprehensive unit and integration tests.

## Architecture

![Agent Service Architecture](media/agent_architecture.png)

## Quickstart

**1. Install Dependencies:**

```bash
# At least one LLM API key is required
echo 'OPENAI_API_KEY=your_openai_api_key' >> .env

# uv is the recommended way to install agent-service-toolkit, but "pip install ." also works
# For uv installation options, see: https://docs.astral.sh/uv/getting-started/installation/
curl -LsSf https://astral.sh/uv/0.7.19/install.sh | sh

# Install dependencies. "uv sync" creates .venv automatically
uv sync --frozen
source .venv/bin/activate
```

**2. Run the Agent Service:**

```bash
python src/run_service.py
```

**3. Run the Streamlit App (in a separate terminal):**

```bash
source .venv/bin/activate
streamlit run src/streamlit_app.py
```

**4. (Optional) Run with Docker:**

```bash
echo 'OPENAI_API_KEY=your_openai_api_key' >> .env
docker compose watch
```

## Try the App!

[Launch the live demo](https://agent-service-toolkit.streamlit.app/)

<img src="media/app_screenshot.png" width="600">

## Key Files

*   `src/agents/`: Contains definitions for various AI agents.
*   `src/schema/`: Defines the data structures and protocols.
*   `src/core/`: Includes core modules like LLM definitions and settings.
*   `src/service/service.py`: The FastAPI service that serves the agents.
*   `src/client/client.py`: Client for interacting with the agent service.
*   `src/streamlit_app.py`: The Streamlit app for the chat interface.
*   `tests/`: Contains unit and integration tests for the entire system.

## Setup and Usage

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/JoshuaC215/agent-service-toolkit.git
    cd agent-service-toolkit
    ```

2.  **Configure Environment Variables:**

    *   Create a `.env` file in the root directory using `.env.example` as a template.  You *must* set at least one LLM API key.  The `.env.example` file offers more options for API keys, authentication, tracing, testing, and other settings.

3.  **Run the agent service and Streamlit app:** using the Quickstart instructions.  Docker setup is recommended.

### Additional Setup

*   [Setting up Ollama](docs/Ollama.md)
*   [Setting up VertexAI](docs/VertexAI.md)
*   [Setting up RAG with ChromaDB](docs/RAG_Assistant.md)

### Customizing Your Agent

1.  Create your new agent in the `src/agents` directory by copying and modifying existing agent files.
2.  Add your agent to the `agents` dictionary in `src/agents/agents.py`.
3.  Adjust the Streamlit app (`src/streamlit_app.py`) to utilize your new agent's capabilities.

### Handling Private Credentials

The `privatecredentials/` directory is provided for storing files needed by LLMs or Agents. This folder is ignored by Git and the Docker build process.  See [Working with File-based Credentials](docs/File_Based_Credentials.md) for details.

### Docker Setup (Recommended)

This project includes a Docker setup for easy development and deployment. The `compose.yaml` file defines three services: `postgres`, `agent_service` and `streamlit_app`. The `Dockerfile` for each service is in their respective directories.

1.  Ensure you have Docker and Docker Compose (>= [v2.23.0](https://docs.docker.com/compose/release-notes/#2230)) installed.
2.  Create a `.env` file with your API keys (copy `.env.example`).
3.  Build and launch the services with `docker compose watch`.
4.  The services auto-reload when you make changes to the code (excluding changes to `pyproject.toml` or `uv.lock`, for which a rebuild is necessary).
5.  Access the Streamlit app at `http://localhost:8501`.
6.  Access the API at `http://0.0.0.0:8080`, and use the OpenAPI docs at `http://0.0.0.0:8080/redoc`.
7.  Use `docker compose down` to stop the services.

### Using the AgentClient

The `src/client/client.AgentClient` provides a flexible interface for interacting with the agent service.  See `src/run_client.py` for example usage.

```python
from client import AgentClient
client = AgentClient()

response = client.invoke("Tell me a brief joke?")
response.pretty_print()
```

### Development with LangGraph Studio

Supports [LangGraph Studio](https://langchain-ai.github.io/langgraph/concepts/langgraph_studio/) for agent development. Run `langgraph dev` after setting up your `.env` file and customizing `langgraph.json`.

### Local Development Without Docker

1.  Create and activate a virtual environment.
2.  Run `python src/run_service.py` in one terminal.
3.  Run `streamlit run src/streamlit_app.py` in another.

## Projects Built With or Inspired By This Toolkit

*   **[PolyRAG](https://github.com/QuentinFuxa/PolyRAG)**: Extends this toolkit with RAG capabilities over PostgreSQL and PDF documents.
*   **[alexrisch/agent-web-kit](https://github.com/alexrisch/agent-web-kit)**: A Next.JS frontend for agent-service-toolkit
*   **[raushan-in/dapa](https://github.com/raushan-in/dapa)**: Enables users to report financial scams and frauds efficiently via a user-friendly platform.

*Please contribute new projects by editing the README or opening a discussion!*

## Contributing

Contributions are welcome!  Follow these steps to test the agent service:

1.  Activate your virtual environment.
2.  Install development dependencies and pre-commit hooks with `uv sync --frozen` and `pre-commit install`.
3.  Run tests using `pytest`.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
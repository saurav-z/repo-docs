# AI Agent Service Toolkit: Build and Deploy Powerful AI Agents with Ease

**Unleash the power of LangGraph, FastAPI, and Streamlit to create and deploy your own cutting-edge AI agent services.** Explore the [Agent Service Toolkit](https://github.com/JoshuaC215/agent-service-toolkit) to build sophisticated AI applications with a user-friendly interface.

[![build status](https://github.com/JoshuaC215/agent-service-toolkit/actions/workflows/test.yml/badge.svg)](https://github.com/JoshuaC215/agent-service-toolkit/actions/workflows/test.yml)
[![codecov](https://codecov.io/github/JoshuaC215/agent-service-toolkit/graph/badge.svg?token=5MTJSYWD05)](https://codecov.io/github/JoshuaC215/agent-service-toolkit)
[![Python Version](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2FJoshuaC215%2Fagent-service-toolkit%2Frefs%2Fheads%2Fmain%2Fpyproject.toml)](https://github.com/JoshuaC215/agent-service-toolkit/blob/main/pyproject.toml)
[![GitHub License](https://img.shields.io/github/license/JoshuaC215/agent-service-toolkit)](https://github.com/JoshuaC215/agent-service-toolkit/blob/main/LICENSE)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_red.svg)](https://agent-service-toolkit.streamlit.app/)

## Key Features:

*   **LangGraph Agent Integration:** Leverage a customizable agent built with the latest LangGraph features (v0.3), including `interrupt()`, `Command`, `Store`, and `langgraph-supervisor`.
*   **FastAPI Service:** Serve your agent efficiently with both streaming and non-streaming endpoints.
*   **Advanced Streaming Support:** Experience a novel approach to token and message-based streaming for enhanced user interaction.
*   **Intuitive Streamlit Interface:** Provide users with a friendly chat interface for effortless interaction with your AI agents.
*   **Multi-Agent Capabilities:** Host multiple agents within your service, accessible via URL paths, with agent information available at `/info`.
*   **Asynchronous Architecture:** Benefit from an async/await design, enabling highly efficient handling of concurrent requests.
*   **Content Moderation:** Integrate LlamaGuard for content moderation (requires a Groq API key).
*   **RAG Agent Implementation:** Includes a basic RAG agent using ChromaDB (see documentation in `docs/RAG_Assistant.md`).
*   **Feedback Mechanism:** Utilize a star-based feedback system integrated with LangSmith for user input and improvement.
*   **Docker Support:** Utilize Dockerfiles and a Docker Compose file for streamlined development and deployment.
*   **Comprehensive Testing:** Includes robust unit and integration tests, ensuring reliability.

## Quickstart

1.  **Prerequisites:** At least one LLM API key (e.g., OpenAI) is required.
2.  **Install Dependencies:** Using `uv` (recommended):

    ```bash
    curl -LsSf https://astral.sh/uv/0.7.19/install.sh | sh
    uv sync --frozen
    source .venv/bin/activate
    ```
    Or using `pip`:
    ```bash
    pip install .
    ```
3.  **Run the Service:**

    ```bash
    python src/run_service.py
    ```
4.  **Run the Streamlit App (in a separate terminal):**

    ```bash
    streamlit run src/streamlit_app.py
    ```

**[Try the App!](https://agent-service-toolkit.streamlit.app/)**

<a href="https://agent-service-toolkit.streamlit.app/"><img src="media/app_screenshot.png" width="600"></a>

## Architecture Overview:

![Agent Architecture Diagram](media/agent_architecture.png)

## Setting Up & Using the Toolkit

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/JoshuaC215/agent-service-toolkit.git
    cd agent-service-toolkit
    ```
2.  **Configure Environment Variables:** Create a `.env` file in the project root.  Consult the [`.env.example`](./.env.example) for available settings and required API keys.

### Further Setup Options:

*   [Setting up Ollama](docs/Ollama.md)
*   [Setting up VertexAI](docs/VertexAI.md)
*   [Setting up RAG with ChromaDB](docs/RAG_Assistant.md)

### Customize Your Agent

1.  Add a new agent to the `src/agents` directory.
2.  Import and add your new agent to the `agents` dictionary in `src/agents/agents.py`.
3.  Adjust the Streamlit interface in `src/streamlit_app.py` as needed.

### Docker Deployment

1.  **Prerequisites:** Docker and Docker Compose (version >= 2.23.0) installed.
2.  Create a `.env` file based on `.env.example` and populate with your API keys.
3.  **Run with Docker Compose Watch:**

    ```bash
    docker compose watch
    ```

    This command automates the build and launches the agent service, Streamlit app, and a PostgreSQL database (for the service).
4.  Access the Streamlit app at `http://localhost:8501` and the API at `http://0.0.0.0:8080`. The OpenAPI docs are accessible at `http://0.0.0.0:8080/redoc`.
5.  Use `docker compose down` to stop the services.

### Local Development (Without Docker)

1.  Create a virtual environment and install dependencies using `uv sync --frozen` or `pip install .`
2.  Run the FastAPI server: `python src/run_service.py`
3.  Run the Streamlit app: `streamlit run src/streamlit_app.py`

### Building on top of the AgentClient

The `src/client/client.AgentClient` is provided to build other apps on top of the agent. See `src/run_client.py` for example usage.

### LangGraph Studio Integration

The toolkit supports [LangGraph Studio](https://langchain-ai.github.io/langgraph/concepts/langgraph_studio/) for agent development.  Run `langgraph dev` to launch the Studio.  Customize the `langgraph.json` file as needed.

## Projects Built with or Inspired by Agent-Service-Toolkit

*   **[PolyRAG](https://github.com/QuentinFuxa/PolyRAG)** - Extends agent-service-toolkit with RAG capabilities over both PostgreSQL databases and PDF documents.
*   **[alexrisch/agent-web-kit](https://github.com/alexrisch/agent-web-kit)** - A Next.JS frontend for agent-service-toolkit
*   **[raushan-in/dapa](https://github.com/raushan-in/dapa)** - Digital Arrest Protection App (DAPA) enables users to report financial scams and frauds efficiently via a user-friendly platform.

**[Add a Project to the List!](https://github.com/JoshuaC215/agent-service-toolkit)**

## Contributing

Contributions are welcome! Please submit a Pull Request.
To run tests:

1.  Activate your virtual environment.
2.  Install development dependencies and pre-commit hooks: `uv sync --frozen` or `pip install .`, then `pre-commit install`.
3.  Run tests with pytest: `pytest`

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**[Visit the Original Repository](https://github.com/JoshuaC215/agent-service-toolkit)**
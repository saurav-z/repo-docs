# AI Agent Service Toolkit: Build and Deploy Powerful AI Agents with Ease

**Build and deploy sophisticated AI agent services quickly using LangGraph, FastAPI, and Streamlit. [Explore the toolkit on GitHub](https://github.com/JoshuaC215/agent-service-toolkit)**

[![build status](https://github.com/JoshuaC215/agent-service-toolkit/actions/workflows/test.yml/badge.svg)](https://github.com/JoshuaC215/agent-service-toolkit/actions/workflows/test.yml)
[![codecov](https://codecov.io/github/JoshuaC215/agent-service-toolkit/graph/badge.svg?token=5MTJSYWD05)](https://codecov.io/github/JoshuaC215/agent-service-toolkit)
[![Python Version](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2FJoshuaC215%2Fagent-service-toolkit%2Frefs%2Fheads%2Fmain%2Fpyproject.toml)](https://github.com/JoshuaC215/agent-service-toolkit/blob/main/pyproject.toml)
[![GitHub License](https://img.shields.io/github/license/JoshuaC215/agent-service-toolkit)](https://github.com/JoshuaC215/agent-service-toolkit/blob/main/LICENSE)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_red.svg)](https://agent-service-toolkit.streamlit.app/)

This comprehensive toolkit provides a complete framework for building, deploying, and interacting with AI agents, offering a streamlined approach to developing LangGraph-based projects. It includes a LangGraph agent, FastAPI service, a client, and a Streamlit app for a user-friendly chat interface. Data structures and settings are built with [Pydantic](https://github.com/pydantic/pydantic).

**[🎥 Watch a video walkthrough of the repo and app](https://www.youtube.com/watch?v=pdYVHw_YCNY)**

## Key Features

*   **LangGraph Agent with Latest Features:** Leverage a customizable agent built with the LangGraph framework, including human-in-the-loop, flow control, long-term memory, and `langgraph-supervisor`.
*   **FastAPI Service:** Efficiently serve your agent with both streaming and non-streaming endpoints.
*   **Advanced Streaming:** Benefit from a novel approach supporting both token-based and message-based streaming.
*   **Streamlit Interface:** Interact with your agent through a user-friendly chat interface.
*   **Multiple Agent Support:** Run multiple agents within the service, accessible via URL paths, with agent and model information available at `/info`.
*   **Asynchronous Design:** Handle concurrent requests efficiently with an async/await design.
*   **Content Moderation:** Implement content moderation using LlamaGuard (requires Groq API key).
*   **RAG Agent:** Explore a basic RAG agent implementation using ChromaDB (see [docs](docs/RAG_Assistant.md)).
*   **Feedback Mechanism:** Integrate a star-based feedback system with LangSmith.
*   **Docker Support:** Simplify development and deployment with Dockerfiles and a docker-compose file.
*   **Comprehensive Testing:** Ensure quality with robust unit and integration tests.

## Architecture

### [Try the App!](https://agent-service-toolkit.streamlit.app/)

<a href="https://agent-service-toolkit.streamlit.app/"><img src="media/app_screenshot.png" width="600"></a>

### Architecture Diagram

<img src="media/agent_architecture.png" width="600">

## Quickstart

### Run Directly in Python

```bash
# At least one LLM API key is required
echo 'OPENAI_API_KEY=your_openai_api_key' >> .env

# uv is the recommended way to install agent-service-toolkit, but "pip install ." also works
# For uv installation options, see: https://docs.astral.sh/uv/getting-started/installation/
curl -LsSf https://astral.sh/uv/0.7.19/install.sh | sh

# Install dependencies. "uv sync" creates .venv automatically
uv sync --frozen
source .venv/bin/activate
python src/run_service.py

# In another shell
source .venv/bin/activate
streamlit run src/streamlit_app.py
```

### Run with Docker

```bash
echo 'OPENAI_API_KEY=your_openai_api_key' >> .env
docker compose watch
```

## Setup and Usage

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/JoshuaC215/agent-service-toolkit.git
    cd agent-service-toolkit
    ```

2.  **Set up environment variables:**

    Create a `.env` file in the root directory. At least one LLM API key or configuration is required. See the [`.env.example` file](./.env.example) for a full list of available environment variables.

3.  **Run the agent service and Streamlit app:** Choose between local Python execution or Docker for a more streamlined setup.

    *   **Docker:** Recommended for easier setup and automatic reloading. Follow the Docker Setup instructions below.
    *   **Local Python:** Requires creating a virtual environment and installing dependencies. See the Local Development without Docker instructions below.

### Additional Setup

*   [Setting up Ollama](docs/Ollama.md)
*   [Setting up VertexAI](docs/VertexAI.md)
*   [Setting up RAG with ChromaDB](docs/RAG_Assistant.md)

### Building or Customizing Your Own Agent

1.  **Add your new agent:** Place your agent in the `src/agents` directory. Customize from `research_assistant.py` or `chatbot.py`.
2.  **Register your agent:** Import and add your agent to the `agents` dictionary in `src/agents/agents.py`.
3.  **Adjust the Streamlit interface:** Customize `src/streamlit_app.py` to match your agent's capabilities.

### Handling Private Credentials

The `privatecredentials/` directory is provided for file-based credential files or certificates, which are ignored by git and docker's build process. See [Working with File-based Credentials](docs/File_Based_Credentials.md).

### Docker Setup

This project includes a Docker setup for easy development and deployment. The `compose.yaml` file defines three services: `postgres`, `agent_service` and `streamlit_app`. The `Dockerfile` for each service is in their respective directories.

For local development, we recommend using [docker compose watch](https://docs.docker.com/compose/file-watch/). This feature allows for a smoother development experience by automatically updating your containers when changes are detected in your source code.

1.  **Prerequisites:** Ensure Docker and Docker Compose (>= [v2.23.0](https://docs.docker.com/compose/release-notes/#2230)) are installed.
2.  **Create `.env`:** From the `.env.example`, create a `.env` file with required LLM API keys.
    ```bash
    cp .env.example .env
    # Edit .env to add your API keys
    ```
3.  **Build and launch:**

    ```bash
    docker compose watch
    ```

    This will automatically start a PostgreSQL database, the agent service, and the Streamlit app.

4.  **Automatic Updates:** Changes in relevant Python files trigger updates for the services. If `pyproject.toml` or `uv.lock` are changed, rebuild with `docker compose up --build`.
5.  **Access:**
    *   Streamlit app: `http://localhost:8501`
    *   Agent service API: `http://0.0.0.0:8080`, OpenAPI docs at `http://0.0.0.0:8080/redoc`
6.  **Stop Services:**  Use `docker compose down`.

### Building Other Apps on the AgentClient

The included `src/client/client.AgentClient` allows interaction with the agent service.  Supports synchronous/asynchronous invocations and streaming/non-streaming requests.
See `src/run_client.py` for examples.

```python
from client import AgentClient
client = AgentClient()

response = client.invoke("Tell me a brief joke?")
response.pretty_print()
# ================================== Ai Message ==================================
#
# A man walked into a library and asked the librarian, "Do you have any books on Pavlov's dogs and Schrödinger's cat?"
# The librarian replied, "It rings a bell, but I'm not sure if it's here or not."
```

### Development with LangGraph Studio

The agent supports [LangGraph Studio](https://langchain-ai.github.io/langgraph/concepts/langgraph_studio/). Add your `.env` file, then launch with `langgraph dev`.  Customize `langgraph.json` as needed. See the [local quickstart](https://langchain-ai.github.io/langgraph/cloud/how-tos/studio/quick_start/#local-development-server).

### Local Development without Docker

1.  **Set up virtual environment:**

    ```bash
    uv sync --frozen
    source .venv/bin/activate
    ```

2.  **Run the FastAPI server:**

    ```bash
    python src/run_service.py
    ```

3.  **Run the Streamlit app:**

    ```bash
    streamlit run src/streamlit_app.py
    ```

    Open your browser to the URL provided by Streamlit.

## Projects Built With or Inspired By This Toolkit

*   **[PolyRAG](https://github.com/QuentinFuxa/PolyRAG)** - RAG capabilities over PostgreSQL databases and PDF documents.
*   **[alexrisch/agent-web-kit](https://github.com/alexrisch/agent-web-kit)** - A Next.JS frontend for agent-service-toolkit
*   **[raushan-in/dapa](https://github.com/raushan-in/dapa)** - Digital Arrest Protection App (DAPA) enables users to report financial scams and frauds efficiently via a user-friendly platform.

**Contribute by adding projects to the list! Create a pull request or open a discussion.**

## Contributing

Contributions are welcome! Submit a Pull Request with your improvements.

To run tests locally:

1.  Ensure you're in the project root and the virtual environment is activated.
2.  Install development dependencies and pre-commit hooks:

    ```bash
    uv sync --frozen
    pre-commit install
    ```

3.  Run tests:

    ```bash
    pytest
    ```

## License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/JoshuaC215/agent-service-toolkit/blob/main/LICENSE) file for details.
# AI Agent Service Toolkit: Build and Deploy Powerful AI Agents

**Create, deploy, and interact with cutting-edge AI agents effortlessly with the AI Agent Service Toolkit – your complete solution for building AI-powered applications.** Explore the original repository: [https://github.com/JoshuaC215/agent-service-toolkit](https://github.com/JoshuaC215/agent-service-toolkit).

[![build status](https://github.com/JoshuaC215/agent-service-toolkit/actions/workflows/test.yml/badge.svg)](https://github.com/JoshuaC215/agent-service-toolkit/actions/workflows/test.yml)
[![codecov](https://codecov.io/github/JoshuaC215/agent-service-toolkit/graph/badge.svg?token=5MTJSYWD05)](https://codecov.io/github/JoshuaC215/agent-service-toolkit)
[![Python Version](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2FJoshuaC215%2Fagent-service-toolkit%2Frefs%2Fheads%2Fmain%2Fpyproject.toml)](https://github.com/JoshuaC215/agent-service-toolkit/blob/main/pyproject.toml)
[![GitHub License](https://img.shields.io/github/license/JoshuaC215/agent-service-toolkit)](https://github.com/JoshuaC215/agent-service-toolkit/blob/main/LICENSE)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_red.svg)](https://agent-service-toolkit.streamlit.app/)

This toolkit provides a robust and comprehensive framework for building AI agent services using LangGraph, FastAPI, and Streamlit.  It offers a streamlined development experience, from defining your agent to deploying a user-friendly interface.  Built with Pydantic for data structures and settings.

**[🎥 Watch a video walkthrough of the repo and app](https://www.youtube.com/watch?v=pdYVHw_YCNY)**

## Key Features

*   **LangGraph-Powered Agents:** Build and customize agents with the latest LangGraph v0.3 features like `interrupt()`, `Command`, `Store`, and `langgraph-supervisor`.
*   **FastAPI Service:**  Deploy your agents with a scalable and efficient FastAPI service, supporting both streaming and non-streaming endpoints.
*   **Advanced Streaming Support:** Experience a novel approach to streaming that supports both token-based and message-based streaming.
*   **Interactive Streamlit Interface:**  Provides an intuitive, user-friendly chat interface for seamless agent interaction.
*   **Multi-Agent Management:** Run multiple agents simultaneously within the service, accessed via URL paths.
*   **Asynchronous Design:** Leverage async/await for efficient handling of concurrent requests and improved performance.
*   **Content Moderation:**  Integrated LlamaGuard for content moderation (requires Groq API key).
*   **RAG Agent Implementation:** A basic Retrieval-Augmented Generation (RAG) agent implementation using ChromaDB (see [docs](docs/RAG_Assistant.md)).
*   **Feedback Mechanism:** Integrate a star-based feedback system with LangSmith for performance tracking.
*   **Docker Support:**  Includes Dockerfiles and a docker compose file for easy development and deployment.
*   **Comprehensive Testing:**  Robust unit and integration tests ensure the reliability of your agents.

## Quickstart

### Try the app!

<a href="https://agent-service-toolkit.streamlit.app/"><img src="media/app_screenshot.png" width="600"></a>

### Run locally (Python)

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

## Architecture Diagram

<img src="media/agent_architecture.png" width="600">

## Core Components & File Structure

The toolkit is structured with the following key files and directories:

*   `src/agents/`: Agent definitions with different capabilities.
*   `src/schema/`: Protocol schemas.
*   `src/core/`: Core modules, including LLM definitions and settings.
*   `src/service/service.py`: FastAPI service for serving the agents.
*   `src/client/client.py`: Client for interacting with the agent service.
*   `src/streamlit_app.py`: Streamlit application for the chat interface.
*   `tests/`: Unit and integration tests.

## Setup and Usage

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/JoshuaC215/agent-service-toolkit.git
    cd agent-service-toolkit
    ```

2.  **Configure Environment Variables:**
    Create a `.env` file in the root directory to store required API keys and configuration settings.  Refer to the [`.env.example`](./.env.example) file for a full list of available variables.

3.  **Run the Service:** Launch both the agent service and the Streamlit app locally using either Docker (recommended) or Python. Docker simplifies setup and offers automatic reloading for code changes.

### Additional setup for specific AI providers

*   [Setting up Ollama](docs/Ollama.md)
*   [Setting up VertexAI](docs/VertexAI.md)
*   [Setting up RAG with ChromaDB](docs/RAG_Assistant.md)

### Building or Customizing Your Own Agent

1.  **Add your agent:** Add your new agent to the `src/agents` directory. You can copy `research_assistant.py` or `chatbot.py` and modify it to change the agent's behavior and tools.
2.  **Register your agent:** Import and add your new agent to the `agents` dictionary in `src/agents/agents.py`. Your agent can then be accessed via the API at `/<your_agent_name>/invoke` or `/<your_agent_name>/stream`.
3.  **Update the UI:** Adjust the Streamlit interface in `src/streamlit_app.py` to match your agent's capabilities.

### Handling Private Credential files

For agents and LLMs requiring file-based credentials or certificates, the `privatecredentials/` directory is provided (excluded from Git and Docker builds).  See [Working with File-based Credentials](docs/File_Based_Credentials.md) for usage recommendations.

## Docker Setup

This project provides a Docker-based development and deployment setup. The `compose.yaml` file defines services for `postgres`, `agent_service`, and `streamlit_app`, with corresponding `Dockerfile`s.

For development, `docker compose watch` is recommended for auto-updating containers on code changes.

1.  Install Docker and Docker Compose (>= v2.23.0).

2.  Create a `.env` file from `.env.example`, providing at least one LLM API key (e.g., `OPENAI_API_KEY`).

    ```bash
    cp .env.example .env
    # Edit .env to add your API keys
    ```

3.  Build and run services in watch mode:

    ```bash
    docker compose watch
    ```

    This will:
    * Start a PostgreSQL database.
    * Start the FastAPI agent service.
    * Start the Streamlit user interface.

4.  Changes to code files will automatically trigger updates.  Changes to `pyproject.toml` or `uv.lock` require a rebuild using `docker compose up --build`.

5.  Access the Streamlit app at `http://localhost:8501`.

6.  Access the agent service API at `http://0.0.0.0:8080`. Use the OpenAPI docs at `http://0.0.0.0:8080/redoc`.

7.  Stop services with `docker compose down`.

## Building Other Apps on the AgentClient

The `src/client/client.AgentClient` provides a flexible interface for interacting with the agent service, supporting synchronous, asynchronous, streaming, and non-streaming requests.

See `src/run_client.py` for usage examples.  A basic example:

```python
from client import AgentClient
client = AgentClient()

response = client.invoke("Tell me a brief joke?")
response.pretty_print()
```

## Development with LangGraph Studio

This toolkit supports LangGraph Studio ([https://langchain-ai.github.io/langgraph/concepts/langgraph_studio/](https://langchain-ai.github.io/langgraph/concepts/langgraph_studio/)), the IDE for developing agents in LangGraph.

After installing `uv sync` and creating an `.env` file as above, you can launch LangGraph Studio with `langgraph dev`. Customize `langgraph.json` as needed. See the [local quickstart](https://langchain-ai.github.io/langgraph/cloud/how-tos/studio/quick_start/#local-development-server) for details.

## Local Development Without Docker

You can also develop locally without Docker, using a Python virtual environment.

1.  Create a virtual environment and install dependencies:

    ```bash
    uv sync --frozen
    source .venv/bin/activate
    ```

2.  Run the FastAPI server:

    ```bash
    python src/run_service.py
    ```

3.  In a separate terminal, run the Streamlit app:

    ```bash
    streamlit run src/streamlit_app.py
    ```

4.  Access the Streamlit app at `http://localhost:8501`.

## Projects Built With or Inspired By

*   **[PolyRAG](https://github.com/QuentinFuxa/PolyRAG)** - RAG capabilities over PostgreSQL databases and PDF documents.
*   **[alexrisch/agent-web-kit](https://github.com/alexrisch/agent-web-kit)** - Next.JS frontend.
*   **[raushan-in/dapa](https://github.com/raushan-in/dapa)** - Digital Arrest Protection App (DAPA) for reporting financial scams.

**Please submit a PR or discussion for any new projects!**

## Contributing

Contributions are welcome! Submit Pull Requests.  To run tests:

1.  Activate your virtual environment.

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

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
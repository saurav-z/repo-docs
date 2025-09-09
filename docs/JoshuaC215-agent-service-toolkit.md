# Agent Service Toolkit: Build Powerful AI Agent Services with LangGraph, FastAPI & Streamlit

**Easily build and deploy robust AI agent services with this comprehensive toolkit built with LangGraph, FastAPI, and Streamlit. Explore the original repo [here](https://github.com/JoshuaC215/agent-service-toolkit).**

[![build status](https://github.com/JoshuaC215/agent-service-toolkit/actions/workflows/test.yml/badge.svg)](https://github.com/JoshuaC215/agent-service-toolkit/actions/workflows/test.yml) [![codecov](https://codecov.io/github/JoshuaC215/agent-service-toolkit/graph/badge.svg?token=5MTJSYWD05)](https://codecov.io/github/JoshuaC215/agent-service-toolkit) [![Python Version](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2FJoshuaC215%2Fagent-service-toolkit%2Frefs%2Fheads%2Fmain%2Fpyproject.toml)](https://github.com/JoshuaC215/agent-service-toolkit/blob/main/pyproject.toml)
[![GitHub License](https://img.shields.io/github/license/JoshuaC215/agent-service-toolkit)](https://github.com/JoshuaC215/agent-service-toolkit/blob/main/LICENSE) [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_red.svg)](https://agent-service-toolkit.streamlit.app/)

## Key Features

*   **LangGraph Agent & Latest Features**: Leverage a customizable agent built on the cutting-edge LangGraph framework. Includes implementations for `interrupt()`, `Command`, `Store`, and `langgraph-supervisor`.
*   **FastAPI Service**: Serve your agent with both streaming and non-streaming endpoints.
*   **Advanced Streaming Support**: Experience a novel streaming approach for token-based and message-based streaming.
*   **Interactive Streamlit Interface**: Provide a user-friendly chat interface for seamless agent interaction.
*   **Multi-Agent Support**: Run multiple agents within the service, accessible by URL path, and view available models via `/info`.
*   **Asynchronous Design**: Benefit from an efficient asynchronous architecture for handling concurrent requests.
*   **Content Moderation**: Includes LlamaGuard integration for content moderation (requires Groq API key).
*   **RAG Agent**: A basic RAG agent implementation using ChromaDB - see [docs](docs/RAG_Assistant.md).
*   **Feedback Integration**: Integrate a star-based feedback system with LangSmith.
*   **Docker Support**: Simplify development and deployment with included Dockerfiles and a Docker Compose file.
*   **Comprehensive Testing**: Ensure code quality with robust unit and integration tests throughout the toolkit.

## Quickstart

### Try the App!

<a href="https://agent-service-toolkit.streamlit.app/"><img src="media/app_screenshot.png" width="600"></a>

### Run with Docker

```bash
echo 'OPENAI_API_KEY=your_openai_api_key' >> .env
docker compose watch
```

### Run directly in python

```sh
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

## Architecture

<img src="media/agent_architecture.png" width="600">

## Key Files

*   `src/agents/`: Contains definitions for various agents.
*   `src/schema/`: Defines the protocol schema.
*   `src/core/`: Core modules, including LLM definitions and settings.
*   `src/service/service.py`: The FastAPI service for serving agents.
*   `src/client/client.py`: Client for interacting with the agent service.
*   `src/streamlit_app.py`: The Streamlit app providing the chat interface.
*   `tests/`: Includes unit and integration tests.

## Setup and Usage

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/JoshuaC215/agent-service-toolkit.git
    cd agent-service-toolkit
    ```

2.  **Set Up Environment Variables:**

    Create a `.env` file in the root directory. At least one LLM API key is required.  See the [`.env.example` file](./.env.example) for a comprehensive list of available environment variables.

3.  Run the agent service and the Streamlit app locally with Docker or just using Python.  Docker is recommended for simpler environment setup.

## Additional Setup & Customization

*   [Setting up Ollama](docs/Ollama.md)
*   [Setting up VertexAI](docs/VertexAI.md)
*   [Setting up RAG with ChromaDB](docs/RAG_Assistant.md)

### Building or Customizing Your Own Agent

1.  Add your new agent to the `src/agents` directory.
2.  Import and add your new agent to the `agents` dictionary in `src/agents/agents.py`.
3.  Adjust the Streamlit interface in `src/streamlit_app.py` to match your agent's capabilities.

### Handling Private Credential files

The `privatecredentials/` directory is provided for storing file-based credentials like API keys and certificates.  It is ignored by git and Docker builds. See [Working with File-based Credentials](docs/File_Based_Credentials.md) for recommended usage.

## Docker Setup

1.  Ensure Docker and Docker Compose are installed.
2.  Create a `.env` file from `.env.example` and add your API keys.
3.  Build and launch with `docker compose watch`.
4.  Access the Streamlit app at `http://localhost:8501`.
5.  Access the agent service API at `http://0.0.0.0:8080`.
6.  Use `docker compose down` to stop the services.

## Building Apps on the AgentClient

The `src/client/client.AgentClient` provides a flexible interface for building applications on top of the agent service.  It supports both synchronous and asynchronous invocations, and streaming and non-streaming requests. See `src/run_client.py` for examples.

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

## Development with LangGraph Studio

The agent supports [LangGraph Studio](https://langchain-ai.github.io/langgraph/concepts/langgraph_studio/), the IDE for developing agents in LangGraph. Launch LangGraph Studio with `langgraph dev`. Customize `langgraph.json` as needed.

## Local Development without Docker

1.  Create a virtual environment:

    ```bash
    uv sync --frozen
    source .venv/bin/activate
    ```

2.  Run the FastAPI server:

    ```bash
    python src/run_service.py
    ```

3.  Run the Streamlit app in a separate terminal:

    ```bash
    streamlit run src/streamlit_app.py
    ```

4.  Open your browser to the Streamlit URL (usually `http://localhost:8501`).

## Projects Built With or Inspired By This Toolkit

*   **[PolyRAG](https://github.com/QuentinFuxa/PolyRAG)** - RAG capabilities over PostgreSQL and PDF documents.
*   **[alexrisch/agent-web-kit](https://github.com/alexrisch/agent-web-kit)** - A Next.JS frontend.
*   **[raushan-in/dapa](https://github.com/raushan-in/dapa)** - Digital Arrest Protection App (DAPA).

## Contributing

Contributions are welcome! Please submit a Pull Request.

To run the tests:

1.  Ensure you're in the project root and have activated your virtual environment.
2.  Install the development dependencies and pre-commit hooks:

    ```bash
    uv sync --frozen
    pre-commit install
    ```

3.  Run the tests:

    ```bash
    pytest
    ```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
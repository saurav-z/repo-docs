# AI Agent Service Toolkit: Build and Deploy Powerful AI Agents

**Create robust and customizable AI agent services with ease using LangGraph, FastAPI, and Streamlit. ** Explore the [original repository](https://github.com/JoshuaC215/agent-service-toolkit) for comprehensive details.

[![Build Status](https://github.com/JoshuaC215/agent-service-toolkit/actions/workflows/test.yml/badge.svg)](https://github.com/JoshuaC215/agent-service-toolkit/actions/workflows/test.yml)
[![Codecov](https://codecov.io/github/JoshuaC215/agent-service-toolkit/graph/badge.svg?token=5MTJSYWD05)](https://codecov.io/github/JoshuaC215/agent-service-toolkit)
[![Python Version](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2FJoshuaC215%2Fagent-service-toolkit%2Frefs%2Fheads%2Fmain%2Fpyproject.toml)](https://github.com/JoshuaC215/agent-service-toolkit/blob/main/pyproject.toml)
[![GitHub License](https://img.shields.io/github/license/JoshuaC215/agent-service-toolkit)](https://github.com/JoshuaC215/agent-service-toolkit/blob/main/LICENSE)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_red.svg)](https://agent-service-toolkit.streamlit.app/)

This toolkit provides a complete framework for developing and deploying AI agent services using cutting-edge technologies. It offers a streamlined approach, going from agent definition to user interface, accelerating your LangGraph-based project.

## Key Features

*   **LangGraph Agent Integration:** Leverages the latest features of LangGraph (v0.3), including human-in-the-loop, flow control, long-term memory, and supervisor capabilities.
*   **FastAPI Service:** Efficiently serves your AI agents, supporting both streaming and non-streaming endpoints.
*   **Advanced Streaming:** Includes a novel approach to support both token-based and message-based streaming.
*   **User-Friendly Streamlit Interface:** Provides an intuitive chat interface for easy interaction with your agents.
*   **Multi-Agent Support:** Run and access multiple agents within the service using URL paths; available agents and models are described in `/info`.
*   **Asynchronous Design:** Utilizes async/await for optimal performance, handling concurrent requests effectively.
*   **Content Moderation:** Integrates LlamaGuard for content moderation (requires a Groq API key).
*   **RAG Agent Implementation:** A basic RAG agent using ChromaDB is included (see [RAG Assistant Docs](docs/RAG_Assistant.md)).
*   **Feedback Mechanism:** Incorporates a star-based feedback system and integrates with LangSmith.
*   **Docker Support:** Simplifies development and deployment with comprehensive Dockerfiles and a Docker Compose file.
*   **Comprehensive Testing:** Includes robust unit and integration tests for reliability.

## Quickstart

1.  **Set up environment variables**
    Create a `.env` file in the root directory with required API keys and configurations. See the [`.env.example`](./.env.example) for available environment variables.
2.  **Install Dependencies and Run**

    ```bash
    # Recommended: Install with uv
    curl -LsSf https://astral.sh/uv/0.7.19/install.sh | sh
    uv sync --frozen
    source .venv/bin/activate

    #Run the service
    python src/run_service.py
    
    # In another shell
    source .venv/bin/activate
    streamlit run src/streamlit_app.py
    ```

3.  **Alternatively, run with Docker**

    ```bash
    docker compose watch
    ```

## Architecture

[<img src="media/agent_architecture.png" width="600">](https://agent-service-toolkit.streamlit.app/)

## Core Components

*   `src/agents/`: Contains definitions for various AI agents.
*   `src/schema/`: Defines the data protocol schema.
*   `src/core/`: Includes core modules like LLM definitions and settings.
*   `src/service/service.py`: The FastAPI service for serving agents.
*   `src/client/client.py`: A client to interact with the agent service.
*   `src/streamlit_app.py`: The Streamlit application providing a user interface.
*   `tests/`: Contains unit and integration tests.

## Getting Started

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/JoshuaC215/agent-service-toolkit.git
    cd agent-service-toolkit
    ```

2.  **Set up your environment**: Follow the Quickstart instructions above.

3.  **Access the application**: Access the Streamlit app via `http://localhost:8501`.
4.  **Access the API**: The agent service API is available at `http://0.0.0.0:8080`. OpenAPI docs at `http://0.0.0.0:8080/redoc`.

## Customization and Usage

### Building or Customizing Your Own Agent

1.  Create your agent in the `src/agents/` directory (e.g., by copying `research_assistant.py` or `chatbot.py`).
2.  Import and add your agent to the `agents` dictionary in `src/agents/agents.py`.
3.  Adjust the Streamlit interface in `src/streamlit_app.py` to suit your agent's functionality.

### Handling Private Credential Files

The `privatecredentials/` directory is available for file-based credentials. Use [Working with File-based Credentials](docs/File_Based_Credentials.md) for instructions.

### Docker Setup

For local development, use `docker compose watch`.

1.  Make sure Docker and Docker Compose are installed.
2.  Create a `.env` file from `.env.example` and populate with your API keys.
3.  Run `docker compose watch` in the terminal. The application will be available at `http://localhost:8501`.
4.  Use `docker compose down` to stop the services.

### Building other apps on the AgentClient

The repo includes a generic `src/client/client.AgentClient` that can be used to interact with the agent service. This client is designed to be flexible and can be used to build other apps on top of the agent. It supports both synchronous and asynchronous invocations, and streaming and non-streaming requests.

See the `src/run_client.py` file for full examples of how to use the `AgentClient`. A quick example:

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

The agent supports [LangGraph Studio](https://langchain-ai.github.io/langgraph/concepts/langgraph_studio/), the IDE for developing agents in LangGraph.

`langgraph-cli[inmem]` is installed with `uv sync`. You can simply add your `.env` file to the root directory as described above, and then launch LangGraph Studio with `langgraph dev`. Customize `langgraph.json` as needed. See the [local quickstart](https://langchain-ai.github.io/langgraph/cloud/how-tos/studio/quick_start/#local-development-server) to learn more.

### Local development without Docker

1.  Create a virtual environment and install dependencies:

    ```sh
    uv sync --frozen
    source .venv/bin/activate
    ```

2.  Run the FastAPI server:

    ```sh
    python src/run_service.py
    ```

3.  In a separate terminal, run the Streamlit app:

    ```sh
    streamlit run src/streamlit_app.py
    ```

4.  Open your browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

## Projects Built With/Inspired By agent-service-toolkit

*   **[PolyRAG](https://github.com/QuentinFuxa/PolyRAG)** - Extends agent-service-toolkit with RAG capabilities over both PostgreSQL databases and PDF documents.
*   **[alexrisch/agent-web-kit](https://github.com/alexrisch/agent-web-kit)** - A Next.JS frontend for agent-service-toolkit
*   **[raushan-in/dapa](https://github.com/raushan-in/dapa)** - Digital Arrest Protection App (DAPA) enables users to report financial scams and frauds efficiently via a user-friendly platform.

**Add your project here! Create a pull request to edit the README or open a discussion to add your project.**

## Contributing

Contributions are welcomed!  See the Contributing section of the original [repo](https://github.com/JoshuaC215/agent-service-toolkit) for details. To run tests:

1.  Ensure you're in the project root directory and have activated your virtual environment.

2.  Install the development dependencies and pre-commit hooks:

    ```sh
    uv sync --frozen
    pre-commit install
    ```

3.  Run the tests using pytest:

    ```sh
    pytest
    ```

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/JoshuaC215/agent-service-toolkit/blob/main/LICENSE) file for details.
# AI Agent Service Toolkit: Build and Deploy Powerful AI Agents with Ease

**Quickly build, deploy, and interact with AI agents using LangGraph, FastAPI, Streamlit, and more.**  Find the original repo [here](https://github.com/JoshuaC215/agent-service-toolkit).

[![build status](https://github.com/JoshuaC215/agent-service-toolkit/actions/workflows/test.yml/badge.svg)](https://github.com/JoshuaC215/agent-service-toolkit/actions/workflows/test.yml) [![codecov](https://codecov.io/github/JoshuaC215/agent-service-toolkit/graph/badge.svg?token=5MTJSYWD05)](https://codecov.io/github/JoshuaC215/agent-service-toolkit) [![Python Version](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2FJoshuaC215%2Fagent-service-toolkit%2Frefs%2Fheads%2Fmain%2Fpyproject.toml)](https://github.com/JoshuaC215/agent-service-toolkit/blob/main/pyproject.toml)
[![GitHub License](https://img.shields.io/github/license/JoshuaC215/agent-service-toolkit)](https://github.com/JoshuaC215/agent-service-toolkit/blob/main/LICENSE) [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_red.svg)](https://agent-service-toolkit.streamlit.app/)

This toolkit provides a comprehensive solution for building and deploying AI agents using cutting-edge technologies. It includes a LangGraph agent, a FastAPI service for deployment, a client for interaction, and a Streamlit app for a user-friendly interface. Built with Pydantic for robust data structures and settings.  This toolkit streamlines the development process, offering a complete setup from agent definition to user interface, making it easier to get started with LangGraph-based projects.

**[🎥 Watch a video walkthrough of the repo and app](https://www.youtube.com/watch?v=pdYVHw_YCNY)**

## Key Features

*   **LangGraph Agent with Latest Features:** Leverage the power of LangGraph with a customizable agent, incorporating the newest features like human-in-the-loop (`interrupt()`), flow control (`Command`), long-term memory (`Store`), and `langgraph-supervisor`.
*   **FastAPI Service:** Deploy your agent with a robust FastAPI service, supporting both streaming and non-streaming endpoints for flexible interaction.
*   **Advanced Streaming:**  Experience a novel approach to streaming, supporting both token-based and message-based streaming.
*   **Streamlit Chat Interface:**  Interact with your agent through an intuitive Streamlit chat interface.
*   **Multi-Agent Support:** Run and manage multiple agents within the service, accessible via URL paths, with agent and model information available at `/info`.
*   **Asynchronous Design:** Benefit from efficient handling of concurrent requests thanks to an asynchronous architecture.
*   **Content Moderation:** Integrate LlamaGuard for content moderation (requires Groq API key).
*   **RAG Agent:**  Includes a basic RAG agent implementation using ChromaDB.  See [docs](docs/RAG_Assistant.md).
*   **Feedback Mechanism:** Implement a star-based feedback system integrated with LangSmith.
*   **Docker Support:**  Simplify development and deployment with pre-configured Dockerfiles and a `docker compose` file.
*   **Comprehensive Testing:** Includes unit and integration tests to ensure reliability.

## Getting Started

### Try the App!

[<img src="media/app_screenshot.png" width="600">](https://agent-service-toolkit.streamlit.app/)

### Quickstart

Run the project locally using Python:

```bash
# Set your OpenAI API key (required)
echo 'OPENAI_API_KEY=your_openai_api_key' >> .env

# Install dependencies using uv (recommended) or pip
curl -LsSf https://astral.sh/uv/0.7.19/install.sh | sh
uv sync --frozen
source .venv/bin/activate

# Run the FastAPI service
python src/run_service.py

# In a separate terminal, run the Streamlit app
source .venv/bin/activate
streamlit run src/streamlit_app.py
```

Run with Docker:

```bash
# Set your OpenAI API key (required)
echo 'OPENAI_API_KEY=your_openai_api_key' >> .env

# Start the services
docker compose watch

# Access the Streamlit app at http://localhost:8501
# The API will be available at http://0.0.0.0:8080
```

### Architecture Diagram

<img src="media/agent_architecture.png" width="600">

## Key Files

*   `src/agents/`: Agent definitions.
*   `src/schema/`: Protocol schemas.
*   `src/core/`: Core modules, LLM definitions, and settings.
*   `src/service/service.py`: FastAPI service.
*   `src/client/client.py`: Agent service client.
*   `src/streamlit_app.py`: Streamlit chat interface.
*   `tests/`: Unit and integration tests.

## Setup and Usage

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/JoshuaC215/agent-service-toolkit.git
    cd agent-service-toolkit
    ```

2.  **Configure Environment Variables:**

    Create a `.env` file in the root directory.  Refer to [`.env.example`](./.env.example) for available options (API keys, authentication, LangSmith tracing, etc.).  At a minimum, you need an LLM API key.

### Additional Configuration

*   [Setting up Ollama](docs/Ollama.md)
*   [Setting up VertexAI](docs/VertexAI.md)
*   [Setting up RAG with ChromaDB](docs/RAG_Assistant.md)

### Customizing Your Agent

1.  **Add Your Agent:**  Place your new agent in the `src/agents` directory, modifying existing examples like `research_assistant.py` or `chatbot.py`.
2.  **Register Your Agent:**  Import and add your agent to the `agents` dictionary in `src/agents/agents.py`. Your agent will then be accessible via `/your_agent_name/invoke` and `/your_agent_name/stream`.
3.  **Update the UI:** Modify `src/streamlit_app.py` to reflect your agent's functionalities and capabilities.

### Handling Private Credentials

The `privatecredentials/` directory (ignored by Git and Docker build) is provided for storing file-based credentials or certificates. See [Working with File-based Credentials](docs/File_Based_Credentials.md) for guidance.

## Docker Setup

This project offers a streamlined Docker setup using `docker compose` for ease of development and deployment. The `compose.yaml` file orchestrates services like `postgres`, `agent_service`, and `streamlit_app`.

**Follow these steps:**

1.  **Prerequisites:** Ensure Docker and Docker Compose (>= v2.23.0) are installed.
2.  **Configure:** Create a `.env` file from `.env.example` and add your LLM API key (required).
3.  **Build and Run:**

    ```bash
    docker compose watch
    ```

    This starts a PostgreSQL database, the FastAPI service, and the Streamlit app, with automatic updates upon code changes.

4.  **Access:**

    *   Streamlit App: `http://localhost:8501`
    *   Agent Service API: `http://0.0.0.0:8080`
    *   OpenAPI Docs: `http://0.0.0.0:8080/redoc`

5.  **Stop:** Use `docker compose down`.

## AgentClient Usage

The `src/client/client.AgentClient` provides a flexible way to interact with the agent service, supporting both synchronous and asynchronous calls, along with streaming and non-streaming capabilities.

Refer to `src/run_client.py` for detailed usage examples.

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

This project supports [LangGraph Studio](https://langchain-ai.github.io/langgraph/concepts/langgraph_studio/) for IDE-based agent development.

*   Ensure you have your `.env` file in place.
*   Run `langgraph dev`.
*   Customize `langgraph.json` as needed.

## Local Development Without Docker

1.  **Set Up Environment:**

    ```bash
    uv sync --frozen
    source .venv/bin/activate
    ```

2.  **Run FastAPI Service:**

    ```bash
    python src/run_service.py
    ```

3.  **Run Streamlit App:**

    ```bash
    streamlit run src/streamlit_app.py
    ```

    Access the app via the URL provided by Streamlit (usually `http://localhost:8501`).

## Projects Built With and Inspired By Agent Service Toolkit

*   **[PolyRAG](https://github.com/QuentinFuxa/PolyRAG)** - Extends with RAG capabilities over PostgreSQL and PDF documents.
*   **[alexrisch/agent-web-kit](https://github.com/alexrisch/agent-web-kit)** - A Next.JS frontend.
*   **[raushan-in/dapa](https://github.com/raushan-in/dapa)** - Digital Arrest Protection App (DAPA).

**Please submit a pull request or open a discussion to add your project to this list!**

## Contributing

Contributions are welcome!

To run the tests (using local development without Docker):

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

This project is licensed under the MIT License. See the [LICENSE](https://github.com/JoshuaC215/agent-service-toolkit/blob/main/LICENSE) file for details.
# 🤖 AI Agent Service Toolkit: Build and Deploy Powerful AI Agents

**Quickly build and deploy powerful AI agents with LangGraph, FastAPI, and Streamlit using this comprehensive toolkit!**  Access the original repo at [https://github.com/JoshuaC215/agent-service-toolkit](https://github.com/JoshuaC215/agent-service-toolkit).

[![build status](https://github.com/JoshuaC215/agent-service-toolkit/actions/workflows/test.yml/badge.svg)](https://github.com/JoshuaC215/agent-service-toolkit/actions/workflows/test.yml) [![codecov](https://codecov.io/github/JoshuaC215/agent-service-toolkit/graph/badge.svg?token=5MTJSYWD05)](https://codecov.io/github/JoshuaC215/agent-service-toolkit) [![Python Version](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2FJoshuaC215%2Fagent-service-toolkit%2Frefs%2Fheads%2Fmain%2Fpyproject.toml)](https://github.com/JoshuaC215/agent-service-toolkit/blob/main/pyproject.toml)
[![GitHub License](https://img.shields.io/github/license/JoshuaC215/agent-service-toolkit)](https://github.com/JoshuaC215/agent-service-toolkit/blob/main/LICENSE) [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_red.svg)](https://agent-service-toolkit.streamlit.app/)

This toolkit provides a robust foundation for building, deploying, and interacting with AI agents, offering a streamlined development experience with LangGraph, FastAPI, and Streamlit. It includes a pre-built LangGraph agent, a FastAPI service, a client for interaction, and a Streamlit app for a user-friendly chat interface.

**[🎥 Watch a video walkthrough of the repo and app](https://www.youtube.com/watch?v=pdYVHw_YCNY)**

## Key Features

*   **LangGraph Agent with Cutting-Edge Features:** Utilize a customizable agent built with LangGraph, leveraging the latest features including human-in-the-loop, flow control, long-term memory, and `langgraph-supervisor`.
*   **FastAPI Service:** Serve your agent with both streaming and non-streaming endpoints for efficient access.
*   **Advanced Streaming Support:** Benefit from a novel approach to streaming that supports both token-based and message-based streaming.
*   **User-Friendly Streamlit Interface:** Interact with your agent through an intuitive chat interface powered by Streamlit.
*   **Multiple Agent Support:**  Run multiple agents within the service, accessible by URL path, with agent and model information available at `/info`.
*   **Asynchronous Design:** Leverage async/await for optimal handling of concurrent requests, ensuring high performance.
*   **Content Moderation:** Integrate LlamaGuard for content moderation, enhancing safety (requires Groq API key).
*   **RAG Agent Implementation:** Includes a basic RAG agent implementation using ChromaDB (see docs/RAG_Assistant.md).
*   **Feedback Mechanism:** Integrated star-based feedback system using LangSmith.
*   **Docker Support:** Provides Dockerfiles and a `docker compose` file for easy setup and deployment.
*   **Comprehensive Testing:** Benefit from extensive unit and integration tests for a reliable and robust codebase.

## Quickstart

### Try the app!

<a href="https://agent-service-toolkit.streamlit.app/"><img src="media/app_screenshot.png" width="600"></a>

### Run with Docker:

```sh
echo 'OPENAI_API_KEY=your_openai_api_key' >> .env
docker compose watch
```

### Run directly in Python:

```sh
# At least one LLM API key is required
echo 'OPENAI_API_KEY=your_openai_api_key' >> .env

# Install dependencies.
uv sync --frozen
source .venv/bin/activate
python src/run_service.py

# In another shell
source .venv/bin/activate
streamlit run src/streamlit_app.py
```

## Architecture Diagram

<img src="media/agent_architecture.png" width="600">

## Project Structure

*   `src/agents/`:  Agent definitions.
*   `src/schema/`:  Protocol schema.
*   `src/core/`:  Core modules.
*   `src/service/service.py`: FastAPI service.
*   `src/client/client.py`: Agent service client.
*   `src/streamlit_app.py`: Streamlit app.
*   `tests/`: Unit and integration tests.

## Setup and Usage

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/JoshuaC215/agent-service-toolkit.git
    cd agent-service-toolkit
    ```

2.  **Set up environment variables:**

    Create a `.env` file with your API keys and configurations, based on the [.env.example](./.env.example) file.

### Additional Setup

*   [Setting up Ollama](docs/Ollama.md)
*   [Setting up VertexAI](docs/VertexAI.md)
*   [Setting up RAG with ChromaDB](docs/RAG_Assistant.md)

## Building & Customizing Your Agent

1.  Add your agent to the `src/agents` directory.
2.  Add to the `agents` dictionary in `src/agents/agents.py`.
3.  Adjust the Streamlit interface in `src/streamlit_app.py` to match your agent's capabilities.

## Private Credentials

Use the `privatecredentials/` directory for file-based credentials; its contents are ignored by Git and Docker builds.  See [Working with File-based Credentials](docs/File_Based_Credentials.md).

## Docker Setup

Use `docker compose watch` for local development.

1.  Install Docker and Docker Compose (>= v2.23.0).
2.  Create a `.env` file from `.env.example`, adding your API keys.
3.  Build and launch the services: `docker compose watch`.
4.  Access the Streamlit app at `http://localhost:8501`.
5.  Access the API at `http://0.0.0.0:8080`.
6.  Stop the services with `docker compose down`.

## Using the AgentClient

The `src/client/client.AgentClient` can be used to interact with the agent service. See `src/run_client.py` for examples.

```python
from client import AgentClient
client = AgentClient()
response = client.invoke("Tell me a brief joke?")
response.pretty_print()
```

## Development with LangGraph Studio

Leverage [LangGraph Studio](https://langchain-ai.github.io/langgraph/concepts/langgraph_studio/) for agent development. Launch with `langgraph dev` after configuring your `.env` file.

## Local Development Without Docker

1.  Create a virtual environment and install dependencies:

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

## Projects Built With or Inspired by Agent-Service-Toolkit

*   **[PolyRAG](https://github.com/QuentinFuxa/PolyRAG)** - Extends agent-service-toolkit with RAG capabilities.
*   **[alexrisch/agent-web-kit](https://github.com/alexrisch/agent-web-kit)** - A Next.JS frontend for agent-service-toolkit.
*   **[raushan-in/dapa](https://github.com/raushan-in/dapa)** - Digital Arrest Protection App.

**Please submit a PR or open a discussion for new projects!**

## Contributing

Contributions are welcome! Run tests after changes:

1.  Activate your virtual environment.
2.  Install development dependencies and pre-commit hooks:

    ```bash
    uv sync --frozen
    pre-commit install
    ```

3.  Run tests: `pytest`.

## License

MIT License - see the [LICENSE](LICENSE) file.
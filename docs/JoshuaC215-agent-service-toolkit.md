# Agent Service Toolkit: Build Powerful AI Agent Services with Ease

**Supercharge your AI projects with the Agent Service Toolkit – a comprehensive framework for building, deploying, and interacting with advanced AI agents.** [Explore the project on GitHub](https://github.com/JoshuaC215/agent-service-toolkit).

[![Build Status](https://github.com/JoshuaC215/agent-service-toolkit/actions/workflows/test.yml/badge.svg)](https://github.com/JoshuaC215/agent-service-toolkit/actions/workflows/test.yml)
[![Codecov](https://codecov.io/github/JoshuaC215/agent-service-toolkit/graph/badge.svg?token=5MTJSYWD05)](https://codecov.io/github/JoshuaC215/agent-service-toolkit)
[![Python Version](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2FJoshuaC215%2Fagent-service-toolkit%2Frefs%2Fheads%2Fmain%2Fpyproject.toml)](https://github.com/JoshuaC215/agent-service-toolkit/blob/main/pyproject.toml)
[![GitHub License](https://img.shields.io/github/license/JoshuaC215/agent-service-toolkit)](https://github.com/JoshuaC215/agent-service-toolkit/blob/main/LICENSE)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_red.svg)](https://agent-service-toolkit.streamlit.app/)

This toolkit provides a robust foundation for building AI agents, including:

*   **LangGraph Integration:** Leverage the power of LangGraph for building and managing sophisticated AI agents.
*   **FastAPI Service:** Easily deploy your agents as a scalable and efficient API service.
*   **User-Friendly Interface:** A Streamlit application offers an intuitive chat interface for interacting with your agents.
*   **Modular Design:** Built with Pydantic for clear data structures and settings.
*   **Full Toolkit:** Provides a full, robust toolkit to get started with LangGraph-based projects.

**[🎥 Watch a video walkthrough of the repo and app](https://www.youtube.com/watch?v=pdYVHw_YCNY)**

## Key Features

*   **LangGraph Agent:** Build customizable agents using the latest LangGraph features (v0.3), including human-in-the-loop, flow control, long-term memory, and `langgraph-supervisor`.
*   **FastAPI Service:** Serves agents with both streaming and non-streaming endpoints for flexible interaction.
*   **Advanced Streaming:** Supports both token-based and message-based streaming for efficient data handling.
*   **Streamlit Interface:** Provides an intuitive chat interface for easy interaction with your agents.
*   **Multiple Agent Support:** Run multiple agents within the service and access them via URL paths.
*   **Asynchronous Design:** Utilizes `async/await` for efficient handling of concurrent requests.
*   **Content Moderation:** Implements LlamaGuard for content moderation (requires Groq API key).
*   **RAG Agent:** Basic RAG agent implementation using ChromaDB (see [docs](docs/RAG_Assistant.md)).
*   **Feedback Mechanism:** Includes a star-based feedback system integrated with LangSmith.
*   **Docker Support:** Includes Dockerfiles and a Docker Compose file for simplified development and deployment.
*   **Comprehensive Testing:** Includes robust unit and integration tests for reliability and quality.

## Quickstart

### Try the app!

<a href="https://agent-service-toolkit.streamlit.app/"><img src="media/app_screenshot.png" width="600"></a>

Choose your preferred setup:

**1. Using Python**

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

**2. Using Docker**

```bash
echo 'OPENAI_API_KEY=your_openai_api_key' >> .env
docker compose watch
```

## Architecture Diagram

<img src="media/agent_architecture.png" width="600">

## Project Structure

*   `src/agents/`: Contains agent definitions.
*   `src/schema/`: Defines data structures (protocol schema).
*   `src/core/`: Contains core modules (e.g., LLM definitions).
*   `src/service/service.py`: The FastAPI service.
*   `src/client/client.py`: Client for interacting with the service.
*   `src/streamlit_app.py`: Streamlit chat interface.
*   `tests/`: Unit and integration tests.

## Setup and Usage

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/JoshuaC215/agent-service-toolkit.git
    cd agent-service-toolkit
    ```

2.  **Configure Environment Variables:**

    *   Create a `.env` file in the project root.
    *   Add required API keys (e.g., OpenAI).  See the [`.env.example`](./.env.example) file for all available variables.

3.  **Run the Service and App:**

    *   Follow either the "Quickstart" instructions above (Python or Docker).

### Additional setup for specific AI providers

-   [Setting up Ollama](docs/Ollama.md)
-   [Setting up VertexAI](docs/VertexAI.md)
-   [Setting up RAG with ChromaDB](docs/RAG_Assistant.md)

## Building and Customizing Agents

1.  **Create a new agent:**  Add a new agent file in the `src/agents` directory (e.g., by copying `research_assistant.py` or `chatbot.py`).
2.  **Register your agent:**  Import your agent in `src/agents/agents.py` and add it to the `agents` dictionary.  Your agent will be accessible via `/your_agent_name/invoke` or `/your_agent_name/stream`.
3.  **Update the UI (Streamlit):**  Modify `src/streamlit_app.py` to reflect your agent's features.

## Docker Setup (Recommended)

*   Install Docker and Docker Compose.
*   Create a `.env` file (see the example).
*   Run `docker compose watch` for development (auto-reloads on code changes).  Use `docker compose up --build` for rebuilds.
*   Access the Streamlit app at `http://localhost:8501`.
*   Access the API at `http://0.0.0.0:8080`, with OpenAPI docs at `http://0.0.0.0:8080/redoc`.
*   Use `docker compose down` to stop the services.

## Developing with AgentClient

The `src/client/client.AgentClient` provides a versatile way to interact with the agent service.

```python
from client import AgentClient
client = AgentClient()

response = client.invoke("Tell me a brief joke?")
response.pretty_print()
```

## Development with LangGraph Studio

The agent supports [LangGraph Studio](https://langchain-ai.github.io/langgraph/concepts/langgraph_studio/), the IDE for developing agents in LangGraph.

`langgraph-cli[inmem]` is installed with `uv sync`. You can simply add your `.env` file to the root directory as described above, and then launch LangGraph Studio with `langgraph dev`. Customize `langgraph.json` as needed. See the [local quickstart](https://langchain-ai.github.io/langgraph/cloud/how-tos/studio/quick_start/#local-development-server) to learn more.

## Local Development Without Docker

1.  Create and activate a virtual environment (`uv sync --frozen` and `source .venv/bin/activate`).
2.  Run the FastAPI server: `python src/run_service.py`
3.  In a separate terminal, run the Streamlit app: `streamlit run src/streamlit_app.py`

## Projects Built With (or Inspired by) Agent Service Toolkit

*   **[PolyRAG](https://github.com/QuentinFuxa/PolyRAG)** - RAG capabilities over PostgreSQL and PDF documents.
*   **[alexrisch/agent-web-kit](https://github.com/alexrisch/agent-web-kit)** - A Next.JS frontend for agent-service-toolkit
*   **[raushan-in/dapa](https://github.com/raushan-in/dapa)** - Digital Arrest Protection App (DAPA)

## Contributing

Contributions are welcome! Please submit a pull request.

1.  Install dependencies and pre-commit hooks: `uv sync --frozen` and `pre-commit install`.
2.  Run tests: `pytest`.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
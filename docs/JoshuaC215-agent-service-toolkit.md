# AI Agent Service Toolkit: Build, Deploy, and Interact with Intelligent Agents

**Effortlessly create and deploy your own AI agent service with a robust toolkit built on LangGraph, FastAPI, and Streamlit. ([Original Repo](https://github.com/JoshuaC215/agent-service-toolkit))**

[![build status](https://github.com/JoshuaC215/agent-service-toolkit/actions/workflows/test.yml/badge.svg)](https://github.com/JoshuaC215/agent-service-toolkit/actions/workflows/test.yml)
[![codecov](https://codecov.io/github/JoshuaC215/agent-service-toolkit/graph/badge.svg?token=5MTJSYWD05)](https://codecov.io/github/JoshuaC215/agent-service-toolkit)
[![Python Version](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2FJoshuaC215%2Fagent-service-toolkit%2Frefs%2Fheads%2Fmain%2Fpyproject.toml)](https://github.com/JoshuaC215/agent-service-toolkit/blob/main/pyproject.toml)
[![GitHub License](https://img.shields.io/github/license/JoshuaC215/agent-service-toolkit)](https://github.com/JoshuaC215/agent-service-toolkit/blob/main/LICENSE)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_red.svg)](https://agent-service-toolkit.streamlit.app/)

This toolkit provides a comprehensive solution for building, deploying, and interacting with AI agents. It includes a LangGraph agent, a FastAPI service, a client for interaction, and a Streamlit app for a user-friendly chat interface. Designed for easy setup and customization, this project allows you to quickly build and run your own LangGraph-based AI projects.

**[🎥 Watch a video walkthrough of the repo and app](https://www.youtube.com/watch?v=pdYVHw_YCNY)**

## Key Features

*   **LangGraph Agent:** Utilize a customizable agent built with the latest LangGraph features, including human-in-the-loop, flow control, long-term memory, and `langgraph-supervisor`.
*   **FastAPI Service:** Serve your agent with streaming and non-streaming endpoints for flexible interaction.
*   **Advanced Streaming:** Leverage a novel approach for both token-based and message-based streaming.
*   **Streamlit Interface:** Interact with your agent through a user-friendly chat interface.
*   **Multiple Agent Support:** Run multiple agents within the service, accessible via URL paths.
*   **Asynchronous Design:** Benefit from async/await for efficient handling of concurrent requests.
*   **Content Moderation:** Integrate LlamaGuard for content moderation (requires a Groq API key).
*   **RAG Agent Implementation:** Includes a basic RAG agent implementation using ChromaDB for Retrieval-Augmented Generation (see [docs](docs/RAG_Assistant.md)).
*   **Feedback Mechanism:** Integrate a star-based feedback system with LangSmith.
*   **Docker Support:** Utilize Dockerfiles and a docker compose file for easy development and deployment.
*   **Comprehensive Testing:** Benefit from robust unit and integration tests.

## Getting Started

### Quickstart

Run directly in python

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

Run with docker

```sh
echo 'OPENAI_API_KEY=your_openai_api_key' >> .env
docker compose watch
```

### Architecture Diagram

<img src="media/agent_architecture.png" width="600">

### Try the App!

<a href="https://agent-service-toolkit.streamlit.app/"><img src="media/app_screenshot.png" width="600"></a>

## Setup and Usage

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/JoshuaC215/agent-service-toolkit.git
    cd agent-service-toolkit
    ```

2.  **Set Up Environment Variables:**

    *   Create a `.env` file in the root directory.
    *   Configure at least one LLM API key. See the [`.env.example` file](./.env.example) for a complete list of available variables.

3.  **Run the Agent Service and Streamlit App:**  You can use either Docker (recommended for simplicity) or a local Python environment.

### Docker Setup (Recommended)

1.  **Prerequisites:** Install Docker and Docker Compose (>= v2.23.0).
2.  **Create `.env` file:** Copy the `.env.example` and fill in your API keys.
    ```bash
    cp .env.example .env
    # Edit .env to add your API keys
    ```
3.  **Run with `docker compose watch`:**
    ```bash
    docker compose watch
    ```
    This will:
    *   Start a PostgreSQL database.
    *   Start the FastAPI agent service.
    *   Start the Streamlit app.

4.  **Access the App:**  Open your browser and go to `http://localhost:8501`.  The API is available at `http://0.0.0.0:8080`, with OpenAPI docs at `http://0.0.0.0:8080/redoc`.
5.  **Stop Services:** `docker compose down`

### Local Development Without Docker

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

4.  Open your browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

## Building and Customizing Your Own Agent

1.  **Add a New Agent:** Add your agent definition to the `src/agents` directory (e.g., by copying `research_assistant.py` or `chatbot.py`).
2.  **Register the Agent:** Import and add your new agent to the `agents` dictionary in `src/agents/agents.py`.  Your agent will be accessible at `/<your_agent_name>/invoke` or `/<your_agent_name>/stream`.
3.  **Adapt the UI:**  Update the Streamlit interface (`src/streamlit_app.py`) to match your agent's capabilities.

## Additional Configuration

*   **Setting up Ollama:**  [docs/Ollama.md](docs/Ollama.md)
*   **Setting up VertexAI:**  [docs/VertexAI.md](docs/VertexAI.md)
*   **Setting up RAG with ChromaDB:**  [docs/RAG_Assistant.md](docs/RAG_Assistant.md)
*   **Handling Private Credential files:** [docs/File_Based_Credentials.md](docs/File_Based_Credentials.md)

## Using the AgentClient

The `src/client/client.AgentClient` provides a flexible way to interact with your agent service. It supports synchronous and asynchronous calls, and streaming or non-streaming responses. See `src/run_client.py` for example usage.

```python
from client import AgentClient
client = AgentClient()
response = client.invoke("Tell me a brief joke?")
response.pretty_print()
```

## Development with LangGraph Studio

The agent supports [LangGraph Studio](https://langchain-ai.github.io/langgraph/concepts/langgraph_studio/), the IDE for developing agents in LangGraph.

Simply add your `.env` file to the root directory as described above, and then launch LangGraph Studio with `langgraph dev`. Customize `langgraph.json` as needed. See the [local quickstart](https://langchain-ai.github.io/langgraph/cloud/how-tos/studio/quick_start/#local-development-server) to learn more.

## Projects built with or inspired by agent-service-toolkit

-   **[PolyRAG](https://github.com/QuentinFuxa/PolyRAG)** - Extends agent-service-toolkit with RAG capabilities over both PostgreSQL databases and PDF documents.
-   **[alexrisch/agent-web-kit](https://github.com/alexrisch/agent-web-kit)** - A Next.JS frontend for agent-service-toolkit
-   **[raushan-in/dapa](https://github.com/raushan-in/dapa)** - Digital Arrest Protection App (DAPA) enables users to report financial scams and frauds efficiently via a user-friendly platform.

**Please submit a pull request or open a discussion to add new projects.**

## Contributing

Contributions are welcome!  Please submit pull requests.

### Running Tests

1.  Ensure you're in the project root and have activated your virtual environment.
2.  Install development dependencies and pre-commit hooks:

    ```bash
    uv sync --frozen
    pre-commit install
    ```
3.  Run the tests:

    ```bash
    pytest
    ```

## License

This project is licensed under the MIT License.  See the [LICENSE](LICENSE) file for details.
# AI Agent Service Toolkit: Build, Deploy, and Scale Your AI Agents

**Effortlessly create and deploy AI agent services with a complete toolkit built on LangGraph, FastAPI, and Streamlit.** [Explore the Agent Service Toolkit on GitHub](https://github.com/JoshuaC215/agent-service-toolkit).

[![build status](https://github.com/JoshuaC215/agent-service-toolkit/actions/workflows/test.yml/badge.svg)](https://github.com/JoshuaC215/agent-service-toolkit/actions/workflows/test.yml)
[![codecov](https://codecov.io/github/JoshuaC215/agent-service-toolkit/graph/badge.svg?token=5MTJSYWD05)](https://codecov.io/github/JoshuaC215/agent-service-toolkit)
[![Python Version](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2FJoshuaC215%2Fagent-service-toolkit%2Frefs%2Fheads%2Fmain%2Fpyproject.toml)](https://github.com/JoshuaC215/agent-service-toolkit/blob/main/pyproject.toml)
[![GitHub License](https://img.shields.io/github/license/JoshuaC215/agent-service-toolkit)](https://github.com/JoshuaC215/agent-service-toolkit/blob/main/LICENSE)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_red.svg)](https://agent-service-toolkit.streamlit.app/)

This toolkit provides a comprehensive foundation for building and deploying AI agent services. It combines the power of LangGraph, FastAPI, and Streamlit to offer a complete solution from agent definition to user interface. With this toolkit, you can quickly prototype, test, and deploy sophisticated AI agents.

**[🎥 Watch a video walkthrough of the repo and app](https://www.youtube.com/watch?v=pdYVHw_YCNY)**

## Key Features

*   **LangGraph-Powered Agents:** Leverage the latest LangGraph v0.3 features, including human-in-the-loop, flow control, long-term memory, and `langgraph-supervisor`.
*   **FastAPI Service:**  A robust FastAPI service with streaming and non-streaming endpoints for efficient agent interaction.
*   **Advanced Streaming:**  Innovative support for both token-based and message-based streaming for real-time agent responses.
*   **Streamlit Chat Interface:**  A user-friendly Streamlit interface provides an intuitive chat experience.
*   **Multiple Agent Support:**  Run multiple agents in a single service, accessible by URL path, and easily see what models and agents are available.
*   **Asynchronous Design:** Optimized for high performance with asynchronous operations for concurrent request handling.
*   **Content Moderation:**  Integrated LlamaGuard for content moderation (requires Groq API key).
*   **RAG Agent:**  A basic Retrieval-Augmented Generation (RAG) agent implementation using ChromaDB.  See [docs](docs/RAG_Assistant.md).
*   **Feedback Mechanism:**  Integrates a star-based feedback system for LangSmith.
*   **Docker Support:** Includes Dockerfiles and a docker compose file for easy development and deployment.
*   **Comprehensive Testing:**  Ensures reliability with thorough unit and integration tests.

## Quick Start

**Try the App!** [https://agent-service-toolkit.streamlit.app/](https://agent-service-toolkit.streamlit.app/)

<a href="https://agent-service-toolkit.streamlit.app/"><img src="media/app_screenshot.png" width="600"></a>

### Run Directly with Python

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

## Architecture

### Architecture Diagram

<img src="media/agent_architecture.png" width="600">

## Key Files

*   `src/agents/`:  Agent definitions with various capabilities.
*   `src/schema/`:  Defines the protocol schema.
*   `src/core/`:  Core modules, including LLM definitions and settings.
*   `src/service/service.py`:  The FastAPI service for serving agents.
*   `src/client/client.py`:  A client to interact with the agent service.
*   `src/streamlit_app.py`:  The Streamlit app for the chat interface.
*   `tests/`:  Unit and integration tests for quality assurance.

## Setup and Usage

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/JoshuaC215/agent-service-toolkit.git
    cd agent-service-toolkit
    ```

2.  **Configure Environment Variables:**

    Create a `.env` file in the root directory.  At least one LLM API key is required to run the agents.  See the [`.env.example` file](./.env.example) for a full list of available environment variables.

3.  **Run the Service:**

    Follow the instructions in the "Quick Start" section above to run the service with either Python or Docker.

### Additional Setup for AI Providers

*   [Setting up Ollama](docs/Ollama.md)
*   [Setting up VertexAI](docs/VertexAI.md)
*   [Setting up RAG with ChromaDB](docs/RAG_Assistant.md)

### Building or Customizing Your Own Agent

1.  **Add Your Agent:** Create your agent in the `src/agents` directory by copying an existing agent file (e.g., `research_assistant.py`, `chatbot.py`).
2.  **Register Your Agent:** Import and add your new agent to the `agents` dictionary in `src/agents/agents.py`. Your new agent will be accessible via a URL such as `/<your_agent_name>/invoke` or `/<your_agent_name>/stream`.
3.  **Update the UI (optional):** Adjust the Streamlit interface in `src/streamlit_app.py` to match your agent's specific functionalities, tools, and interactions.

### Handling Private Credential Files

The `privatecredentials/` directory is provided for file-based credentials (e.g., API keys, certificates), which are excluded from Git and Docker builds for security. See [Working with File-based Credentials](docs/File_Based_Credentials.md) for more information.

### Docker Setup

The project includes Docker and Docker Compose support for streamlined development and deployment. Docker Compose's "watch" feature facilitates automatic updates upon code changes.

**Prerequisites:**  Ensure Docker and Docker Compose are installed (version >= 2.23.0 recommended).

1.  **Create `.env`:**  Copy `.env.example` to `.env` and fill in your LLM API keys and other necessary configurations.
2.  **Build and Launch:**  Run `docker compose watch`.
3.  **Access the App:**  Open your web browser and go to `http://localhost:8501`.
4.  **Access the API:**  The FastAPI service API will be accessible at `http://0.0.0.0:8080`, and the OpenAPI docs can be found at `http://0.0.0.0:8080/redoc`.
5.  **Stop the Services:**  Use `docker compose down`.

## Building Other Apps on the AgentClient

The toolkit provides a flexible `src/client/client.AgentClient` that can be utilized to build other applications on top of your agents. The client supports synchronous and asynchronous invocations, as well as streaming and non-streaming requests.

See `src/run_client.py` for detailed examples of using `AgentClient`.

## Development with LangGraph Studio

The agent supports [LangGraph Studio](https://langchain-ai.github.io/langgraph/concepts/langgraph_studio/)

Launch LangGraph Studio with `langgraph dev`.

## Local Development Without Docker

1.  **Set up environment:**

    ```bash
    uv sync --frozen
    source .venv/bin/activate
    ```

2.  **Run the FastAPI server:**

    ```bash
    python src/run_service.py
    ```

3.  **Run the Streamlit app in a separate terminal:**

    ```bash
    streamlit run src/streamlit_app.py
    ```

    Open your browser to the URL provided by Streamlit (usually `http://localhost:8501`).

## Projects Built With or Inspired By agent-service-toolkit

*   **[PolyRAG](https://github.com/QuentinFuxa/PolyRAG)** - Extends agent-service-toolkit with RAG capabilities over both PostgreSQL databases and PDF documents.
*   **[alexrisch/agent-web-kit](https://github.com/alexrisch/agent-web-kit)** - A Next.JS frontend for agent-service-toolkit
*   **[raushan-in/dapa](https://github.com/raushan-in/dapa)** - Digital Arrest Protection App (DAPA) enables users to report financial scams and frauds efficiently via a user-friendly platform.

**Please create a pull request or open a discussion to add any new projects that have been built using this toolkit!**

## Contributing

We welcome contributions! Please submit a Pull Request.

### Testing

1.  Ensure you are in the project root directory and have activated your virtual environment.
2.  Install development dependencies and pre-commit hooks:

    ```bash
    uv sync --frozen
    pre-commit install
    ```

3.  Run the tests using pytest:

    ```bash
    pytest
    ```

## License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/JoshuaC215/agent-service-toolkit/blob/main/LICENSE) file for details.
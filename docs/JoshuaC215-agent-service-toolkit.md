# AI Agent Service Toolkit: Build and Deploy Powerful AI Agents with Ease

**Quickly build and deploy your own AI agent services using LangGraph, FastAPI, and Streamlit with this comprehensive toolkit.  Explore the code on GitHub: [https://github.com/JoshuaC215/agent-service-toolkit](https://github.com/JoshuaC215/agent-service-toolkit)**

This toolkit provides a complete, ready-to-use foundation for building and deploying AI agent services. It leverages the power of LangGraph for agent orchestration, FastAPI for service creation, and Streamlit for an intuitive user interface, all built with Pydantic for data structure and settings. Whether you're a beginner or an experienced developer, this toolkit simplifies the process of creating and deploying LangGraph-based projects, enabling you to focus on your agent's capabilities.

**[🎥 Watch a video walkthrough of the repo and app](https://www.youtube.com/watch?v=pdYVHw_YCNY)**

## Key Features:

*   **LangGraph Agent:** Develop and customize AI agents using the latest LangGraph v0.3 features including human-in-the-loop, flow control, long-term memory, and `langgraph-supervisor`.
*   **FastAPI Service:** Provides a robust and scalable service for your agents, supporting both streaming and non-streaming endpoints.
*   **Advanced Streaming:** Innovative approach to support both token-based and message-based streaming.
*   **Streamlit Interface:** A user-friendly chat interface for effortless interaction with your agents.
*   **Multiple Agent Support:** Easily manage and deploy multiple agents within a single service, accessible via URL path. Get agent information at `/info`.
*   **Asynchronous Design:** Built with async/await for efficient handling of concurrent requests, ensuring optimal performance.
*   **Content Moderation:** Integrated LlamaGuard for content moderation (requires Groq API key).
*   **RAG Agent Implementation:**  Basic RAG agent using ChromaDB - explore the [docs](docs/RAG_Assistant.md) for more information.
*   **Feedback Mechanism:** Integrate a star-based feedback system with LangSmith for continuous improvement.
*   **Docker Support:** Simplified development and deployment with Dockerfiles and a `docker compose` file.
*   **Comprehensive Testing:** Includes robust unit and integration tests for reliability and quality.

## Getting Started: Quickstart and Setup

### Try the app!

[https://agent-service-toolkit.streamlit.app/](https://agent-service-toolkit.streamlit.app/)

<img src="media/app_screenshot.png" width="600">

### Running the Toolkit

Follow these simple steps to get your AI agent service up and running:

**Prerequisites:**

*   Python 3.8+
*   An LLM API key (e.g., OpenAI, Cohere, etc.)

**Using `uv` (Recommended):**

1.  Set your API key(s) as an environment variable in a `.env` file:

    ```bash
    echo 'OPENAI_API_KEY=your_openai_api_key' >> .env
    ```

2.  Install uv (if not already installed - see the uv docs for options):

    ```bash
    curl -LsSf https://astral.sh/uv/0.7.19/install.sh | sh
    ```

3.  Install dependencies, create a virtual environment, and activate it:

    ```bash
    uv sync --frozen
    source .venv/bin/activate
    ```

4.  Run the FastAPI service:

    ```bash
    python src/run_service.py
    ```

5.  In a separate terminal, activate the virtual environment (if you're not still in it) and run the Streamlit app:

    ```bash
    source .venv/bin/activate
    streamlit run src/streamlit_app.py
    ```

**Using Docker (Recommended for Simplicity):**

1.  Create a `.env` file with your API keys (see `.env.example`).

    ```bash
    cp .env.example .env
    # Edit .env to add your API keys
    ```

2.  Run the application:

    ```bash
    docker compose watch
    ```

    This starts all necessary services, including the FastAPI server, Streamlit app, and a PostgreSQL database.  The `watch` command automatically updates your containers when code changes are detected.

### Architecture Diagram

<img src="media/agent_architecture.png" width="600">

## Customization and Development

### Building or Customizing Your Own Agent

1.  Create your new agent in the `src/agents` directory (e.g., by copying `research_assistant.py` or `chatbot.py`).
2.  Import and add your new agent to the `agents` dictionary in `src/agents/agents.py`.  It will be accessible via `/<your_agent_name>/invoke` or `/<your_agent_name>/stream`.
3.  Adjust the Streamlit interface in `src/streamlit_app.py` to match your agent's capabilities.

### Additional Setup

*   [Setting up Ollama](docs/Ollama.md)
*   [Setting up VertexAI](docs/VertexAI.md)
*   [Setting up RAG with ChromaDB](docs/RAG_Assistant.md)
*   [Working with File-based Credentials](docs/File_Based_Credentials.md)

## Using the AgentClient

The `src/client/client.AgentClient` provides a flexible way to interact with your agent service. It supports both synchronous and asynchronous calls, with streaming and non-streaming options. See `src/run_client.py` for example usage.

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

This toolkit supports [LangGraph Studio](https://langchain-ai.github.io/langgraph/concepts/langgraph_studio/) for simplified agent development.  Use `langgraph dev` after adding your `.env` file and customizing the `langgraph.json` configuration.  See the [local quickstart](https://langchain-ai.github.io/langgraph/cloud/how-tos/studio/quick_start/#local-development-server) for more information.

## Projects Built With/Inspired By agent-service-toolkit

*   **[PolyRAG](https://github.com/QuentinFuxa/PolyRAG)** - RAG capabilities over PostgreSQL and PDF documents.
*   **[alexrisch/agent-web-kit](https://github.com/alexrisch/agent-web-kit)** - A Next.JS frontend for agent-service-toolkit.
*   **[raushan-in/dapa](https://github.com/raushan-in/dapa)** - Digital Arrest Protection App (DAPA) for reporting financial scams.

## Contributing

Contributions are highly encouraged! Please submit a Pull Request or open a discussion with any new ones to be added!

**Testing:**

1.  Ensure you're in the project root directory and have activated your virtual environment.

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

This project is licensed under the MIT License - see the `LICENSE` file for details.
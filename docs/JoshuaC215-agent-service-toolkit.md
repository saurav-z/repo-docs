# AI Agent Service Toolkit: Build and Deploy AI Agents with Ease

**Quickly create and deploy powerful AI agent services using LangGraph, FastAPI, and Streamlit with this comprehensive toolkit. [See the original repository](https://github.com/JoshuaC215/agent-service-toolkit) for the complete source code.**

This toolkit empowers you to build, deploy, and interact with sophisticated AI agents. It provides a complete framework from agent definition to user interface, streamlining the development process.

*   **[Try the Live Demo](https://agent-service-toolkit.streamlit.app/)**

## Key Features

*   **LangGraph Agent**: Develop customizable agents using the latest LangGraph v0.3 features, including human-in-the-loop, flow control, long-term memory, and `langgraph-supervisor`.
*   **FastAPI Service**: Serves your agent with both streaming and non-streaming endpoints for efficient interaction.
*   **Advanced Streaming**:  A novel approach to both token and message-based streaming.
*   **Streamlit Interface**: Provides a user-friendly chat interface for effortless agent interaction.
*   **Multiple Agent Support**: Run multiple agents within the service, accessible via URL paths. Available agents and models are described in `/info`.
*   **Asynchronous Design**:  Built with async/await for optimal performance and concurrent request handling.
*   **Content Moderation**:  Integrates LlamaGuard for content moderation (requires Groq API key).
*   **RAG Agent**:  Includes a basic Retrieval-Augmented Generation (RAG) agent using ChromaDB. See the [RAG Assistant documentation](docs/RAG_Assistant.md).
*   **Feedback Mechanism**:  Integrates a star-based feedback system with LangSmith for performance evaluation.
*   **Docker Support**:  Includes Dockerfiles and a `docker compose` file for simplified development and deployment.
*   **Robust Testing**:  Features extensive unit and integration tests for reliability.

## Architecture Overview

*   **LangGraph**: The core framework for building the AI agent, enabling complex workflows.
*   **FastAPI**: Serves the agent via a RESTful API with streaming and non-streaming capabilities.
*   **Streamlit**: Offers a user-friendly chat interface for interacting with the agent.
*   **Pydantic**:  Used for data structure and settings configuration.
*   **[🎥 Watch a video walkthrough of the repo and app](https://www.youtube.com/watch?v=pdYVHw_YCNY)**

## Quick Start

Get up and running in minutes, either locally or with Docker.  Ensure you have Python 3.11+ installed.

**Local Setup (Python)**

```bash
# Configure LLM API Key - Requires at least one API key to run
echo 'OPENAI_API_KEY=your_openai_api_key' >> .env

# uv is the recommended dependency manager
curl -LsSf https://astral.sh/uv/0.7.19/install.sh | sh

# Install dependencies and activate the virtual environment
uv sync --frozen
source .venv/bin/activate

# Run the FastAPI service
python src/run_service.py

# In a separate terminal, run the Streamlit UI
source .venv/bin/activate
streamlit run src/streamlit_app.py
```

**Docker Setup**

```bash
# Configure LLM API Key
echo 'OPENAI_API_KEY=your_openai_api_key' >> .env

# Run with docker compose watch
docker compose watch
```

## Usage and Customization

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/JoshuaC215/agent-service-toolkit.git
    cd agent-service-toolkit
    ```

2.  **Configure environment variables:**

    Create a `.env` file in the root directory.  Refer to the [`.env.example` file](./.env.example) for available options including API keys, authentication, tracing, and more.

3.  **Build Your Own Agent:**
    *   Add your new agent to the `src/agents` directory.
    *   Modify `src/agents/agents.py` to include your agent in the service.
    *   Adjust `src/streamlit_app.py` to match your agent's capabilities.

## Advanced Configuration

*   **[Setting up Ollama](docs/Ollama.md)**
*   **[Setting up VertexAI](docs/VertexAI.md)**
*   **[Setting up RAG with ChromaDB](docs/RAG_Assistant.md)**
*   **[Working with File-based Credentials](docs/File_Based_Credentials.md)**

## Additional Resources

*   **[LangGraph Studio](https://langchain-ai.github.io/langgraph/concepts/langgraph_studio/)** - Develop agents with a visual IDE.

## Projects Built with or Inspired By Agent-Service-Toolkit

*   **[PolyRAG](https://github.com/QuentinFuxa/PolyRAG)** - Enhances the toolkit with RAG capabilities over PostgreSQL and PDF documents.
*   **[alexrisch/agent-web-kit](https://github.com/alexrisch/agent-web-kit)** - A Next.JS frontend for the agent-service-toolkit.
*   **[raushan-in/dapa](https://github.com/raushan-in/dapa)** - Digital Arrest Protection App (DAPA) facilitates efficient reporting of financial scams.

## Contributing

Contributions are welcome! Please submit a pull request. Run tests using the local development setup (without Docker).

1.  Activate your virtual environment and install pre-commit hooks:

    ```bash
    uv sync --frozen
    pre-commit install
    ```

2.  Run tests:

    ```bash
    pytest
    ```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
# AI Agent Service Toolkit: Build, Deploy, and Scale Your AI Agents

**Unleash the power of LangGraph, FastAPI, and Streamlit to build robust AI agent services with this comprehensive toolkit.  [Explore the repository on GitHub](https://github.com/JoshuaC215/agent-service-toolkit)**

This toolkit provides a complete solution for building and deploying AI agent services, offering a streamlined path from agent definition to a user-friendly interface. It's designed with LangGraph at its core and leverages FastAPI for a robust service layer, a client for easy interaction, and a Streamlit app for a user-friendly chat interface. 

**[Try the Live Demo!](https://agent-service-toolkit.streamlit.app/)**

## Key Features

*   **Advanced LangGraph Agent:** Built using the latest LangGraph framework features, including human-in-the-loop, flow control, long-term memory, and `langgraph-supervisor`.
*   **FastAPI Service:**  Serves your agents with both streaming and non-streaming endpoints, optimized for performance.
*   **Sophisticated Streaming Support:**  A novel approach to handle both token-based and message-based streaming for a better user experience.
*   **User-Friendly Streamlit Interface:**  Provides an intuitive chat interface, making agent interaction easy.
*   **Multiple Agent Support:** Easily manage and run multiple agents within the service, accessible via URL paths.
*   **Asynchronous Design:**  Leverages async/await for efficient handling of concurrent requests and scalability.
*   **Content Moderation:** Integrates LlamaGuard for content moderation (requires Groq API key).
*   **RAG Agent Implementation:** A basic Retrieval Augmented Generation (RAG) agent implementation using ChromaDB (see [docs/RAG\_Assistant.md](docs/RAG_Assistant.md)).
*   **Feedback Mechanism:** Includes a star-based feedback system that integrates with LangSmith for evaluating agent performance.
*   **Docker Support:** Streamlined Docker setup with Dockerfiles and `docker compose` for easy development and deployment.
*   **Comprehensive Testing:** Includes a suite of unit and integration tests to ensure code quality and reliability.

## Quickstart Guide

Get up and running in minutes!

**Prerequisites:** Python 3.9+

**1. Set up your environment:**
   ```bash
   # Create a .env file and add your API key
   echo 'OPENAI_API_KEY=your_openai_api_key' >> .env
   # Install dependencies (uv is recommended):
   curl -LsSf https://astral.sh/uv/0.7.19/install.sh | sh
   uv sync --frozen
   source .venv/bin/activate
   ```

**2. Run the Agent Service:**
   ```bash
   python src/run_service.py
   ```

**3. Run the Streamlit App (in a separate terminal):**
   ```bash
   source .venv/bin/activate
   streamlit run src/streamlit_app.py
   ```
   This will launch the Streamlit app, which you can access in your web browser at `http://localhost:8501`.

**4. Run with Docker (Recommended)**

   ```bash
   echo 'OPENAI_API_KEY=your_openai_api_key' >> .env
   docker compose watch
   ```

## Architecture

The architecture diagram visually represents the components and interactions within the AI Agent Service Toolkit.
<img src="media/agent_architecture.png" width="600">

## Key Files

*   `src/agents/`: Contains the definitions of various AI agents.
*   `src/schema/`: Defines the data structures and protocols used.
*   `src/core/`: Includes core modules, such as LLM definitions and configuration.
*   `src/service/service.py`: The FastAPI service responsible for serving the agents.
*   `src/client/client.py`: A client for interacting with the agent service.
*   `src/streamlit_app.py`: The Streamlit application providing a chat interface.
*   `tests/`: Contains unit and integration tests to ensure code quality.

## Advanced Usage

### Setting Up AI Providers
Detailed instructions for integrating different AI providers (like Ollama and VertexAI) are available in the `docs/` directory:

*   [Setting up Ollama](docs/Ollama.md)
*   [Setting up VertexAI](docs/VertexAI.md)
*   [Setting up RAG with ChromaDB](docs/RAG_Assistant.md)

### Building and Customizing Agents

1.  **Create a New Agent:** Add your agent code to the `src/agents/` directory, using the provided examples as a template (e.g., `research_assistant.py`, `chatbot.py`).
2.  **Register Your Agent:** Import your new agent into `src/agents/agents.py` and add it to the `agents` dictionary.  Your agent will then be accessible via the API at  `/<your_agent_name>/invoke` or `/<your_agent_name>/stream`.
3.  **Update the User Interface:**  Adjust the Streamlit app (`src/streamlit_app.py`) to align with your agent's specific capabilities and interface needs.

### Using Private Credentials
For agents or LLMs requiring file-based credentials, the `privatecredentials/` directory is provided.  Contents within are ignored by Git and Docker builds, and are described in [Working with File-based Credentials](docs/File_Based_Credentials.md).

### Docker Setup

The provided `docker compose` configuration streamlines development and deployment.  Follow these steps:

1.  Ensure you have Docker and Docker Compose (>= v2.23.0) installed.
2.  Create a `.env` file (based on `.env.example`) and populate with necessary API keys.
3.  Start the services in "watch" mode: `docker compose watch`  This will automatically rebuild and restart containers upon code changes.
4.  Access the Streamlit app at `http://localhost:8501`.  The API documentation is available at `http://0.0.0.0:8080/redoc`.
5.  Use `docker compose down` to shut down the services.

### Building on the AgentClient

The toolkit includes a flexible `src/client/client.AgentClient` for interacting with the agent service.  This client supports synchronous and asynchronous invocations, and streaming and non-streaming requests.  Explore `src/run_client.py` for examples.

### Development with LangGraph Studio
Integrate seamlessly with [LangGraph Studio](https://langchain-ai.github.io/langgraph/concepts/langgraph_studio/) for agent development using `langgraph-cli[inmem]`.  

### Local Development (Without Docker)

If you prefer to run without Docker:

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

## Projects built with or inspired by agent-service-toolkit
Here are some example projects which use or were inspired by this repository.  Pull requests adding more projects are welcome!
*   **[PolyRAG](https://github.com/QuentinFuxa/PolyRAG)** - Extends agent-service-toolkit with RAG capabilities over both PostgreSQL databases and PDF documents.
*   **[alexrisch/agent-web-kit](https://github.com/alexrisch/agent-web-kit)** - A Next.JS frontend for agent-service-toolkit
*   **[raushan-in/dapa](https://github.com/raushan-in/dapa)** - Digital Arrest Protection App (DAPA) enables users to report financial scams and frauds efficiently via a user-friendly platform.

## Contributing

Contributions are welcome! Please submit pull requests.  

### Running Tests

1.  Navigate to the project root and activate your virtual environment.
2.  Install pre-commit hooks:
    ```bash
    uv sync --frozen
    pre-commit install
    ```
3.  Run tests:
    ```bash
    pytest
    ```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
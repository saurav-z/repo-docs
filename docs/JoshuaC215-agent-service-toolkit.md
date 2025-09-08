# 🤖 Build AI Agent Services with Ease: Agent Service Toolkit

**Quickly build and deploy powerful AI agents with the Agent Service Toolkit, a comprehensive framework built on LangGraph, FastAPI, and Streamlit. [[Original Repo](https://github.com/JoshuaC215/agent-service-toolkit)]**

This toolkit provides everything you need to create, serve, and interact with AI agents. It offers a complete, robust solution for building LangGraph-based projects, allowing you to go from agent definition to a user interface with minimal effort.

## Key Features:

*   🚀 **LangGraph Agent Integration:** Leverage the power of LangGraph for building customizable agents, including the latest v0.3 features such as human-in-the-loop, flow control, long-term memory, and the LangGraph supervisor.
*   ⚙️ **FastAPI Service:** Serve your agents efficiently with a robust FastAPI backend, supporting both streaming and non-streaming endpoints.
*   📡 **Advanced Streaming Capabilities:** Implement a novel approach to streaming that works seamlessly with both token-based and message-based streaming.
*   💬 **Streamlit Chat Interface:** Provide a user-friendly chat interface for easy agent interaction, ideal for demos and end users.
*   🌐 **Multi-Agent Support:** Easily deploy and manage multiple agents within your service, accessible via URL path.
*   🔄 **Asynchronous Design:** Benefit from an asynchronous architecture for optimal performance and scalability in handling concurrent requests.
*   🛡️ **Content Moderation:** Integrate LlamaGuard for content moderation to ensure responsible AI usage (requires Groq API key).
*   🧠 **RAG Agent Implementation:** Includes a basic RAG agent implementation using ChromaDB for information retrieval, see [docs](docs/RAG_Assistant.md).
*   ⭐ **Feedback Mechanism:** Includes a star-based feedback system integrated with LangSmith to collect user feedback.
*   🐳 **Docker Support:** Streamline your development and deployment with pre-configured Dockerfiles and a docker-compose setup.
*   ✅ **Comprehensive Testing:** Ensures quality and reliability with extensive unit and integration tests.

## Getting Started

### Quickstart

1.  **Set up your environment variables:**

    ```bash
    echo 'OPENAI_API_KEY=your_openai_api_key' >> .env
    ```
    (Replace `your_openai_api_key` with your actual API key.)

2.  **Install dependencies:**

    ```bash
    # uv is the recommended way, but "pip install ." also works
    curl -LsSf https://astral.sh/uv/0.7.19/install.sh | sh
    uv sync --frozen
    source .venv/bin/activate
    ```

3.  **Run the service and Streamlit app:**

    ```bash
    python src/run_service.py
    # In another shell
    streamlit run src/streamlit_app.py
    ```

### Alternative: Run with Docker

1.  **Configure .env:**

    ```bash
    echo 'OPENAI_API_KEY=your_openai_api_key' >> .env
    ```

2.  **Run Docker Compose:**

    ```bash
    docker compose watch
    ```

    Access the Streamlit app at `http://localhost:8501` and the API docs at `http://0.0.0.0:8080/redoc`.

### Architecture

*   **[Architecture Diagram](media/agent_architecture.png)**
*   **[Try the app!](https://agent-service-toolkit.streamlit.app/)**

    <a href="https://agent-service-toolkit.streamlit.app/"><img src="media/app_screenshot.png" width="600"></a>

### Key Files

*   `src/agents/`: Agent definitions
*   `src/schema/`: Protocol schema
*   `src/core/`: Core modules (LLM, settings)
*   `src/service/service.py`: FastAPI service
*   `src/client/client.py`: Agent service client
*   `src/streamlit_app.py`: Streamlit UI
*   `tests/`: Unit and integration tests

## Customization

### Building or customizing your own agent

1.  Add your new agent to the `src/agents` directory.
2.  Import and add your new agent to the `agents` dictionary in `src/agents/agents.py`.
3.  Adjust the Streamlit interface in `src/streamlit_app.py` to match your agent's capabilities.

### Additional setup for specific AI providers

-   [Setting up Ollama](docs/Ollama.md)
-   [Setting up VertexAI](docs/VertexAI.md)
-   [Setting up RAG with ChromaDB](docs/RAG_Assistant.md)
### Handling Private Credential files
See [Working with File-based Credentials](docs/File_Based_Credentials.md)

## Projects built with or inspired by agent-service-toolkit
*   **[PolyRAG](https://github.com/QuentinFuxa/PolyRAG)** - Extends agent-service-toolkit with RAG capabilities over both PostgreSQL databases and PDF documents.
*   **[alexrisch/agent-web-kit](https://github.com/alexrisch/agent-web-kit)** - A Next.JS frontend for agent-service-toolkit
*   **[raushan-in/dapa](https://github.com/raushan-in/dapa)** - Digital Arrest Protection App (DAPA) enables users to report financial scams and frauds efficiently via a user-friendly platform.

**Please create a pull request editing the README or open a discussion with any new ones to be added!** Would love to include more projects.

## Contributing

We welcome contributions! Follow these steps to run the tests:

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

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
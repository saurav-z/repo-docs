# AI Agent Service Toolkit: Build, Deploy, and Scale Your AI Agents

**Quickly build and deploy powerful AI agents with a robust toolkit built on LangGraph, FastAPI, and Streamlit. [Check out the original repo](https://github.com/JoshuaC215/agent-service-toolkit) for the full source code.**

This toolkit provides a comprehensive framework for developing and deploying AI agents, offering a complete solution from agent definition to user interface.  It utilizes cutting-edge technologies like LangGraph, FastAPI, Streamlit, and Pydantic to streamline the development process.  

**[🎥 Watch a video walkthrough of the repo and app](https://www.youtube.com/watch?v=pdYVHw_YCNY)**

## Key Features

*   **LangGraph Agent & Latest Features:** Utilize a customizable agent built with LangGraph, incorporating advanced features like human-in-the-loop interactions with `interrupt()`, flow control using `Command`, persistent memory via `Store`, and the `langgraph-supervisor`.
*   **FastAPI Service:** Provides a robust FastAPI service for serving agents, supporting both streaming and non-streaming endpoints.
*   **Advanced Streaming:** Implements a novel approach to streaming, supporting both token-based and message-based streaming for optimal user experience.
*   **Streamlit Interface:** Includes a user-friendly chat interface built with Streamlit, making it easy to interact with and test your agents.
*   **Multiple Agent Support:**  Run multiple agents within the service, accessible by their unique URL paths, with available agents and models listed in `/info`.
*   **Asynchronous Design:** Leverages async/await for efficient handling of concurrent requests, improving performance and scalability.
*   **Content Moderation:** Integrates LlamaGuard for content moderation to help ensure safety and responsible AI.
*   **RAG Agent:** Includes a sample RAG agent implementation using ChromaDB for retrieval-augmented generation capabilities.
*   **Feedback Mechanism:** Includes a star-based feedback system integrated with LangSmith for gathering user feedback.
*   **Docker Support:** Provides Dockerfiles and a Docker Compose file for easy development, deployment, and containerization.
*   **Comprehensive Testing:**  Includes thorough unit and integration tests to ensure the reliability and quality of your agent service.

## Quickstart Guide

1.  **Set up your environment:** Create a `.env` file and populate it with your API keys (e.g., OPENAI_API_KEY). See the [`.env.example` file](./.env.example) for a full list of available environment variables.

2.  **Choose your setup:**
    *   **Docker (Recommended):**
        ```bash
        echo 'OPENAI_API_KEY=your_openai_api_key' >> .env
        docker compose watch
        ```

    *   **Local (Python):**
        ```bash
        uv sync --frozen
        source .venv/bin/activate
        python src/run_service.py
        # In another shell
        source .venv/bin/activate
        streamlit run src/streamlit_app.py
        ```

3.  **Access the Streamlit app:** Navigate to `http://localhost:8501` in your browser.

## Architecture

![Architecture Diagram](media/agent_architecture.png)

## Deep Dive & Customization

### Key Files

*   `src/agents/`: Contains definitions for different agent types and their capabilities.
*   `src/schema/`: Defines data structures and protocols used throughout the application.
*   `src/core/`: Holds core modules, including LLM definitions and settings.
*   `src/service/service.py`: The FastAPI service that serves the AI agents.
*   `src/client/client.py`: Provides a client for interacting with the agent service.
*   `src/streamlit_app.py`:  The Streamlit application that provides a user-friendly chat interface.
*   `tests/`: Contains unit and integration tests for the project.

### Building or Customizing Your Own Agent

1.  Add your new agent code to the `src/agents` directory.
2.  Import your agent in `src/agents/agents.py` and add it to the `agents` dictionary. Your agent will be accessible via the service at `/<your_agent_name>/invoke` or `/<your_agent_name>/stream`.
3.  Adapt the Streamlit interface in `src/streamlit_app.py` to match your agent's unique features and functions.

### Additional Setup & Configuration

*   [Setting up Ollama](docs/Ollama.md)
*   [Setting up VertexAI](docs/VertexAI.md)
*   [Setting up RAG with ChromaDB](docs/RAG_Assistant.md)
*   [Working with File-based Credentials](docs/File_Based_Credentials.md)

## Projects Built With and Inspired By agent-service-toolkit

*   **[PolyRAG](https://github.com/QuentinFuxa/PolyRAG)** - Extends agent-service-toolkit with RAG capabilities over both PostgreSQL databases and PDF documents.
*   **[alexrisch/agent-web-kit](https://github.com/alexrisch/agent-web-kit)** - A Next.JS frontend for agent-service-toolkit
*   **[raushan-in/dapa](https://github.com/raushan-in/dapa)** - Digital Arrest Protection App (DAPA) enables users to report financial scams and frauds efficiently via a user-friendly platform.

**Please create a pull request editing the README or open a discussion with any new ones to be added!** Would love to include more projects.

## Contributing

Contributions are warmly welcomed! See the [Contributing](CONTRIBUTING.md) file for detailed instructions on submitting pull requests.

To run tests:

1.  Activate your virtual environment.
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

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
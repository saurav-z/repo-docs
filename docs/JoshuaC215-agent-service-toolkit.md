# AI Agent Service Toolkit: Build and Deploy Powerful AI Agents

**Quickly build and deploy your own AI agent service with LangGraph, FastAPI, and Streamlit using this comprehensive toolkit!** 
[Go to the Original Repo](https://github.com/JoshuaC215/agent-service-toolkit)

This toolkit provides a robust foundation for developing and deploying AI agents. It combines [LangGraph](https://langchain-ai.github.io/langgraph/) for agent orchestration, [FastAPI](https://fastapi.tiangolo.com/) for service creation, and [Streamlit](https://streamlit.io/) for a user-friendly chat interface. Built with [Pydantic](https://github.com/pydantic/pydantic) for data structures and settings, it offers a complete solution for building and running your LangGraph-based projects.

**[🎥 Watch a video walkthrough of the repo and app](https://www.youtube.com/watch?v=pdYVHw_YCNY)**

## Key Features:

*   **LangGraph Agent and Latest Features:** Customizable agents built with the LangGraph framework, including human-in-the-loop (`interrupt()`), flow control (`Command`), long-term memory (`Store`), and `langgraph-supervisor`.
*   **FastAPI Service:** Serves your agents with both streaming and non-streaming endpoints for optimal performance and flexibility.
*   **Advanced Streaming:**  A unique approach supports both token-based and message-based streaming.
*   **Streamlit Interface:**  Provides an intuitive chat interface, enabling easy interaction with your AI agents.
*   **Multiple Agent Support:**  Run multiple agents concurrently, accessible via unique URL paths (e.g., `/agent_name/invoke`).  The `/info` endpoint lists available agents and model details.
*   **Asynchronous Design:** Utilizes `async/await` for efficient handling of concurrent requests.
*   **Content Moderation:** Integrates LlamaGuard for content moderation, enhancing safety (requires Groq API key).
*   **RAG Agent:**  Includes a basic RAG (Retrieval-Augmented Generation) agent implementation using ChromaDB (see [docs](docs/RAG_Assistant.md)).
*   **Feedback Mechanism:**  Integrates a star-based feedback system with LangSmith for continuous improvement.
*   **Docker Support:**  Includes Dockerfiles and a docker-compose file for simplified development and deployment.
*   **Comprehensive Testing:**  Features robust unit and integration tests to ensure code quality and reliability.

## Quickstart:

###  Run with Docker (Recommended)

1.  Create a `.env` file in the project root (copy from `.env.example`) and add your API keys.
2.  Run `docker compose watch` to build and launch the services.  The services will auto-reload on code changes.
3.  Access the Streamlit app at `http://localhost:8501`.
4.  The FastAPI service API will be available at `http://0.0.0.0:8080`.
   OpenAPI docs are at `http://0.0.0.0:8080/redoc`.

### Run Locally (without Docker)

1.  Set up environment variables by creating a `.env` file (see `.env.example` for options).
2.  Install Dependencies with `uv sync --frozen` and activate the virtual environment.
3.  Run the FastAPI server: `python src/run_service.py`
4.  In a separate terminal, run the Streamlit app: `streamlit run src/streamlit_app.py`

## Architecture:

```
<img src="media/agent_architecture.png" width="600">
```

## Key Files:

*   `src/agents/`: Agent definitions with different capabilities.
*   `src/schema/`: Protocol schema definitions.
*   `src/core/`: Core modules including LLM definitions and settings.
*   `src/service/service.py`: FastAPI service for serving agents.
*   `src/client/client.py`: Client for interacting with the agent service.
*   `src/streamlit_app.py`: Streamlit app for the chat interface.
*   `tests/`: Unit and integration tests.

## Setup and Usage:

Detailed instructions, including setting up AI providers like Ollama and VertexAI, are in the original README, or see below:

1.  Clone the repository:

    ```sh
    git clone https://github.com/JoshuaC215/agent-service-toolkit.git
    cd agent-service-toolkit
    ```

2.  Set up environment variables as described above.

3.  Choose your preferred setup method (Docker or local).

## Customization and Development:

*   **Building Your Own Agent:**
    1.  Add a new agent to the `src/agents` directory, copying and modifying existing agent files (e.g., `research_assistant.py`).
    2.  Import your agent and add it to the `agents` dictionary in `src/agents/agents.py`.
    3.  Update the Streamlit interface (`src/streamlit_app.py`) to align with your agent's functionality.

*   **Handling Private Credential Files:** Use the `privatecredentials/` directory.  Files within are ignored by Git and Docker builds. See [Working with File-based Credentials](docs/File_Based_Credentials.md) for details.

*   **Building other apps on the AgentClient:**  The `src/client/client.AgentClient` is a generic client designed to be flexible and used to build other apps on top of the agent. It supports both synchronous and asynchronous invocations, and streaming and non-streaming requests. See `src/run_client.py` for examples.

*   **Development with LangGraph Studio:**  Integrate with [LangGraph Studio](https://langchain-ai.github.io/langgraph/concepts/langgraph_studio/) for an improved development experience.

## Projects Built With or Inspired By agent-service-toolkit:

*   **[PolyRAG](https://github.com/QuentinFuxa/PolyRAG)** - Enhances with RAG over PostgreSQL databases and PDF documents.
*   **[alexrisch/agent-web-kit](https://github.com/alexrisch/agent-web-kit)** -  A Next.JS frontend for agent-service-toolkit.
*   **[raushan-in/dapa](https://github.com/raushan-in/dapa)** -  Digital Arrest Protection App (DAPA).

Please create a pull request editing the README or open a discussion with any new ones to be added!

## Contributing:

Contributions are welcome!  Follow these steps to contribute:

1.  Activate your virtual environment and install development dependencies and pre-commit hooks: `uv sync --frozen` then `pre-commit install`.
2.  Run tests using `pytest`.

## License:

This project is licensed under the MIT License (see the `LICENSE` file).
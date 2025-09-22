# AI Agent Service Toolkit: Build and Deploy Powerful AI Agents

**Quickly build, deploy, and experiment with AI agents using LangGraph, FastAPI, and Streamlit – the ultimate toolkit for your AI agent projects.**  Explore the original repo: [https://github.com/JoshuaC215/agent-service-toolkit](https://github.com/JoshuaC215/agent-service-toolkit)

[![build status](https://github.com/JoshuaC215/agent-service-toolkit/actions/workflows/test.yml/badge.svg)](https://github.com/JoshuaC215/agent-service-toolkit/actions/workflows/test.yml)
[![codecov](https://codecov.io/github/JoshuaC215/agent-service-toolkit/graph/badge.svg?token=5MTJSYWD05)](https://codecov.io/github/JoshuaC215/agent-service-toolkit)
[![Python Version](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2FJoshuaC215%2Fagent-service-toolkit%2Frefs%2Fheads%2Fmain%2Fpyproject.toml)](https://github.com/JoshuaC215/agent-service-toolkit/blob/main/pyproject.toml)
[![GitHub License](https://img.shields.io/github/license/JoshuaC215/agent-service-toolkit)](https://github.com/JoshuaC215/agent-service-toolkit/blob/main/LICENSE)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_red.svg)](https://agent-service-toolkit.streamlit.app/)

This toolkit provides a comprehensive framework for developing and deploying AI agent services. It leverages the power of [LangGraph](https://langchain-ai.github.io/langgraph/) for agent construction, [FastAPI](https://fastapi.tiangolo.com/) for service creation, and [Streamlit](https://streamlit.io/) for a user-friendly interface. This allows for quick prototyping and robust deployment of LangGraph-based AI agents.

**[🎥 Watch a video walkthrough](https://www.youtube.com/watch?v=pdYVHw_YCNY)**

## Key Features

*   **LangGraph Agent:** Build customizable agents using the latest LangGraph features, including human-in-the-loop interactions, flow control, long-term memory, and `langgraph-supervisor`.
*   **FastAPI Service:** Deploy your agents with a production-ready FastAPI service that supports both streaming and non-streaming endpoints.
*   **Advanced Streaming:** Innovative streaming approach that supports both token-based and message-based streaming for enhanced user experience.
*   **Streamlit Interface:**  A Streamlit application provides an intuitive chat interface for easy agent interaction.
*   **Multi-Agent Support:**  Run multiple agents within the service and access them via URL paths. Agent information available at `/info`.
*   **Asynchronous Design:**  Efficiently handle concurrent requests with asynchronous/await.
*   **Content Moderation:**  Integrates LlamaGuard for content moderation (Groq API key required).
*   **RAG Agent:** Includes a basic RAG agent implementation using ChromaDB (see [docs](docs/RAG_Assistant.md)).
*   **Feedback Mechanism:** Integrated star-based feedback system connected with LangSmith.
*   **Docker Support:** Includes Dockerfiles and a docker compose file for streamlined development and deployment.
*   **Robust Testing:** Includes comprehensive unit and integration tests.

## Quickstart

### Try the App!

[<img src="media/app_screenshot.png" width="600" alt="Streamlit App Screenshot">](https://agent-service-toolkit.streamlit.app/)

### Run Locally

1.  **Set up your environment variables.**  At a minimum, set your OpenAI API Key:
    ```bash
    echo 'OPENAI_API_KEY=your_openai_api_key' >> .env
    ```

2.  **Install dependencies:**  We recommend using `uv`:
    ```bash
    curl -LsSf https://astral.sh/uv/0.7.19/install.sh | sh
    uv sync --frozen
    source .venv/bin/activate
    ```

3.  **Run the service:**
    ```bash
    python src/run_service.py
    ```

4.  **Run the Streamlit app in a separate terminal:**
    ```bash
    streamlit run src/streamlit_app.py
    ```

### Run with Docker

1.  **Set your API key:**
    ```bash
    echo 'OPENAI_API_KEY=your_openai_api_key' >> .env
    ```

2.  **Run Docker Compose:**
    ```bash
    docker compose watch
    ```

### Architecture Diagram

<img src="media/agent_architecture.png" width="600" alt="Agent Architecture Diagram">

## Setup and Usage

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/JoshuaC215/agent-service-toolkit.git
    cd agent-service-toolkit
    ```

2.  **Configure Environment Variables:** Create a `.env` file in the root directory and add your API keys and settings. Refer to the [`.env.example` file](./.env.example) for available options.

## Additional Setup

*   [Setting up Ollama](docs/Ollama.md)
*   [Setting up VertexAI](docs/VertexAI.md)
*   [Setting up RAG with ChromaDB](docs/RAG_Assistant.md)

## Building or Customizing Agents

1.  **Add a New Agent:**  Create a new agent file in the `src/agents` directory.  You can copy `research_assistant.py` or `chatbot.py` and modify it.
2.  **Register the Agent:**  Import your new agent in `src/agents/agents.py` and add it to the `agents` dictionary. The agent will then be available at `/your_agent_name/invoke` or `/your_agent_name/stream`.
3.  **Adjust the UI:**  Update `src/streamlit_app.py` to match your agent's capabilities and interface.

## Working with Private Credentials

The `privatecredentials/` directory is provided for storing credential files, excluded from Git and the Docker build. See [Working with File-based Credentials](docs/File_Based_Credentials.md).

## Docker Setup

This project includes a Docker setup for easy development and deployment. The `compose.yaml` file defines three services: `postgres`, `agent_service` and `streamlit_app`. The `Dockerfile` for each service is in their respective directories.

For local development, we recommend using [docker compose watch](https://docs.docker.com/compose/file-watch/). This feature allows for a smoother development experience by automatically updating your containers when changes are detected in your source code.

1. Make sure you have Docker and Docker Compose (>= [v2.23.0](https://docs.docker.com/compose/release-notes/#2230)) installed on your system.

2. Create a `.env` file from the `.env.example`. At minimum, you need to provide an LLM API key (e.g., OPENAI_API_KEY).
   ```sh
   cp .env.example .env
   # Edit .env to add your API keys
   ```

3. Build and launch the services in watch mode:

   ```sh
   docker compose watch
   ```

   This will automatically:
   - Start a PostgreSQL database service that the agent service connects to
   - Start the agent service with FastAPI
   - Start the Streamlit app for the user interface

4. The services will now automatically update when you make changes to your code:
   - Changes in the relevant python files and directories will trigger updates for the relevant services.
   - NOTE: If you make changes to the `pyproject.toml` or `uv.lock` files, you will need to rebuild the services by running `docker compose up --build`.

5. Access the Streamlit app by navigating to `http://localhost:8501` in your web browser.

6. The agent service API will be available at `http://0.0.0.0:8080`. You can also use the OpenAPI docs at `http://0.0.0.0:8080/redoc`.

7. Use `docker compose down` to stop the services.

This setup allows you to develop and test your changes in real-time without manually restarting the services.

## Building Other Apps with the AgentClient

The repository includes a flexible `src/client/client.AgentClient` class that can be used to build other applications that interact with the agent service. It supports synchronous and asynchronous calls, with both streaming and non-streaming capabilities.

See `src/run_client.py` for complete usage examples.

## Development with LangGraph Studio

This toolkit integrates with [LangGraph Studio](https://langchain-ai.github.io/langgraph/concepts/langgraph_studio/), the IDE for LangGraph agent development.
Configure your `.env` file and then launch Studio with `langgraph dev`. Customize `langgraph.json` as needed.

## Local Development Without Docker

1.  **Create a Virtual Environment:**

    ```bash
    uv sync --frozen
    source .venv/bin/activate
    ```

2.  **Run the FastAPI Server:**

    ```bash
    python src/run_service.py
    ```

3.  **Run the Streamlit App:**

    ```bash
    streamlit run src/streamlit_app.py
    ```

    Access the app via the URL provided by Streamlit (usually `http://localhost:8501`).

## Projects Built with or Inspired By This Toolkit

*   **[PolyRAG](https://github.com/QuentinFuxa/PolyRAG)** - RAG capabilities over PostgreSQL databases and PDF documents.
*   **[alexrisch/agent-web-kit](https://github.com/alexrisch/agent-web-kit)** - A Next.JS frontend for agent-service-toolkit
*   **[raushan-in/dapa](https://github.com/raushan-in/dapa)** - Digital Arrest Protection App (DAPA) enables users to report financial scams.

**Contribute by suggesting projects to add!**

## Contributing

Contributions are welcome!  Follow these steps to contribute:

1.  Ensure you are in the project root and have activated the virtual environment.
2.  Install development dependencies and pre-commit hooks:

    ```bash
    uv sync --frozen
    pre-commit install
    ```

3.  Run tests using pytest:

    ```bash
    pytest
    ```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
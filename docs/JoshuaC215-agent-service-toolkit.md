# AI Agent Service Toolkit: Build, Deploy, and Scale Your AI Agents

**Quickly build and deploy powerful AI agents with LangGraph, FastAPI, and Streamlit – your complete toolkit for agent development.**  Explore the [original repository](https://github.com/JoshuaC215/agent-service-toolkit) for more details.

[![build status](https://github.com/JoshuaC215/agent-service-toolkit/actions/workflows/test.yml/badge.svg)](https://github.com/JoshuaC215/agent-service-toolkit/actions/workflows/test.yml)
[![codecov](https://codecov.io/github/JoshuaC215/agent-service-toolkit/graph/badge.svg?token=5MTJSYWD05)](https://codecov.io/github/JoshuaC215/agent-service-toolkit)
[![Python Version](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2FJoshuaC215%2Fagent-service-toolkit%2Frefs%2Fheads%2Fmain%2Fpyproject.toml)](https://github.com/JoshuaC215/agent-service-toolkit/blob/main/pyproject.toml)
[![GitHub License](https://img.shields.io/github/license/JoshuaC215/agent-service-toolkit)](https://github.com/JoshuaC215/agent-service-toolkit/blob/main/LICENSE)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_red.svg)](https://agent-service-toolkit.streamlit.app/)

This toolkit provides a robust and flexible foundation for building, deploying, and interacting with AI agents using cutting-edge technologies.  It combines [LangGraph](https://langchain-ai.github.io/langgraph/) for agent orchestration, [FastAPI](https://fastapi.tiangolo.com/) for a scalable service, [Streamlit](https://streamlit.io/) for a user-friendly interface, and [Pydantic](https://github.com/pydantic/pydantic) for data management, enabling you to rapidly prototype and deploy AI-powered applications.

**[🎥 Watch a video walkthrough of the repo and app](https://www.youtube.com/watch?v=pdYVHw_YCNY)**

## Key Features

*   **LangGraph Agent with Latest Features**: Leverage the power of LangGraph with features like human-in-the-loop (`interrupt()`), flow control (`Command`), persistent memory (`Store`), and `langgraph-supervisor`.
*   **FastAPI Service**:  Serve your agents efficiently with both streaming and non-streaming endpoints, optimized for performance and scalability.
*   **Advanced Streaming**: Experience a novel approach to support token-based and message-based streaming, enhancing user experience.
*   **User-Friendly Streamlit Interface**:  Easily interact with your agents through an intuitive and customizable chat interface.
*   **Multiple Agent Support**:  Run multiple agents within the service and access them via URL paths, `/info` endpoint describes available agents and models.
*   **Asynchronous Design**: Benefit from asynchronous architecture for handling concurrent requests efficiently.
*   **Content Moderation (Optional)**: Implement content moderation using LlamaGuard (requires a Groq API key).
*   **RAG Agent**:  Includes a basic Retrieval-Augmented Generation (RAG) agent implementation using ChromaDB - see [docs](docs/RAG_Assistant.md).
*   **Feedback Mechanism**:  Integrates a star-based feedback system with LangSmith for iterative improvement.
*   **Docker Support**:  Includes Dockerfiles and a docker compose file for streamlined development and deployment.
*   **Comprehensive Testing**:  Ensure code quality with robust unit and integration tests throughout the project.

## Getting Started

### Try the app!

[Link to Streamlit App](https://agent-service-toolkit.streamlit.app/)

<a href="https://agent-service-toolkit.streamlit.app/"><img src="media/app_screenshot.png" width="600"></a>

### Quickstart

1.  **Prerequisites**: Ensure you have Python, `uv` (or pip), and Docker (optional, but recommended for ease of use) installed.

2.  **Clone the Repository**:

    ```bash
    git clone https://github.com/JoshuaC215/agent-service-toolkit.git
    cd agent-service-toolkit
    ```

3.  **Set up Environment Variables**:

    Create a `.env` file in the root directory with your API keys (e.g., OpenAI).  See the  [`.env.example` file](./.env.example) for all available options.

    ```bash
    echo 'OPENAI_API_KEY=your_openai_api_key' >> .env
    ```

4.  **Install Dependencies & Run:**

    ```bash
    # Install Dependencies (using uv - recommended)
    curl -LsSf https://astral.sh/uv/0.7.19/install.sh | sh
    uv sync --frozen
    source .venv/bin/activate

    # Run FastAPI service
    python src/run_service.py

    # In a separate terminal, run the Streamlit App
    source .venv/bin/activate
    streamlit run src/streamlit_app.py
    ```

    Or, use Docker for simpler setup:

    ```bash
    echo 'OPENAI_API_KEY=your_openai_api_key' >> .env
    docker compose watch
    ```

### Architecture Diagram

<img src="media/agent_architecture.png" width="600">

## Customization & Usage

### Building or Customizing Your Own Agent

1.  **Create a New Agent**:  Add your custom agent definition to the `src/agents` directory.  You can adapt the provided templates (`research_assistant.py` or `chatbot.py`).
2.  **Register Your Agent**: Import your new agent into `src/agents/agents.py` and add it to the `agents` dictionary.  Your agent will then be accessible via the API (e.g., `/<your_agent_name>/invoke` or `/<your_agent_name>/stream`).
3.  **Update the UI (Optional)**:  Modify `src/streamlit_app.py` to tailor the Streamlit interface to your agent's unique capabilities and features.

### Advanced Setup

*   [Setting up Ollama](docs/Ollama.md)
*   [Setting up VertexAI](docs/VertexAI.md)
*   [Setting up RAG with ChromaDB](docs/RAG_Assistant.md)
*   [Working with File-based Credentials](docs/File_Based_Credentials.md)

### Handling Private Credential files

If your agents or chosen LLM require file-based credential files or certificates, the `privatecredentials/` has been provided for your development convenience. All contents, excluding the `.gitkeep` files, are ignored by git and docker's build process. See [Working with File-based Credentials](docs/File_Based_Credentials.md) for suggested use.

### Docker Setup

This project includes a Docker setup for easy development and deployment. The `compose.yaml` file defines three services: `postgres`, `agent_service` and `streamlit_app`. The `Dockerfile` for each service is in their respective directories.

For local development, we recommend using [docker compose watch](https://docs.docker.com/compose/file-watch/). This feature allows for a smoother development experience by automatically updating your containers when changes are detected in your source code.

1.  Make sure you have Docker and Docker Compose (>= [v2.23.0](https://docs.docker.com/compose/release-notes/#2230)) installed on your system.

2.  Create a `.env` file from the `.env.example`. At minimum, you need to provide an LLM API key (e.g., OPENAI_API_KEY).
    ```sh
    cp .env.example .env
    # Edit .env to add your API keys
    ```

3.  Build and launch the services in watch mode:

    ```sh
    docker compose watch
    ```

    This will automatically:
    - Start a PostgreSQL database service that the agent service connects to
    - Start the agent service with FastAPI
    - Start the Streamlit app for the user interface

4.  The services will now automatically update when you make changes to your code:
    - Changes in the relevant python files and directories will trigger updates for the relevant services.
    - NOTE: If you make changes to the `pyproject.toml` or `uv.lock` files, you will need to rebuild the services by running `docker compose up --build`.

5.  Access the Streamlit app by navigating to `http://localhost:8501` in your web browser.

6.  The agent service API will be available at `http://0.0.0.0:8080`. You can also use the OpenAPI docs at `http://0.0.0.0:8080/redoc`.

7.  Use `docker compose down` to stop the services.

This setup allows you to develop and test your changes in real-time without manually restarting the services.

### Building other apps on the AgentClient

The repo includes a generic `src/client/client.AgentClient` that can be used to interact with the agent service. This client is designed to be flexible and can be used to build other apps on top of the agent. It supports both synchronous and asynchronous invocations, and streaming and non-streaming requests.

See the `src/run_client.py` file for full examples of how to use the `AgentClient`. A quick example:

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

### Development with LangGraph Studio

The agent supports [LangGraph Studio](https://langchain-ai.github.io/langgraph/concepts/langgraph_studio/), the IDE for developing agents in LangGraph.

`langgraph-cli[inmem]` is installed with `uv sync`. You can simply add your `.env` file to the root directory as described above, and then launch LangGraph Studio with `langgraph dev`. Customize `langgraph.json` as needed. See the [local quickstart](https://langchain-ai.github.io/langgraph/cloud/how-tos/studio/quick_start/#local-development-server) to learn more.

### Local development without Docker

You can also run the agent service and the Streamlit app locally without Docker, just using a Python virtual environment.

1. Create a virtual environment and install dependencies:

   ```sh
   uv sync --frozen
   source .venv/bin/activate
   ```

2. Run the FastAPI server:

   ```sh
   python src/run_service.py
   ```

3. In a separate terminal, run the Streamlit app:

   ```sh
   streamlit run src/streamlit_app.py
   ```

4. Open your browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

## Projects Built With or Inspired By This Toolkit

This repository has served as a foundation for several exciting projects:

*   **[PolyRAG](https://github.com/QuentinFuxa/PolyRAG)** -  Extends this toolkit with advanced RAG capabilities, integrating with PostgreSQL and PDF documents.
*   **[alexrisch/agent-web-kit](https://github.com/alexrisch/agent-web-kit)** -  A Next.JS frontend designed to work seamlessly with this toolkit.
*   **[raushan-in/dapa](https://github.com/raushan-in/dapa)** - Digital Arrest Protection App (DAPA) enables users to report financial scams and frauds efficiently via a user-friendly platform.

**If you've built something using this toolkit, please submit a PR or open a discussion to add your project!**

## Contributing

We welcome contributions! To contribute:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Make your changes and commit them.
4.  Run tests (ensure you've installed dev dependencies and pre-commit):

    ```bash
    uv sync --frozen
    pre-commit install
    pytest
    ```

5.  Submit a pull request.

## License

This project is licensed under the MIT License.  See the `LICENSE` file for details.
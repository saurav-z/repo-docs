# AI Agent Service Toolkit: Build, Deploy, and Manage Powerful AI Agents

**Effortlessly create and manage AI agent services with this comprehensive toolkit built on LangGraph, FastAPI, and Streamlit. ([Original Repo](https://github.com/JoshuaC215/agent-service-toolkit))**

This toolkit provides everything you need to get started with AI agent development, from agent definition to user interface. It's a complete solution for building and deploying advanced AI agents.

## Key Features

*   **LangGraph Agent & Latest Features:** Leverage the power of LangGraph, including human-in-the-loop, flow control, long-term memory, and `langgraph-supervisor`.
*   **FastAPI Service:** Serve your agents efficiently with both streaming and non-streaming endpoints.
*   **Advanced Streaming:** Innovative streaming capabilities to support both token-based and message-based streaming.
*   **Streamlit Interface:** A user-friendly chat interface for easy interaction with your agents.
*   **Multiple Agent Support:** Manage and call multiple agents within the service, accessible via URL paths.
*   **Asynchronous Design:** Optimized for performance with an async/await architecture.
*   **Content Moderation:** Integrated LlamaGuard for responsible AI implementation (requires Groq API key).
*   **RAG Agent:** A basic RAG agent implementation using ChromaDB, with documentation to help you build your own (see [docs](docs/RAG_Assistant.md)).
*   **Feedback Mechanism:**  Integrate a star-based feedback system with LangSmith for continuous improvement.
*   **Docker Support:** Includes Dockerfiles and a docker compose file for simple development and deployment.
*   **Extensive Testing:**  Includes thorough unit and integration tests for reliability.

## Quickstart

Get up and running quickly with our pre-built setup:

**Try the app!**

[Try the app!](https://agent-service-toolkit.streamlit.app/)
<a href="https://agent-service-toolkit.streamlit.app/"><img src="media/app_screenshot.png" width="600"></a>

**Python Setup**
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

**Docker Setup**

```sh
echo 'OPENAI_API_KEY=your_openai_api_key' >> .env
docker compose watch
```

## Architecture

![Agent Architecture Diagram](media/agent_architecture.png)

## Key Files

*   `src/agents/`: Defines various AI agents.
*   `src/schema/`: Defines the data structures and protocol schema.
*   `src/core/`: Core modules including LLM definition and settings.
*   `src/service/service.py`: FastAPI service for serving the agents.
*   `src/client/client.py`: Client for interacting with the agent service.
*   `src/streamlit_app.py`: Streamlit application providing the user interface.
*   `tests/`: Unit and integration tests.

## Setup and Usage

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/JoshuaC215/agent-service-toolkit.git
    cd agent-service-toolkit
    ```

2.  **Set up environment variables:**

    *   Create a `.env` file in the root directory (see `.env.example` for reference).  You'll need at least one LLM API key to get started.

3.  **Run the service and app:** Follow the instructions in the Quickstart section above for either a Python or Docker-based setup.

### Additional Setup and Guides

*   [Setting up Ollama](docs/Ollama.md)
*   [Setting up VertexAI](docs/VertexAI.md)
*   [Setting up RAG with ChromaDB](docs/RAG_Assistant.md)
*   [Working with File-based Credentials](docs/File_Based_Credentials.md)

### Building or customizing your own agent

1.  Add your new agent to the `src/agents` directory.
2.  Import and add your new agent to the `agents` dictionary in `src/agents/agents.py`.
3.  Adjust the Streamlit interface in `src/streamlit_app.py` to match your agent's capabilities.

## Docker Setup and Development

The project includes a Docker Compose setup to streamline development and deployment.

1.  Install Docker and Docker Compose.
2.  Create a `.env` file, ensuring you include the necessary API keys and configurations (See the `.env.example` file).
3.  Run `docker compose watch` to build, launch, and automatically update your services as you make code changes.
4.  Access the Streamlit app at `http://localhost:8501` and the API at `http://0.0.0.0:8080`.
5.  Use `docker compose down` to stop services.

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

Integrate with [LangGraph Studio](https://langchain-ai.github.io/langgraph/concepts/langgraph_studio/) to develop and debug your agents. Launch Studio with `langgraph dev` after adding your `.env` file to the root. Customize `langgraph.json` as needed.

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

## Projects Built With or Inspired By agent-service-toolkit

*   **[PolyRAG](https://github.com/QuentinFuxa/PolyRAG)** - Extends agent-service-toolkit with RAG capabilities over both PostgreSQL databases and PDF documents.
*   **[alexrisch/agent-web-kit](https://github.com/alexrisch/agent-web-kit)** - A Next.JS frontend for agent-service-toolkit
*   **[raushan-in/dapa](https://github.com/raushan-in/dapa)** - Digital Arrest Protection App (DAPA) enables users to report financial scams and frauds efficiently via a user-friendly platform.

**Please contribute links to your projects via a pull request!**

## Contributing

Contributions are highly encouraged!  To contribute, create a Pull Request with your changes after running tests.

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

This project is licensed under the MIT License.  See the `LICENSE` file for details.
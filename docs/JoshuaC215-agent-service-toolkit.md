# AI Agent Service Toolkit: Build and Deploy Powerful AI Agents with Ease

**Quickly build, deploy, and interact with AI agents using LangGraph, FastAPI, and Streamlit, offering a complete toolkit for your AI agent service needs.**  [Check out the original repository!](https://github.com/JoshuaC215/agent-service-toolkit)

This toolkit provides a comprehensive framework for building and deploying AI agents, offering a streamlined approach from agent definition to user interface. Leverage LangGraph's power with a FastAPI service, a client for interaction, and a Streamlit app for a user-friendly chat experience.

**[🎥 Watch a video walkthrough of the repo and app](https://www.youtube.com/watch?v=pdYVHw_YCNY)**

## Key Features:

*   **LangGraph Agent:**  Customizable agents built with the latest LangGraph features, including human-in-the-loop, flow control, long-term memory, and `langgraph-supervisor`.
*   **FastAPI Service:**  Efficiently serves your agent with both streaming and non-streaming endpoints, optimized for performance.
*   **Advanced Streaming:** A novel approach to support both token-based and message-based streaming.
*   **Streamlit Interface:** A user-friendly chat interface built with Streamlit to allow you to interact with your agent.
*   **Multiple Agent Support:**  Run multiple agents, accessible via URL pathing with easy-to-find `/info` endpoint to get available agents and models.
*   **Asynchronous Design:** Built with asynchronous/await for efficient concurrent request handling.
*   **Content Moderation:** Implements LlamaGuard for content moderation (requires Groq API key).
*   **RAG Agent:**  Includes a basic RAG agent implementation using ChromaDB, see [docs](docs/RAG_Assistant.md).
*   **Feedback Mechanism:** A star-based feedback system with integration with LangSmith.
*   **Docker Support:** Includes Dockerfiles and a docker compose file for effortless development and deployment.
*   **Comprehensive Testing:** Includes robust unit and integration tests to ensure reliability.

## Getting Started

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/JoshuaC215/agent-service-toolkit.git
    cd agent-service-toolkit
    ```

2.  **Set Up Environment Variables:**
    Create a `.env` file in the root directory and add your API keys.  See the [`.env.example` file](./.env.example) for a complete list of available variables.

3.  **Run the service**
    You can either run the app with Docker or using Python in a virtual environment.  Docker is the recommended approach for easy environment setup.

    **Docker:**

    ```bash
    echo 'OPENAI_API_KEY=your_openai_api_key' >> .env
    docker compose watch
    ```

    **Local Development (Python):**

    ```bash
    uv sync --frozen
    source .venv/bin/activate
    python src/run_service.py
    # In another shell
    source .venv/bin/activate
    streamlit run src/streamlit_app.py
    ```

4.  **Access the App:**  If using Docker, your Streamlit app will be at `http://localhost:8501`.  The API will be at `http://0.0.0.0:8080` and the openAPI docs are available at `http://0.0.0.0:8080/redoc`.

## Architecture

```mermaid
graph LR
    A[User (Streamlit)] --> B(Agent Service (FastAPI))
    B --> C(LangGraph Agent)
    C --> D(LLM / Tools)
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style B fill:#ccf,stroke:#333,stroke-width:2px
    style C fill:#fcc,stroke:#333,stroke-width:2px
    style D fill:#cff,stroke:#333,stroke-width:2px
```

## Customization

*   **Building Your Own Agent:**

    1.  Add your new agent to the `src/agents` directory.
    2.  Import and add your agent to the `agents` dictionary in `src/agents/agents.py`.
    3.  Adjust the Streamlit interface in `src/streamlit_app.py` to match your agent's capabilities.

*   **Handling Credentials:**  Use the `privatecredentials/` directory for file-based credentials (ignored by Git and Docker builds) - see [Working with File-based Credentials](docs/File_Based_Credentials.md).

## Additional Setup Guides

*   [Setting up Ollama](docs/Ollama.md)
*   [Setting up VertexAI](docs/VertexAI.md)
*   [Setting up RAG with ChromaDB](docs/RAG_Assistant.md)

## Projects Built With or Inspired By

*   [PolyRAG](https://github.com/QuentinFuxa/PolyRAG): RAG capabilities with PostgreSQL and PDF documents.
*   [alexrisch/agent-web-kit](https://github.com/alexrisch/agent-web-kit): A Next.JS frontend.
*   [raushan-in/dapa](https://github.com/raushan-in/dapa): Digital Arrest Protection App (DAPA) for reporting scams.

## Contribute

Contributions are welcome! Submit a Pull Request.  To run tests:

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

This project is licensed under the MIT License.  See the `LICENSE` file for details.
# AI Agent Service Toolkit: Build, Deploy, and Experiment with Powerful AI Agents

**Unlock the power of AI agents with a robust toolkit built with LangGraph, FastAPI, and Streamlit. Check out the [original repository](https://github.com/JoshuaC215/agent-service-toolkit) for the full details and to contribute!**

This toolkit provides a comprehensive framework for creating and deploying AI agents, offering a streamlined approach from agent definition to user interface.  Built with LangGraph, FastAPI, and Streamlit, it empowers developers to rapidly prototype, test, and deploy their own AI agent services.  It serves as a great template and starting point for your own LangGraph projects, providing a full and robust toolkit.

## Key Features

*   **LangGraph Agent with Latest Features:** Leverage the latest advancements in LangGraph v0.3, including human-in-the-loop (`interrupt()`), flow control (`Command`), long-term memory (`Store`), and `langgraph-supervisor`.
*   **FastAPI Service:**  Deploy your agent with a scalable FastAPI service, offering both streaming and non-streaming endpoints for optimal performance.
*   **Advanced Streaming:**  Benefit from a novel approach to support both token-based and message-based streaming, enhancing the user experience.
*   **Streamlit Interface:**  Provide a user-friendly chat interface built with Streamlit, making agent interaction intuitive and accessible.
*   **Multiple Agent Support:** Easily manage and run multiple agents within the service, accessible via URL paths. See `/info` for available agents and models.
*   **Asynchronous Design:**  Experience efficient handling of concurrent requests with an asynchronous design.
*   **Content Moderation:**  Integrate LlamaGuard for content moderation (requires Groq API key).
*   **RAG Agent:**  Includes a basic RAG agent implementation using ChromaDB - explore the [RAG Assistant documentation](docs/RAG_Assistant.md).
*   **Feedback Mechanism:**  Integrate a star-based feedback system connected to LangSmith for performance analysis.
*   **Docker Support:** Simplify development and deployment with included Dockerfiles and a `docker compose` file.
*   **Comprehensive Testing:**  Ensure stability and reliability with robust unit and integration tests.

## Key Technologies

*   **LangGraph:** Define and orchestrate complex AI agent workflows.
*   **FastAPI:**  Build fast and efficient APIs to serve your agents.
*   **Streamlit:**  Create interactive and user-friendly interfaces for your agents.
*   **Pydantic:** Manage data structures and settings with ease.

## Quickstart

Get up and running in minutes with either Python or Docker:

**Python:**

```bash
# Configure API keys (at least one LLM API key is required)
echo 'OPENAI_API_KEY=your_openai_api_key' >> .env

# Install dependencies
uv sync --frozen
source .venv/bin/activate
python src/run_service.py

# In another terminal
source .venv/bin/activate
streamlit run src/streamlit_app.py
```

**Docker:**

```bash
# Configure API keys (at least one LLM API key is required)
echo 'OPENAI_API_KEY=your_openai_api_key' >> .env
docker compose watch
```

## Architecture

![Architecture Diagram](media/agent_architecture.png)

## Additional Resources

*   **[Try the App!](https://agent-service-toolkit.streamlit.app/)**
    <a href="https://agent-service-toolkit.streamlit.app/"><img src="media/app_screenshot.png" width="600"></a>
*   **[🎥 Video Walkthrough](https://www.youtube.com/watch?v=pdYVHw_YCNY)**
*   **Setup Guides:**
    *   [Setting up Ollama](docs/Ollama.md)
    *   [Setting up VertexAI](docs/VertexAI.md)
    *   [Setting up RAG with ChromaDB](docs/RAG_Assistant.md)
*   **File-based Credentials:** [Working with File-based Credentials](docs/File_Based_Credentials.md)

## Contributing

We welcome contributions!  Please review the [Contributing guidelines](CONTRIBUTING.md).

1.  Install development dependencies and pre-commit hooks:

    ```bash
    uv sync --frozen
    pre-commit install
    ```
2.  Run tests:

    ```bash
    pytest
    ```

## Projects Built on agent-service-toolkit (or inspired by it)

*   [PolyRAG](https://github.com/QuentinFuxa/PolyRAG)
*   [alexrisch/agent-web-kit](https://github.com/alexrisch/agent-web-kit)
*   [raushan-in/dapa](https://github.com/raushan-in/dapa)

## License

This project is licensed under the MIT License.  See the [LICENSE](LICENSE) file for details.
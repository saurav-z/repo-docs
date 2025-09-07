# AI Agent Service Toolkit: Build, Deploy, and Scale Your AI Agents

**Quickly create and deploy robust AI agent services with this comprehensive toolkit, featuring LangGraph, FastAPI, and Streamlit.**  [Explore the Agent Service Toolkit on GitHub](https://github.com/JoshuaC215/agent-service-toolkit)

This toolkit provides a complete framework for building and deploying AI agent services, leveraging the power of LangGraph, FastAPI, and Streamlit. It offers a streamlined approach from agent definition to a user-friendly interface.

**Key Features:**

*   **LangGraph Integration:** Build and manage agents with the latest LangGraph features, including `interrupt()`, `Command`, `Store`, and `langgraph-supervisor`.
*   **FastAPI Service:** Provides a scalable FastAPI service with both streaming and non-streaming endpoints.
*   **Advanced Streaming:** Novel approach to support both token-based and message-based streaming.
*   **User-Friendly Interface:** Streamlit app provides an intuitive chat interface.
*   **Multiple Agent Support:** Run multiple agents, accessible via URL paths and described in `/info`.
*   **Asynchronous Design:** Efficiently handles concurrent requests with async/await.
*   **Content Moderation:** Integrates LlamaGuard for content moderation (requires Groq API key).
*   **RAG Agent Implementation:** Basic RAG agent implemented with ChromaDB (see [docs](docs/RAG_Assistant.md)).
*   **Feedback Mechanism:** Includes a star-based feedback system integrated with LangSmith.
*   **Docker Support:** Dockerfiles and a docker compose file for easy development and deployment.
*   **Comprehensive Testing:** Robust unit and integration tests ensure reliability.

## Get Started

### Try the App!

[<img src="https://user-images.githubusercontent.com/35504121/273220636-cf56210c-0a6a-441c-a763-187868495802.png" width="600">](https://agent-service-toolkit.streamlit.app/)

### Quickstart

Choose your preferred setup:

**1. Local Development with Python**

```bash
# Set LLM API key (required)
echo 'OPENAI_API_KEY=your_openai_api_key' >> .env

# Install dependencies (uv is recommended)
curl -LsSf https://astral.sh/uv/0.7.19/install.sh | sh
uv sync --frozen
source .venv/bin/activate

# Run the service
python src/run_service.py

# In a separate terminal, run the Streamlit app
source .venv/bin/activate
streamlit run src/streamlit_app.py
```

**2. Docker**

```bash
# Set LLM API key (required)
echo 'OPENAI_API_KEY=your_openai_api_key' >> .env

# Run with docker compose watch
docker compose watch
```

### Architecture Diagram

<img src="media/agent_architecture.png" width="600">

## Key Files

*   `src/agents/`: Agent definitions.
*   `src/schema/`: Data structures and schema definitions.
*   `src/core/`: Core modules, including LLM configuration.
*   `src/service/service.py`: FastAPI service.
*   `src/client/client.py`: Agent service client.
*   `src/streamlit_app.py`: Streamlit chat interface.
*   `tests/`: Unit and integration tests.

## Setting Up and Using the Toolkit

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/JoshuaC215/agent-service-toolkit.git
    cd agent-service-toolkit
    ```

2.  **Configure Environment Variables:**

    *   Create a `.env` file (see `.env.example` for options). At minimum, provide an LLM API key.

3.  **Choose your deployment strategy:**
    *   Follow the Quickstart instructions above, depending on the environment you want to use.

### Additional Setups

*   [Setting up Ollama](docs/Ollama.md)
*   [Setting up VertexAI](docs/VertexAI.md)
*   [Setting up RAG with ChromaDB](docs/RAG_Assistant.md)

### Customizing Your Agent

1.  Add your new agent to the `src/agents` directory.
2.  Import and add your agent to the `agents` dictionary in `src/agents/agents.py`.
3.  Adjust the Streamlit interface in `src/streamlit_app.py` as needed.

### Handling Private Credentials

Use the `privatecredentials/` directory to store file-based credentials, which are ignored by Git and the Docker build process.  See [Working with File-based Credentials](docs/File_Based_Credentials.md)

###  Developing with LangGraph Studio

The agent supports [LangGraph Studio](https://langchain-ai.github.io/langgraph/concepts/langgraph_studio/), for agent development.

*   Add your `.env` file to the root directory, and run `langgraph dev`.
*   Customize `langgraph.json` as needed.

## Projects Using or Inspired By This Toolkit

*   **[PolyRAG](https://github.com/QuentinFuxa/PolyRAG)**: Enhances agent-service-toolkit with RAG capabilities over PostgreSQL and PDF documents.
*   **[alexrisch/agent-web-kit](https://github.com/alexrisch/agent-web-kit)**: A Next.JS frontend for agent-service-toolkit
*   **[raushan-in/dapa](https://github.com/raushan-in/dapa)**: Digital Arrest Protection App (DAPA) enables users to report financial scams and frauds efficiently.

**Contribute by submitting a Pull Request or starting a discussion to add your project to this list!**

## Contributing

Contributions are welcome!

1.  Follow the setup instructions to install development dependencies.
2.  Run the tests with `pytest`.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
# IntentKit: Build and Manage Autonomous AI Agents

**Unlock the power of AI with IntentKit, an open-source framework for creating intelligent agents that can interact with the world, from blockchain to social media.**  ([Original Repo](https://github.com/crestalnetwork/intentkit))

<div align="center">
  <img src="docs/images/intentkit_banner.png" alt="IntentKit by Crestal" width="100%" />
</div>
<br>

IntentKit is an autonomous agent framework designed to simplify the development and management of AI agents capable of diverse tasks.  It empowers developers to create agents that can interact with various platforms, including blockchain, social media, and custom-built skills.

## Key Features

*   **🤖 Multiple Agent Support:**  Easily manage and deploy multiple autonomous agents.
*   **🔄 Autonomous Agent Management:** Streamline the lifecycle of your AI agents.
*   **🔗 Blockchain Integration:** Interact with EVM-compatible blockchains to execute transactions and access data.
*   **🐦 Social Media Integration:** Connect with platforms like Twitter and Telegram to manage social media presence and automate tasks.
*   **🛠️ Extensible Skill System:** Build and integrate custom skills to expand agent capabilities.
*   **🔌 MCP (WIP):**  (Placeholder for future functionality)

## Architecture

IntentKit is built around a modular architecture, enabling flexible and scalable agent development.  Core components include:

*   **Entrypoints:**  Interact with agents through various channels (Twitter, Telegram, Web, etc.).
*   **Agent Core:** Powered by LangGraph, the central processing unit for your agents.
*   **Storage:**  Manages agent configuration, credentials, personality, and memory.
*   **Skills:**  Provide specific functionalities, such as blockchain interaction, social media management, internet searches, and image processing.

For a more in-depth understanding, refer to the detailed [Architecture](docs/architecture.md) documentation.

## Package Manager Update

Important:  This project has migrated to `uv` from `poetry`.  To update your environment:

```bash
rm -rf .venv
uv sync
```

## Development

Get started with IntentKit by following the [Development Guide](DEVELOPMENT.md) for setup instructions.

## Documentation

Comprehensive documentation is available to help you get the most out of IntentKit.  Explore the [Documentation](docs/) to learn more.

## Project Structure

The project is organized into two main components:

*   **`intentkit/`**: The core IntentKit package (published to PyPI)
    *   `abstracts/`: Abstract classes and interfaces
    *   `clients/`: Clients for external services
    *   `config/`: System-level configurations
    *   `core/`: Core agent system
    *   `models/`: Entity models
    *   `skills/`: Extensible skills system
    *   `utils/`: Utility functions
*   **`app/`**: The IntentKit application (API server, autonomous runner, scheduler)
    *   `admin/`: Admin APIs and agent management
    *   `entrypoints/`: Agent interaction entrypoints
    *   `services/`: Service implementations (e.g., Telegram, Twitter)
    *   `api.py`: REST API server
    *   `autonomous.py`: Autonomous agent runner
    *   `checker.py`: Health and credit checking
    *   `readonly.py`: Read-only entrypoint
    *   `scheduler.py`: Background task scheduler
    *   `singleton.py`: Singleton agent manager
    *   `telegram.py`: Telegram integration
    *   `twitter.py`: Twitter integration
*   `docs/`: Documentation
*   `scripts/`: Scripts for management and migrations

## Agent API

Integrate your applications seamlessly with IntentKit agents using the comprehensive REST API.

*   **Explore the API:** [Agent API Documentation](docs/agent_api.md)

## Contributing

We welcome contributions!  Please review the [Contributing Guidelines](CONTRIBUTING.md) before submitting a pull request.

### Contributing Skills

Extend IntentKit's capabilities by contributing new skills.

1.  Review the [Wishlist](docs/contributing/wishlist.md) for existing requests.
2.  Follow the [Skill Development Guide](docs/contributing/skills.md) to get started.

### Developer Community

Join the IntentKit developer community for discussions and support:

*   **Discord:** [Discord](https://discord.com/invite/crestal) - Request an intentkit dev role for access.

## License

This project is licensed under the MIT License.  See the [LICENSE](LICENSE) file for details.
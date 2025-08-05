# IntentKit: Build and Manage Powerful AI Agents

**IntentKit** is an open-source framework that empowers you to create and control AI agents with a wide range of capabilities, including blockchain interaction, social media management, and custom skill integration.  Check out the [original repository](https://github.com/crestalnetwork/intentkit) for more details and to contribute.

<div align="center">
  <img src="docs/images/intentkit_banner.png" alt="IntentKit by Crestal" width="100%" />
</div>
<br>

## Key Features

*   **Multi-Agent Support:** Manage and orchestrate multiple AI agents simultaneously.
*   **Autonomous Agent Management:**  Control and oversee your agents with automated workflows.
*   **Blockchain Integration:** Interact with EVM-compatible blockchains and execute on-chain actions.
*   **Social Media Integration:**  Connect to platforms like Twitter, Telegram, and more to engage with users.
*   **Extensible Skill System:**  Customize agent capabilities with a flexible skill system, utilizing LangChain tools.
*   **Modular Architecture:** Built with a robust architecture, using LangGraph, for scalability and maintainability.

## Package Manager Migration Notice

This project has migrated to `uv` from `poetry`. To set up your environment:

```bash
rm -rf .venv
uv sync
```

## Architecture Overview

IntentKit's architecture is designed for flexibility and extensibility.  Agents interact with various entry points (like Twitter, Telegram) and leverage a core engine powered by LangGraph. This engine connects to storage (agent configuration, credentials, memory), skills (blockchain interaction, internet search, etc.), and various configurations.  For a deeper dive, see the detailed [Architecture](docs/architecture.md) documentation.

## Project Structure

The project is organized into two main components:

*   **`intentkit/`**: The core IntentKit package (pip installable) including:
    *   `abstracts/`: Abstract classes and interfaces
    *   `clients/`: Clients for external services
    *   `config/`: System configurations
    *   `core/`: Core agent system (LangGraph)
    *   `models/`: Entity models (Pydantic, SQLAlchemy)
    *   `skills/`: Extensible skills system (LangChain)
    *   `utils/`: Utility functions
*   **`app/`**: The IntentKit application (API server, autonomous runner, scheduler) including:
    *   `admin/`: Admin APIs and agent generation
    *   `entrypoints/`: Agent interaction entrypoints (web, Telegram, Twitter)
    *   `services/`: Implementations for Telegram, Twitter, etc.
    *   `api.py`: REST API server
    *   `autonomous.py`: Autonomous agent runner
    *   `checker.py`: Health and credit checking
    *   `readonly.py`: Readonly entrypoint
    *   `scheduler.py`: Background task scheduler
    *   `singleton.py`: Singleton agent manager
    *   `telegram.py`: Telegram integration
    *   `twitter.py`: Twitter integration
*   `docs/`: Documentation
*   `scripts/`: Management and migration scripts

## Agent API

IntentKit provides a comprehensive REST API for programmatic interaction with your agents.  Use this API to build custom applications, integrate with other systems, or create tailored interfaces.  Explore the [Agent API Documentation](docs/agent_api.md) to get started.

## Development

To set up your development environment, consult the [Development Guide](DEVELOPMENT.md).

## Documentation

Comprehensive documentation is available; start with the [Documentation](docs/) section.

## Contributing

We welcome contributions!  Please review our [Contributing Guidelines](CONTRIBUTING.md) before submitting a pull request.

### Contribute Skills

Check the [Wishlist](docs/contributing/wishlist.md) for active skill requests. Then, consult the [Skill Development Guide](docs/contributing/skills.md) to start building your own.

### Developer Community

Join our active developer community on [Discord](https://discord.com/invite/crestal). Open a support ticket to request an intentkit developer role. There's a discussion channel available for collaboration.

## License

This project is licensed under the MIT License; see the [LICENSE](LICENSE) file for details.
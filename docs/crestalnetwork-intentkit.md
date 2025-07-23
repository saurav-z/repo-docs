# IntentKit: Build Autonomous AI Agents with Ease

<div align="center">
  <img src="docs/images/intentkit_banner.png" alt="IntentKit by Crestal" width="100%" />
</div>
<br>

**IntentKit is an open-source framework empowering you to build and manage sophisticated AI agents capable of interacting with blockchains, social media, and more.**  This framework provides a robust foundation for creating autonomous agents with diverse capabilities.  [Explore the original repository](https://github.com/crestalnetwork/intentkit) for more details.

## Key Features

*   **Multi-Agent Support:** Manage and orchestrate multiple AI agents within a single framework.
*   **Autonomous Agent Management:**  Provides tools for the lifecycle management and operation of autonomous agents.
*   **Blockchain Integration:** Seamlessly interact with EVM-compatible blockchains (with expansion planned).
*   **Social Media Integration:** Connect your agents to platforms like Twitter and Telegram.
*   **Extensible Skill System:** Easily add custom skills and functionality to your agents.
*   **MCP (WIP):** (Mention of MCP - details needed for a complete description.)

## Architecture Overview

IntentKit's architecture is designed for modularity and extensibility.  Agents are the central components, receiving input from entrypoints such as social media platforms and interacting with a variety of skills, including blockchain integration, internet search, and image processing. The system utilizes a core driven by LangGraph for efficient execution and management.

[See detailed architecture](docs/architecture.md) for a comprehensive look.

## Development

Get started quickly with the [Development Guide](DEVELOPMENT.md).

## Documentation

Comprehensive documentation is available to guide you through the framework.  Explore the [Documentation](docs/) to begin.

## Project Structure

The project is structured into core components:

*   **`intentkit/`:** The core IntentKit package (available via pip).
    *   `abstracts/`: Abstract classes and interfaces.
    *   `clients/`: Clients for interacting with external services.
    *   `config/`: System-level configurations.
    *   `core/`: Core agent system based on LangGraph.
    *   `models/`: Entity models using Pydantic and SQLAlchemy.
    *   `skills/`: Extensible skill system using LangChain tools.
    *   `utils/`: Utility functions.

*   **`app/`:** The IntentKit application, including:
    *   `admin/`: Admin APIs and agent generation tools.
    *   `entrypoints/`: Interfaces for agent interaction (web, Telegram, Twitter, etc.).
    *   `services/`: Implementations for integrations (Telegram, Twitter).
    *   `api.py`: REST API server.
    *   `autonomous.py`: Autonomous agent runner.
    *   `checker.py`: Health and credit checking logic.
    *   `readonly.py`: Read-only entrypoint.
    *   `scheduler.py`: Background task scheduler.
    *   `singleton.py`: Singleton agent manager.
    *   `telegram.py`: Telegram integration.
    *   `twitter.py`: Twitter integration.

*   `docs/`: Documentation
*   `scripts/`: Scripts for management and migrations.

## Agent API

The Agent API provides programmatic access to your agents.  This REST API allows you to build custom applications and integrate your agents with existing systems.  [Explore the Agent API Documentation](docs/agent_api.md).

## Contributing

Contributions are welcome! Please review the [Contributing Guidelines](CONTRIBUTING.md) before submitting a pull request.

### Contribute Skills

Check the [Wishlist](docs/contributing/wishlist.md) for feature requests and development ideas.

See the [Skill Development Guide](docs/contributing/skills.md) for details on building new skills.

### Developer Chat

Join our developer community on [Discord](https://discord.com/invite/crestal). Open a support ticket to request access to the IntentKit dev role and engage with other developers.

## License

This project is licensed under the [MIT License](LICENSE).
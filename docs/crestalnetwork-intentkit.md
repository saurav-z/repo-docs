# IntentKit: Build and Manage Autonomous AI Agents

**Create powerful AI agents with ease using IntentKit, an open-source framework designed for blockchain interaction, social media management, and custom skill integration.** Learn more on the [original repo](https://github.com/crestalnetwork/intentkit).

<div align="center">
  <img src="docs/images/intentkit_banner.png" alt="IntentKit by Crestal" width="100%" />
</div>
<br>

IntentKit is an open-source framework for developing and managing autonomous AI agents, giving you control over their capabilities and actions. This framework allows you to build AI agents with various capabilities, including blockchain interaction, social media management, and custom skill integration.

## Key Features

*   🤖 **Multi-Agent Support:** Manage and deploy multiple AI agents simultaneously.
*   🔄 **Autonomous Agent Management:** Control and monitor your agents' lifecycle.
*   🔗 **Blockchain Integration:** Interact with EVM-compatible blockchain networks.
*   🐦 **Social Media Integration:** Connect with platforms like Twitter, Telegram, and more.
*   🛠️ **Extensible Skill System:** Customize agents with a wide range of skills based on LangChain tools.
*   🔌 **MCP (WIP):** In development.

## Architecture Overview

IntentKit's architecture is designed for flexibility and extensibility. The system integrates entrypoints for communication (Twitter, Telegram, etc.) with a central agent core powered by LangGraph. This core utilizes storage for agent configuration, credentials, memory, and skill state, and utilizes skills such as chain integration and internet search.

For more details, see the [Architecture](docs/architecture.md) section.

## Development

Get started with IntentKit by following the [Development Guide](DEVELOPMENT.md).

## Documentation

Explore comprehensive documentation to learn more about IntentKit: [Documentation](docs/).

## Project Structure

The project is divided into the core package and the application:

*   **[intentkit/](intentkit/)**: The IntentKit package (published as a pip package)
    *   [abstracts/](intentkit/abstracts/): Abstract classes and interfaces for core and skills
    *   [clients/](intentkit/clients/): Clients for external services
    *   [config/](intentkit/config/): System level configurations
    *   [core/](intentkit/core/): Core agent system, driven by LangGraph
    *   [models/](intentkit/models/): Entity models using Pydantic and SQLAlchemy
    *   [skills/](intentkit/skills/): Extensible skills system, based on LangChain tools
    *   [utils/](intentkit/utils/): Utility functions
*   **[app/](app/)**: The IntentKit app (API server, autonomous runner, and background scheduler)
    *   [admin/](app/admin/): Admin APIs, agent generators, and related functionality
    *   [entrypoints/](app/entrypoints/): Entrypoints for interacting with agents (web, Telegram, Twitter, etc.)
    *   [services/](app/services/): Service implementations for Telegram, Twitter, etc.
    *   [api.py](app/api.py): REST API server
    *   [autonomous.py](app/autonomous.py): Autonomous agent runner
    *   [checker.py](app/checker.py): Health and credit checking logic
    *   [readonly.py](app/readonly.py): Readonly entrypoint
    *   [scheduler.py](app/scheduler.py): Background task scheduler
    *   [singleton.py](app/singleton.py): Singleton agent manager
    *   [telegram.py](app/telegram.py): Telegram integration
    *   [twitter.py](app/twitter.py): Twitter integration
*   [docs/](docs/): Documentation
*   [scripts/](scripts/): Operation and temporary scripts for management and migrations

## Agent API

Use the Agent API for programmatic access to your agents to build applications, integrate with existing systems, or create custom interfaces.

**Explore the Agent API:** [Agent API Documentation](docs/agent_api.md)

## Contributing

We welcome contributions! Please read our [Contributing Guidelines](CONTRIBUTING.md) before submitting a pull request.

### Contribute Skills

1.  Check the [Wishlist](docs/contributing/wishlist.md) for active requests.
2.  See the [Skill Development Guide](docs/contributing/skills.md) for more information.

### Developer Community

Join our [Discord](https://discord.com/invite/crestal) to connect with other developers. Open a support ticket to apply for an IntentKit dev role.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
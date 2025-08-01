# IntentKit: Build and Manage Autonomous AI Agents

[<img src="docs/images/intentkit_banner.png" alt="IntentKit by Crestal" width="100%">](https://github.com/crestalnetwork/intentkit)

**IntentKit empowers you to create and control intelligent AI agents that can perform complex tasks, bridging the gap between your ideas and automated solutions.**

## Key Features

*   🤖 **Multi-Agent Support:** Manage and orchestrate multiple AI agents.
*   🔄 **Autonomous Agent Management:** Automate agent workflows and decision-making.
*   🔗 **Blockchain Integration:** Interact with EVM-compatible blockchain networks.
*   🐦 **Social Media Integration:** Connect with platforms like Twitter and Telegram.
*   🛠️ **Extensible Skill System:** Customize agents with a wide range of abilities.
*   🔌 **MCP (WIP):**  (More Coming Soon)

## Architecture Overview

IntentKit's architecture is built around a core agent system driven by LangGraph, enabling agents to interact with various entrypoints (Twitter, Telegram, etc.) and utilize skills for tasks like blockchain interaction, internet search, and more. For a more detailed view, please refer to the [Architecture documentation](docs/architecture.md).

## Project Structure

IntentKit is organized into two main components:

*   **`intentkit/`**: The core Python package, available for installation.
    *   `abstracts/`: Abstract classes and interfaces.
    *   `clients/`: Clients for external services.
    *   `config/`: System-level configurations.
    *   `core/`: Core agent system.
    *   `models/`: Entity models.
    *   `skills/`: Extensible skills system.
    *   `utils/`: Utility functions.
*   **`app/`**: The application layer, providing API server, agent runner, and scheduler.
    *   `admin/`: Admin APIs and agent generation.
    *   `entrypoints/`: Agent interaction entrypoints.
    *   `services/`: Service implementations.
    *   `api.py`: REST API server.
    *   `autonomous.py`: Autonomous agent runner.
    *   `checker.py`: Health and credit checking.
    *   `readonly.py`: Readonly entrypoint.
    *   `scheduler.py`: Background task scheduler.
    *   `singleton.py`: Singleton agent manager.
    *   `telegram.py`: Telegram integration.
    *   `twitter.py`: Twitter integration.
*   `docs/`: Project documentation.
*   `scripts/`: Operation and management scripts.

## Agent API

Interact with your agents programmatically using the comprehensive REST API.

*   **[Agent API Documentation](docs/agent_api.md)**

## Development

Get started by reading the [Development Guide](DEVELOPMENT.md) for setup instructions.

## Documentation

Explore the full documentation before you start.

*   **[Documentation](docs/)**

## Contributing

We welcome contributions!

*   Read our [Contributing Guidelines](CONTRIBUTING.md)
*   Check the [Wishlist](docs/contributing/wishlist.md) for feature requests.
*   See our [Skill Development Guide](docs/contributing/skills.md) for information on contributing skills.

### Developer Chat

Join our community on [Discord](https://discord.com/invite/crestal). Open a support ticket to apply for an intentkit dev role.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
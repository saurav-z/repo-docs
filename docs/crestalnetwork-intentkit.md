<!-- Improved README for IntentKit -->
# IntentKit: Build Autonomous AI Agents for Web3 and Beyond

<div align="center">
  <img src="docs/images/intentkit_banner.png" alt="IntentKit by Crestal" width="100%" />
</div>
<br>

**IntentKit empowers developers to create and manage powerful AI agents capable of interacting with blockchain, social media, and more.**

Explore the original repository: [https://github.com/crestalnetwork/intentkit](https://github.com/crestalnetwork/intentkit)

## Key Features

*   **🤖 Multi-Agent Support:** Manage multiple AI agents concurrently.
*   **🔄 Autonomous Agent Management:** Orchestrate and oversee agent operations.
*   **🔗 Blockchain Integration:** Interact with EVM-compatible blockchains.
*   **🐦 Social Media Integration:** Connect with platforms like Twitter and Telegram.
*   **🛠️ Extensible Skill System:** Customize agent capabilities with a modular skill architecture.
*   **🔌 MCP (WIP):** (Mention what this is when you know)

## Architecture Overview

IntentKit's architecture facilitates flexible and powerful agent development, integrating diverse functionalities.

[Image of the architecture diagram from original README, or a summarized version]

For more detailed information, see the [Architecture](docs/architecture.md) section in the original repo.

## Development & Getting Started

### Package Manager Migration
*   **Important:** To ensure proper operation, delete the `.venv` folder and run `uv sync`.
```bash
rm -rf .venv
uv sync
```

### Development Guide
Refer to the [Development Guide](DEVELOPMENT.md) to start your development environment.

### Documentation
Check out [Documentation](docs/) before you start.

## Project Structure

IntentKit is structured into a core package and an application:

*   **`intentkit/`**: The IntentKit package (installable via pip)
    *   `abstracts/`: Core and skills abstract classes and interfaces
    *   `clients/`: Clients for external services
    *   `config/`: System-level configurations
    *   `core/`: The central agent system, powered by LangGraph
    *   `models/`: Data models using Pydantic and SQLAlchemy
    *   `skills/`: Extendable skills system, based on LangChain tools
    *   `utils/`: Utility functions
*   **`app/`**: The IntentKit application (API server, autonomous runner, and scheduler)
    *   `admin/`: Admin APIs and agent generation
    *   `entrypoints/`: Interfaces for agent interaction (web, Telegram, Twitter, etc.)
    *   `services/`: Service implementations for platforms like Telegram and Twitter
    *   `api.py`: REST API server
    *   `autonomous.py`: Autonomous agent runner
    *   `checker.py`: Health and credit checking logic
    *   `readonly.py`: Readonly entrypoint
    *   `scheduler.py`: Background task scheduler
    *   `singleton.py`: Singleton agent manager
    *   `telegram.py`: Telegram integration
    *   `twitter.py`: Twitter integration
*   `docs/`: Documentation
*   `scripts/`: Management and migration scripts

## Agent API

IntentKit offers a comprehensive REST API for easy interaction with your AI agents, enabling integration and custom application development.

**Get Started:** [Agent API Documentation](docs/agent_api.md)

## Contributing

Contributions are welcome!

*   Read our [Contributing Guidelines](CONTRIBUTING.md).
*   Check the [Wishlist](docs/contributing/wishlist.md) for active skill requests.
*   For skill development, see the [Skill Development Guide](docs/contributing/skills.md).

### Developer Community

*   Join our [Discord](https://discord.com/invite/crestal) and apply for an "intentkit dev" role in a support ticket to join the dev channels.

## License

IntentKit is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
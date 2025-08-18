# IntentKit: Build and Manage Autonomous AI Agents

**Unleash the power of AI agents with IntentKit, a flexible framework for creating intelligent, autonomous systems capable of interacting with blockchain, social media, and more.** ([View the original repository](https://github.com/crestalnetwork/intentkit))

<div align="center">
  <img src="docs/images/intentkit_banner.png" alt="IntentKit by Crestal" width="100%" />
</div>
<br>

IntentKit is a powerful framework designed to empower developers to create and manage sophisticated AI agents. This framework allows for extensive customization and integration across various platforms, making it ideal for automating tasks, managing social media, and interacting with blockchain technology.

## Key Features

*   **🤖 Multi-Agent Support:** Manage and deploy multiple autonomous agents simultaneously.
*   **🔄 Autonomous Agent Management:** Effortlessly control and orchestrate your AI agents.
*   **🔗 Blockchain Integration:** Interact with EVM-compatible blockchains and manage wallets.
*   **🐦 Social Media Integration:** Connect with platforms like Twitter and Telegram.
*   **🛠️ Extensible Skill System:** Easily add custom skills and functionality to your agents.
*   **🔌 MCP (WIP):** (More details to come)

## Architecture

IntentKit is built on a modular architecture, allowing for easy expansion and customization. The system includes:

*   **Entrypoints:** Interfaces for interacting with agents (e.g., Twitter, Telegram).
*   **Agent Core:** The central component powered by LangGraph.
*   **Skills:** Including blockchain interaction, social media management, and more.
*   **Storage:** Managing agent configurations, credentials, personality and memory.

For a detailed overview of the architecture, please refer to the [Architecture](docs/architecture.md) documentation.

## Development

Get started with IntentKit development by following the instructions in the [Development Guide](DEVELOPMENT.md).

## Documentation

Comprehensive documentation is available to guide you through the framework. Explore the [Documentation](docs/) for in-depth information.

## Project Structure

The IntentKit project is organized into two main components:

*   **[intentkit/](intentkit/)**: The core Python package.
    *   [abstracts/](intentkit/abstracts/): Abstract classes and interfaces
    *   [clients/](intentkit/clients/): Clients for external services
    *   [config/](intentkit/config/): System level configurations
    *   [core/](intentkit/core/): Core agent system
    *   [models/](intentkit/models/): Entity models
    *   [skills/](intentkit/skills/): Extensible skills system
    *   [utils/](intentkit/utils/): Utility functions

*   **[app/](app/)**: The application layer (API server, autonomous runner, scheduler).
    *   [admin/](app/admin/): Admin APIs, agent generators
    *   [entrypoints/](app/entrypoints/): Agent interaction entrypoints
    *   [services/](app/services/): Service implementations (Telegram, Twitter, etc.)
    *   [api.py](app/api.py): REST API server
    *   [autonomous.py](app/autonomous.py): Autonomous agent runner
    *   [checker.py](app/checker.py): Health and credit checking logic
    *   [readonly.py](app/readonly.py): Readonly entrypoint
    *   [scheduler.py](app/scheduler.py): Background task scheduler
    *   [singleton.py](app/singleton.py): Singleton agent manager
    *   [telegram.py](app/telegram.py): Telegram integration
    *   [twitter.py](app/twitter.py): Twitter integration

*   [docs/](docs/): Documentation
*   [scripts/](scripts/): Management and migration scripts

## Agent API

Integrate with your agents programmatically using the IntentKit REST API. Create custom interfaces and build powerful applications.

**Get Started:** [Agent API Documentation](docs/agent_api.md)

## Contributing

Contribute to the IntentKit project and help build a robust AI agent framework.

*   Read the [Contributing Guidelines](CONTRIBUTING.md) before submitting pull requests.

### Contribute Skills

*   Check the [Wishlist](docs/contributing/wishlist.md) for active skill requests.
*   Follow the [Skill Development Guide](docs/contributing/skills.md) for building new skills.

### Developer Chat

Join the IntentKit community on [Discord](https://discord.com/invite/crestal) and connect with other developers. Request an intentkit dev role in the support ticket system.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Package Manager Migration Warning

If you are migrating to uv from poetry, make sure to:
*   Delete the `.venv` folder
*   Run `uv sync` to create a new virtual environment.
```bash
rm -rf .venv
uv sync
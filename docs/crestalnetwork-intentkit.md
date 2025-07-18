# IntentKit: Build Autonomous AI Agents with Ease

**IntentKit** is an innovative framework enabling developers to effortlessly create and manage powerful, autonomous AI agents capable of interacting with blockchains, social media, and custom skills. ([View on GitHub](https://github.com/crestalnetwork/intentkit))

<div align="center">
  <img src="docs/images/intentkit_banner.png" alt="IntentKit by Crestal" width="100%" />
</div>
<br>

## Key Features

*   🤖 **Multi-Agent Support:** Manage and deploy multiple AI agents simultaneously.
*   🔄 **Autonomous Agent Management:** Automate agent workflows and decision-making.
*   🔗 **Blockchain Integration:** Interact with EVM-compatible blockchains (with plans for more).
*   🐦 **Social Media Integration:** Connect with platforms like Twitter and Telegram.
*   🛠️ **Extensible Skill System:** Easily integrate custom functionalities and expand agent capabilities.
*   🔌 **MCP (WIP):** (Placeholder for future functionality)

## Architecture Overview

IntentKit's architecture is designed for flexibility and extensibility. Agents are built using LangGraph and are connected to various entrypoints (e.g., Twitter, Telegram). They leverage a core system incorporating:

*   **Agent Configuration & Memory:**  Stores agent profiles and data.
*   **Skills:** Includes capabilities for blockchain interaction, wallet management, on-chain actions, internet search, and image processing.
*   **Entrypoints:** Entrypoints for interacting with agents (web, Telegram, Twitter, etc.)

For more detailed information, please refer to the [Architecture](docs/architecture.md) section.

## Development and Setup

### Package Manager Migration Warning

**Important:** The project has migrated to `uv` from `poetry`.
   1. Delete the old virtual environment: `rm -rf .venv`
   2. Create a new virtual environment: `uv sync`

### Getting Started

*   **Development Guide:** Consult the [Development Guide](DEVELOPMENT.md) for setup instructions.
*   **Documentation:** Explore the comprehensive [Documentation](docs/) to get started.

## Project Structure

The project is structured into two main parts:

*   **[intentkit/](intentkit/)**: The core Python package, published to PyPI
    *   [abstracts/](intentkit/abstracts/)
    *   [clients/](intentkit/clients/)
    *   [config/](intentkit/config/)
    *   [core/](intentkit/core/)
    *   [models/](intentkit/models/)
    *   [skills/](intentkit/skills/)
    *   [utils/](intentkit/utils/)
*   **[app/](app/)**: The application, containing API server, autonomous runner, and background scheduler.
    *   [admin/](app/admin/)
    *   [entrypoints/](app/entrypoints/)
    *   [services/](app/services/)
    *   [api.py](app/api.py)
    *   [autonomous.py](app/autonomous.py)
    *   [checker.py](app/checker.py)
    *   [readonly.py](app/readonly.py)
    *   [scheduler.py](app/scheduler.py)
    *   [singleton.py](app/singleton.py)
    *   [telegram.py](app/telegram.py)
    *   [twitter.py](app/twitter.py)
*   [docs/](docs/): Documentation
*   [scripts/](scripts/): Operation and temporary scripts

## Agent API

IntentKit offers a robust REST API for programmatic access to your AI agents. Build custom integrations and interfaces.

*   **API Documentation:** Explore the [Agent API Documentation](docs/agent_api.md).

## Contribute

We welcome contributions!

*   **Contributing Guidelines:** Review our [Contributing Guidelines](CONTRIBUTING.md).
*   **Contribute Skills:** Check the [Wishlist](docs/contributing/wishlist.md) for feature requests, then see the [Skill Development Guide](docs/contributing/skills.md).
*   **Developer Chat:** Join our [Discord](https://discord.com/invite/crestal) and request an "intentkit dev" role.
## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
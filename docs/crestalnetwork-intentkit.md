# IntentKit: Build Autonomous AI Agents with Ease

Create and manage powerful AI agents for blockchain interaction, social media, and custom tasks with **IntentKit**, a cutting-edge autonomous agent framework.  [Learn more at the original repository](https://github.com/crestalnetwork/intentkit).

<div align="center">
  <img src="docs/images/intentkit_banner.png" alt="IntentKit by Crestal" width="100%" />
</div>
<br>

## Key Features

*   **Multi-Agent Support:** Manage and deploy multiple AI agents within a single framework.
*   **Autonomous Agent Management:** Oversee the lifecycle and operations of your AI agents.
*   **Blockchain Integration:** Interact with EVM-compatible blockchains to perform actions and retrieve data.
*   **Social Media Integration:** Seamlessly integrate with platforms like Twitter and Telegram.
*   **Extensible Skill System:** Customize agent capabilities by adding new skills and functionalities.
*   **[WIP] MCP:** Coming soon!

## Architecture Overview

IntentKit's architecture is designed for flexibility and extensibility:

*   **Entrypoints:** Interact with agents through various entrypoints such as Twitter, Telegram, and more.
*   **Core Agent:** Powered by LangGraph, the central hub for agent logic.
*   **Skills:** Enable diverse capabilities including chain integration, wallet management, on-chain actions, internet search, and image processing.
*   **Data Storage:** Agent config, credentials, personality, memory, skill states.

For a detailed understanding of the architecture, please refer to the [Architecture](docs/architecture.md) section.

## Getting Started

### Package Manager Migration Warning

We just migrated to uv from poetry.
You need to delete the .venv folder and run `uv sync` to create a new virtual environment. (one time)
```bash
rm -rf .venv
uv sync
```

### Development

*   **Development Guide:** Learn how to set up your development environment by reading the [Development Guide](DEVELOPMENT.md).
*   **Documentation:** Explore the comprehensive [Documentation](docs/) to get started.

## Project Structure

The project is organized into core components:

*   **[intentkit/](intentkit/)**: Core package.
    *   [abstracts/](intentkit/abstracts/): Abstract classes and interfaces.
    *   [clients/](intentkit/clients/): Clients for external services.
    *   [config/](intentkit/config/): System configurations.
    *   [core/](intentkit/core/): Core agent system.
    *   [models/](intentkit/models/): Entity models.
    *   [skills/](intentkit/skills/): Extensible skills system.
    *   [utils/](intentkit/utils/): Utility functions.
*   **[app/](app/)**: The IntentKit application.
    *   [admin/](app/admin/): Admin APIs and agent generation.
    *   [entrypoints/](app/entrypoints/): Agent interaction entrypoints.
    *   [services/](app/services/): Service implementations.
    *   [api.py](app/api.py): REST API server.
    *   [autonomous.py](app/autonomous.py): Autonomous agent runner.
    *   [checker.py](app/checker.py): Health and credit checking logic.
    *   [readonly.py](app/readonly.py): Readonly entrypoint.
    *   [scheduler.py](app/scheduler.py): Background task scheduler.
    *   [singleton.py](app/singleton.py): Singleton agent manager.
    *   [telegram.py](app/telegram.py): Telegram integration.
    *   [twitter.py](app/twitter.py): Twitter integration.
*   [docs/](docs/): Documentation
*   [scripts/](scripts/): Scripts for management and migrations

## Agent API

IntentKit provides a robust REST API for programmatic interaction with your agents.

*   **Agent API Documentation:** Consult the [Agent API Documentation](docs/agent_api.md) to get started.

## Contribute

We welcome contributions to IntentKit!

*   **Contributing Guidelines:** Review our [Contributing Guidelines](CONTRIBUTING.md) for details.
*   **Contribute Skills:** Check the [Wishlist](docs/contributing/wishlist.md) for feature requests and see the [Skill Development Guide](docs/contributing/skills.md) to get started.

### Developer Community

*   **Discord:** Join our [Discord](https://discord.com/invite/crestal) and apply for an intentkit dev role.
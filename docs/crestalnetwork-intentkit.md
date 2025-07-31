# IntentKit: Build Powerful AI Agents for the Modern Web

[![IntentKit by Crestal](docs/images/intentkit_banner.png)](https://github.com/crestalnetwork/intentkit)

**IntentKit empowers you to build and manage intelligent AI agents capable of interacting with blockchains, social media, and custom integrations.**

## Key Features

*   🤖 **Multi-Agent Support:** Easily create and manage multiple AI agents.
*   🔄 **Autonomous Agent Management:** Orchestrate and control agent behaviors.
*   🔗 **Blockchain Integration:** Interact with EVM-compatible blockchains.
*   🐦 **Social Media Integration:** Manage Twitter, Telegram, and more.
*   🛠️ **Extensible Skill System:** Customize agents with a wide range of skills.
*   🔌 **MCP (WIP):** (WIP - More details in the future.)

## Architecture Overview

IntentKit leverages LangGraph to create a flexible and scalable agent framework. Agents can interact with various entry points (Twitter, Telegram, etc.), utilize storage for configurations and data (Agent Config, Credentials, Personality, Memory, Skill State), and access a diverse set of skills, including:

*   Chain Integration (Wallet Management, On-Chain Actions)
*   Internet Search
*   Image Processing

**For a deeper understanding of the architecture, please refer to the [Architecture Documentation](docs/architecture.md).**

## Package Manager Migration Warning

We've recently migrated from Poetry to `uv`. To ensure a smooth setup, please remove your existing virtual environment and run `uv sync`:

```bash
rm -rf .venv
uv sync
```

## Project Structure

The project is structured into the core package and the application:

*   **[intentkit/](intentkit/)**: The IntentKit package (installable via pip)
    *   [abstracts/](intentkit/abstracts/): Abstract classes and interfaces
    *   [clients/](intentkit/clients/): Clients for external services
    *   [config/](intentkit/config/): System configurations
    *   [core/](intentkit/core/): Core agent system, powered by LangGraph
    *   [models/](intentkit/models/): Entity models using Pydantic and SQLAlchemy
    *   [skills/](intentkit/skills/): Extensible skills system based on LangChain tools
    *   [utils/](intentkit/utils/): Utility functions
*   **[app/](app/)**: The IntentKit application (API server, autonomous runner, and background scheduler)
    *   [admin/](app/admin/): Admin APIs and agent generators
    *   [entrypoints/](app/entrypoints/): Agent interaction entrypoints (web, Telegram, Twitter, etc.)
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
*   [scripts/](scripts/): Operation and temporary scripts

## Agent API

IntentKit provides a REST API for programmatic control over your agents.

**Access the Agent API Documentation:** [Agent API Documentation](docs/agent_api.md)

## Development

Get started with development by reading the [Development Guide](DEVELOPMENT.md).

## Documentation

Explore the full documentation before you start: [Documentation](docs/)

## Contribute

We welcome contributions!  Please review the following resources:

*   **Contributing Guidelines:** [Contributing Guidelines](CONTRIBUTING.md)
*   **Contribute Skills:** Review the [Wishlist](docs/contributing/wishlist.md) and then the [Skill Development Guide](docs/contributing/skills.md).

### Developer Chat

Join our [Discord](https://discord.com/invite/crestal) and apply for an intentkit dev role.  You can then join the developer discussion channel.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**[Visit the original repository on GitHub](https://github.com/crestalnetwork/intentkit)**
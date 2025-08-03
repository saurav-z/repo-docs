# IntentKit: Build Autonomous AI Agents with Ease

IntentKit is an open-source framework empowering developers to create and manage sophisticated AI agents capable of blockchain interactions, social media management, and much more. [Explore the original repository](https://github.com/crestalnetwork/intentkit).

<div align="center">
  <img src="docs/images/intentkit_banner.png" alt="IntentKit by Crestal" width="100%" />
</div>
<br>

## Key Features

*   **Multi-Agent Support:** Manage and deploy multiple AI agents within a single framework.
*   **Autonomous Agent Management:** Oversee the lifecycle and operations of your AI agents with ease.
*   **Blockchain Integration:** Interact with EVM-compatible blockchains to execute transactions and retrieve data.
*   **Social Media Integration:** Connect your agents to platforms like Twitter and Telegram for automated content management and interaction.
*   **Extensible Skill System:** Customize your agents with a wide range of skills and capabilities.
*   **[WIP] MCP (Multi-Chain Protocol):** Expanding capabilities to support multiple blockchains.

## Architecture Overview

IntentKit's architecture is designed for flexibility and extensibility. Agents leverage LangGraph to orchestrate tasks, drawing upon various components for functionality:

*   **Entrypoints:** (Twitter/Telegram & more) - Interfaces for agent interaction.
*   **Agent Configuration:** Defines agent behavior and capabilities.
*   **Skills:** Chain Integration, Wallet Management, On-Chain Actions, Internet Search, Image Processing, etc.
*   **The Agent:** The central processing unit, powered by LangGraph.
*   **Storage:** Agent Config, Credentials, Personality, Memory, Skill State
*   **Agent Config & Memory:** Configure and store agent data.

For a detailed understanding, please see the [Architecture](docs/architecture.md) section within the documentation.

## Getting Started

### Package Manager Migration Warning

We have migrated to `uv` from `poetry`. You will need to delete your `.venv` and run `uv sync` to update your environment.

```bash
rm -rf .venv
uv sync
```

### Development

*   **Setup:** Follow the [Development Guide](DEVELOPMENT.md) to configure your development environment.
*   **Documentation:** Consult the comprehensive [Documentation](docs/) for detailed information.

## Project Structure

*   **[intentkit/](intentkit/)**: Core IntentKit package
    *   [abstracts/](intentkit/abstracts/): Abstract classes and interfaces.
    *   [clients/](intentkit/clients/): Clients for external services.
    *   [config/](intentkit/config/): System configurations.
    *   [core/](intentkit/core/): Core agent system.
    *   [models/](intentkit/models/): Data models.
    *   [skills/](intentkit/skills/): Extensible skills system.
    *   [utils/](intentkit/utils/): Utility functions.
*   **[app/](app/)**: IntentKit application
    *   [admin/](app/admin/): Admin APIs and agent management.
    *   [entrypoints/](app/entrypoints/): Agent interaction entrypoints.
    *   [services/](app/services/): Service implementations.
    *   [api.py](app/api.py): REST API server.
    *   [autonomous.py](app/autonomous.py): Autonomous agent runner.
    *   [checker.py](app/checker.py): Health and credit checks.
    *   [readonly.py](app/readonly.py): Readonly entrypoint.
    *   [scheduler.py](app/scheduler.py): Background task scheduler.
    *   [singleton.py](app/singleton.py): Singleton agent manager.
    *   [telegram.py](app/telegram.py): Telegram integration.
    *   [twitter.py](app/twitter.py): Twitter integration.
*   [docs/](docs/): Documentation
*   [scripts/](scripts/): Utility scripts

## Agent API

IntentKit provides a comprehensive REST API.

*   **API Documentation:** [Agent API Documentation](docs/agent_api.md)

## Contributing

We welcome contributions!

*   **Contribution Guidelines:** Review the [Contributing Guidelines](CONTRIBUTING.md).
*   **Contribute Skills:** Check the [Wishlist](docs/contributing/wishlist.md) for skill requests and follow the [Skill Development Guide](docs/contributing/skills.md).

### Developer Community

*   **Discord:** Join the [Discord](https://discord.com/invite/crestal) and request an intentkit dev role.

## License

This project is licensed under the [MIT License](LICENSE).
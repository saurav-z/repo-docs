# IntentKit: Build Autonomous AI Agents with Ease

**IntentKit** empowers you to create and manage powerful AI agents for blockchain interaction, social media management, and more. ([See the original repository](https://github.com/crestalnetwork/intentkit))

## Key Features:

*   🤖 **Multiple Agent Support:** Manage and orchestrate multiple autonomous agents.
*   🔄 **Autonomous Agent Management:** Streamline the lifecycle of your AI agents.
*   🔗 **Blockchain Integration:** Interact with EVM-compatible blockchains.
*   🐦 **Social Media Integration:** Connect with platforms like Twitter and Telegram.
*   🛠️ **Extensible Skill System:** Customize agents with a flexible skill architecture.
*   🔌 **MCP (WIP):** (Mention of MCP - Mobile Control Panel)

## Architecture Overview

[Image of the IntentKit architecture - removed for brevity, but keep in mind the image should still be here.]

IntentKit's architecture is designed to be modular and extensible. At its core, the system is powered by LangGraph. Agents interact with various entrypoints (Twitter, Telegram, etc.) and leverage a range of skills, including:

*   Chain Integration
*   Wallet Management
*   On-Chain Actions
*   Internet Search
*   Image Processing
*   And more...

Detailed architectural information can be found in the [Architecture](docs/architecture.md) section of the documentation.

## Package Manager Migration Warning

We've recently migrated to `uv` from `poetry`. To update your environment:

```bash
rm -rf .venv
uv sync
```

## Development

Get started with IntentKit development by following the instructions in the [Development Guide](DEVELOPMENT.md).

## Documentation

Explore the full capabilities of IntentKit. Start with the [Documentation](docs/) to understand the platform.

## Project Structure

The project is structured into a core package and an application layer:

*   **`intentkit/`:** The IntentKit package (pip installable).
    *   `abstracts/`: Abstract classes and interfaces.
    *   `clients/`: Clients for external services.
    *   `config/`: System-level configurations.
    *   `core/`: Core agent system (LangGraph-driven).
    *   `models/`: Data models (Pydantic and SQLAlchemy).
    *   `skills/`: Extensible skills system.
    *   `utils/`: Utility functions.
*   **`app/`:** The IntentKit application (API server, autonomous runner, etc.).
    *   `admin/`: Admin APIs and agent generation.
    *   `entrypoints/`: Entrypoints for agent interaction.
    *   `services/`: Service implementations (Telegram, Twitter).
    *   `api.py`: REST API server.
    *   `autonomous.py`: Autonomous agent runner.
    *   `checker.py`: Health and credit checking.
    *   `readonly.py`: Read-only entrypoint.
    *   `scheduler.py`: Background task scheduler.
    *   `singleton.py`: Singleton agent manager.
    *   `telegram.py`: Telegram integration.
    *   `twitter.py`: Twitter integration.
*   `docs/`: Documentation
*   `scripts/`: Operation and utility scripts.

## Agent API

Integrate with your agents programmatically using the IntentKit REST API.

**Get Started:** [Agent API Documentation](docs/agent_api.md)

## Contributing

We welcome contributions! Please review the [Contributing Guidelines](CONTRIBUTING.md) before submitting pull requests.

### Contribute Skills

Check the [Wishlist](docs/contributing/wishlist.md) for requests.

See the [Skill Development Guide](docs/contributing/skills.md) for how to contribute.

### Developer Chat

Join our community on [Discord](https://discord.com/invite/crestal) and apply for an intentkit dev role for access to the developer discussion channel.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
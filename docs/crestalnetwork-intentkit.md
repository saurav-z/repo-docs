# IntentKit: Build Autonomous AI Agents with Ease

**Unlock the power of AI agents to automate tasks and interact with the world with IntentKit, a versatile framework for building and managing intelligent agents.** ([Original Repository](https://github.com/crestalnetwork/intentkit))

<div align="center">
  <img src="docs/images/intentkit_banner.png" alt="IntentKit by Crestal" width="100%" />
</div>
<br>

IntentKit empowers developers to create sophisticated AI agents capable of interacting with blockchains, social media, and other external services.

## Key Features of IntentKit

*   🤖 **Multi-Agent Support:** Manage and orchestrate multiple AI agents.
*   🔄 **Autonomous Agent Management:** Control and monitor your agents' behavior.
*   🔗 **Blockchain Integration:** Interact with EVM-compatible blockchains.
*   🐦 **Social Media Integration:** Connect with platforms like Twitter and Telegram.
*   🛠️ **Extensible Skill System:** Customize your agents with a wide range of skills.
*   🔌 **MCP (WIP):** (Mentioned, but needs more detail in the original doc)

## Architecture Overview

IntentKit's architecture is designed for flexibility and extensibility.  Agents interact with entrypoints (like Twitter/Telegram), utilize storage for agent configurations, credentials, personality, and memory, and leverage skills to perform actions. These skills include blockchain interaction, wallet management, on-chain actions, internet search, and image processing.

A more detailed explanation can be found in the [Architecture](docs/architecture.md) section.

## Package Manager Migration Warning

**Important:** We've moved to `uv` from `poetry`. Follow these steps to update your environment:

```bash
rm -rf .venv
uv sync
```

## Development

Get started with IntentKit development by following the instructions in the [Development Guide](DEVELOPMENT.md).

## Documentation & Resources

*   **Comprehensive Documentation:** Explore the full capabilities of IntentKit in the [Documentation](docs/).
*   **Agent API:** Use the comprehensive REST API for programmatic access to your agents.
    *   [Agent API Documentation](docs/agent_api.md)

## Project Structure

The project is divided into two main parts:

*   **`intentkit/` (Python Package):**  The core IntentKit library.  Contains:
    *   `abstracts/`: Abstract classes and interfaces.
    *   `clients/`: Clients for external services.
    *   `config/`: System configurations.
    *   `core/`: Core agent system (powered by LangGraph).
    *   `models/`: Data models (Pydantic & SQLAlchemy).
    *   `skills/`: Extensible skills.
    *   `utils/`: Utility functions.

*   **`app/` (Application):**  The IntentKit application.  Includes:
    *   `admin/`: Admin APIs and agent generators.
    *   `entrypoints/`: Agent entrypoints (web, Telegram, Twitter, etc.).
    *   `services/`: Service implementations (Telegram, Twitter, etc.).
    *   `api.py`: REST API server.
    *   `autonomous.py`: Autonomous agent runner.
    *   `checker.py`: Health/credit checking.
    *   `readonly.py`: Readonly entrypoint.
    *   `scheduler.py`: Background task scheduler.
    *   `singleton.py`: Singleton agent manager.
    *   `telegram.py`: Telegram integration.
    *   `twitter.py`: Twitter integration.

*   `docs/`: Documentation
*   `scripts/`:  Scripts for management and migrations.

## Contributing

We welcome contributions! Please review the [Contributing Guidelines](CONTRIBUTING.md) before submitting a pull request.

### Contribute Skills

1.  Check the [Wishlist](docs/contributing/wishlist.md) for existing skill requests.
2.  Follow the [Skill Development Guide](docs/contributing/skills.md) to build and integrate your skills.

### Developer Community

Join the Crestal community on [Discord](https://discord.com/invite/crestal) and request an IntentKit dev role to participate in discussions and get support.
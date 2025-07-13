# IntentKit: Build Autonomous AI Agents with Ease

**IntentKit** is an open-source framework that empowers you to create, manage, and deploy sophisticated AI agents capable of interacting with blockchains, social media, and more. [View the original repository on GitHub](https://github.com/crestalnetwork/intentkit).

<div align="center">
  <img src="docs/images/intentkit_banner.png" alt="IntentKit by Crestal" width="100%" />
</div>
<br>

## Key Features of IntentKit

*   🤖 **Multi-Agent Support:** Manage and orchestrate multiple autonomous AI agents.
*   🔄 **Autonomous Agent Management:**  Simplified lifecycle management and control over your agents.
*   🔗 **Blockchain Integration:** Seamlessly interact with EVM-compatible blockchains (and more to come).
*   🐦 **Social Media Integration:**  Connect with platforms like Twitter and Telegram to interact with your audience.
*   🛠️ **Extensible Skill System:** Easily add new capabilities and customize agent behavior.
*   🔌 **MCP (WIP):** (Work In Progress)

## Architecture

IntentKit's architecture is designed for flexibility and extensibility. It allows agents to interact with various entrypoints (social media, etc.), leverage a robust skill system, and utilize storage for configuration, credentials, and memory management, all powered by LangGraph.

*   See [Architecture](docs/architecture.md) for more details.

## Development

### Package Manager Migration Warning

**Important:** We've switched to `uv` from poetry.

1.  Delete the existing virtual environment: `rm -rf .venv`
2.  Create a new virtual environment: `uv sync`

### Getting Started

*   Read the [Development Guide](DEVELOPMENT.md) to set up your environment.
*   Explore the [Documentation](docs/) to learn how to use IntentKit.

## Project Structure

*   **[abstracts/](intentkit/abstracts/)**: Abstract classes and interfaces
*   **[app/](app/)**: Core application code
    *   [core/](intentkit/core/): Core modules
    *   [services/](app/services/): Services
    *   [entrypoints/](app/entrypoints/): Entrypoints
    *   [admin/](app/admin/): Admin logic
    *   [config/](intentkit/config/): Configurations
    *   [api.py](app/api.py): REST API server
    *   [autonomous.py](app/autonomous.py): Autonomous agent scheduler
    *   [singleton.py](app/singleton.py): Singleton agent scheduler
    *   [scheduler.py](app/scheduler.py): Scheduler for periodic tasks
    *   [readonly.py](app/readonly.py): Readonly entrypoint
    *   [telegram.py](app/telegram.py): Telegram listener
*   **[clients/](intentkit/clients/)**: Clients for external services
*   **[docs/](docs/)**: Documentation
*   **[models/](intentkit/models/)**: Database models
*   **[scripts/](scripts/)**: Scripts for agent management
*   **[skills/](intentkit/skills/)**: Skill implementations
*   **[utils/](intentkit/utils/)**: Utility functions

## Contributing

We welcome contributions!  Please review the following:

*   Read our [Contributing Guidelines](CONTRIBUTING.md).
*   Check the [Wishlist](docs/contributing/wishlist.md) for active skill requests.
*   See the [Skill Development Guide](docs/contributing/skills.md) for creating new skills.

### Developer Community

*   Join our [Discord](https://discord.com/invite/crestal) and request an IntentKit dev role.
*   Use the discussion channel for collaboration.

## License

IntentKit is licensed under the [MIT License](LICENSE).
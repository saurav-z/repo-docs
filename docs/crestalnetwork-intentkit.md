# IntentKit: Build Intelligent Autonomous Agents with Ease

[![IntentKit by Crestal](docs/images/intentkit_banner.png)](https://github.com/crestalnetwork/intentkit)

**IntentKit empowers you to create and manage AI agents capable of complex tasks, including blockchain interactions and social media engagement.**

## Key Features

*   🤖 **Multi-Agent Support:** Manage and deploy multiple autonomous agents.
*   🔄 **Autonomous Agent Management:** Orchestrate and oversee agent operations.
*   🔗 **Blockchain Integration:** Interact with EVM-compatible blockchains.
*   🐦 **Social Media Integration:** Connect agents with platforms like Twitter and Telegram.
*   🛠️ **Extensible Skill System:** Customize agent capabilities with a flexible skill architecture.
*   🔌 **MCP (WIP):** (Mention of the work in progress feature.)

## Architecture Overview

IntentKit's architecture enables agents to interact with various entrypoints (Twitter, Telegram, etc.) and leverage skills like blockchain integration, internet search, and image processing. Agents are powered by LangGraph and utilize storage for configurations, credentials, personality, memory, and skill states.

[View detailed architecture in the documentation](docs/architecture.md)

## Package Manager Migration Warning

*   **Important:** You need to delete the `.venv` folder and run `uv sync` to create a new virtual environment. (one time)
    ```bash
    rm -rf .venv
    uv sync
    ```

## Project Structure

*   **[intentkit/](intentkit/)**: The IntentKit package (published as a pip package)
    *   [abstracts/](intentkit/abstracts/): Abstract classes and interfaces for core and skills
    *   [clients/](intentkit/clients/): Clients for external services
    *   [config/](intentkit/config/): System level configurations
    *   [core/](intentkit/core/): Core agent system, driven by LangGraph
    *   [models/](intentkit/models/): Entity models using Pydantic and SQLAlchemy
    *   [skills/](intentkit/skills/): Extensible skills system, based on LangChain tools
    *   [utils/](intentkit/utils/): Utility functions

*   **[app/](app/)**: The IntentKit app (API server, autonomous runner, and background scheduler)
    *   [admin/](app/admin/): Admin APIs, agent generators, and related functionality
    *   [entrypoints/](app/entrypoints/): Entrypoints for interacting with agents (web, Telegram, Twitter, etc.)
    *   [services/](app/services/): Service implementations for Telegram, Twitter, etc.
    *   [api.py](app/api.py): REST API server
    *   [autonomous.py](app/autonomous.py): Autonomous agent runner
    *   [checker.py](app/checker.py): Health and credit checking logic
    *   [readonly.py](app/readonly.py): Readonly entrypoint
    *   [scheduler.py](app/scheduler.py): Background task scheduler
    *   [singleton.py](app/singleton.py): Singleton agent manager
    *   [telegram.py](app/telegram.py): Telegram integration
    *   [twitter.py](app/twitter.py): Twitter integration

*   [docs/](docs/): Documentation
*   [scripts/](scripts/): Operation and temporary scripts for management and migrations

## Agent API

Access and control your agents programmatically using the comprehensive REST API.

**[Explore the Agent API Documentation](docs/agent_api.md)**

## Getting Started

*   **[Development Guide](DEVELOPMENT.md):** Learn how to set up your development environment.
*   **[Documentation](docs/):** Dive into the comprehensive documentation.

## Contribute

Your contributions are highly valued!

*   **[Contributing Guidelines](CONTRIBUTING.md):** Learn how to submit your pull requests.
*   **[Skill Development Guide](docs/contributing/skills.md):** Build and contribute new skills.
*   **[Wishlist](docs/contributing/wishlist.md):** Check out active requests.

## Developer Community

*   **[Discord](https://discord.com/invite/crestal):** Join the Crestal Discord server and request an intentkit dev role for developer discussions.

## License

This project is licensed under the [MIT License](LICENSE).

**[View the Original Repository](https://github.com/crestalnetwork/intentkit)**
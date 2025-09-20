# IntentKit: Build Intelligent AI Agents for the Web3 World

**Empower your projects with autonomous AI agents capable of blockchain interaction, social media management, and custom skill integration.**

[<img src="docs/images/intentkit_banner.png" alt="IntentKit by Crestal" width="100%">](https://github.com/crestalnetwork/intentkit)

IntentKit is a powerful, open-source framework designed for building and managing AI agents. These agents can perform a wide range of tasks, from interacting with EVM-compatible blockchains to managing your social media presence, and integrating with custom-built skills. Leverage IntentKit to automate complex workflows and create intelligent applications.

## Key Features of IntentKit

*   🤖 **Multiple Agent Support:** Manage and deploy multiple AI agents simultaneously.
*   🔄 **Autonomous Agent Management:** Orchestrate and control your agents' behavior.
*   🔗 **Blockchain Integration:** Seamlessly interact with EVM-compatible blockchains.
*   🐦 **Social Media Integration:** Connect and manage your presence on platforms like Twitter and Telegram.
*   🛠️ **Extensible Skill System:** Easily add new skills and functionalities to your agents.
*   🔌 **MCP (WIP):**  [Awaiting clarification/expansion on this feature.]

## Architecture Overview

[Include a simplified version of the provided architecture diagram here, or a text description if a visual is not possible.]

IntentKit utilizes a modular architecture built around a core agent system, powered by LangGraph. This allows agents to process various inputs, utilize different skills, and interact with external services. Detailed architecture information can be found in the [Architecture](docs/architecture.md) section.

## Getting Started

### Package Manager Migration Warning

Due to recent changes, you must delete the `.venv` folder and run `uv sync` to create a new virtual environment. This is a one-time requirement.

```bash
rm -rf .venv
uv sync
```

### Development Setup

Get up and running quickly with the detailed instructions in our [Development Guide](DEVELOPMENT.md).

### Comprehensive Documentation

Explore all the features and functionality of IntentKit with our in-depth [Documentation](docs/).

## Project Structure

Understand the organization of the IntentKit project to easily navigate the codebase:

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

**Interact with your agents programmatically through our comprehensive REST API.** Build custom applications and integrate with existing systems.

**Explore the Agent API:** [Agent API Documentation](docs/agent_api.md)

## Contribute

**We welcome contributions!**  Review our [Contributing Guidelines](CONTRIBUTING.md) to get started.

### Contribute Skills

*   Check the [Wishlist](docs/contributing/wishlist.md) for active requests.
*   Review the [Skill Development Guide](docs/contributing/skills.md) for guidance.

### Developer Community

Join our developer community on [Discord](https://discord.com/invite/crestal) and request the "intentkit dev" role to engage in discussions and support.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
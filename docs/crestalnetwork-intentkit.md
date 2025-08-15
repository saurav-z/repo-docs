# IntentKit: Build Powerful AI Agents with Ease

**Unlock the potential of autonomous AI agents for blockchain, social media, and more with IntentKit!** ([See the original repository](https://github.com/crestalnetwork/intentkit))

<div align="center">
  <img src="docs/images/intentkit_banner.png" alt="IntentKit by Crestal" width="100%" />
</div>
<br>

IntentKit is an open-source framework designed to simplify the creation, management, and deployment of AI agents.  It provides the building blocks for autonomous agents capable of interacting with various services, including blockchain platforms, social media networks, and custom integrations.

## Key Features

*   🤖 **Multi-Agent Support:** Manage multiple AI agents simultaneously.
*   🔄 **Autonomous Agent Management:** Easily control and orchestrate agent behaviors.
*   🔗 **Blockchain Integration:** Interact with EVM-compatible blockchain networks.
*   🐦 **Social Media Integration:** Connect with platforms like Twitter and Telegram.
*   🛠️ **Extensible Skill System:**  Add custom functionalities to your agents with ease.
*   🔌 **MCP (WIP):** (Mention of ongoing development - consider removing if not central to user's needs)

## Architecture

IntentKit employs a modular architecture, enabling flexibility and scalability. The core is powered by LangGraph, allowing for complex agent workflows. Key components include:

*   **Entrypoints:**  Handle external interactions (e.g., Twitter, Telegram)
*   **Skills:** Provide agent capabilities (e.g., chain integration, internet search).
*   **The Agent:** The central processing unit, powered by LangGraph.
*   **Storage:** Manage agent configurations, credentials, and memory.

For a more detailed overview, please refer to the [Architecture](docs/architecture.md) documentation.

## Development & Setup

To get started with IntentKit:

1.  **Package Manager Migration Warning**: You'll need to delete your `.venv` folder and run `uv sync`.
    ```bash
    rm -rf .venv
    uv sync
    ```
2.  Explore the [Development Guide](DEVELOPMENT.md) for setup instructions.
3.  Consult the [Documentation](docs/) for comprehensive guidance.

## Project Structure

IntentKit is organized into two primary components:

*   **[intentkit/](intentkit/)**: The core IntentKit package (published as a pip package).  Key sub-directories:
    *   [abstracts/](intentkit/abstracts/)
    *   [clients/](intentkit/clients/)
    *   [config/](intentkit/config/)
    *   [core/](intentkit/core/)
    *   [models/](intentkit/models/)
    *   [skills/](intentkit/skills/)
    *   [utils/](intentkit/utils/)

*   **[app/](app/)**: The IntentKit application (API server, autonomous runner, and background scheduler). Key sub-directories:
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

## Agent API

IntentKit offers a robust REST API for programmatic interaction with your AI agents.  This allows you to integrate your agents into various applications, create custom interfaces, and automate agent workflows.

*   **Explore the [Agent API Documentation](docs/agent_api.md)**

## Contributing

We welcome contributions!  Please review the [Contributing Guidelines](CONTRIBUTING.md) before submitting a pull request.

### Contribute Skills

*   Check the [Wishlist](docs/contributing/wishlist.md) for active skill requests.
*   Follow the [Skill Development Guide](docs/contributing/skills.md) to get started.

### Developer Community

Join our developer community on [Discord](https://discord.com/invite/crestal) and apply for an intentkit dev role in the support ticket.
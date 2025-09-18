# IntentKit: Build Autonomous AI Agents with Ease

**Unleash the power of autonomous AI agents with IntentKit, a flexible framework for blockchain interaction, social media management, and custom skill integration.**  [Explore the Original Repo](https://github.com/crestalnetwork/intentkit)

<div align="center">
  <img src="docs/images/intentkit_banner.png" alt="IntentKit by Crestal" width="100%" />
</div>
<br>

## Key Features

*   **🤖 Multi-Agent Support:** Manage and orchestrate multiple autonomous agents.
*   **🔄 Autonomous Agent Management:** Easily create, configure, and deploy AI agents.
*   **🔗 Blockchain Integration:** Interact with EVM-compatible blockchains.
*   **🐦 Social Media Integration:** Connect to platforms like Twitter and Telegram.
*   **🛠️ Extensible Skill System:** Customize agent capabilities with a modular skill architecture.
*   **🔌 MCP (WIP):**  [Please provide more detail on this feature]

## Architecture

IntentKit leverages a modular architecture for flexible and scalable agent development.  Agents interact with various entrypoints (Twitter, Telegram, etc.) and utilize a core system powered by LangGraph.  Key components include:

*   **Agent Configuration & Memory:**  Stores agent settings and persistent data.
*   **Skills:**  Modular functionalities, including:
    *   Chain Integration (Wallet Management, On-Chain Actions)
    *   Internet Search
    *   Image Processing
    *   ... and more!

For a more detailed view, see the [Architecture](docs/architecture.md) documentation.

## Getting Started

### Package Manager Migration

**Important:** We've migrated to `uv` from `poetry`. To set up your environment:

```bash
rm -rf .venv
uv sync
```

### Development

*   **Development Guide:**  Follow the [Development Guide](DEVELOPMENT.md) to get started.
*   **Documentation:**  Explore the comprehensive [Documentation](docs/) to understand the framework.

## Project Structure

IntentKit is divided into two main parts:

*   **`intentkit/` (Python Package):** The core framework.
    *   `abstracts/`: Abstract classes and interfaces.
    *   `clients/`: Clients for external services.
    *   `config/`: System configurations.
    *   `core/`: Core agent system.
    *   `models/`: Entity models.
    *   `skills/`: Extensible skills system.
    *   `utils/`: Utility functions.

*   **`app/` (Application):**  The application layer with:
    *   `admin/`: Admin APIs and agent management.
    *   `entrypoints/`: Agent interaction entrypoints (web, Telegram, etc.).
    *   `services/`: Service implementations for integrations.
    *   `api.py`: REST API server.
    *   `autonomous.py`: Autonomous agent runner.
    *   `checker.py`: Health and credit checking logic.
    *   `readonly.py`: Readonly entrypoint
    *   `scheduler.py`: Background task scheduler.
    *   `singleton.py`: Singleton agent manager.
    *   `telegram.py`: Telegram integration.
    *   `twitter.py`: Twitter integration.

*   `docs/`: Documentation
*   `scripts/`: Scripts for management and migrations

## Agent API

Integrate your AI agents with other applications using the comprehensive REST API.

*   **Agent API Documentation:**  [Agent API Documentation](docs/agent_api.md)

## Contribute

We welcome contributions!

*   **Contributing Guidelines:** Review our [Contributing Guidelines](CONTRIBUTING.md).

### Contribute Skills

*   **Wishlist:** Check the [Wishlist](docs/contributing/wishlist.md) for active skill requests.
*   **Skill Development Guide:** See the [Skill Development Guide](docs/contributing/skills.md) for details.

### Developer Community

*   **Discord:** Join our [Discord](https://discord.com/invite/crestal) and apply for an intentkit dev role.

## License

*   **MIT License:** This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
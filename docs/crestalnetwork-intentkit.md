# IntentKit: Build and Manage Autonomous AI Agents

**IntentKit empowers you to create and orchestrate intelligent AI agents, unlocking powerful automation capabilities.**  [View the original repository on GitHub](https://github.com/crestalnetwork/intentkit)

<div align="center">
  <img src="docs/images/intentkit_banner.png" alt="IntentKit by Crestal" width="100%" />
</div>
<br>

## Key Features

*   🤖 **Multi-Agent Support:**  Manage and deploy multiple autonomous AI agents.
*   🔄 **Autonomous Agent Management:**  Control and oversee agent behavior and lifecycles.
*   🔗 **Blockchain Integration:** Seamlessly interact with EVM-compatible blockchains (with broader support planned).
*   🐦 **Social Media Integration:**  Connect with and manage social media platforms like Twitter and Telegram.
*   🛠️ **Extensible Skill System:**  Customize agents with a wide range of skills and capabilities.
*   🔌 **MCP (WIP):** (Details on this feature coming soon.)

## Architecture Overview

IntentKit is designed with a modular architecture, enabling flexibility and scalability.  The core agent system, powered by LangGraph, interacts with various entrypoints (like social media) and leverages a robust set of components:

*   **Entrypoints:** Handles interactions with the outside world (Twitter, Telegram, etc.).
*   **Storage:** Manages agent configurations, credentials, and memory.
*   **Skills:**  Provides agents with capabilities like chain integration, wallet management, on-chain actions, internet search, and image processing.

For a more in-depth understanding, please refer to the [Architecture](docs/architecture.md) documentation.

## Getting Started

### Package Manager Migration Warning

We've recently migrated to `uv` from `poetry`. To set up your environment:

```bash
rm -rf .venv
uv sync
```

### Development

See the [Development Guide](DEVELOPMENT.md) for detailed setup instructions.

### Documentation

Explore the comprehensive [Documentation](docs/) to learn more.

## Project Structure

The project is structured into a core package and the application:

**`intentkit/` (Core Package)**

*   `abstracts/`: Abstract classes and interfaces.
*   `clients/`: Clients for external services.
*   `config/`: System configurations.
*   `core/`: Core agent system.
*   `models/`: Entity models.
*   `skills/`: Extensible skills system.
*   `utils/`: Utility functions.

**`app/` (Application)**

*   `admin/`: Admin APIs and agent generation.
*   `entrypoints/`: Agent interaction entrypoints.
*   `services/`: Service implementations.
*   `api.py`: REST API server.
*   `autonomous.py`: Autonomous agent runner.
*   `checker.py`: Health and credit checking logic.
*   `readonly.py`: Readonly entrypoint.
*   `scheduler.py`: Background task scheduler.
*   `singleton.py`: Singleton agent manager.
*   `telegram.py`: Telegram integration.
*   `twitter.py`: Twitter integration.

Additionally, the project includes:

*   `docs/`: Documentation.
*   `scripts/`: Management and migration scripts.

## Agent API

Access and control your agents programmatically through the IntentKit REST API.

**Explore the Agent API:** [Agent API Documentation](docs/agent_api.md)

## Contributing

We welcome contributions!

*   **Contributing Guidelines:** Read our [Contributing Guidelines](CONTRIBUTING.md)
*   **Contribute Skills:** Check the [Wishlist](docs/contributing/wishlist.md) and then the [Skill Development Guide](docs/contributing/skills.md).

## Developer Community

*   **Join us on Discord:** [Discord](https://discord.com/invite/crestal) and apply for the intentkit dev role in the support ticket.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
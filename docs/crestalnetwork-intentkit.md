# IntentKit: Build Autonomous AI Agents with Ease

[![IntentKit Banner](docs/images/intentkit_banner.png)](https://github.com/crestalnetwork/intentkit)

**IntentKit empowers you to build and manage powerful, autonomous AI agents capable of interacting with blockchains, social media, and custom skills.**

[View the original repository on GitHub](https://github.com/crestalnetwork/intentkit)

## Key Features

*   **🤖 Multiple Agent Support:** Manage and deploy multiple AI agents simultaneously.
*   **🔄 Autonomous Agent Management:**  Control and orchestrate the behavior of your agents.
*   **🔗 Blockchain Integration:** Interact with EVM-compatible blockchains for on-chain actions.
*   **🐦 Social Media Integration:** Connect with and manage your presence on platforms like Twitter and Telegram.
*   **🛠️ Extensible Skill System:**  Customize your agents with a modular system for integrating new capabilities.
*   **🔌 MCP (WIP):** *[Mention the feature if it's still relevant]*

## Architecture

[Diagram of Architecture from Original README]

IntentKit's architecture is designed for flexibility and scalability.  For a deeper dive, explore the [Architecture Documentation](docs/architecture.md).

## Getting Started

### Package Manager Migration Warning

We've recently transitioned to `uv` from `poetry`. To set up your environment:

```bash
rm -rf .venv
uv sync
```

### Development

Follow our [Development Guide](DEVELOPMENT.md) for setup instructions.

### Documentation

Access comprehensive documentation to get you started: [Documentation](docs/)

## Project Structure

IntentKit is structured into two main components: the core package and the application:

**IntentKit Package (`intentkit/`)**

*   `abstracts/`: Abstract classes and interfaces.
*   `clients/`: Clients for external services.
*   `config/`: System-level configurations.
*   `core/`: Core agent system powered by LangGraph.
*   `models/`: Entity models using Pydantic and SQLAlchemy.
*   `skills/`: Extensible skills system based on LangChain tools.
*   `utils/`: Utility functions.

**IntentKit Application (`app/`)**

*   `admin/`: Admin APIs and agent generation tools.
*   `entrypoints/`: Entrypoints for interacting with agents (web, Telegram, Twitter, etc.).
*   `services/`: Service implementations (Telegram, Twitter, etc.).
*   `api.py`: REST API server.
*   `autonomous.py`: Autonomous agent runner.
*   `checker.py`: Health and credit checking logic.
*   `readonly.py`: Readonly entrypoint.
*   `scheduler.py`: Background task scheduler.
*   `singleton.py`: Singleton agent manager.
*   `telegram.py`: Telegram integration.
*   `twitter.py`: Twitter integration.

*   `docs/`: Documentation.
*   `scripts/`: Operation and temporary scripts.

## Agent API

IntentKit offers a robust REST API for interacting with your AI agents programmatically.

*   **Agent API Documentation:** [Agent API Documentation](docs/agent_api.md)

## Contribute

We welcome contributions!

*   **Contributing Guidelines:** Read our [Contributing Guidelines](CONTRIBUTING.md).
*   **Contribute Skills:** Check the [Wishlist](docs/contributing/wishlist.md) for active requests, and then see the [Skill Development Guide](docs/contributing/skills.md).

### Developer Community

*   **Discord:** Join our [Discord](https://discord.com/invite/crestal) and apply for a dev role to join the discussion channel.

## License

This project is licensed under the MIT License.  See the [LICENSE](LICENSE) file for details.
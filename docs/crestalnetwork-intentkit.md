# IntentKit: Build Autonomous AI Agents with Ease

**Unleash the power of AI agents for blockchain, social media, and more with IntentKit, a flexible and extensible framework.**  [Explore the original repository on GitHub](https://github.com/crestalnetwork/intentkit).

## Key Features

*   🤖 **Multiple Agent Support:** Manage and deploy numerous AI agents.
*   🔄 **Autonomous Agent Management:**  Orchestrate and control your agents' actions.
*   🔗 **Blockchain Integration:** Interact with EVM-compatible blockchains.
*   🐦 **Social Media Integration:** Connect with platforms like Twitter and Telegram.
*   🛠️ **Extensible Skill System:**  Customize your agents with a wide range of skills.
*   🔌 **MCP (WIP):** (Brief description or remove if not relevant).

## Architecture Overview

IntentKit employs a modular architecture that enables agents to interact with various external services and internal systems. It leverages LangGraph to power the core agent logic, allowing for complex workflows and decision-making.

**(Simplified Diagram -  Original Diagram from README can be included here)**

**Key Components:**

*   **Entrypoints:** Handle incoming requests from platforms like Twitter, Telegram, and web interfaces.
*   **The Agent:**  The central processing unit, driven by LangGraph, responsible for decision-making and action execution.
*   **Skills:**  Provide specific functionalities, such as blockchain interaction, social media posting, internet searches, and image processing.
*   **Agent Config & Memory:** Stores agent configurations and memory.

*For more detailed information, refer to the [Architecture](docs/architecture.md) section.*

## Development

Get started with your setup by reading the [Development Guide](DEVELOPMENT.md).

## Documentation

Comprehensive documentation is available to help you understand and utilize IntentKit.  Start with the [Documentation](docs/) to explore the framework's capabilities.

## Project Structure

The project is organized into two primary components:

*   **`intentkit/`:** The core IntentKit package (installable via pip) containing:
    *   `abstracts/`: Abstract classes and interfaces.
    *   `clients/`: Clients for external services.
    *   `config/`: System-level configurations.
    *   `core/`: Core agent system (LangGraph).
    *   `models/`: Entity models (Pydantic and SQLAlchemy).
    *   `skills/`: Extensible skill system (LangChain tools).
    *   `utils/`: Utility functions.
*   **`app/`:** The IntentKit application (API server, runner, scheduler):
    *   `admin/`: Admin APIs, agent generators.
    *   `entrypoints/`: Agent interaction entrypoints (web, Telegram, Twitter).
    *   `services/`: Service implementations (Telegram, Twitter).
    *   `api.py`: REST API server.
    *   `autonomous.py`: Autonomous agent runner.
    *   `checker.py`: Health and credit checking logic.
    *   `readonly.py`: Readonly entrypoint.
    *   `scheduler.py`: Background task scheduler.
    *   `singleton.py`: Singleton agent manager.
    *   `telegram.py`: Telegram integration.
    *   `twitter.py`: Twitter integration.
*   `docs/`: Documentation.
*   `scripts/`: Operational and temporary scripts.

## Agent API

IntentKit provides a robust REST API for programmatic agent access. This allows developers to create custom interfaces, integrate with existing systems, and build powerful applications on top of IntentKit.

**Explore the [Agent API Documentation](docs/agent_api.md) to get started.**

## Contributing

We welcome contributions! Please review the [Contributing Guidelines](CONTRIBUTING.md) before submitting a pull request.

### Contribute Skills

1.  Check the [Wishlist](docs/contributing/wishlist.md) for feature requests.
2.  Review the [Skill Development Guide](docs/contributing/skills.md) to get started.

### Developer Chat

Join the community on [Discord](https://discord.com/invite/crestal) and request an IntentKit dev role in the support channel.

## License

This project is licensed under the [MIT License](LICENSE).
# IntentKit: Build Intelligent AI Agents with Ease

[![IntentKit Banner](docs/images/intentkit_banner.png)](https://github.com/crestalnetwork/intentkit)

**IntentKit** is an open-source framework designed to simplify the creation and management of powerful, autonomous AI agents capable of interacting with the real world, from blockchains to social media.

[View the original repository](https://github.com/crestalnetwork/intentkit)

## Key Features

*   🤖 **Multi-Agent Support:** Manage and orchestrate multiple AI agents.
*   🔄 **Autonomous Agent Management:** Automate agent workflows and decision-making.
*   🔗 **Blockchain Integration:** Seamless interaction with EVM-compatible blockchains.
*   🐦 **Social Media Integration:** Connect with Twitter, Telegram, and more.
*   🛠️ **Extensible Skill System:** Easily add custom skills and functionalities.
*   🔌 **MCP (WIP):**  (Mention the purpose of this feature)

## Architecture

IntentKit leverages a modular architecture built upon the LangGraph framework, allowing for flexible agent design and integration.
[See detailed architecture](docs/architecture.md)

## Project Structure

The project is organized into two main components:

*   **intentkit/:** The core package, available as a pip package. Includes:
    *   Abstracts for core and skills.
    *   Clients for external services.
    *   Configuration files.
    *   Core agent system (LangGraph).
    *   Entity models using Pydantic and SQLAlchemy.
    *   Extensible skills based on LangChain tools.
    *   Utility functions.

*   **app/:**  The application layer, including:
    *   Admin APIs and agent generators.
    *   Entrypoints for agent interaction.
    *   Service implementations.
    *   API Server, Autonomous Runner, and background scheduler.

## Agent API

Access and control your agents programmatically with our comprehensive REST API.
[Agent API Documentation](docs/agent_api.md)

## Get Started

*   **Development:** [Development Guide](DEVELOPMENT.md)
*   **Documentation:** [Documentation](docs/)

## Contribute

We welcome contributions!

*   **Contributing Guidelines:** [CONTRIBUTING.md](CONTRIBUTING.md)
*   **Contribute Skills:**  Refer to the [Wishlist](docs/contributing/wishlist.md) and [Skill Development Guide](docs/contributing/skills.md).
*   **Developer Chat:** Join us on [Discord](https://discord.com/invite/crestal) for support and discussions.

## License

This project is licensed under the MIT License.
[LICENSE](LICENSE)
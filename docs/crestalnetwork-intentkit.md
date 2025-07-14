# IntentKit: Build Autonomous AI Agents with Ease

[<img src="docs/images/intentkit_banner.png" alt="IntentKit by Crestal" width="100%">](https://github.com/crestalnetwork/intentkit)

**IntentKit is a powerful framework that allows developers to create and manage sophisticated AI agents capable of interacting with various platforms and executing complex tasks.**

## Key Features

*   🤖 **Multi-Agent Support:** Manage and orchestrate multiple AI agents simultaneously.
*   🔄 **Autonomous Agent Management:** Built-in capabilities for autonomous operation and decision-making.
*   🔗 **Blockchain Integration:** Interact with EVM-compatible blockchains for on-chain actions and data analysis.
*   🐦 **Social Media Integration:** Connect with platforms like Twitter, Telegram, and more to manage social presence and engage with users.
*   🛠️ **Extensible Skill System:** Easily integrate custom skills and functionalities to expand agent capabilities.
*   🔌 **MCP (WIP):** (Briefly explain what this is when it's no longer WIP)

## Architecture Overview

IntentKit's architecture leverages LangGraph to provide a robust foundation for building intelligent agents. The system integrates various components, including:

*   **Entrypoints:**  Interfaces for interacting with agents (e.g., Twitter, Telegram).
*   **Core Agent System:** Powered by LangGraph, enabling complex agent workflows.
*   **Skills:**  Modules for performing specific tasks, such as blockchain interaction, social media management, and more.
*   **Storage:** Managing agent configurations, credentials, personality, memory, and skill state.

For a detailed understanding, refer to the [Architecture](docs/architecture.md) section in the documentation.

## Project Structure

The project is structured into two main parts:

*   **IntentKit Package (`intentkit/`):**  The core library, published as a pip package.  Includes:
    *   Abstract classes and interfaces
    *   Clients for external services
    *   System-level configurations
    *   Core agent system
    *   Entity models
    *   Extensible skills system
    *   Utility functions
*   **IntentKit App (`app/`):**  The application layer, featuring an API server, autonomous agent runner, and background scheduler.  Includes:
    *   Admin APIs
    *   Entrypoints (web, Telegram, Twitter, etc.)
    *   Service implementations
    *   REST API server
    *   Autonomous agent runner
    *   Health and credit checking logic
    *   Readonly entrypoint
    *   Background task scheduler
    *   Singleton agent manager
    *   Telegram and Twitter integrations

## Development

Get started with development by reading the [Development Guide](DEVELOPMENT.md).

## Documentation

Comprehensive documentation is available in the [Documentation](docs/) directory.

## Contributing

We welcome contributions!  Please review the [Contributing Guidelines](CONTRIBUTING.md) before submitting a pull request.

### Contribute Skills

Explore the [Wishlist](docs/contributing/wishlist.md) for active skill requests and follow the [Skill Development Guide](docs/contributing/skills.md) for guidance.

### Developer Chat

Join our [Discord](https://discord.com/invite/crestal) and request an IntentKit dev role to connect with the community.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
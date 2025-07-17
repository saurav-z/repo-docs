# IntentKit: Build and Manage Powerful AI Agents

**IntentKit empowers you to create and deploy autonomous AI agents capable of complex tasks, making it easy to interact with blockchains, social media, and more.** ([View the original repository](https://github.com/crestalnetwork/intentkit))

<div align="center">
  <img src="docs/images/intentkit_banner.png" alt="IntentKit by Crestal" width="100%" />
</div>
<br>

## Key Features of IntentKit

*   **Multi-Agent Support:** Manage and orchestrate multiple AI agents simultaneously.
*   **Autonomous Agent Management:**  Control and oversee the lifecycle of your intelligent agents.
*   **Blockchain Integration:** Seamlessly interact with EVM-compatible blockchains for on-chain actions.
*   **Social Media Integration:** Connect with platforms like Twitter and Telegram to manage your online presence.
*   **Extensible Skill System:** Easily expand agent capabilities with custom skills built upon LangChain tools.
*   **Under Development:** MCP (Work in progress)

## Architecture Overview

IntentKit's architecture facilitates diverse agent functionalities by allowing for interaction with various entrypoints, including social media platforms. This enables the agents to perform on-chain actions, manage digital assets and connect with the world through custom skills.

*   **Entrypoints**: Twitter, Telegram, and more
*   **Core Functionality**: The Agent, built with LangGraph.
*   **Skills**: Chain integration, wallet management, on-chain actions, internet search, image processing, and more.
*   **Supporting components:** Agent Config & Memory, Credentials, Personality, and Skill State.

For a more detailed understanding, refer to the [Architecture](docs/architecture.md) section.

## Development

To begin developing with IntentKit, consult the [Development Guide](DEVELOPMENT.md).

## Documentation

Comprehensive documentation is available to guide your implementation. Explore the [Documentation](docs/) to get started.

## Project Structure

The project is organized into core and application components:

*   **`intentkit/`**: The IntentKit package, includes:
    *   `abstracts/`: Abstract classes and interfaces
    *   `clients/`: Clients for external services
    *   `config/`: System level configurations
    *   `core/`: Core agent system
    *   `models/`: Entity models
    *   `skills/`: Extensible skills system
    *   `utils/`: Utility functions
*   **`app/`**: The IntentKit application:
    *   `admin/`: Admin APIs and agent generators
    *   `entrypoints/`: Agent interaction entrypoints
    *   `services/`: Service implementations
    *   `api.py`: REST API server
    *   `autonomous.py`: Autonomous agent runner
    *   `checker.py`: Health and credit checking logic
    *   `readonly.py`: Readonly entrypoint
    *   `scheduler.py`: Background task scheduler
    *   `singleton.py`: Singleton agent manager
    *   `telegram.py`: Telegram integration
    *   `twitter.py`: Twitter integration
*   `docs/`: Documentation
*   `scripts/`: Operation and temporary scripts

## Agent API

IntentKit offers a REST API for programmatic agent interaction. Learn how to leverage the API by accessing the [Agent API Documentation](docs/agent_api.md).

## Contributing

Contributions are welcome! Please review the [Contributing Guidelines](CONTRIBUTING.md) before submitting your pull requests.

### Contribute Skills

Check the [Wishlist](docs/contributing/wishlist.md) for active requests.

See the [Skill Development Guide](docs/contributing/skills.md) for developing new skills.

### Developer Chat

Join our [Discord](https://discord.com/invite/crestal) and request a developer role for support.
<!--
  SPDX-License-Identifier: MIT
-->

# IntentKit: Build Autonomous AI Agents with Ease

IntentKit empowers developers to create and manage sophisticated, autonomous AI agents capable of interacting with the blockchain, social media, and custom skills. ([Original Repository](https://github.com/crestalnetwork/intentkit))

<div align="center">
  <img src="docs/images/intentkit_banner.png" alt="IntentKit by Crestal" width="100%" />
</div>
<br>

## Key Features

*   🤖 **Multiple Agent Support:** Manage and deploy numerous autonomous agents simultaneously.
*   🔄 **Autonomous Agent Management:** Streamlined control and oversight of your AI agents.
*   🔗 **Blockchain Integration:** Interact with EVM-compatible blockchains for on-chain actions.
*   🐦 **Social Media Integration:** Connect with and manage platforms like Twitter and Telegram.
*   🛠️ **Extensible Skill System:** Customize agents with a vast array of skills using LangChain tools.
*   🔌 **MCP (WIP):**  (Work in Progress)

## Architecture Overview

IntentKit's architecture enables agents to receive input from various entrypoints (Twitter, Telegram, etc.), process information using core components like agent configuration, memory, and skills, and take actions. Skills can encompass blockchain interaction, internet search, and image processing, among others. More detailed information can be found in the [Architecture Documentation](docs/architecture.md).

## Development and Documentation

*   **Get Started:** Learn how to set up your development environment with our [Development Guide](DEVELOPMENT.md).
*   **Comprehensive Documentation:** Explore detailed documentation for comprehensive information [Documentation](docs/).

## Project Structure

The project is structured into two primary components:

*   **`intentkit/`**: The core IntentKit package (installable via pip):
    *   `abstracts/`: Abstract classes and interfaces.
    *   `clients/`: Clients for external services.
    *   `config/`: System-level configurations.
    *   `core/`: Core agent system (LangGraph-driven).
    *   `models/`: Entity models (Pydantic & SQLAlchemy).
    *   `skills/`: Extensible skill system (LangChain tools).
    *   `utils/`: Utility functions.
*   **`app/`**: The IntentKit application (API server, autonomous runner, scheduler):
    *   `admin/`: Admin APIs and agent management.
    *   `entrypoints/`: Agent interaction entrypoints (web, Telegram, Twitter).
    *   `services/`: Service implementations for various platforms.
    *   `api.py`: REST API server.
    *   `autonomous.py`: Autonomous agent runner.
    *   `checker.py`: Health and credit checking.
    *   `readonly.py`: Readonly entrypoint.
    *   `scheduler.py`: Background task scheduler.
    *   `singleton.py`: Singleton agent manager.
    *   `telegram.py`: Telegram integration.
    *   `twitter.py`: Twitter integration.

Plus directories for documentation, scripts, and other supporting files.

## Agent API

Integrate your applications and systems with IntentKit's powerful [Agent API](docs/agent_api.md) for programmatic agent access.

## Contributing

We welcome contributions! Please review our [Contributing Guidelines](CONTRIBUTING.md) before submitting pull requests.

*   **Contribute Skills:** Check our [Wishlist](docs/contributing/wishlist.md) for active requests and follow the [Skill Development Guide](docs/contributing/skills.md).
*   **Join the Community:** Connect with developers on our [Discord](https://discord.com/invite/crestal) and apply for an intentkit dev role to join the discussion channel.

## License

This project is licensed under the [MIT License](LICENSE).
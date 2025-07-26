# IntentKit: Build Autonomous AI Agents with Ease

**IntentKit** is a powerful framework for creating and managing autonomous AI agents, designed to interact with various platforms and perform complex tasks. [Explore the original repository](https://github.com/crestalnetwork/intentkit) for more details.

## Key Features

*   🤖 **Multiple Agent Support:** Create and manage numerous AI agents.
*   🔄 **Autonomous Agent Management:**  Effortlessly control the lifecycle of your agents.
*   🔗 **Blockchain Integration:**  Interact with EVM-compatible blockchains.
*   🐦 **Social Media Integration:**  Connect with platforms like Twitter and Telegram.
*   🛠️ **Extensible Skill System:**  Customize agent capabilities with a flexible skill system.
*   🔌 **MCP (WIP):**  More features are constantly being added.

## Architecture Overview

IntentKit's architecture allows for flexible integration of various modules:

*   **Entrypoints:** Handle interactions with external services (Twitter, Telegram, etc.).
*   **Skills:** Enable on-chain actions, internet searches, image processing, and more.
*   **Core Agent:** Powered by LangGraph, managing agent behavior and interactions.
*   **Storage:**  Manages agent configurations, credentials, personality, and memory.

For a detailed understanding of the architecture, refer to the [Architecture](docs/architecture.md) documentation.

## Package Manager Migration Warning

If you are setting up this project for the first time, ensure you have deleted the old virtual environment, and installed the new one:

```bash
rm -rf .venv
uv sync
```

## Development

Get started with development by following the [Development Guide](DEVELOPMENT.md).

## Documentation

Comprehensive documentation is available to help you get started: [Documentation](docs/)

## Project Structure

The project is organized into the following main components:

*   **[intentkit/](intentkit/)**: The core IntentKit package (pip installable):
    *   [abstracts/](intentkit/abstracts/)
    *   [clients/](intentkit/clients/)
    *   [config/](intentkit/config/)
    *   [core/](intentkit/core/)
    *   [models/](intentkit/models/)
    *   [skills/](intentkit/skills/)
    *   [utils/](intentkit/utils/)
*   **[app/](app/)**: The IntentKit application:
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
*   [docs/](docs/): Documentation
*   [scripts/](scripts/):  Scripts for management and migrations.

## Agent API

Access and control your agents programmatically using the comprehensive REST API.

**Learn more:** [Agent API Documentation](docs/agent_api.md)

## Contributing

We welcome contributions!  Please review the [Contributing Guidelines](CONTRIBUTING.md) before submitting pull requests.

### Contribute Skills

Check the [Wishlist](docs/contributing/wishlist.md) for active skill requests.

Refer to the [Skill Development Guide](docs/contributing/skills.md) for guidance.

### Developer Chat

Join the developer community on [Discord](https://discord.com/invite/crestal) and apply for the IntentKit dev role.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
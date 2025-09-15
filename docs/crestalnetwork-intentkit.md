# IntentKit: Build and Manage Autonomous AI Agents

**Unleash the power of AI with IntentKit, an open-source framework for creating and orchestrating intelligent autonomous agents.** ([View on GitHub](https://github.com/crestalnetwork/intentkit))

<div align="center">
  <img src="docs/images/intentkit_banner.png" alt="IntentKit by Crestal" width="100%" />
</div>
<br>

IntentKit empowers developers to build and manage sophisticated AI agents capable of interacting with various services, including blockchain, social media, and custom integrations. This framework simplifies the development process and provides a robust foundation for creating autonomous systems.

## Key Features

*   🤖 **Multi-Agent Support:** Manage multiple AI agents simultaneously.
*   🔄 **Autonomous Agent Management:**  Efficiently handle agent lifecycles and operations.
*   🔗 **Blockchain Integration:** Interact with EVM-compatible blockchain networks.
*   🐦 **Social Media Integration:** Seamlessly connect with platforms like Twitter and Telegram.
*   🛠️ **Extensible Skill System:** Easily integrate custom skills to expand agent capabilities.
*   🔌 **MCP (WIP):** (Mention of MCP, awaiting more information)

## Architecture Overview

IntentKit's architecture is designed for modularity and extensibility. Agents leverage a LangGraph core, interacting with various entrypoints (Twitter, Telegram, etc.) and integrating with external services via skills.

```
                                                                                    
                                 Entrypoints                                        
                       │                             │                              
                       │   Twitter/Telegram & more   │                              
                       └──────────────┬──────────────┘                              
                                      │                                             
  Storage:  ────┐                     │                      ┌──── Skills:          
                │                     │                      │                      
  Agent Config  │     ┌───────────────▼────────────────┐     │  Chain Integration   
                │     │                                │     │                      
  Credentials   │     │                                │     │  Wallet Management   
                │     │           The Agent            │     │                      
  Personality   │     │                                │     │  On-Chain Actions    
                │     │                                │     │                      
  Memory        │     │      Powered by LangGraph      │     │  Internet Search     
                │     │                                │     │                      
  Skill State   │     └────────────────────────────────┘     │  Image Processing    
            ────┘                                            └────                  
                                                                                    
                                                                More and More...    
                         ┌──────────────────────────┐                               
                         │                          │                               
                         │  Agent Config & Memory   │                               
                         │                          │                               
                         └──────────────────────────┘                               
                                                                                    
```

For a more detailed understanding, explore the [Architecture Documentation](docs/architecture.md).

## Quick Start: Package Manager Migration

**Important:** This project has migrated to `uv` from `poetry`.

1.  **Remove existing virtual environment:** `rm -rf .venv`
2.  **Create a new virtual environment:** `uv sync`

## Development

Get started with setting up your development environment.  Refer to the [Development Guide](DEVELOPMENT.md).

## Documentation

Comprehensive documentation is available to guide you through using IntentKit. Review the [Documentation](docs/).

## Project Structure

The project is organized for clarity and maintainability, with key components including:

*   **[intentkit/](intentkit/)**: The core IntentKit package (installable via pip)
    *   [abstracts/](intentkit/abstracts/): Abstract classes and interfaces.
    *   [clients/](intentkit/clients/): Clients for external services.
    *   [config/](intentkit/config/): System configurations.
    *   [core/](intentkit/core/): Core agent system.
    *   [models/](intentkit/models/): Entity models using Pydantic and SQLAlchemy.
    *   [skills/](intentkit/skills/): Extensible skills system.
    *   [utils/](intentkit/utils/): Utility functions.

*   **[app/](app/)**: The IntentKit application (API server, runner, and scheduler)
    *   [admin/](app/admin/): Admin APIs and related functionality.
    *   [entrypoints/](app/entrypoints/): Entrypoints for agent interaction.
    *   [services/](app/services/): Service implementations.
    *   [api.py](app/api.py): REST API server.
    *   [autonomous.py](app/autonomous.py): Autonomous agent runner.
    *   [checker.py](app/checker.py): Health and credit checking.
    *   [readonly.py](app/readonly.py): Readonly entrypoint.
    *   [scheduler.py](app/scheduler.py): Background task scheduler.
    *   [singleton.py](app/singleton.py): Singleton agent manager.
    *   [telegram.py](app/telegram.py): Telegram integration.
    *   [twitter.py](app/twitter.py): Twitter integration.

*   [docs/](docs/): Documentation
*   [scripts/](scripts/): Management and migration scripts

## Agent API

IntentKit offers a robust REST API, enabling programmatic interaction with your agents. Explore the [Agent API Documentation](docs/agent_api.md) to build custom applications and integrations.

## Contributing

We welcome contributions! Please review the [Contributing Guidelines](CONTRIBUTING.md) before submitting a pull request.

### Contribute Skills

Check the [Wishlist](docs/contributing/wishlist.md) for active feature requests. Then, follow the [Skill Development Guide](docs/contributing/skills.md) to create new skills.

### Developer Chat

Join our developer community on [Discord](https://discord.com/invite/crestal) and apply for an IntentKit dev role to join the discussion channel.
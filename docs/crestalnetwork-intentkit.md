# IntentKit: Build and Manage Intelligent AI Agents

**Unlock the power of autonomous AI with IntentKit, an open-source framework for creating and deploying sophisticated AI agents.** ([Original Repository](https://github.com/crestalnetwork/intentkit))

<div align="center">
  <img src="docs/images/intentkit_banner.png" alt="IntentKit by Crestal" width="100%" />
</div>
<br>

IntentKit empowers you to build AI agents capable of complex tasks, from blockchain interactions to social media management and custom skill integration.

## Key Features of IntentKit

*   **🤖 Multiple Agent Support:** Manage and deploy multiple autonomous agents.
*   **🔄 Autonomous Agent Management:** Robust framework for controlling and monitoring your AI agents.
*   **🔗 Blockchain Integration:** Interact with EVM-compatible blockchains.
*   **🐦 Social Media Integration:** Connect with platforms like Twitter and Telegram.
*   **🛠️ Extensible Skill System:** Easily add custom skills to extend agent capabilities.
*   **🔌 MCP (WIP):** Placeholder for future functionality.

## Architecture Overview

IntentKit's architecture is designed for flexibility and extensibility.  Agents are powered by LangGraph and integrate with a variety of services and tools.

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

For a more in-depth understanding, explore the [Architecture Documentation](docs/architecture.md).

## Getting Started

### Package Manager Migration Warning

If you are updating from a previous version, you will need to delete your `.venv` folder and run `uv sync` to create a new virtual environment.

```bash
rm -rf .venv
uv sync
```

### Development Setup

To begin your development journey, consult the [Development Guide](DEVELOPMENT.md).

### Comprehensive Documentation

Dive into the full documentation before you start: [Documentation](docs/)

## Project Structure

The IntentKit project is structured into distinct components:

*   **[intentkit/](intentkit/)**: The core IntentKit package (available on PyPI).
    *   [abstracts/](intentkit/abstracts/): Abstract classes and interfaces.
    *   [clients/](intentkit/clients/): Clients for external services.
    *   [config/](intentkit/config/): System configurations.
    *   [core/](intentkit/core/): Core agent system.
    *   [models/](intentkit/models/): Data models using Pydantic and SQLAlchemy.
    *   [skills/](intentkit/skills/): Extensible skills system.
    *   [utils/](intentkit/utils/): Utility functions.
*   **[app/](app/)**: The IntentKit application (API server, autonomous runner, scheduler).
    *   [admin/](app/admin/): Admin APIs and agent generators.
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
*   [docs/](docs/): Documentation.
*   [scripts/](scripts/): Management and migration scripts.

## Agent API

Access and manage your agents programmatically through the comprehensive REST API.  Use the Agent API to integrate with existing systems or build custom interfaces.

*   **Explore the Agent API:** [Agent API Documentation](docs/agent_api.md)

## Contributing

We welcome contributions! Please review the [Contributing Guidelines](CONTRIBUTING.md) before submitting a pull request.

### Contributing Skills

1.  Check the [Wishlist](docs/contributing/wishlist.md) for active requests.
2.  Follow the [Skill Development Guide](docs/contributing/skills.md) for instructions.

### Developer Community

Connect with other developers and get support:

*   **Join our Discord:** [Discord](https://discord.com/invite/crestal)
*   Apply for an IntentKit dev role by opening a support ticket.
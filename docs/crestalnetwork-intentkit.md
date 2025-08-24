# IntentKit: Build & Manage Powerful AI Agents

**Unleash the power of AI with IntentKit, an open-source framework for creating and deploying autonomous agents that seamlessly interact with the digital world.**  ([View on GitHub](https://github.com/crestalnetwork/intentkit))

<div align="center">
  <img src="docs/images/intentkit_banner.png" alt="IntentKit by Crestal" width="100%" />
</div>
<br>

IntentKit provides a robust and flexible framework for building AI agents capable of a wide range of tasks, including blockchain interaction, social media management, and custom skill integration. Leverage LangGraph to create, manage, and extend the capabilities of your AI agents.

## Key Features

*   **🤖 Multiple Agent Support:** Easily manage and deploy multiple independent AI agents.
*   **🔄 Autonomous Agent Management:** Monitor and control your agents' behavior and tasks.
*   **🔗 Blockchain Integration:** Interact with EVM-compatible blockchains for decentralized application development and automation.
*   **🐦 Social Media Integration:** Connect with platforms like Twitter and Telegram to engage and manage your social presence.
*   **🛠️ Extensible Skill System:** Expand agent capabilities with custom skills based on LangChain tools.
*   **🔌 MCP (WIP):** (Mention if this is a key differentiator or feature)

## Architecture Overview

IntentKit's architecture is designed for flexibility and extensibility:

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

For a more detailed view, see the [Architecture](docs/architecture.md) documentation.

## Getting Started

### Package Manager Migration Warning

We recently migrated to `uv` from `poetry`. To ensure a smooth setup:

1.  Delete your existing virtual environment: `rm -rf .venv`
2.  Create a new virtual environment: `uv sync`

### Development

Start your development journey with our [Development Guide](DEVELOPMENT.md).

### Documentation

Comprehensive documentation is available in the [Documentation](docs/) section.

## Project Structure

The project is organized for maintainability and modularity:

*   **[intentkit/](intentkit/)**: The core IntentKit package.
    *   [abstracts/](intentkit/abstracts/): Abstract classes and interfaces.
    *   [clients/](intentkit/clients/): Clients for external services.
    *   [config/](intentkit/config/): System-level configurations.
    *   [core/](intentkit/core/): Core agent system (LangGraph-driven).
    *   [models/](intentkit/models/): Entity models (Pydantic and SQLAlchemy).
    *   [skills/](intentkit/skills/): Extensible skill system (LangChain-based).
    *   [utils/](intentkit/utils/): Utility functions.

*   **[app/](app/)**: The IntentKit application.
    *   [admin/](app/admin/): Admin APIs and agent management.
    *   [entrypoints/](app/entrypoints/): Entrypoints for agent interaction (web, Telegram, Twitter, etc.).
    *   [services/](app/services/): Service implementations (Telegram, Twitter, etc.).
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

**Programmatically control your agents using our comprehensive REST API.**

*   **Explore the API:** [Agent API Documentation](docs/agent_api.md)

## Contribute

**We welcome contributions!**

*   **Contributing Guidelines:** Read our [Contributing Guidelines](CONTRIBUTING.md).

### Skill Development

*   **Feature Requests:** Review the [Wishlist](docs/contributing/wishlist.md) for active requests.
*   **Skill Development Guide:** [Skill Development Guide](docs/contributing/skills.md) for details on contributing skills.

### Developer Community

*   **Join the Conversation:** Connect with other developers on our [Discord](https://discord.com/invite/crestal). Request an intentkit dev role in the support ticket.
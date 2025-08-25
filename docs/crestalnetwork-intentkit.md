# IntentKit: Build and Manage Autonomous AI Agents

**Create powerful AI agents for blockchain interaction, social media management, and more with the versatile IntentKit framework.** ([See the original repo](https://github.com/crestalnetwork/intentkit))

<div align="center">
  <img src="docs/images/intentkit_banner.png" alt="IntentKit by Crestal" width="100%" />
</div>
<br>

IntentKit is an innovative autonomous agent framework designed to empower developers to build and manage sophisticated AI agents with a wide range of capabilities. From interacting with blockchains to managing social media and integrating custom skills, IntentKit provides the tools you need to bring your AI agent ideas to life.

## Key Features

*   **🤖 Multi-Agent Support:** Manage and orchestrate multiple AI agents within a single framework.
*   **🔄 Autonomous Agent Management:** Easily control and monitor the lifecycle of your autonomous agents.
*   **🔗 Blockchain Integration:** Seamlessly interact with EVM-compatible blockchains.
*   **🐦 Social Media Integration:** Connect with platforms like Twitter, Telegram, and more.
*   **🛠️ Extensible Skill System:** Customize your agents with a wide range of skills, and easily create new ones.
*   **🔌 MCP (WIP):** (Mention the purpose/functionality of this if possible. If not, remove.)

## Architecture Overview

IntentKit's architecture is designed for flexibility and scalability. The core of the system is driven by LangGraph, providing a robust foundation for agent management and interaction.

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

For a deeper understanding of the architecture, please refer to the [Architecture](docs/architecture.md) section in the documentation.

## Getting Started

### Development

Get up and running with IntentKit by following the instructions in the [Development Guide](DEVELOPMENT.md).

### Documentation

Comprehensive documentation is available to guide you through the framework: [Documentation](docs/)

## Project Structure

The project is organized into the following key components:

*   **[intentkit/](intentkit/)**: The core IntentKit package.
    *   [abstracts/](intentkit/abstracts/): Abstract classes and interfaces.
    *   [clients/](intentkit/clients/): Clients for external services.
    *   [config/](intentkit/config/): System-level configurations.
    *   [core/](intentkit/core/): Core agent system.
    *   [models/](intentkit/models/): Entity models using Pydantic and SQLAlchemy.
    *   [skills/](intentkit/skills/): Extensible skills system.
    *   [utils/](intentkit/utils/): Utility functions.

*   **[app/](app/)**: The IntentKit application.
    *   [admin/](app/admin/): Admin APIs and agent generation.
    *   [entrypoints/](app/entrypoints/): Agent interaction entrypoints.
    *   [services/](app/services/): Service implementations.
    *   [api.py](app/api.py): REST API server.
    *   [autonomous.py](app/autonomous.py): Autonomous agent runner.
    *   [checker.py](app/checker.py): Health and credit checking logic.
    *   [readonly.py](app/readonly.py): Readonly entrypoint.
    *   [scheduler.py](app/scheduler.py): Background task scheduler.
    *   [singleton.py](app/singleton.py): Singleton agent manager.
    *   [telegram.py](app/telegram.py): Telegram integration.
    *   [twitter.py](app/twitter.py): Twitter integration.

*   [docs/](docs/): Documentation
*   [scripts/](scripts/): Management and migration scripts

## Agent API

Interact with your agents programmatically using the comprehensive REST API.

**Explore the API:** [Agent API Documentation](docs/agent_api.md)

## Contributing

We welcome contributions!  Review our [Contributing Guidelines](CONTRIBUTING.md) to get started.

### Contribute Skills

Check the [Wishlist](docs/contributing/wishlist.md) for skill requests and then read the [Skill Development Guide](docs/contributing/skills.md) to contribute.

### Developer Community

Join the conversation on our [Discord](https://discord.com/invite/crestal) and apply for an intentkit dev role to gain access to the developer channel.

## License

IntentKit is licensed under the [MIT License](LICENSE).
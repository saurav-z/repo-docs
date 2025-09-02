# IntentKit: Build Autonomous AI Agents with Ease

**IntentKit empowers developers to create and manage sophisticated AI agents capable of interacting with the blockchain, social media, and more.** ([See the original repo](https://github.com/crestalnetwork/intentkit))

<div align="center">
  <img src="docs/images/intentkit_banner.png" alt="IntentKit by Crestal" width="100%" />
</div>
<br>

## Key Features

*   🤖 **Multi-Agent Support:** Manage and deploy multiple AI agents simultaneously.
*   🔄 **Autonomous Agent Management:**  Automate agent workflows and decision-making processes.
*   🔗 **Blockchain Integration:** Interact with EVM-compatible blockchains.
*   🐦 **Social Media Integration:**  Connect and manage agents on platforms like Twitter and Telegram.
*   🛠️ **Extensible Skill System:** Customize agent capabilities with a flexible skill framework.
*   🔌 **MCP (WIP):** Placeholder for upcoming features

## Architecture

IntentKit leverages a modular architecture for maximum flexibility.  The core agent system, powered by LangGraph, integrates seamlessly with various entrypoints and skills.

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

For a more detailed understanding, refer to the [Architecture](docs/architecture.md) section.

## Getting Started

### Package Manager Migration

**Important:**  The project has migrated from Poetry to `uv`. To set up your environment:

```bash
rm -rf .venv
uv sync
```

### Development

Follow the [Development Guide](DEVELOPMENT.md) to configure your development environment.

### Documentation

Explore the comprehensive [Documentation](docs/) for detailed information.

## Project Structure

The project is organized into two main parts: the `intentkit` package and the `app`.

*   **[intentkit/](intentkit/)**: The IntentKit package (published as a pip package)
    *   [abstracts/](intentkit/abstracts/): Abstract classes and interfaces
    *   [clients/](intentkit/clients/): Clients for external services
    *   [config/](intentkit/config/): System configurations
    *   [core/](intentkit/core/): Core agent system
    *   [models/](intentkit/models/): Entity models
    *   [skills/](intentkit/skills/): Extensible skills system
    *   [utils/](intentkit/utils/): Utility functions

*   **[app/](app/)**: The IntentKit app (API server, autonomous runner, etc.)
    *   [admin/](app/admin/): Admin APIs
    *   [entrypoints/](app/entrypoints/): Entrypoints for interacting with agents
    *   [services/](app/services/): Service implementations
    *   [api.py](app/api.py): REST API server
    *   [autonomous.py](app/autonomous.py): Autonomous agent runner
    *   [checker.py](app/checker.py): Health and credit checking logic
    *   [readonly.py](app/readonly.py): Readonly entrypoint
    *   [scheduler.py](app/scheduler.py): Background task scheduler
    *   [singleton.py](app/singleton.py): Singleton agent manager
    *   [telegram.py](app/telegram.py): Telegram integration
    *   [twitter.py](app/twitter.py): Twitter integration

*   [docs/](docs/): Documentation
*   [scripts/](scripts/): Operation and temporary scripts

## Agent API

Access and control your agents programmatically using the REST API.

**Learn More:** [Agent API Documentation](docs/agent_api.md)

## Contributing

We welcome contributions! Please review the [Contributing Guidelines](CONTRIBUTING.md).

### Contribute Skills

Consult the [Wishlist](docs/contributing/wishlist.md) for active skill requests.
See the [Skill Development Guide](docs/contributing/skills.md) for instructions on creating skills.

### Developer Chat

Join the community on [Discord](https://discord.com/invite/crestal) and apply for a dev role. There's a discussion channel there for collaboration.

## License

This project is licensed under the [MIT License](LICENSE).
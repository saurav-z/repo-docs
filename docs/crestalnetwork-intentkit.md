# IntentKit: Build Intelligent Autonomous AI Agents

**Unleash the power of AI with IntentKit, a cutting-edge framework for creating and managing autonomous agents that can interact with the world.**

[View the original repository](https://github.com/crestalnetwork/intentkit)

<div align="center">
  <img src="docs/images/intentkit_banner.png" alt="IntentKit by Crestal" width="100%" />
</div>
<br>

IntentKit is a robust framework designed for building and managing AI agents with diverse capabilities, including blockchain interaction, social media management, and custom skill integration.  It empowers developers to create intelligent, autonomous systems that can automate complex tasks and interact with various platforms.

## Key Features of IntentKit

*   **🤖 Multi-Agent Support:**  Create and manage multiple AI agents simultaneously.
*   **🔄 Autonomous Agent Management:**  Easily control and orchestrate your agents' behavior.
*   **🔗 Blockchain Integration:** Seamlessly interact with EVM-compatible blockchains.
*   **🐦 Social Media Integration:**  Connect with platforms like Twitter, Telegram, and more.
*   **🛠️ Extensible Skill System:**  Customize agent capabilities with a flexible skill system.
*   **🔌 MCP (WIP):** Explore Machine Coordination Protocol integration.

## Architecture Overview

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

For a more detailed understanding, refer to the [Architecture](docs/architecture.md) documentation.

## Package Manager Migration Notice

We recently migrated to `uv` from `poetry`.  To ensure compatibility, please delete your `.venv` folder and run `uv sync` to create a new virtual environment:

```bash
rm -rf .venv
uv sync
```

## Project Structure

IntentKit is organized into core package and an application:

*   **[intentkit/](intentkit/)**: The IntentKit package (published as a pip package)
    *   [abstracts/](intentkit/abstracts/): Abstract classes and interfaces for core and skills
    *   [clients/](intentkit/clients/): Clients for external services
    *   [config/](intentkit/config/): System level configurations
    *   [core/](intentkit/core/): Core agent system, driven by LangGraph
    *   [models/](intentkit/models/): Entity models using Pydantic and SQLAlchemy
    *   [skills/](intentkit/skills/): Extensible skills system, based on LangChain tools
    *   [utils/](intentkit/utils/): Utility functions

*   **[app/](app/)**: The IntentKit app (API server, autonomous runner, and background scheduler)
    *   [admin/](app/admin/): Admin APIs, agent generators, and related functionality
    *   [entrypoints/](app/entrypoints/): Entrypoints for interacting with agents (web, Telegram, Twitter, etc.)
    *   [services/](app/services/): Service implementations for Telegram, Twitter, etc.
    *   [api.py](app/api.py): REST API server
    *   [autonomous.py](app/autonomous.py): Autonomous agent runner
    *   [checker.py](app/checker.py): Health and credit checking logic
    *   [readonly.py](app/readonly.py): Readonly entrypoint
    *   [scheduler.py](app/scheduler.py): Background task scheduler
    *   [singleton.py](app/singleton.py): Singleton agent manager
    *   [telegram.py](app/telegram.py): Telegram integration
    *   [twitter.py](app/twitter.py): Twitter integration

*   [docs/](docs/): Documentation
*   [scripts/](scripts/): Operation and temporary scripts for management and migrations

## Agent API

Access and control your AI agents programmatically with the comprehensive REST API. Build custom applications, integrate with existing systems, and create unique interfaces.

**Get Started:** [Agent API Documentation](docs/agent_api.md)

## Development

Learn how to set up your development environment:  [Development Guide](DEVELOPMENT.md)

## Documentation

Comprehensive documentation is available: [Documentation](docs/)

## Contribute

We welcome contributions! Please review the [Contributing Guidelines](CONTRIBUTING.md) before submitting a pull request.

### Contribute Skills

1.  Check the [Wishlist](docs/contributing/wishlist.md) for open requests.
2.  Follow the [Skill Development Guide](docs/contributing/skills.md) to create your own skills.

### Developer Community

Join our [Discord](https://discord.com/invite/crestal) and apply for the intentkit dev role in a support ticket. We also have a discussion channel for developers.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
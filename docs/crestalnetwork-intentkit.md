# IntentKit: Build and Manage Autonomous AI Agents for Any Task

[<img src="docs/images/intentkit_banner.png" alt="IntentKit by Crestal" width="100%" />](https://github.com/crestalnetwork/intentkit)

IntentKit is an innovative framework empowering you to create, deploy, and manage powerful autonomous AI agents capable of a wide range of tasks, from interacting with blockchains to managing social media and integrating custom skills.

## Key Features

*   🤖 **Multi-Agent Support:** Manage and orchestrate multiple AI agents.
*   🔄 **Autonomous Agent Management:**  Effortlessly control and monitor your AI agents.
*   🔗 **Blockchain Integration:** Interact with EVM-compatible blockchains (with plans for expansion).
*   🐦 **Social Media Integration:** Connect with users via platforms like Twitter, Telegram, and more.
*   🛠️ **Extensible Skill System:** Easily add new capabilities to your agents.
*   🔌 **MCP (WIP):**  (Details to come)

## Architecture

IntentKit's architecture is designed for flexibility and extensibility, enabling you to build sophisticated AI agent applications. The core is built on LangGraph, and supports a variety of inputs/outputs and skills.

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

For a deeper dive into the architecture, please see the [Architecture](docs/architecture.md) documentation.

## Development

Get started with IntentKit development by reviewing the [Development Guide](DEVELOPMENT.md).

## Documentation

Comprehensive documentation is available to guide you through the framework.  Explore the [Documentation](docs/) to learn more.

## Project Structure

The project is organized into a core package and an application, each with specific responsibilities:

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

## Contributing

We welcome contributions!  Please review the [Contributing Guidelines](CONTRIBUTING.md) before submitting a pull request.

### Contribute Skills

Check the [Wishlist](docs/contributing/wishlist.md) for open skill requests.

The [Skill Development Guide](docs/contributing/skills.md) contains information to help you get started.

### Developer Chat

Join our [Discord](https://discord.com/invite/crestal) and open a support ticket to apply for an IntentKit dev role.

## License

IntentKit is licensed under the [MIT License](LICENSE).
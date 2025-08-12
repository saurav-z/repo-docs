# IntentKit: Build and Manage AI Agents with Ease

**IntentKit empowers you to create and manage intelligent AI agents capable of interacting with the blockchain, social media, and more.** ([View the original repository](https://github.com/crestalnetwork/intentkit))

<div align="center">
  <img src="docs/images/intentkit_banner.png" alt="IntentKit by Crestal" width="100%" />
</div>
<br>

IntentKit is an open-source framework designed for building and deploying autonomous AI agents. It provides a robust foundation for developing agents that can interact with various platforms and services, streamlining complex tasks and automating workflows.

## Key Features

*   🤖 **Multi-Agent Support:** Manage and orchestrate multiple AI agents simultaneously.
*   🔄 **Autonomous Agent Management:** Control and monitor agent behavior with built-in automation capabilities.
*   🔗 **Blockchain Integration:** Interact with EVM-compatible blockchains for on-chain actions.
*   🐦 **Social Media Integration:** Connect with and manage popular social media platforms like Twitter and Telegram.
*   🛠️ **Extensible Skill System:** Customize your agents with a wide range of skills.
*   🔌 **MCP (WIP):** (Feature description coming soon)

## Quick Start - Package Manager Migration (Important!)

**Attention:** This project has migrated to `uv` from `poetry`. You'll need to perform a one-time setup to ensure compatibility:

```bash
rm -rf .venv
uv sync
```

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

For a more detailed explanation, refer to the [Architecture](docs/architecture.md) documentation.

## Development

*   **Getting Started:**  Read the [Development Guide](DEVELOPMENT.md) to set up your environment.
*   **Documentation:** Explore the comprehensive [Documentation](docs/) for in-depth information.

## Project Structure

*   **[intentkit/](intentkit/)**: The core IntentKit package
    *   [abstracts/](intentkit/abstracts/): Abstract classes and interfaces
    *   [clients/](intentkit/clients/): Clients for external services
    *   [config/](intentkit/config/): System configurations
    *   [core/](intentkit/core/): Core agent system
    *   [models/](intentkit/models/): Entity models
    *   [skills/](intentkit/skills/): Extensible skills system
    *   [utils/](intentkit/utils/): Utility functions
*   **[app/](app/)**: The IntentKit application
    *   [admin/](app/admin/): Admin APIs and agent generators
    *   [entrypoints/](app/entrypoints/): Agent entrypoints (web, Telegram, etc.)
    *   [services/](app/services/): Service implementations
    *   [api.py](app/api.py): REST API server
    *   [autonomous.py](app/autonomous.py): Autonomous agent runner
    *   [checker.py](app/checker.py): Health and credit checking
    *   [readonly.py](app/readonly.py): Readonly entrypoint
    *   [scheduler.py](app/scheduler.py): Background task scheduler
    *   [singleton.py](app/singleton.py): Singleton agent manager
    *   [telegram.py](app/telegram.py): Telegram integration
    *   [twitter.py](app/twitter.py): Twitter integration
*   [docs/](docs/): Documentation
*   [scripts/](scripts/): Utility scripts

## Agent API

IntentKit provides a comprehensive REST API for programmatically interacting with your AI agents.

*   **API Documentation:** Access the [Agent API Documentation](docs/agent_api.md) to learn more.

## Contributing

We welcome contributions!

*   **Contribution Guidelines:** Review the [Contributing Guidelines](CONTRIBUTING.md).
*   **Contribute Skills:**  Check the [Wishlist](docs/contributing/wishlist.md) and the [Skill Development Guide](docs/contributing/skills.md).
*   **Community:** Join the developer chat on [Discord](https://discord.com/invite/crestal). Request the intentkit dev role in a support ticket.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
# IntentKit: Build Powerful Autonomous AI Agents

**Unleash the power of AI with IntentKit, a cutting-edge framework for creating and managing autonomous agents that can interact with the blockchain, social media, and much more.** [View the original repository](https://github.com/crestalnetwork/intentkit)

<div align="center">
  <img src="docs/images/intentkit_banner.png" alt="IntentKit by Crestal" width="100%" />
</div>
<br>

IntentKit empowers developers to build sophisticated AI agents with diverse capabilities. It offers a flexible and extensible framework for creating intelligent systems that can automate tasks, interact with various platforms, and execute complex workflows.

## Key Features

*   🤖 **Multiple Agent Support:** Manage and orchestrate several agents simultaneously.
*   🔄 **Autonomous Agent Management:** Provides tools for monitoring, managing, and scaling your agents.
*   🔗 **Blockchain Integration:** Interact seamlessly with EVM-compatible blockchains.
*   🐦 **Social Media Integration:** Engage with audiences on platforms like Twitter and Telegram.
*   🛠️ **Extensible Skill System:** Add custom functionality by integrating new skills.
*   🔌 **MCP (WIP):** (Mention if you want to share)

## Architecture Overview

IntentKit's architecture is designed for modularity and scalability. The core of the system is built around LangGraph, enabling the creation of sophisticated agent workflows.

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

For a deeper dive into the system's design, refer to the [Architecture](docs/architecture.md) documentation.

## Development

Get started with your setup by consulting the [Development Guide](DEVELOPMENT.md).

## Documentation

Comprehensive documentation is available to guide you through the framework. Explore the [Documentation](docs/) to learn more.

## Project Structure

The project is divided into core components:

*   **[intentkit/](intentkit/)**: The IntentKit package.
    *   [abstracts/](intentkit/abstracts/): Abstract classes and interfaces.
    *   [clients/](intentkit/clients/): Clients for external services.
    *   [config/](intentkit/config/): System configurations.
    *   [core/](intentkit/core/): Core agent system.
    *   [models/](intentkit/models/): Entity models.
    *   [skills/](intentkit/skills/): Extensible skills system.
    *   [utils/](intentkit/utils/): Utility functions.
*   **[app/](app/)**: The IntentKit application.
    *   [admin/](app/admin/): Admin APIs.
    *   [entrypoints/](app/entrypoints/): Entrypoints.
    *   [services/](app/services/): Service implementations.
    *   [api.py](app/api.py): REST API server.
    *   [autonomous.py](app/autonomous.py): Autonomous agent runner.
    *   [checker.py](app/checker.py): Health and credit checking logic.
    *   [readonly.py](app/readonly.py): Readonly entrypoint.
    *   [scheduler.py](app/scheduler.py): Background task scheduler.
    *   [singleton.py](app/singleton.py): Singleton agent manager.
    *   [telegram.py](app/telegram.py): Telegram integration.
    *   [twitter.py](app/twitter.py): Twitter integration.
*   [docs/](docs/): Documentation.
*   [scripts/](scripts/): Operation scripts.

## Agent API

Interact with your agents programmatically using the provided REST API. Build applications, integrate with existing systems, or create custom interfaces.

*   **Agent API Documentation:** [Agent API Documentation](docs/agent_api.md)

## Contributing

Contributions are welcome! Refer to the following guidelines:

*   **Contributing Guidelines:** [Contributing Guidelines](CONTRIBUTING.md)
*   **Contribute Skills:**
    *   Review the [Wishlist](docs/contributing/wishlist.md) for active requests.
    *   Follow the [Skill Development Guide](docs/contributing/skills.md) to contribute.

## Community

*   **Developer Chat:** Join our [Discord](https://discord.com/invite/crestal) to get involved in discussions and receive support.

## Package Manager Migration Warning

We just migrated to uv from poetry.
You need to delete the .venv folder and run `uv sync` to create a new virtual environment. (one time)

```bash
rm -rf .venv
uv sync
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
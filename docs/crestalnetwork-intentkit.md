# IntentKit: Build and Manage Autonomous AI Agents

**IntentKit empowers you to create intelligent, autonomous AI agents capable of interacting with the world, from blockchain to social media.** [Visit the original repository](https://github.com/crestalnetwork/intentkit)

<div align="center">
  <img src="docs/images/intentkit_banner.png" alt="IntentKit by Crestal" width="100%" />
</div>
<br>

## Key Features

*   🤖 **Multiple Agent Support:** Manage and orchestrate multiple AI agents within a single framework.
*   🔄 **Autonomous Agent Management:**  Enable your agents to operate independently and make decisions.
*   🔗 **Blockchain Integration:** Interact with EVM-compatible blockchains for on-chain actions and data access.
*   🐦 **Social Media Integration:** Connect with platforms like Twitter and Telegram to engage with audiences.
*   🛠️ **Extensible Skill System:**  Add new capabilities to your agents using a flexible skill-based architecture.
*   🔌 **MCP (WIP):**  (To be updated with further info)

## Architecture Overview

IntentKit's architecture is designed for flexibility and extensibility, with LangGraph at its core.  Agents can receive input from various entrypoints (e.g., Twitter, Telegram) and leverage a range of skills to perform tasks.

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

Get started with the IntentKit framework by reading the [Development Guide](DEVELOPMENT.md).

## Documentation

Comprehensive documentation is available at [Documentation](docs/) to help you get started.

## Project Structure

*   **[intentkit/](intentkit/)**: The core IntentKit package
    *   [abstracts/](intentkit/abstracts/): Abstract classes and interfaces
    *   [clients/](intentkit/clients/): Clients for external services
    *   [config/](intentkit/config/): System level configurations
    *   [core/](intentkit/core/): Core agent system
    *   [models/](intentkit/models/): Entity models
    *   [skills/](intentkit/skills/): Extensible skills system
    *   [utils/](intentkit/utils/): Utility functions
*   **[app/](app/)**: The IntentKit application
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

Interact with your agents programmatically using the IntentKit REST API.

**Agent API Documentation:** [docs/agent_api.md](docs/agent_api.md)

## Contributing

We welcome contributions! Please review our [Contributing Guidelines](CONTRIBUTING.md).

### Contribute Skills

1.  Check the [Wishlist](docs/contributing/wishlist.md) for existing requests.
2.  Follow the [Skill Development Guide](docs/contributing/skills.md) to start building.

### Developer Community

Join the discussion on our [Discord](https://discord.com/invite/crestal), and apply for a developer role.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
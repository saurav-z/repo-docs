# IntentKit: Build Autonomous AI Agents with Ease

**IntentKit empowers you to create and manage intelligent AI agents capable of blockchain interaction, social media engagement, and more!** Explore the power of autonomous agents with this versatile framework. [Check out the original repo](https://github.com/crestalnetwork/intentkit) for more details.

<div align="center">
  <img src="docs/images/intentkit_banner.png" alt="IntentKit by Crestal" width="100%" />
</div>
<br>

## Key Features

*   🤖 **Multi-Agent Support:** Manage and deploy multiple AI agents simultaneously.
*   🔄 **Autonomous Agent Management:** Control and orchestrate your agents with ease.
*   🔗 **Blockchain Integration:** Interact with EVM-compatible blockchains (with plans for more).
*   🐦 **Social Media Integration:** Connect to Twitter, Telegram, and other platforms.
*   🛠️ **Extensible Skill System:** Customize your agents with a wide range of skills.
*   🔌 **MCP (WIP):** (Mention of MCP, which is under development)

## Architecture

IntentKit leverages a modular architecture based on LangGraph, providing a flexible foundation for building complex agents.

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

For a deeper dive into the architecture, please refer to the [Architecture](docs/architecture.md) section in the documentation.

## Development

Get started with IntentKit development by following the instructions in the [Development Guide](DEVELOPMENT.md).

## Documentation

Comprehensive documentation is available to guide you. Explore the [Documentation](docs/) to learn more.

## Project Structure

The project is organized into two main components:

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
    *   [entrypoints/](app/entrypoints/): Agent entrypoints (web, Telegram, Twitter, etc.)
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
*   [scripts/](scripts/): Operational and temporary scripts

## Agent API

Programmatically interact with your agents using the comprehensive REST API.

**Explore the Agent API:** [Agent API Documentation](docs/agent_api.md)

## Contributing

We welcome contributions! Please review the [Contributing Guidelines](CONTRIBUTING.md) before submitting pull requests.

### Contribute Skills

Check the [Wishlist](docs/contributing/wishlist.md) for skill requests.

Get started with skill development by reading the [Skill Development Guide](docs/contributing/skills.md).

### Developer Chat

Join the community on [Discord](https://discord.com/invite/crestal) and apply for an IntentKit dev role to join the discussion channel.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
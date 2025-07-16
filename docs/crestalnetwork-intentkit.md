# IntentKit: Build Autonomous AI Agents with Ease

**Unlock the power of AI agents with IntentKit, a flexible framework for creating and managing autonomous agents that can interact with blockchain, social media, and custom integrations.** [View the original repository on GitHub](https://github.com/crestalnetwork/intentkit)

<div align="center">
  <img src="docs/images/intentkit_banner.png" alt="IntentKit by Crestal" width="100%" />
</div>
<br>

IntentKit empowers developers to build sophisticated AI agents with a wide range of capabilities. Leverage its modular design to create agents tailored to your specific needs.

## Key Features

*   🤖 **Multiple Agent Support:** Manage and deploy numerous AI agents.
*   🔄 **Autonomous Agent Management:**  Simplify the lifecycle of your AI agents.
*   🔗 **Blockchain Integration:** Interact with EVM-compatible blockchains.
*   🐦 **Social Media Integration:** Seamlessly connect with platforms like Twitter and Telegram.
*   🛠️ **Extensible Skill System:** Add custom functionality through a flexible skill architecture.
*   🔌 **MCP (WIP):** [Placeholder for feature being developed]

## Architecture Overview

IntentKit's architecture is designed for modularity and extensibility. The core agent system, powered by LangGraph, integrates with various entrypoints (Twitter, Telegram, etc.), storage, and skills.  For a more detailed understanding, explore the [Architecture](docs/architecture.md) documentation.

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

## Development and Documentation

*   **Getting Started:** Follow the [Development Guide](DEVELOPMENT.md) for setup instructions.
*   **Comprehensive Documentation:** Dive deeper into IntentKit with the official [Documentation](docs/).

## Project Structure

The project is structured into core components, enabling a clear and organized approach to development:

*   **[intentkit/](intentkit/)**: The IntentKit package (pip package)
    *   [abstracts/](intentkit/abstracts/): Abstract classes and interfaces
    *   [clients/](intentkit/clients/): Clients for external services
    *   [config/](intentkit/config/): System level configurations
    *   [core/](intentkit/core/): Core agent system, driven by LangGraph
    *   [models/](intentkit/models/): Entity models using Pydantic and SQLAlchemy
    *   [skills/](intentkit/skills/): Extensible skills system, based on LangChain tools
    *   [utils/](intentkit/utils/): Utility functions
*   **[app/](app/)**: The IntentKit app (API server, autonomous runner, and background scheduler)
    *   [admin/](app/admin/): Admin APIs, agent generators, and related functionality
    *   [entrypoints/](app/entrypoints/): Entrypoints for interacting with agents
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
*   [scripts/](scripts/): Operation and temporary scripts

## Agent API

**Effortlessly control and interact with your agents using the comprehensive REST API.** Build applications, integrate with existing systems, or create custom interfaces with ease.  Explore the [Agent API Documentation](docs/agent_api.md) to get started.

## Contributing

We welcome contributions!

*   **Guidelines:** Review our [Contributing Guidelines](CONTRIBUTING.md).
*   **Skill Development:** Learn how to contribute new skills via the [Skill Development Guide](docs/contributing/skills.md).
*   **Feature Requests:** Check the [Wishlist](docs/contributing/wishlist.md) for active feature requests.
*   **Community Chat:** Join the discussion on [Discord](https://discord.com/invite/crestal) (apply for an intentkit dev role).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
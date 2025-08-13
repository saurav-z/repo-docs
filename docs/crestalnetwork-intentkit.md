# IntentKit: Build and Manage Autonomous AI Agents

**Unleash the power of autonomous agents with IntentKit, an open-source framework designed for AI-driven automation and intelligent interaction.**  ([Original Repo](https://github.com/crestalnetwork/intentkit))

<div align="center">
  <img src="docs/images/intentkit_banner.png" alt="IntentKit by Crestal" width="100%" />
</div>
<br>

IntentKit empowers you to create and manage AI agents with a diverse range of capabilities, from blockchain interactions to social media management, all within a flexible and extensible framework.

## Key Features

*   🤖 **Multi-Agent Support:** Manage multiple autonomous agents.
*   🔄 **Autonomous Agent Management:** Orchestrate and control the lifecycle of your agents.
*   🔗 **Blockchain Integration:** Interact with EVM-compatible blockchains.
*   🐦 **Social Media Integration:** Connect with platforms like Twitter and Telegram.
*   🛠️ **Extensible Skill System:** Customize agent behavior with a plug-and-play skill system.
*   🔌 **MCP (WIP):**  (Details to come)

## Architecture

IntentKit's architecture is designed for modularity and scalability. The core agent system, powered by LangGraph, sits at the center, interacting with various entrypoints (e.g., Twitter, Telegram) and leveraging a rich set of skills. This includes functionalities such as:

*   Agent Configuration and Memory
*   Chain Integration and Wallet Management
*   Social Media Integration and Communication
*   Internet Search and Information Retrieval
*   Image Processing and Data Analysis

A simplified architectural diagram is provided below. For a deeper dive, refer to the [Architecture](docs/architecture.md) section.

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

## Package Manager Update (Important!)

The package manager has been updated to `uv`.  To update your environment:

```bash
rm -rf .venv
uv sync
```

## Development

Start building and customizing with IntentKit! See the [Development Guide](DEVELOPMENT.md) for setup instructions and best practices.

## Documentation

Comprehensive documentation is available to guide you through the framework. Explore the [Documentation](docs/) to get started.

## Project Structure

The project is organized into distinct modules for easy navigation and development:

*   **[intentkit/](intentkit/)**: The core IntentKit package (pip installable)
    *   [abstracts/](intentkit/abstracts/): Abstract classes and interfaces
    *   [clients/](intentkit/clients/): Clients for external services
    *   [config/](intentkit/config/): System configurations
    *   [core/](intentkit/core/): Core agent system
    *   [models/](intentkit/models/): Entity models
    *   [skills/](intentkit/skills/): Extensible skills system
    *   [utils/](intentkit/utils/): Utility functions

*   **[app/](app/)**: The IntentKit application (API server, autonomous runner, scheduler)
    *   [admin/](app/admin/): Admin APIs and agent management
    *   [entrypoints/](app/entrypoints/): Agent entrypoints (web, Telegram, Twitter, etc.)
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
*   [scripts/](scripts/): Management and migration scripts

## Agent API

Interact with your agents programmatically using the comprehensive REST API.

**Explore the Agent API:** [Agent API Documentation](docs/agent_api.md)

## Contributing

We welcome contributions! Review the [Contributing Guidelines](CONTRIBUTING.md) before submitting your pull requests.

### Contribute Skills

Contribute new skills to extend the capabilities of your agents!

1.  Check the [Wishlist](docs/contributing/wishlist.md) for active requests.
2.  Follow the [Skill Development Guide](docs/contributing/skills.md) for instructions.

### Developer Community

Join the IntentKit developer community and collaborate with other contributors!

*   Join our [Discord](https://discord.com/invite/crestal) and apply for a developer role.

## License

IntentKit is released under the MIT License. See the [LICENSE](LICENSE) file for details.
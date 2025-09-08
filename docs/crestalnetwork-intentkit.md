# IntentKit: Build Autonomous AI Agents with Ease

**Unleash the power of AI with IntentKit, a flexible framework for creating and managing intelligent agents that can interact with the world.**  [View the original repository on GitHub](https://github.com/crestalnetwork/intentkit)

<div align="center">
  <img src="docs/images/intentkit_banner.png" alt="IntentKit by Crestal" width="100%" />
</div>
<br>

IntentKit is an open-source framework designed to simplify the development and deployment of autonomous AI agents.  It allows you to build agents with a wide range of capabilities, from blockchain interactions to social media management and beyond.

## Key Features

*   🤖 **Multi-Agent Support:**  Create and manage multiple independent AI agents.
*   🔄 **Autonomous Agent Management:**  Orchestrate the entire lifecycle of your agents.
*   🔗 **Blockchain Integration:**  Seamlessly interact with EVM-compatible blockchains (with expansion to other chains).
*   🐦 **Social Media Integration:**  Connect your agents to platforms like Twitter and Telegram (and more).
*   🛠️ **Extensible Skill System:**  Easily add new skills and capabilities to your agents.
*   🔌 **MCP (WIP):**  Ongoing development for advanced features.

## Architecture Overview

IntentKit's architecture is designed for flexibility and extensibility.  At its core, the system is powered by LangGraph, enabling sophisticated agent workflows.  Agents interact with the world through various entrypoints (e.g., social media platforms) and leverage a range of skills to perform tasks.

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

For a more in-depth look at the architecture, please see the [Architecture Documentation](docs/architecture.md).

## Getting Started

### Package Manager Migration
If this is your first time running the project, you will need to run the following to create a new virtual environment.
```bash
rm -rf .venv
uv sync
```

### Development

Begin developing with IntentKit!  The [Development Guide](DEVELOPMENT.md) provides detailed instructions for setting up your environment and getting started.

### Documentation

Explore the comprehensive [Documentation](docs/) to fully understand IntentKit's capabilities and how to use them.

## Project Structure

IntentKit is organized into a core package and an application layer:

*   **[intentkit/](intentkit/)**: The core IntentKit package (available via pip):
    *   [abstracts/](intentkit/abstracts/): Abstract classes and interfaces
    *   [clients/](intentkit/clients/): Clients for external services
    *   [config/](intentkit/config/): System configurations
    *   [core/](intentkit/core/): Core agent system (LangGraph-driven)
    *   [models/](intentkit/models/): Entity models (Pydantic & SQLAlchemy)
    *   [skills/](intentkit/skills/): Extensible skills system (LangChain tools)
    *   [utils/](intentkit/utils/): Utility functions

*   **[app/](app/)**: The IntentKit application (API server, autonomous runner, scheduler):
    *   [admin/](app/admin/): Admin APIs and agent generators
    *   [entrypoints/](app/entrypoints/): Agent interaction entrypoints (web, Telegram, etc.)
    *   [services/](app/services/): Service implementations (Telegram, Twitter, etc.)
    *   [api.py](app/api.py): REST API server
    *   [autonomous.py](app/autonomous.py): Autonomous agent runner
    *   [checker.py](app/checker.py): Health and credit checking
    *   [readonly.py](app/readonly.py): Readonly entrypoint
    *   [scheduler.py](app/scheduler.py): Background task scheduler
    *   [singleton.py](app/singleton.py): Singleton agent manager
    *   [telegram.py](app/telegram.py): Telegram integration
    *   [twitter.py](app/twitter.py): Twitter integration

*   [docs/](docs/): Documentation
*   [scripts/](scripts/): Scripts for management and migrations

## Agent API

Unlock programmatic access to your agents with IntentKit's robust REST API.  Build custom applications and integrate with existing systems.

*   **Learn More:** [Agent API Documentation](docs/agent_api.md)

## Contributing

We welcome contributions!

*   **Contribution Guidelines:**  Please read the [Contributing Guidelines](CONTRIBUTING.md) before submitting pull requests.
*   **Contribute Skills:** Check out the [Wishlist](docs/contributing/wishlist.md) to find the most up-to-date skill requests.  Once ready, the [Skill Development Guide](docs/contributing/skills.md) will help you get started.
*   **Developer Chat:** Join our [Discord](https://discord.com/invite/crestal) and apply for the IntentKit developer role to join the discussion.

## License

This project is licensed under the MIT License.  See the [LICENSE](LICENSE) file for more information.
# IntentKit: Build and Manage Autonomous AI Agents

**IntentKit empowers you to effortlessly create and manage sophisticated AI agents capable of interacting with blockchains, social media, and custom skills.**

[![IntentKit Banner](docs/images/intentkit_banner.png)](https://github.com/crestalnetwork/intentkit)

## Key Features

*   🤖 **Multiple Agent Support:** Create and manage multiple autonomous AI agents.
*   🔄 **Autonomous Agent Management:**  Automate agent workflows and decision-making.
*   🔗 **Blockchain Integration:** Seamlessly interact with EVM-compatible blockchains.
*   🐦 **Social Media Integration:**  Connect with platforms like Twitter and Telegram.
*   🛠️ **Extensible Skill System:**  Extend agent capabilities with custom skills and integrations.
*   🔌 **MCP (WIP):** Multi-chain provider integration (Work in Progress).

## Architecture Overview

IntentKit's architecture allows agents to interact with the outside world through various entrypoints (Twitter, Telegram, etc.) and leverage a suite of skills.  These skills, such as blockchain interaction, social media management, and internet search, are integrated with the core agent system, powered by LangGraph.  See the [detailed architecture documentation](docs/architecture.md) for more information.

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

## Development & Getting Started

1.  **Package Manager Migration:**  After migrating to `uv` from `poetry`, you'll need to delete the existing virtual environment.
    ```bash
    rm -rf .venv
    uv sync
    ```

2.  **Development Guide:**  Refer to the [Development Guide](DEVELOPMENT.md) to set up your development environment.

3.  **Documentation:**  Explore the comprehensive [documentation](docs/) for in-depth information.

## Project Structure

*   **[intentkit/](intentkit/)**: The core IntentKit package.
    *   [abstracts/](intentkit/abstracts/): Abstract classes and interfaces.
    *   [clients/](intentkit/clients/): Clients for external services.
    *   [config/](intentkit/config/): System-level configurations.
    *   [core/](intentkit/core/): Core agent system (LangGraph).
    *   [models/](intentkit/models/): Entity models (Pydantic & SQLAlchemy).
    *   [skills/](intentkit/skills/): Extensible skills system (LangChain).
    *   [utils/](intentkit/utils/): Utility functions.

*   **[app/](app/)**: The IntentKit application (API server, autonomous runner, etc.).
    *   [admin/](app/admin/): Admin APIs and agent generators.
    *   [entrypoints/](app/entrypoints/): Agent entrypoints (web, Telegram, Twitter, etc.).
    *   [services/](app/services/): Service implementations.
    *   [api.py](app/api.py): REST API server.
    *   [autonomous.py](app/autonomous.py): Autonomous agent runner.
    *   [checker.py](app/checker.py): Health and credit checking.
    *   [readonly.py](app/readonly.py): Readonly entrypoint.
    *   [scheduler.py](app/scheduler.py): Background task scheduler.
    *   [singleton.py](app/singleton.py): Singleton agent manager.
    *   [telegram.py](app/telegram.py): Telegram integration.
    *   [twitter.py](app/twitter.py): Twitter integration.

*   [docs/](docs/): Documentation
*   [scripts/](scripts/): Operation and temporary scripts

## Agent API

**Programmatically control your agents with the IntentKit REST API.**  Explore the [Agent API Documentation](docs/agent_api.md) to get started.

## Contributing

We welcome contributions! Please review the [Contributing Guidelines](CONTRIBUTING.md) before submitting pull requests.

### Contribute Skills

1.  Review the [Wishlist](docs/contributing/wishlist.md) for active requests.
2.  See the [Skill Development Guide](docs/contributing/skills.md) for implementation details.

### Developer Community

Join our [Discord](https://discord.com/invite/crestal) to connect with other developers and get support. Apply for an intentkit dev role in the `#support-ticket` channel.

## License

This project is licensed under the [MIT License](LICENSE).

---
**[View the original repository on GitHub](https://github.com/crestalnetwork/intentkit)**
# IntentKit: Build Autonomous AI Agents with Ease

**Empower your projects with intelligent, autonomous agents capable of interacting with blockchains, social media, and more using IntentKit.** ([View on GitHub](https://github.com/crestalnetwork/intentkit))

<div align="center">
  <img src="docs/images/intentkit_banner.png" alt="IntentKit by Crestal" width="100%" />
</div>
<br>

IntentKit is an open-source framework designed to simplify the creation and management of AI agents. These agents can perform complex tasks autonomously, including interacting with blockchains, managing social media, and leveraging custom skills.

## Key Features

*   🤖 **Multiple Agent Support:** Manage and orchestrate multiple AI agents.
*   🔄 **Autonomous Agent Management:** Effortlessly oversee the lifecycle and actions of your agents.
*   🔗 **Blockchain Integration:** Interact with EVM-compatible blockchains.
*   🐦 **Social Media Integration:** Connect with platforms like Twitter and Telegram.
*   🛠️ **Extensible Skill System:** Easily add custom skills to extend agent capabilities using LangChain tools.
*   🔌 **MCP (WIP):** (Mention of future feature)

## Package Manager Migration Warning

**Important:** You must delete the `.venv` folder and run `uv sync` to create a new virtual environment after migrating to `uv` from poetry.

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

For a more detailed understanding, explore the [Architecture](docs/architecture.md) documentation.

## Project Structure

*   **[intentkit/](intentkit/)**: Core IntentKit package.
    *   [abstracts/](intentkit/abstracts/): Abstract classes and interfaces.
    *   [clients/](intentkit/clients/): Clients for external services.
    *   [config/](intentkit/config/): System-level configurations.
    *   [core/](intentkit/core/): Core agent system, powered by LangGraph.
    *   [models/](intentkit/models/): Entity models using Pydantic and SQLAlchemy.
    *   [skills/](intentkit/skills/): Extensible skills system.
    *   [utils/](intentkit/utils/): Utility functions.

*   **[app/](app/)**: IntentKit app (API server, runner, and scheduler).
    *   [admin/](app/admin/): Admin APIs, agent generators, and related functionality.
    *   [entrypoints/](app/entrypoints/): Entrypoints for agent interaction.
    *   [services/](app/services/): Service implementations (Telegram, Twitter, etc.).
    *   [api.py](app/api.py): REST API server.
    *   [autonomous.py](app/autonomous.py): Autonomous agent runner.
    *   [checker.py](app/checker.py): Health and credit checking logic.
    *   [readonly.py](app/readonly.py): Readonly entrypoint.
    *   [scheduler.py](app/scheduler.py): Background task scheduler.
    *   [singleton.py](app/singleton.py): Singleton agent manager.
    *   [telegram.py](app/telegram.py): Telegram integration.
    *   [twitter.py](app/twitter.py): Twitter integration.

*   [docs/](docs/): Documentation.
*   [scripts/](scripts/): Operation and temporary scripts.

## Agent API

IntentKit provides a comprehensive REST API for programmatic access to your agents.

**Explore the API:** [Agent API Documentation](docs/agent_api.md)

## Development

**Get Started:** Read the [Development Guide](DEVELOPMENT.md) to set up your environment.

**Comprehensive Documentation:** Explore the [Documentation](docs/) for detailed information.

## Contributing

We welcome contributions! Please review the [Contributing Guidelines](CONTRIBUTING.md) before submitting pull requests.

### Contribute Skills

1.  Check the [Wishlist](docs/contributing/wishlist.md) for active skill requests.
2.  Follow the [Skill Development Guide](docs/contributing/skills.md) for detailed instructions.

### Developer Community

Join our [Discord](https://discord.com/invite/crestal) and apply for an intentkit dev role to engage with the development team.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
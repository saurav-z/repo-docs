# IntentKit: Build and Manage Autonomous AI Agents

**Create powerful AI agents for blockchain interaction, social media, and more with IntentKit, an open-source framework.** Learn more on [GitHub](https://github.com/crestalnetwork/intentkit).

<div align="center">
  <img src="docs/images/intentkit_banner.png" alt="IntentKit by Crestal" width="100%" />
</div>
<br>

IntentKit empowers developers to build and manage sophisticated AI agents capable of a wide range of tasks. From interacting with blockchains to managing social media presence and integrating custom skills, IntentKit provides the tools you need to create intelligent, autonomous systems.

## Key Features

*   🤖 **Multi-Agent Support:** Easily create and manage multiple AI agents.
*   🔄 **Autonomous Agent Management:** Orchestrate and oversee agent operations with built-in management tools.
*   🔗 **Blockchain Integration:** Interact with EVM-compatible blockchains (with plans to expand).
*   🐦 **Social Media Integration:** Connect with platforms like Twitter and Telegram.
*   🛠️ **Extensible Skill System:** Integrate custom skills and functionalities to expand agent capabilities.
*   🔌 **MCP (WIP):**  (More context needed - consider adding a short explanation here if possible).

## Architecture

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

This diagram provides a simplified overview of IntentKit's architecture.  For more detailed information, please refer to the [Architecture](docs/architecture.md) section.

## Package Manager Migration Warning

**Important:** We have migrated to `uv` from `poetry`. To ensure a clean virtual environment, delete your existing `.venv` folder and run `uv sync`:

```bash
rm -rf .venv
uv sync
```

## Getting Started

*   **Development:** Consult the [Development Guide](DEVELOPMENT.md) to set up your environment.
*   **Documentation:** Explore the comprehensive [Documentation](docs/) to understand IntentKit's capabilities.
*   **Agent API:** Access and interact with your agents programmatically using the [Agent API Documentation](docs/agent_api.md).

## Project Structure

The project is organized into two main parts: the core IntentKit package and the application itself.

*   **[intentkit/](intentkit/)**: The core IntentKit package (published as a pip package). Includes:
    *   [abstracts/](intentkit/abstracts/): Abstract classes and interfaces.
    *   [clients/](intentkit/clients/): Clients for external services.
    *   [config/](intentkit/config/): System-level configurations.
    *   [core/](intentkit/core/): The core agent system, powered by LangGraph.
    *   [models/](intentkit/models/): Entity models using Pydantic and SQLAlchemy.
    *   [skills/](intentkit/skills/): Extensible skills system, based on LangChain tools.
    *   [utils/](intentkit/utils/): Utility functions.
*   **[app/](app/)**: The IntentKit application (API server, autonomous runner, and background scheduler). Includes:
    *   [admin/](app/admin/): Admin APIs, agent generators, and related functionality.
    *   [entrypoints/](app/entrypoints/): Entrypoints for interacting with agents (web, Telegram, Twitter, etc.).
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

1.  Check the [Wishlist](docs/contributing/wishlist.md) for existing skill requests.
2.  Follow the [Skill Development Guide](docs/contributing/skills.md) to create and submit your skills.

### Developer Community

Join our community on [Discord](https://discord.com/invite/crestal) and open a support ticket to request the "intentkit dev" role.

## License

IntentKit is licensed under the [MIT License](LICENSE).
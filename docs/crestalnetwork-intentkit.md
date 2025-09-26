# IntentKit: Build and Manage Intelligent AI Agents

**Unleash the power of autonomous AI agents with IntentKit, a powerful framework for building intelligent applications.** ([Original Repo](https://github.com/crestalnetwork/intentkit))

<div align="center">
  <img src="docs/images/intentkit_banner.png" alt="IntentKit by Crestal" width="100%" />
</div>
<br>

IntentKit is an open-source framework designed for creating and managing AI agents with diverse capabilities, enabling you to automate complex tasks and build intelligent systems. Whether you need to interact with blockchains, manage social media, or integrate custom skills, IntentKit provides the tools you need.

## Key Features

*   **Multi-Agent Support:** Manage multiple AI agents, each with unique roles and functionalities.
*   **Autonomous Agent Management:** Control and orchestrate your agents with ease.
*   **Blockchain Integration:** Interact with EVM-compatible blockchain networks for on-chain actions.
*   **Social Media Integration:** Connect with platforms like Twitter and Telegram to automate social interactions.
*   **Extensible Skill System:** Customize your agents with a wide range of skills, including internet search, image processing, and more.
*   **Agent API:** Interact programmatically with your agents, build applications, and integrate with other systems using our REST API.

## Package Manager Migration Warning

Please note that we have migrated from Poetry to `uv`. To update your environment, please remove the existing virtual environment and run `uv sync`.

```bash
rm -rf .venv
uv sync
```

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

For a more detailed overview of the architecture, please refer to the [Architecture](docs/architecture.md) section in the documentation.

## Development

Get started with IntentKit development by following our [Development Guide](DEVELOPMENT.md).

## Documentation

Explore the comprehensive [Documentation](docs/) to learn more about IntentKit and its features.

## Project Structure

The project is structured as follows:

*   **[intentkit/](intentkit/)**: The IntentKit package (published as a pip package)
    *   [abstracts/](intentkit/abstracts/): Abstract classes and interfaces for core and skills
    *   [clients/](intentkit/clients/): Clients for external services
    *   [config/](intentkit/config/): System level configurations
    *   [core/](intentkit/core/): Core agent system, driven by LangGraph
    *   [models/](intentkit/models/): Entity models using Pydantic and SQLAlchemy
    *   [skills/](intentkit/skills/): Extensible skills system, based on LangChain tools
    *   [utils/](intentkit/utils/): Utility functions

*   **[app/](app/)**: The IntentKit app (API server, autonomous runner, and background scheduler)
    *   [admin/](app/admin/): Admin APIs, agent generators, and related functionality
    *   [entrypoints/](app/entrypoints/): Entrypoints for interacting with agents (web, Telegram, Twitter, etc.)
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

## Agent API

IntentKit provides a comprehensive REST API for programmatic access to your agents. Build applications, integrate with existing systems, or create custom interfaces using our Agent API.

**Get Started:** [Agent API Documentation](docs/agent_api.md)

## Contributing

We welcome contributions! Please review our [Contributing Guidelines](CONTRIBUTING.md) before submitting a pull request.

### Contribute Skills

First, check the [Wishlist](docs/contributing/wishlist.md) for active requests.

Once you are ready, see the [Skill Development Guide](docs/contributing/skills.md) for more information.

### Developer Chat

Join our [Discord](https://discord.com/invite/crestal) to connect with other developers. Open a support ticket to request an IntentKit dev role.
There is a discussion channel available for developers.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
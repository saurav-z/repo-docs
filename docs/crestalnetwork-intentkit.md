# IntentKit: Build Autonomous AI Agents with Ease

**Unleash the power of AI with IntentKit, a cutting-edge framework for creating and managing intelligent agents that can interact with the blockchain, social media, and more.**  (Original Repo: [https://github.com/crestalnetwork/intentkit](https://github.com/crestalnetwork/intentkit))

<div align="center">
  <img src="docs/images/intentkit_banner.png" alt="IntentKit by Crestal" width="100%" />
</div>
<br>

IntentKit is an autonomous agent framework designed to empower developers to build and deploy AI agents with a wide range of capabilities. Whether you're interested in blockchain interactions, social media management, or integrating custom skills, IntentKit provides the tools you need to succeed.

## Key Features of IntentKit:

*   **🤖 Multi-Agent Support:**  Easily manage and orchestrate multiple AI agents.
*   **🔄 Autonomous Agent Management:**  Automated agent operation and control.
*   **🔗 Blockchain Integration:** Interact with EVM-compatible blockchains.
*   **🐦 Social Media Integration:** Connect with platforms like Twitter and Telegram.
*   **🛠️ Extensible Skill System:**  Customize agent capabilities with a modular skill system.
*   **🔌 MCP (WIP):**  (Brief description/placeholder for future details)

## Architecture Overview

IntentKit employs a modular architecture that allows for flexible agent design and extensibility.

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

For a more in-depth understanding, please refer to the [Architecture](docs/architecture.md) section.

## Getting Started

### Package Manager Migration Warning

*   **Important:**  We have migrated to `uv` from `poetry`. Follow these steps to update your environment:

```bash
rm -rf .venv
uv sync
```

### Development

*   Start building with the [Development Guide](DEVELOPMENT.md).

### Documentation

*   Explore the [Documentation](docs/) for comprehensive information.

## Project Structure Breakdown:

*   **[intentkit/](intentkit/)**: The IntentKit package
    *   [abstracts/](intentkit/abstracts/): Abstract classes and interfaces
    *   [clients/](intentkit/clients/): External service clients
    *   [config/](intentkit/config/): System configurations
    *   [core/](intentkit/core/): Core agent system
    *   [models/](intentkit/models/): Entity models
    *   [skills/](intentkit/skills/): Extensible skills system
    *   [utils/](intentkit/utils/): Utility functions

*   **[app/](app/)**: The IntentKit app (API server, runner, scheduler)
    *   [admin/](app/admin/): Admin APIs and agent generators
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

IntentKit provides a powerful REST API for seamless programmatic agent interaction.

*   **Explore the API:**  Consult the [Agent API Documentation](docs/agent_api.md).

## Contributing

We welcome contributions from the community!

*   **Contribution Guidelines:** Read the [Contributing Guidelines](CONTRIBUTING.md).
*   **Skill Development:**  If you'd like to contribute skills, check out the [Wishlist](docs/contributing/wishlist.md) and the [Skill Development Guide](docs/contributing/skills.md).

### Community

*   **Developer Chat:** Join our [Discord](https://discord.com/invite/crestal) to collaborate. Apply for a dev role to participate in discussions.

## License

*   This project is licensed under the MIT License.  See the [LICENSE](LICENSE) file for details.
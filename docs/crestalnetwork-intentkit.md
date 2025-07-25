# IntentKit: Build and Manage Autonomous AI Agents

**Empower your projects with IntentKit, the open-source framework for creating and managing powerful, autonomous AI agents.** ([See the original repository](https://github.com/crestalnetwork/intentkit))

<div align="center">
  <img src="docs/images/intentkit_banner.png" alt="IntentKit by Crestal" width="100%" />
</div>
<br>

IntentKit is your go-to solution for building AI agents with a wide range of capabilities, including blockchain interaction, social media management, and custom skill integration. Leverage the power of LangGraph to create sophisticated, autonomous systems tailored to your specific needs.

## Key Features

*   🤖 **Multiple Agent Support:** Manage and orchestrate multiple AI agents simultaneously.
*   🔄 **Autonomous Agent Management:**  Control the lifecycle and behavior of your AI agents.
*   🔗 **Blockchain Integration:** Seamlessly interact with EVM-compatible blockchain networks.
*   🐦 **Social Media Integration:** Connect and manage agents across Twitter, Telegram, and more platforms.
*   🛠️ **Extensible Skill System:** Easily add custom skills to expand agent capabilities.
*   🔌 **MCP (WIP):**  (Details to come)

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

For a more detailed architectural overview, please consult the [Architecture](docs/architecture.md) section of the documentation.

## Development

Get started with your IntentKit setup by following the instructions in the [Development Guide](DEVELOPMENT.md).

## Documentation

Explore the comprehensive [Documentation](docs/) to learn more about IntentKit's features and capabilities.

## Project Structure

The project is organized into two primary parts: the IntentKit package and the application itself:

*   **[intentkit/](intentkit/)**: The core IntentKit package (installable via pip)
    *   [abstracts/](intentkit/abstracts/): Abstract classes and interfaces.
    *   [clients/](intentkit/clients/): Clients for external services.
    *   [config/](intentkit/config/): System-level configurations.
    *   [core/](intentkit/core/): Core agent system (powered by LangGraph).
    *   [models/](intentkit/models/): Entity models (Pydantic and SQLAlchemy).
    *   [skills/](intentkit/skills/): Extensible skill system.
    *   [utils/](intentkit/utils/): Utility functions.
*   **[app/](app/)**: The IntentKit application (API server, autonomous runner, and scheduler).
    *   [admin/](app/admin/): Admin APIs and agent generators.
    *   [entrypoints/](app/entrypoints/): Entrypoints for agent interaction (web, Telegram, Twitter, etc.).
    *   [services/](app/services/): Service implementations for various platforms.
    *   [api.py](app/api.py): REST API server.
    *   [autonomous.py](app/autonomous.py): Autonomous agent runner.
    *   [checker.py](app/checker.py): Health and credit checking.
    *   [readonly.py](app/readonly.py): Read-only entrypoint.
    *   [scheduler.py](app/scheduler.py): Background task scheduler.
    *   [singleton.py](app/singleton.py): Singleton agent manager.
    *   [telegram.py](app/telegram.py): Telegram integration.
    *   [twitter.py](app/twitter.py): Twitter integration.
*   [docs/](docs/): Documentation.
*   [scripts/](scripts/): Operational and temporary scripts.

## Agent API

IntentKit offers a comprehensive REST API, allowing you to programmatically access and control your agents. This empowers you to build custom applications, integrate with existing systems, and create bespoke interfaces.

**Learn more:** [Agent API Documentation](docs/agent_api.md)

## Contributing

We welcome contributions! Please review the [Contributing Guidelines](CONTRIBUTING.md) before submitting your pull requests.

### Contribute Skills

Consult the [Wishlist](docs/contributing/wishlist.md) for a list of actively requested skills.

For detailed guidance on developing skills, see the [Skill Development Guide](docs/contributing/skills.md).

### Developer Community

Join our vibrant community on [Discord](https://discord.com/invite/crestal).  To gain access to the IntentKit developer role, open a support ticket within the Discord server.

We encourage collaboration and discussion in our dedicated developer channel.

## License

This project is licensed under the MIT License.  See the [LICENSE](LICENSE) file for complete details.
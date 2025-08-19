# IntentKit: Build and Manage Autonomous AI Agents

**Unlock the power of AI automation with IntentKit, a robust framework for creating and deploying intelligent agents.**  [See the original repository](https://github.com/crestalnetwork/intentkit)

<div align="center">
  <img src="docs/images/intentkit_banner.png" alt="IntentKit by Crestal" width="100%" />
</div>
<br>

IntentKit is an open-source framework that empowers you to build and manage AI agents with a wide range of capabilities. This framework allows for easy integration with blockchain, social media, and custom skills.

## Key Features:

*   **🤖 Multiple Agent Support:** Create and manage various AI agents simultaneously.
*   **🔄 Autonomous Agent Management:** Orchestrate and oversee the autonomous operation of your agents.
*   **🔗 Blockchain Integration:** Interact with EVM-compatible blockchains for various on-chain actions.
*   **🐦 Social Media Integration:** Seamlessly connect with platforms like Twitter and Telegram to publish, engage, and automate social media interactions.
*   **🛠️ Extensible Skill System:** Integrate and create new skills to customize your agent's capabilities, expanding the range of tasks it can perform.
*   **🔌 MCP (WIP):** (Note: This feature is currently a work in progress.)

## Architecture

IntentKit utilizes a modular architecture centered around a central "Agent" component, powered by LangGraph, with clearly defined entry points and access to a skill system.

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
For a more in-depth understanding, please refer to the [Architecture](docs/architecture.md) section in the documentation.

## Package Manager Migration Warning

We have recently migrated from Poetry to `uv`. After this migration, follow the steps below to get started:
1.  Delete the existing virtual environment:
    ```bash
    rm -rf .venv
    ```
2.  Create a new virtual environment:
    ```bash
    uv sync
    ```

## Development

For detailed instructions on setting up your development environment, consult the [Development Guide](DEVELOPMENT.md).

## Documentation

Comprehensive documentation is available to guide you through IntentKit's features and functionalities.  Explore the documentation at [Documentation](docs/).

## Project Structure

IntentKit is organized into core package and application components:

**IntentKit Package (Published as a Pip Package - `intentkit/`)**
*   `intentkit/abstracts/`: Abstract classes and interfaces
*   `intentkit/clients/`: Clients for interacting with external services
*   `intentkit/config/`: System-level configurations
*   `intentkit/core/`: The core agent system
*   `intentkit/models/`: Data models using Pydantic and SQLAlchemy
*   `intentkit/skills/`: Extensible skills system, based on LangChain tools
*   `intentkit/utils/`: Utility functions

**IntentKit App (`app/`)**
*   `app/admin/`: Admin APIs and agent generation
*   `app/entrypoints/`: Agent interaction entrypoints (web, Telegram, Twitter)
*   `app/services/`: Service implementations for Telegram, Twitter, etc.
*   `app/api.py`: REST API server
*   `app/autonomous.py`: Autonomous agent runner
*   `app/checker.py`: Health and credit checks
*   `app/readonly.py`: Readonly entrypoint
*   `app/scheduler.py`: Background task scheduler
*   `app/singleton.py`: Singleton agent manager
*   `app/telegram.py`: Telegram integration
*   `app/twitter.py`: Twitter integration
*   `docs/`: Project documentation
*   `scripts/`: Management and migration scripts

## Agent API

IntentKit provides a REST API for programmatic access to your agents. Build custom applications, integrate with existing systems, or create custom interfaces using our Agent API.
**Get Started:** [Agent API Documentation](docs/agent_api.md)

## Contributing

We welcome contributions!  Please review the [Contributing Guidelines](CONTRIBUTING.md) before submitting a pull request.

### Contribute Skills

Check the [Wishlist](docs/contributing/wishlist.md) for current feature requests.
Once you are ready, consult the [Skill Development Guide](docs/contributing/skills.md) to understand how to contribute new skills.

### Developer Chat

Join our active community on [Discord](https://discord.com/invite/crestal), and request an `intentkit dev` role to access the developer discussion channel.

## License

IntentKit is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
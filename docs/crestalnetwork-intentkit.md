<div align="center">
  <img src="docs/images/intentkit_banner.png" alt="IntentKit by Crestal" width="100%" />
</div>
<br>

# IntentKit: Build and Manage Autonomous AI Agents with Ease

**IntentKit** empowers developers to create and manage sophisticated AI agents for a variety of tasks, from blockchain interaction to social media management. [Learn more and explore the code on GitHub](https://github.com/crestalnetwork/intentkit).

## Key Features of IntentKit

*   🤖 **Multiple Agent Support:** Manage and orchestrate multiple AI agents simultaneously.
*   🔄 **Autonomous Agent Management:** Control and monitor your agents' behavior and actions.
*   🔗 **Blockchain Integration:** Interact with EVM-compatible blockchains for on-chain actions.
*   🐦 **Social Media Integration:** Seamlessly connect with platforms like Twitter and Telegram.
*   🛠️ **Extensible Skill System:** Easily add new capabilities and integrate custom skills.
*   🔌 **MCP (WIP):** (Mention the purpose briefly or remove until it's ready)

## Architecture Overview

IntentKit is designed with a modular architecture, allowing for easy customization and extension.  At its core, the Agent is powered by LangGraph, integrating with various services through entrypoints and skill sets:

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
                │     │                                │     │  Image Processing    
  Skill State   │     └────────────────────────────────┘     │                      
            ────┘                                            └────                  
                                                                                    
                                                                More and More...    
                         ┌──────────────────────────┐                               
                         │                          │                               
                         │  Agent Config & Memory   │                               
                         │                          │                               
                         └──────────────────────────┘                               
                                                                                    
```

For a more in-depth understanding of the architecture, please refer to the [Architecture](docs/architecture.md) documentation.

## Development & Setup

*   **Package Manager Migration Warning:**  To update to the latest version, you'll need to delete your existing virtual environment and run `uv sync`.

    ```bash
    rm -rf .venv
    uv sync
    ```

*   **Getting Started:** Explore the [Development Guide](DEVELOPMENT.md) to set up your development environment.
*   **Documentation:** Comprehensive documentation is available in the [docs/](docs/) directory.

## Project Structure

The project is structured into core components:

*   **[intentkit/](intentkit/)**: The core IntentKit package.
    *   `abstracts/`, `clients/`, `config/`, `core/`, `models/`, `skills/`, `utils/`
*   **[app/](app/)**: The IntentKit application (API server, agent runner, and scheduler).
    *   `admin/`, `entrypoints/`, `services/`, `api.py`, `autonomous.py`, `checker.py`, `readonly.py`, `scheduler.py`, `singleton.py`, `telegram.py`, `twitter.py`
*   `docs/`, `scripts/`

## Agent API

IntentKit provides a robust REST API for interacting with your agents.

*   **Agent API Documentation:** Access and integrate with your agents programmatically via the [Agent API Documentation](docs/agent_api.md).

## Contributing

We welcome contributions!

*   **Contributing Guidelines:** Please review our [Contributing Guidelines](CONTRIBUTING.md) before submitting pull requests.
*   **Contribute Skills:** First check [Wishlist](docs/contributing/wishlist.md).
*   **Skill Development Guide:** To contribute new skills, see [Skill Development Guide](docs/contributing/skills.md)

## Community & Support

*   **Developer Chat:** Join our [Discord](https://discord.com/invite/crestal), open a support ticket to apply for an intentkit dev role.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
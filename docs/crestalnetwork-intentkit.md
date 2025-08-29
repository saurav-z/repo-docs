# IntentKit: Build and Manage Intelligent AI Agents

**Unleash the power of autonomous AI with IntentKit, a versatile framework for creating and deploying AI agents with diverse capabilities.** ([View the original repository](https://github.com/crestalnetwork/intentkit))

<div align="center">
  <img src="docs/images/intentkit_banner.png" alt="IntentKit by Crestal" width="100%" />
</div>
<br>

IntentKit empowers developers to build and manage sophisticated AI agents capable of interacting with the real world through:

## Key Features

*   🤖 **Multiple Agent Support:** Manage numerous AI agents within a single framework.
*   🔄 **Autonomous Agent Management:**  Control and orchestrate agent workflows seamlessly.
*   🔗 **Blockchain Integration:**  Interact with EVM-compatible blockchain networks.
*   🐦 **Social Media Integration:**  Connect with audiences on platforms like Twitter and Telegram.
*   🛠️ **Extensible Skill System:**  Customize agent capabilities with a modular skill architecture.
*   🔌 **MCP (WIP):** (More details to come!)

## Architecture Overview

IntentKit's architecture is designed for flexibility and extensibility:

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

For a deeper dive into the architecture, explore the [Architecture documentation](docs/architecture.md).

## Development and Setup

1.  **Dependency Management:**  We've migrated to `uv` from `poetry`.  To set up your environment:

    ```bash
    rm -rf .venv
    uv sync
    ```
2.  **Get Started:** Consult the [Development Guide](DEVELOPMENT.md) for detailed setup instructions.

## API and Documentation

*   **Agent API:**  Programmatically access and control your agents using our comprehensive REST API.
    *   [Agent API Documentation](docs/agent_api.md)
*   **General Documentation:**  Find in-depth information and guides in the [Documentation](docs/) directory.

## Project Structure

*   **[intentkit/](intentkit/)**: The IntentKit package (pip package)
    *   [abstracts/](intentkit/abstracts/)
    *   [clients/](intentkit/clients/)
    *   [config/](intentkit/config/)
    *   [core/](intentkit/core/)
    *   [models/](intentkit/models/)
    *   [skills/](intentkit/skills/)
    *   [utils/](intentkit/utils/)
*   **[app/](app/)**: The IntentKit app (API server, autonomous runner, scheduler)
    *   [admin/](app/admin/)
    *   [entrypoints/](app/entrypoints/)
    *   [services/](app/services/)
    *   [api.py](app/api.py)
    *   [autonomous.py](app/autonomous.py)
    *   [checker.py](app/checker.py)
    *   [readonly.py](app/readonly.py)
    *   [scheduler.py](app/scheduler.py)
    *   [singleton.py](app/singleton.py)
    *   [telegram.py](app/telegram.py)
    *   [twitter.py](app/twitter.py)
*   [docs/](docs/): Documentation
*   [scripts/](scripts/): Scripts for management

## Contributing

We welcome contributions!  Please review the following resources:

*   **Contribution Guidelines:**  [CONTRIBUTING.md](CONTRIBUTING.md)
*   **Contribute Skills:**
    *   Check the [Wishlist](docs/contributing/wishlist.md) for active requests.
    *   See the [Skill Development Guide](docs/contributing/skills.md) for guidance.
*   **Developer Chat:** Join our [Discord](https://discord.com/invite/crestal) and apply for the intentkit dev role for discussion.

## License

This project is licensed under the MIT License.  See the [LICENSE](LICENSE) file for details.
<div align="center">
  <a href="https://swarms.world">
    <img src="https://github.com/kyegomez/swarms/blob/master/images/swarmslogobanner.png" style="margin: 15px; max-width: 700px" width="100%" alt="Swarms Logo">
  </a>
</div>

<p align="center">
  <strong>Swarms: Build, orchestrate, and scale enterprise-grade multi-agent AI applications with ease.</strong>
</p>

<p align="center">
    <a href="https://pypi.org/project/swarms/" target="_blank">
        <img alt="Python" src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" />
        <img alt="Version" src="https://img.shields.io/pypi/v/swarms?style=for-the-badge&color=3670A0">
    </a>
    <a href="https://github.com/kyegomez/swarms">
        <img src="https://img.shields.io/github/stars/kyegomez/swarms?style=for-the-badge&logo=github" alt="GitHub stars">
    </a>
</p>

<p align="center">
    <a href="https://twitter.com/swarms_corp/">🐦 Twitter</a>
    <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
    <a href="https://discord.gg/jM3Z6M9uMq">📢 Discord</a>
    <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
    <a href="https://swarms.ai">Swarms Website</a>
    <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
    <a href="https://docs.swarms.world">📙 Documentation</a>
    <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
    <a href="https://swarms.world"> Swarms Marketplace</a>
</p>

## What is Swarms?

Swarms is a production-ready, enterprise-grade framework for building and deploying multi-agent systems. It provides the infrastructure, tools, and architectures you need to create powerful, scalable, and reliable AI applications.  [Check out the original repo!](https://github.com/kyegomez/swarms)

## Key Features

*   **Enterprise-Grade Architecture:** Designed for high availability, scalability, and seamless integration into existing systems.
*   **Multi-Agent Orchestration:** Orchestrate complex workflows with hierarchical agent swarms, parallel processing, and flexible agent networks.
*   **Flexible Integrations:** Supports multiple model providers, custom agent development, and integration with popular frameworks like LangChain, AutoGen, and CrewAI.
*   **Scalability & Performance:** Features concurrent multi-agent processing, intelligent resource management, and horizontal scaling capabilities for optimal performance.
*   **Developer-Friendly Experience:** Provides an intuitive API, comprehensive documentation, an active community, and helpful tools for rapid development and deployment.

## Quickstart: Install and Run Your First Agent

### Installation

*   **Using `pip`:**
    ```bash
    $ pip3 install -U swarms
    ```
*   **Using `uv` (Recommended):**
    ```bash
    # Install uv
    $ curl -LsSf https://astral.sh/uv/install.sh | sh
    # Install swarms using uv
    $ uv pip install swarms
    ```
*   **Using `poetry`:**
    ```bash
    # Install poetry if you haven't already
    $ curl -sSL https://install.python-poetry.org | python3 -
    # Add swarms to your project
    $ poetry add swarms
    ```
*   **From Source:**
    ```bash
    # Clone the repository
    $ git clone https://github.com/kyegomez/swarms.git
    $ cd swarms
    # Install with pip
    $ pip install -e .
    ```

### Environment Configuration

Set your API keys:
```
OPENAI_API_KEY=""
WORKSPACE_DIR="agent_workspace"
ANTHROPIC_API_KEY=""
GROQ_API_KEY=""
```

### Run Your First Agent
```python
from swarms import Agent

# Initialize an agent
agent = Agent(
    model_name="gpt-4o-mini", # Specify the LLM
    max_loops=1,              # Set the number of interactions
    interactive=True,         # Enable interactive mode
)
# Run the agent
agent.run("What are the key benefits of using a multi-agent system?")
```

## Multi-Agent Architectures for Production

Swarms offers a variety of pre-built architectures:

*   **[SequentialWorkflow](https://docs.swarms.world/en/latest/swarms/structs/sequential_workflow/)**: Linear task execution (e.g., data pipelines).
*   **[ConcurrentWorkflow](https://docs.swarms.world/en/latest/swarms/structs/concurrent_workflow/)**: Parallel task execution (e.g., batch processing).
*   **[AgentRearrange](https://docs.swarms.world/en/latest/swarms/structs/agent_rearrange/)**: Dynamic agent relationships (e.g., task distribution).
*   **[GraphWorkflow](https://docs.swarms.world/en/latest/swarms/structs/graph_workflow/)**: DAG-based workflows (e.g., software builds).
*   **[MixtureOfAgents (MoA)](https://docs.swarms.world/en/latest/swarms/structs/moa/)**: Parallel expert agents with aggregation (e.g., complex problem-solving).
*   **[GroupChat](https://docs.swarms.world/en/latest/swarms/structs/group_chat/)**: Conversational agent collaboration (e.g., brainstorming).
*   **[ForestSwarm](https://docs.swarms.world/en/latest/swarms/structs/forest_swarm/)**: Dynamic agent selection (e.g., task routing).
*   **[SpreadSheetSwarm](https://docs.swarms.world/en/latest/swarms/structs/spreadsheet_swarm/)**: Massive parallel operations.
*   **[SwarmRouter](https://docs.swarms.world/en/latest/swarms/structs/swarm_router/)**: Universal orchestrator for all swarm types.

## Documentation

Comprehensive documentation is available at: [docs.swarms.world](https://docs.swarms.world)

## Guides and Walkthroughs

*   **Installation:** Complete setup guide:  [Installation](https://docs.swarms.world/en/latest/swarms/install/install/)
*   **Quickstart:** Run your first swarm in minutes: [Get Started](https://docs.swarms.world/en/latest/swarms/install/quickstart/)
*   **Agent Internals:** Deep dive into agent architecture: [Agent Architecture](https://docs.swarms.world/en/latest/swarms/framework/agents_explained/)
*   **Agent API:** Complete reference for the Agent class: [Agent API](https://docs.swarms.world/en/latest/swarms/structs/agent/)
*   **External Agents:** Integrate with other frameworks: [Integrating External APIs](https://docs.swarms.world/en/latest/swarms/agents/external_party_agents/)
*   **Agents from YAML:** Define agents with YAML files: [Creating Agents from YAML](https://docs.swarms.world/en/latest/swarms/agents/create_agents_yaml/)
*   **Why Swarms?:** Benefits of multi-agent collaboration: [Why Multi-Agent Collaboration is Necessary](https://docs.swarms.world/en/latest/swarms/concept/why/)
*   **Architectures Analysis:** Guide to selecting the right swarm architecture: [Multi-Agent Architectures](https://docs.swarms.world/en/latest/swarms/concept/swarm_architectures/)
*   **Business Problem Guide:** Optimize for your business needs:  [Business Problem Guide](https://docs.swarms.world/en/latest/swarms/concept/swarm_architectures/)
*   **AgentRearrange:** Dynamic agent rearrangement and workflow optimization: [AgentRearrange API](https://docs.swarms.world/en/latest/swarms/structs/agent_rearrange/)

## 🫶 Contribute to Swarms

Join the community and help shape the future of multi-agent AI!

*   **[Contributing Project Board](https://github.com/users/kyegomez/projects/1)**: Find issues to tackle.
*   **[Bug Reports & Feature Requests](https://github.com/kyegomez/swarms/issues)**: Report issues or suggest new features.
*   **[Contribution Guidelines](https://github.com/kyegomez/swarms/blob/master/CONTRIBUTING.md)**: Review our guidelines before contributing.
*   **[Code Cleanliness Guide](https://docs.swarms.world/en/latest/swarms/framework/code_cleanliness/)**:  Learn how to contribute code.
*   **[Discord](https://discord.gg/jM3Z6M9uMq)**: Join the community for discussions and support.

<a href="https://github.com/kyegomez/swarms/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=kyegomez/swarms" />
</a>

## Connect With Us

*   **Documentation:** [docs.swarms.world](https://docs.swarms.world)
*   **Blog:** [Medium](https://medium.com/@kyeg)
*   **Discord:** [Join Discord](https://discord.gg/jM3Z6M9uMq)
*   **Twitter:** [@kyegomez](https://twitter.com/kyegomez)
*   **LinkedIn:** [The Swarm Corporation](https://www.linkedin.com/company/the-swarm-corporation)
*   **YouTube:** [Swarms Channel](https://www.youtube.com/channel/UC9yXyitkbU_WSy7bd_41SqQ)
*   **Events:** [Sign up here](https://lu.ma/5p2jnc2v)
*   **Onboarding:** [Book Session](https://cal.com/swarms/swarms-onboarding-session)

## Citation

If you use **swarms** in your research, please cite the project:
```bibtex
@misc{SWARMS_2022,
  author  = {Gomez, Kye and Pliny and More, Harshal and Swarms Community},
  title   = {{Swarms: Production-Grade Multi-Agent Infrastructure Platform}},
  year    = {2022},
  howpublished = {\url{https://github.com/kyegomez/swarms}},
  note    = {Documentation available at \url{https://docs.swarms.world}},
  version = {latest}
}
```

## License

APACHE
```
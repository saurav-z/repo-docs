<p align="center">
  <a href="https://github.com/lastmile-ai/mcp-agent">
    <img src="https://github.com/user-attachments/assets/c8d059e5-bd56-4ea2-a72d-807fb4897bde" alt="MCP Agent Logo" width="300" />
  </a>
</p>

<p align="center">
  <em>Build powerful, composable AI agents with Model Context Protocol (MCP).</em>
</p>

<p align="center">
  <a href="https://github.com/lastmile-ai/mcp-agent/tree/main/examples" target="_blank"><strong>Examples</strong></a>
  |
  <a href="https://www.anthropic.com/research/building-effective-agents" target="_blank"><strong>Building Effective Agents</strong></a>
  |
  <a href="https://modelcontextprotocol.io/introduction" target="_blank"><strong>MCP</strong></a>
</p>

<p align="center">
<a href="https://docs.mcp-agent.com"><img src="https://img.shields.io/badge/docs-8F?style=flat&link=https%3A%2F%2Fdocs.mcp-agent.com%2F" /><a/>
<a href="https://pypi.org/project/mcp-agent/"><img src="https://img.shields.io/pypi/v/mcp-agent?color=%2334D058&label=pypi" /></a>
<a href="https://github.com/lastmile-ai/mcp-agent/issues"><img src="https://img.shields.io/github/issues-raw/lastmile-ai/mcp-agent" /></a>
<img alt="Pepy Total Downloads" src="https://img.shields.io/pepy/dt/mcp-agent?label=pypi%20%7C%20downloads"/>
<a href="https://github.com/lastmile-ai/mcp-agent/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg"/></a>
<a href="https://lmai.link/discord/mcp-agent"><img src="https://img.shields.io/badge/Discord-%235865F2.svg?logo=discord&logoColor=white" alt="discord"/></a>
</p>

<p align="center">
<a href="https://trendshift.io/repositories/13216" target="_blank"><img src="https://trendshift.io/api/badge/repositories/13216" alt="lastmile-ai%2Fmcp-agent | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

## 🚀 Why Use `mcp-agent`?

**`mcp-agent` is a Python framework designed to simplify the creation of robust and composable AI agents leveraging the power of the Model Context Protocol (MCP).** Built upon Anthropic's "Building Effective Agents" principles, `mcp-agent` provides a lightweight, agent-pattern library that lets you build complex AI applications with ease, and provides a model-agnostic implementation of the OpenAI Swarm pattern. This allows you to easily build agents that can leverage any tool exposed by MCP servers.

## ✨ Key Features

*   **Simplified MCP Integration:** Handles MCP server lifecycle management, making it easy to connect and interact with MCP-compatible services.
*   **Composable Workflows:** Implements Anthropic's agent patterns, allowing you to chain and combine patterns for sophisticated agent behavior.
*   **Model-Agnostic Design:** Flexible and adaptable, allowing you to choose the LLM and tools that best suit your needs.
*   **Multi-Agent Orchestration:** Includes an implementation of OpenAI's Swarm pattern for orchestrating multiple agents.
*   **Human-in-the-Loop:** Supports human input and signaling within workflows for enhanced control and review.

## 🛠️ Core Components

*   **[MCPApp](src/mcp_agent/app.py):** Manages global application state and configuration.
*   **MCP Server Management:** Utilities for easily connecting and interacting with MCP servers, including `gen_client` and `MCPConnectionManager`.
*   **[Agent](src/mcp_agent/agents/agent.py):** Represents an autonomous entity with access to MCP servers, enabling tool use by LLMs.
*   **[AugmentedLLM](src/mcp_agent/workflows/llm/augmented_llm.py):** An LLM enhanced with tools and functionalities via Agents and MCP servers.

## 💡 Workflows

`mcp-agent` provides implementations for key agent patterns from Anthropic's "Building Effective Agents" and OpenAI's Swarm, all exposed as composable `AugmentedLLM` instances.

*   **[AugmentedLLM](src/mcp_agent/workflows/llm/augmented_llm.py):** LLMs enhanced with tools accessible via Agents and MCP servers.
*   **[Parallel](src/mcp_agent/workflows/parallel/parallel_llm.py):** Executes tasks in parallel using multiple sub-agents.
*   **[Router](src/mcp_agent/workflows/router/):** Directs requests to the most relevant category or agent.
*   **[IntentClassifier](src/mcp_agent/workflows/intent_classifier/):** Classifies user intents for improved task handling.
*   **[Evaluator-Optimizer](src/mcp_agent/workflows/evaluator_optimizer/evaluator_optimizer.py):** Refines responses through iterative evaluation and optimization.
*   **[Orchestrator-Workers](src/mcp_agent/workflows/orchestrator/orchestrator.py):** Enables higher-level planning and coordination of sub-agents.
*   **[Swarm](src/mcp_agent/workflows/swarm/swarm.py):** Implements OpenAI's Swarm pattern for multi-agent collaboration.

## 🎬 Examples

Discover the power of `mcp-agent` with these real-world application examples.

### Claude Desktop

Integrate `mcp-agent` applications into MCP clients like Claude Desktop.

#### mcp-agent server

This application exposes agents and workflows that Claude Desktop can invoke to service user requests, showcasing a multi-agent evaluation task.

**Link to code**: [examples/basic/mcp_server_aggregator](./examples/basic/mcp_server_aggregator)

### Streamlit

Deploy `mcp-agent` apps using Streamlit.

#### Gmail agent

Perform read and write actions on Gmail via text prompts.

**Link to code**: [gmail-mcp-server](https://github.com/jasonsum/gmail-mcp-server/blob/add-mcp-agent-streamlit/streamlit_app.py)

#### Simple RAG Chatbot

Build a Q&A chatbot using a Qdrant vector database.

**Link to code**: [examples/usecases/streamlit_mcp_rag_agent](./examples/usecases/streamlit_mcp_rag_agent/)

### Marimo

Run your agents in the reactive Python notebook [Marimo](https://github.com/marimo-team/marimo).
Here's the "file finder" agent from [Quickstart](#quickstart) implemented in Marimo:

<img src="https://github.com/user-attachments/assets/139a95a5-e3ac-4ea7-9c8f-bad6577e8597" width="400"/>

**Link to code**: [examples/usecases/marimo_mcp_basic_agent](./examples/usecases/marimo_mcp_basic_agent/)

### Python

Run `mcp-agent` apps as Python scripts or Jupyter notebooks.

#### Swarm

Demonstrates a multi-agent setup for handling customer service requests in an airline context using the Swarm workflow pattern.

**Link to code**: [examples/workflows/workflow_swarm/main.py](./examples/workflows/workflow_swarm/)

## 🚀 Get Started

### Installation

We recommend using [uv](https://docs.astral.sh/uv/) to manage your Python projects:

```bash
uv add "mcp-agent"
```

Alternatively:

```bash
pip install mcp-agent
```

### Quickstart

> [!TIP]
> The [`examples`](/examples) directory contains many applications to get started with.
> To run an example, clone this repo, then:
>
> ```bash
> cd examples/basic/mcp_basic_agent # Or any other example
> # Option A: secrets YAML
> # cp mcp_agent.secrets.yaml.example mcp_agent.secrets.yaml && edit mcp_agent.secrets.yaml
> # Option B: .env
> cp .env.example .env && edit .env
> uv run main.py
> ```

Here is a basic "finder" agent that uses the fetch and filesystem servers to look up a file, read a blog and write a tweet. [Example link](./examples/basic/mcp_basic_agent/):

<details open>
<summary>finder_agent.py</summary>

```python
import asyncio
import os

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM

app = MCPApp(name="hello_world_agent")

async def example_usage():
    async with app.run() as mcp_agent_app:
        logger = mcp_agent_app.logger
        # This agent can read the filesystem or fetch URLs
        finder_agent = Agent(
            name="finder",
            instruction="""You can read local files or fetch URLs.
                Return the requested information when asked.""",
            server_names=["fetch", "filesystem"], # MCP servers this Agent can use
        )

        async with finder_agent:
            # Automatically initializes the MCP servers and adds their tools for LLM use
            tools = await finder_agent.list_tools()
            logger.info(f"Tools available:", data=tools)

            # Attach an OpenAI LLM to the agent (defaults to GPT-4o)
            llm = await finder_agent.attach_llm(OpenAIAugmentedLLM)

            # This will perform a file lookup and read using the filesystem server
            result = await llm.generate_str(
                message="Show me what's in README.md verbatim"
            )
            logger.info(f"README.md contents: {result}")

            # Uses the fetch server to fetch the content from URL
            result = await llm.generate_str(
                message="Print the first two paragraphs from https://www.anthropic.com/research/building-effective-agents"
            )
            logger.info(f"Blog intro: {result}")

            # Multi-turn interactions by default
            result = await llm.generate_str("Summarize that in a 128-char tweet")
            logger.info(f"Tweet: {result}")

if __name__ == "__main__":
    asyncio.run(example_usage())

```

</details>

<details>
<summary>mcp_agent.config.yaml</summary>

```yaml
execution_engine: asyncio
logger:
  transports: [console] # You can use [file, console] for both
  level: debug
  path: "logs/mcp-agent.jsonl" # Used for file transport
  # For dynamic log filenames:
  # path_settings:
  #   path_pattern: "logs/mcp-agent-{unique_id}.jsonl"
  #   unique_id: "timestamp"  # Or "session_id"
  #   timestamp_format: "%Y%m%d_%H%M%S"

mcp:
  servers:
    fetch:
      command: "uvx"
      args: ["mcp-server-fetch"]
    filesystem:
      command: "npx"
      args:
        [
          "-y",
          "@modelcontextprotocol/server-filesystem",
          "<add_your_directories>",
        ]

openai:
  # Secrets (API keys, etc.) are stored in an mcp_agent.secrets.yaml file which can be gitignored
  default_model: gpt-4o
```

</details>

<details>
<summary>Agent output</summary>
<img width="2398" alt="Image" src="https://github.com/user-attachments/assets/eaa60fdf-bcc6-460b-926e-6fa8534e9089" />
</details>

## ⚙️ Advanced

### Composability

Demonstrates using an Evaluator-Optimizer workflow as the planner LLM inside the Orchestrator workflow.

```python
optimizer = Agent(name="plan_optimizer", server_names=[...], instruction="Generate a plan given an objective ...")
evaluator = Agent(name="plan_evaluator", instruction="Evaluate logic, ordering and precision of plan......")

planner_llm = EvaluatorOptimizerLLM(
    optimizer=optimizer,
    evaluator=evaluator,
    llm_factory=OpenAIAugmentedLLM,
    min_rating=QualityRating.EXCELLENT,
)

orchestrator = Orchestrator(
    llm_factory=AnthropicAugmentedLLM,
    available_agents=[finder_agent, writer_agent, proofreader, fact_checker, style_enforcer],
    planner=planner_llm # It's that simple
)

...
```

### Signaling and Human Input

**Signaling**: The framework can pause/resume tasks. The agent or LLM might “signal” that it needs user input, so the workflow awaits. A developer may signal during a workflow to seek approval or review before continuing with a workflow.

**Human Input**: If an Agent has a `human_input_callback`, the LLM can call a `__human_input__` tool to request user input mid-workflow.

<details>
<summary>Example</summary>

The [Swarm example](examples/workflows/workflow_swarm/main.py) shows this in action.

```python
from mcp_agent.human_input.handler import console_input_callback

lost_baggage = SwarmAgent(
    name="Lost baggage traversal",
    instruction=lambda context_variables: f"""
        {
        FLY_AIR_AGENT_PROMPT.format(
            customer_context=context_variables.get("customer_context", "None"),
            flight_context=context_variables.get("flight_context", "None"),
        )
    }\n Lost baggage policy: policies/lost_baggage_policy.md""",
    functions=[
        escalate_to_agent,
        initiate_baggage_search,
        transfer_to_triage,
        case_resolved,
    ],
    server_names=["fetch", "filesystem"],
    human_input_callback=console_input_callback, # Request input from the console
)
```

</details>

### App Config

Define secrets in an [`mcp_agent.config.yaml`](/schema/mcp-agent.config.schema.json) and define secrets via either a gitignored [`mcp_agent.secrets.yaml`](./examples/basic/mcp_basic_agent/mcp_agent.secrets.yaml.example) or a local [`.env`](./examples/basic/mcp_basic_agent/.env.example). In production, prefer `MCP_APP_SETTINGS_PRELOAD` to avoid writing plaintext secrets to disk.

### MCP server management

mcp-agent makes it trivial to connect to MCP servers. Create an [`mcp_agent.config.yaml`](/schema/mcp-agent.config.schema.json) to define server configuration under the `mcp` section:

```yaml
mcp:
  servers:
    fetch:
      command: "uvx"
      args: ["mcp-server-fetch"]
      description: "Fetch content at URLs from the world wide web"
```

#### [`gen_client`](src/mcp_agent/mcp/gen_client.py)

Manage the lifecycle of an MCP server within an async context manager:

```python
from mcp_agent.mcp.gen_client import gen_client

async with gen_client("fetch") as fetch_client:
    # Fetch server is initialized and ready to use
    result = await fetch_client.list_tools()

# Fetch server is automatically disconnected/shutdown
```

The gen_client function makes it easy to spin up connections to MCP servers.

#### Persistent server connections

In many cases, you want an MCP server to stay online for persistent use (e.g. in a multi-step tool use workflow).
For persistent connections, use:

- [`connect`](<(src/mcp_agent/mcp/gen_client.py)>) and [`disconnect`](src/mcp_agent/mcp/gen_client.py)

```python
from mcp_agent.mcp.gen_client import connect, disconnect

fetch_client = None
try:
     fetch_client = connect("fetch")
     result = await fetch_client.list_tools()
finally:
     disconnect("fetch")
```

- [`MCPConnectionManager`](src/mcp_agent/mcp/mcp_connection_manager.py)
  For even more fine-grained control over server connections, you can use the MCPConnectionManager.

<details>
<summary>Example</summary>

```python
from mcp_agent.context import get_current_context
from mcp_agent.mcp.mcp_connection_manager import MCPConnectionManager

context = get_current_context()
connection_manager = MCPConnectionManager(context.server_registry)

async with connection_manager:
fetch_client = await connection_manager.get_server("fetch") # Initializes fetch server
result = fetch_client.list_tool()
fetch_client2 = await connection_manager.get_server("fetch") # Reuses same server connection

# All servers managed by connection manager are automatically disconnected/shut down
```

</details>

#### MCP Server Aggregator

[`MCPAggregator`](src/mcp_agent/mcp/mcp_aggregator.py) acts as a "server-of-servers".
It provides a single MCP server interface for interacting with multiple MCP servers.
This allows you to expose tools from multiple servers to LLM applications.

<details>
<summary>Example</summary>

```python
from mcp_agent.mcp.mcp_aggregator import MCPAggregator

aggregator = await MCPAggregator.create(server_names=["fetch", "filesystem"])

async with aggregator:
   # combined list of tools exposed by 'fetch' and 'filesystem' servers
   tools = await aggregator.list_tools()

   # namespacing -- invokes the 'fetch' server to call the 'fetch' tool
   fetch_result = await aggregator.call_tool(name="fetch-fetch", arguments={"url": "https://www.anthropic.com/research/building-effective-agents"})

   # no namespacing -- first server in the aggregator exposing that tool wins
   read_file_result = await aggregator.call_tool(name="read_file", arguments={})
```

</details>

## 🤝 Contributing

We welcome all contributions!  Check out the [CONTRIBUTING guidelines](./CONTRIBUTING.md) to learn how you can help.

### Special Mentions

Huge thanks to these community contributors driving the project forward:

-   [Shaun Smith (@evalstate)](https://github.com/evalstate)
-   [Jerron Lim (@StreetLamb)](https://github.com/StreetLamb)
-   [Jason Summer (@jasonsum)](https://github.com/jasonsum)

## 🗺️ Roadmap

We are actively developing features to enhance mcp-agent, including:

*   **Durable Execution:** Support workflow pausing, resuming, and state serialization via [Temporal](src/mcp_agent/executor/temporal.py).
*   **Memory:** Implementing long-term memory capabilities.
*   **Streaming:** Add streaming listeners for real-time progress updates.
*   **Expanded MCP Support:**  Enhancing MCP capabilities beyond tool calls, including resources, prompts, and notifications.

## ❓ FAQs

### What are the core benefits of using mcp-agent?

mcp-agent offers a streamlined approach to building AI agents using capabilities exposed by **MCP** (Model Context Protocol) servers.

**MCP** is a low-level standard.  `mcp-agent` simplifies that, handling the complexities of connecting to servers, managing LLMs, human input, and persisting state.  This allows developers to focus on building AI agents.

Key benefits:

*   🤝 **Interoperability**: Ensures that tools exposed by any number of MCP servers can seamlessly plug into your agents.
*   ⛓️ **Composability & Customizability**: Built on well-defined workflows that can be combined and fully customized.
*   💻 **Programmatic Control Flow**: Simplified with standard coding structures (e.g., `if` statements, `while` loops), removing the need for complex graph structures.
*   🖐️ **Human Input & Signals**: Support for workflows to pause for external signals, such as human input.

### Do you need an MCP client to use mcp-agent?

No!  `mcp-agent` handles MCP client creation, which lets you use it anywhere.

**Here's how you can set up your mcp-agent application:**

*   **MCP-Agent Server**: Expose `mcp-agent` applications as MCP servers.
*   **MCP Client or Host**: Embed `mcp-agent` in an existing MCP client.
*   **Standalone**: Use `mcp-agent` applications independently (as in the [examples](/examples/)).

### Tell me a fun fact

I debated naming this project _silsila_ (سلسلہ), which means chain of events in Urdu. mcp-agent is more matter-of-fact, but there's still an easter egg in the project paying homage to silsila.
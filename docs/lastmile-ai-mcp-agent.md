<p align="center">
  <a href="https://github.com/lastmile-ai/mcp-agent"><img src="https://github.com/user-attachments/assets/c8d059e5-bd56-4ea2-a72d-807fb4897bde" alt="MCP Agent Logo" width="300" /></a>
</p>

<p align="center">
  <em>**Build powerful, composable AI agents with Model Context Protocol (MCP) using mcp-agent, the easiest way to leverage MCP.**</em>
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

## 🚀 **mcp-agent: Build Robust AI Agents with Model Context Protocol**

`mcp-agent` is a Python framework designed to simplify building powerful AI agents that leverage the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/introduction).  It provides a clean, composable approach for creating agents using simple patterns, making it easier to build production-ready AI applications.  Get started with `mcp-agent` on [GitHub](https://github.com/lastmile-ai/mcp-agent)!

### 🔑 Key Features

*   **Simplified MCP Integration:** Handles MCP server connection management, so you can focus on agent logic.
*   **Composable Agent Patterns:** Implements the patterns from Anthropic's "Building Effective Agents" in a modular way.
*   **Model-Agnostic Design:**  Works with various LLMs and MCP servers.
*   **Multi-Agent Orchestration:**  Includes an implementation of OpenAI's Swarm pattern for collaborative agent workflows.
*   **Easy Setup:**  Simple installation and quickstart examples to get you up and running fast.

### 💡 Why Use `mcp-agent`?

`mcp-agent` is the only framework built specifically for MCP.  It's a lightweight agent pattern library and provides:

*   **Interoperability**: Integrates seamlessly with any MCP server to leverage its exposed tools.
*   **Composability & Customization**: Enables complex, compound workflows through well-defined, composable building blocks.
*   **Programmatic Control**:  Employs simple code, avoiding complex graph-based approaches.
*   **Human-in-the-Loop**: Allows for agent pausing and human input via tool calls.

### 🚦 Get Started

Easily manage your Python projects using [uv](https://docs.astral.sh/uv/):

```bash
uv add "mcp-agent"
```

Or, install with pip:

```bash
pip install mcp-agent
```

#### Quickstart:

Explore the [`examples`](/examples) directory for hands-on experience. To run an example, clone the repository and follow these steps:

```bash
cd examples/basic/mcp_basic_agent # Or any other example
cp mcp_agent.secrets.yaml.example mcp_agent.secrets.yaml # Update API keys
uv run main.py
```

Here's a basic "finder" agent example:

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

## 📚 Table of Contents

-   [Core Concepts](#core-components)
-   [Workflows Patterns](#workflows)
    -   [Augmented LLM](#augmentedllm)
    -   [Parallel](#parallel)
    -   [Router](#router)
    -   [Intent-Classifier](#intentclassifier)
    -   [Evaluator-Optimizer](#evaluator-optimizer)
    -   [Orchestrator-Workers](#orchestrator-workers)
    -   [Swarm](#swarm)
-   [Advanced](#advanced)
    -   [Composability](#composability)
    -   [Signaling and Human Input](#signaling-and-human-input)
    -   [App Config](#app-config)
    -   [MCP Server Management](#mcp-server-management)
-   [Examples](#examples)
    -   [Claude Desktop](#claude-desktop)
    -   [Streamlit](#streamlit)
    -   [Marimo](#marimo)
    -   [Python](#python)
-   [Contributing](#contributing)
-   [Roadmap](#roadmap)
-   [FAQs](#faqs)

## 🧩 Core Components

These are the foundational elements of `mcp-agent`:

*   **[MCPApp](./src/mcp_agent/app.py)**: Global state and application configuration.
*   **MCP Server Management**:  [`gen_client`](./src/mcp_agent/mcp/gen_client.py) and [`MCPConnectionManager`](./src/mcp_agent/mcp/mcp_connection_manager.py) for easy MCP server connections.
*   **[Agent](./src/mcp_agent/agents/agent.py)**: Entities with access to MCP servers, exposed as tools to an LLM, with a defined name and purpose.
*   **[AugmentedLLM](./src/mcp_agent/workflows/llm/augmented_llm.py)**: An LLM enhanced with tools provided by MCP servers.

## ⚙️ Workflows

`mcp-agent` implements the core agent patterns from Anthropic's "Building Effective Agents" and the OpenAI Swarm pattern. Each workflow is model-agnostic and built on the `AugmentedLLM` interface for easy composition.

### Augmented LLM

[AugmentedLLM](./src/mcp_agent/workflows/llm/augmented_llm.py) enhances an LLM with access to MCP servers and agent tools.  LLM providers implement the `AugmentedLLM` interface.

*   `generate`: Generates messages using tools from MCP servers, possibly over multiple iterations.
*   `generate_str`: Returns results as a string.
*   `generate_structured`:  Uses [Instructor](https://github.com/instructor-ai/instructor) to return results as Pydantic models.
*   Supports short and long-term memory for context.

<details>
<summary>Example</summary>

```python
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm_anthropic import AnthropicAugmentedLLM

finder_agent = Agent(
    name="finder",
    instruction="You are an agent with filesystem + fetch access. Return the requested file or URL contents.",
    server_names=["fetch", "filesystem"],
)

async with finder_agent:
   llm = await finder_agent.attach_llm(AnthropicAugmentedLLM)

   result = await llm.generate_str(
      message="Print the first 2 paragraphs of https://www.anthropic.com/research/building-effective-agents",
      # Can override model, tokens and other defaults
   )
   logger.info(f"Result: {result}")

   # Multi-turn conversation
   result = await llm.generate_str(
      message="Summarize those paragraphs in a 128 character tweet",
   )
   logger.info(f"Result: {result}")
```

</details>

### [Parallel](src/mcp_agent/workflows/parallel/parallel_llm.py)

![Parallel workflow (Image credit: Anthropic)](https://www.anthropic.com/_next/image?url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2F406bb032ca007fd1624f261af717d70e6ca86286-2401x1000.png&w=3840&q=75)

Executes tasks concurrently using sub-agents, then merges the results.

> [!NOTE]
>
> **[Link to full example](examples/workflows/workflow_parallel/main.py)**

<details>
<summary>Example</summary>

```python
proofreader = Agent(name="proofreader", instruction="Review grammar...")
fact_checker = Agent(name="fact_checker", instruction="Check factual consistency...")
style_enforcer = Agent(name="style_enforcer", instruction="Enforce style guidelines...")

grader = Agent(name="grader", instruction="Combine feedback into a structured report.")

parallel = ParallelLLM(
    fan_in_agent=grader,
    fan_out_agents=[proofreader, fact_checker, style_enforcer],
    llm_factory=OpenAIAugmentedLLM,
)

result = await parallel.generate_str("Student short story submission: ...", RequestParams(model="gpt4-o"))
```

</details>

### [Router](src/mcp_agent/workflows/router/)

![Router workflow (Image credit: Anthropic)](https://www.anthropic.com/_next/image?url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2F5c0c0e9fe4def0b584c04d37849941da55e5e71c-2401x1000.png&w=3840&q=75)

Routes an input to the most relevant category, which can be an Agent, MCP server, or regular function.

mcp-agent provides:

*   [`EmbeddingRouter`](src/mcp_agent/workflows/router/router_embedding.py):  Uses embedding models for classification.
*   [`LLMRouter`](src/mcp_agent/workflows/router/router_llm.py): Uses LLMs for classification.

> [!NOTE]
>
> **[Link to full example](examples/workflows/workflow_router/main.py)**

<details>
<summary>Example</summary>

```python
def print_hello_world:
     print("Hello, world!")

finder_agent = Agent(name="finder", server_names=["fetch", "filesystem"])
writer_agent = Agent(name="writer", server_names=["filesystem"])

llm = OpenAIAugmentedLLM()
router = LLMRouter(
    llm=llm,
    agents=[finder_agent, writer_agent],
    functions=[print_hello_world],
)

results = await router.route( # Also available: route_to_agent, route_to_server
    request="Find and print the contents of README.md verbatim",
    top_k=1
)
chosen_agent = results[0].result
async with chosen_agent:
    ...
```

</details>

### [IntentClassifier](src/mcp_agent/workflows/intent_classifier/)

Similar to a Router, it identifies the top intents that match a given input. Uses embedding or LLM-based classifiers.

### [Evaluator-Optimizer](src/mcp_agent/workflows/evaluator_optimizer/evaluator_optimizer.py)

![Evaluator-optimizer workflow (Image credit: Anthropic)](https://www.anthropic.com/_next/image?url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2F14f51e6406ccb29e695da48b17017e899a6119c7-2401x1000.png&w=3840&q=75)

An Optimizer refines a response, while an Evaluator critiques it, iterating until quality criteria are met.

> [!NOTE]
>
> **[Link to full example](examples/workflows/workflow_evaluator_optimizer/main.py)**

<details>
<summary>Example</summary>

```python
optimizer = Agent(name="cover_letter_writer", server_names=["fetch"], instruction="Generate a cover letter ...")
evaluator = Agent(name="critiquer", instruction="Evaluate clarity, specificity, relevance...")

eo_llm = EvaluatorOptimizerLLM(
    optimizer=optimizer,
    evaluator=evaluator,
    llm_factory=OpenAIAugmentedLLM,
    min_rating=QualityRating.EXCELLENT, # Keep iterating until the minimum quality bar is reached
)

result = await eo_llm.generate_str("Write a job cover letter for an AI framework developer role at LastMile AI.")
print("Final refined cover letter:", result)
```

</details>

### [Orchestrator-workers](src/mcp_agent/workflows/orchestrator/orchestrator.py)

![Orchestrator workflow (Image credit: Anthropic)](https://www.anthropic.com/_next/image?url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2F8985fc683fae4780fb34eab1365ab78c7e51bc8e-2401x1000.png&w=3840&q=75)

A higher-level LLM creates a plan, assigns tasks to sub-agents, and integrates the results, parallelizing where possible.

> [!NOTE]
>
> **[Link to full example](examples/workflows/workflow_orchestrator_worker/main.py)**

<details>
<summary>Example</summary>

```python
finder_agent = Agent(name="finder", server_names=["fetch", "filesystem"])
writer_agent = Agent(name="writer", server_names=["filesystem"])
proofreader = Agent(name="proofreader", ...)
fact_checker = Agent(name="fact_checker", ...)
style_enforcer = Agent(name="style_enforcer", instructions="Use APA style guide from ...", server_names=["fetch"])

orchestrator = Orchestrator(
    llm_factory=AnthropicAugmentedLLM,
    available_agents=[finder_agent, writer_agent, proofreader, fact_checker, style_enforcer],
)

task = "Load short_story.md, evaluate it, produce a graded_report.md with multiple feedback aspects."
result = await orchestrator.generate_str(task, RequestParams(model="gpt-4o"))
print(result)
```

</details>

### [Swarm](src/mcp_agent/workflows/swarm/swarm.py)

Implements the OpenAI Swarm pattern.

<img src="https://github.com/openai/swarm/blob/main/assets/swarm_diagram.png?raw=true" width=500 />

The mcp-agent Swarm works seamlessly with MCP servers and is exposed as an `AugmentedLLM` for easy integration.

> [!NOTE]
>
> **[Link to full example](examples/workflows/workflow_swarm/main.py)**

<details>
<summary>Example</summary>

```python
triage_agent = SwarmAgent(...)
flight_mod_agent = SwarmAgent(...)
lost_baggage_agent = SwarmAgent(...)

# The triage agent decides whether to route to flight_mod_agent or lost_baggage_agent
swarm = AnthropicSwarm(agent=triage_agent, context_variables={...})

test_input = "My bag was not delivered!"
result = await swarm.generate_str(test_input)
print("Result:", result)
```

</details>

## 💡 Advanced Features

### Composability

Compose workflows, like using an [Evaluator-Optimizer](#evaluator-optimizer) within an [Orchestrator](#orchestrator-workers).  This is straightforward since workflows are implemented as `AugmentedLLM` instances.

<details>
<summary>Example</summary>

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

</details>

### Signaling and Human Input

*   **Signaling:**  Workflows can be paused/resumed.  Agents or LLMs can "signal" for user input.  Developers can also signal mid-workflow to seek approvals.
*   **Human Input:**  If an Agent has a `human_input_callback`, the LLM can call a `__human_input__` tool to request user input during a workflow.

<details>
<summary>Example</summary>

See the [Swarm example](examples/workflows/workflow_swarm/main.py) for an example.

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

Customize your application using [`mcp_agent.config.yaml`](/schema/mcp-agent.config.schema.json) and a gitignored [`mcp_agent.secrets.yaml`](./examples/basic/mcp_basic_agent/mcp_agent.secrets.yaml.example). Configure logging, execution settings, LLM provider APIs, and MCP server details.

### MCP Server Management

`mcp-agent` simplifies connecting to MCP servers.  Configure servers in your `mcp_agent.config.yaml` under the `mcp` section.

```yaml
mcp:
  servers:
    fetch:
      command: "uvx"
      args: ["mcp-server-fetch"]
      description: "Fetch content at URLs from the world wide web"
```

#### [`gen_client`](src/mcp_agent/mcp/gen_client.py)

Manage MCP server lifecycles using an async context manager:

```python
from mcp_agent.mcp.gen_client import gen_client

async with gen_client("fetch") as fetch_client:
    # Fetch server is initialized and ready to use
    result = await fetch_client.list_tools()

# Fetch server is automatically disconnected/shutdown
```

#### Persistent Server Connections

For persistent server usage, use:

*   [`connect`](<(src/mcp_agent/mcp/gen_client.py)>) and [`disconnect`](src/mcp_agent/mcp/gen_client.py)

```python
from mcp_agent.mcp.gen_client import connect, disconnect

fetch_client = None
try:
     fetch_client = connect("fetch")
     result = await fetch_client.list_tools()
finally:
     disconnect("fetch")
```

*   [`MCPConnectionManager`](src/mcp_agent/mcp/mcp_connection_manager.py)  for fine-grained control.

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

[`MCPAggregator`](src/mcp_agent/mcp/mcp_aggregator.py) aggregates multiple MCP servers, exposing a single interface.

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

## 🧑‍💻 Examples

Explore how to build AI applications with `mcp-agent`:

### Claude Desktop

Integrate `mcp-agent` apps into MCP clients like Claude Desktop.  See: [examples/basic/mcp_server_aggregator](./examples/basic/mcp_server_aggregator)

### Streamlit

Deploy `mcp-agent` apps using Streamlit.  See:

*   [gmail-mcp-server](https://github.com/jasonsum/gmail-mcp-server/blob/add-mcp-agent-streamlit/streamlit_app.py)
*   [examples/usecases/streamlit_mcp_rag_agent](./examples/usecases/streamlit_mcp_rag_agent/)

### Marimo

Build reactive Python notebooks with `mcp-agent` using Marimo. See: [examples/usecases/marimo_mcp_basic_agent](./examples/usecases/marimo_mcp_basic_agent/)

### Python

Write `mcp-agent` apps as Python scripts or Jupyter notebooks.  See: [examples/workflows/workflow_swarm](./examples/workflows/workflow_swarm/)

## 🤝 Contributing

We welcome your contributions!  See the [CONTRIBUTING guidelines](./CONTRIBUTING.md) for details.

### Special Mentions

Thank you to our community contributors:

*   [Shaun Smith (@evalstate)](https://github.com/evalstate)
*   [Jerron Lim (@StreetLamb)](https://github.com/StreetLamb)
*   [Jason Summer (@jasonsum)](https://github.com/jasonsum)

## 🗺️ Roadmap

(A detailed roadmap is coming, driven by your feedback.) Current priorities:

*   **Durable Execution:**  Allow workflows to pause/resume and serialize state. Integrating [Temporal](./src/mcp_agent/executor/temporal.py).
*   **Memory:**  Add support for long-term memory.
*   **Streaming:**  Implement streaming listeners for iterative progress.
*   **Expanded MCP capabilities:** Support resources, prompts, and notifications.

## ❓ FAQs

### What are the core benefits of using mcp-agent?

`mcp-agent` simplifies building AI agents using MCP servers.  Key advantages:

*   **Interoperability**: Connects to tools from any MCP server.
*   **Composability & Customizability**: Build complex workflows with modular patterns.
*   **Programmatic Control**: Write code instead of managing complex graphs.
*   **Human Input & Signals**: Supports pausing workflows for user interaction.

### Do you need an MCP client to use mcp-agent?

No, `mcp-agent` handles MCPClient creation, enabling use outside of MCP hosts.

#### Setup options:

*   **MCP-Agent Server**: Expose `mcp-agent` apps as MCP servers (server-of-servers).
*   **MCP Client or Host**: Embed `mcp-agent` in an MCP client.
*   **Standalone**: Use `mcp-agent` independently (examples are standalone).

### Tell me a fun fact

The project was almost named _silsila_ (Urdu for "chain of events"), but there's an Easter egg referencing it in the code.
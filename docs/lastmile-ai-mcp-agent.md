<p align="center">
  <a href="https://github.com/lastmile-ai/mcp-agent">
    <img src="https://github.com/user-attachments/assets/c8d059e5-bd56-4ea2-a72d-807fb4897bde" alt="mcp-agent Logo" width="300" />
  </a>
</p>

<p align="center">
  <em>Build Powerful, Composable AI Agents with the Model Context Protocol.</em>
</p>

<p align="center">
  <a href="https://github.com/lastmile-ai/mcp-agent/tree/main/examples" target="_blank"><strong>Examples</strong></a>
  |
  <a href="https://www.anthropic.com/research/building-effective-agents" target="_blank"><strong>Building Effective Agents</strong></a>
  |
  <a href="https://modelcontextprotocol.io/introduction" target="_blank"><strong>MCP</strong></a>
</p>

<p align="center">
  <a href="https://docs.mcp-agent.com"><img src="https://img.shields.io/badge/docs-8F?style=flat&link=https%3A%2F%2Fdocs.mcp-agent.com%2F" /></a>
  <a href="https://pypi.org/project/mcp-agent/"><img src="https://img.shields.io/pypi/v/mcp-agent?color=%2334D058&label=pypi" /></a>
  <a href="https://github.com/lastmile-ai/mcp-agent/issues"><img src="https://img.shields.io/github/issues-raw/lastmile-ai/mcp-agent" /></a>
  <a href="https://lmai.link/discord/mcp-agent"><img src="https://shields.io/discord/1089284610329952357" alt="discord" /></a>
  <img alt="Pepy Total Downloads" src="https://img.shields.io/pepy/dt/mcp-agent?label=pypi%20%7C%20downloads"/>
  <a href="https://github.com/lastmile-ai/mcp-agent/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/mcp-agent" /></a>
  <a href="https://trendshift.io/repositories/13216" target="_blank"><img src="https://trendshift.io/api/badge/repositories/13216" alt="lastmile-ai%2Fmcp-agent | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

## mcp-agent: Build Intelligent Agents with Ease

**mcp-agent** is a Python framework designed to simplify the creation of AI agents, leveraging the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/introduction). It provides a composable and flexible architecture, inspired by Anthropic's "Building Effective Agents," to build robust and production-ready AI applications.  [Explore the mcp-agent repository](https://github.com/lastmile-ai/mcp-agent).

**Key Features:**

*   ✅ **Simplified MCP Integration:** Handles MCP server connections, allowing you to focus on agent logic.
*   ✅ **Composable Workflows:** Implements and extends patterns from "Building Effective Agents" in a modular way.
*   ✅ **Model Agnostic:** Compatible with various LLMs.
*   ✅ **Multi-Agent Orchestration:** Includes OpenAI's Swarm pattern for model-agnostic multi-agent coordination.
*   ✅ **Human-in-the-Loop Support:** Allows for user input and approval during agent workflows.
*   ✅ **Easy Setup:** Simple installation via pip or uv.

## Get Started

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
> The [`examples`](/examples) directory has several example applications to get started with.
> To run an example, clone this repo, then:
>
> ```bash
> cd examples/basic/mcp_basic_agent # Or any other example
> cp mcp_agent.secrets.yaml.example mcp_agent.secrets.yaml # Update API keys
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

## Table of Contents

-   [Why use mcp-agent?](#why-use-mcp-agent)
-   [Examples](#examples)
    -   [Claude Desktop](#claude-desktop)
    -   [Streamlit](#streamlit)
        -   [Gmail Agent](#gmail-agent)
        -   [RAG](#simple-rag-chatbot)
    -   [Marimo](#marimo)
    -   [Python](#python)
        -   [Swarm (CLI)](#swarm)
-   [Core Concepts](#core-components)
-   [Workflows Patterns](#workflows)
    -   [Augmented LLM](#augmentedllm)
    -   [Parallel](#parallel)
    -   [Router](#router)
    -   [Intent-Classifier](#intentclassifier)
    -   [Orchestrator-Workers](#orchestrator-workers)
    -   [Evaluator-Optimizer](#evaluator-optimizer)
    -   [Swarm](#swarm-1)
-   [Advanced](#advanced)
    -   [Composing multiple workflows](#composability)
    -   [Signaling and Human input](#signaling-and-human-input)
    -   [App Config](#app-config)
    -   [MCP Server Management](#mcp-server-management)
-   [Contributing](#contributing)
-   [Roadmap](#roadmap)
-   [FAQs](#faqs)

## Why Use mcp-agent?

mcp-agent offers a streamlined, lightweight approach to building robust AI agents. It is tailored for the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/introduction), allowing you to create intelligent applications that can leverage any MCP-compatible service with ease.

## Examples

Discover what you can build with mcp-agent. From multi-agent workflows to RAG pipelines, the possibilities are vast.

### Claude Desktop

Integrate mcp-agent applications with MCP clients like Claude Desktop.

#### mcp-agent Server

This example wraps an mcp-agent app within an MCP server, exposing agents and workflows for interaction with Claude Desktop.

https://github.com/user-attachments/assets/7807cffd-dba7-4f0c-9c70-9482fd7e0699

**Details**: This demo showcases a multi-agent evaluation task, where agents evaluate an input poem, and an aggregator summarizes the findings.

**Link to code**: [examples/basic/mcp_server_aggregator](./examples/basic/mcp_server_aggregator)

> \[!NOTE]
>
> Huge thanks to [Jerron Lim (@StreetLamb)](https://github.com/StreetLamb)
> for developing and contributing this example!

### Streamlit

Deploy mcp-agent apps using Streamlit.

#### Gmail agent

This app manages Gmail, performing read, write, and other actions with text prompts.

https://github.com/user-attachments/assets/54899cac-de24-4102-bd7e-4b2022c956e3

**Link to code**: [gmail-mcp-server](https://github.com/jasonsum/gmail-mcp-server/blob/add-mcp-agent-streamlit/streamlit_app.py)

> \[!NOTE]
>
> Huge thanks to [Jason Summer (@jasonsum)](https://github.com/jasonsum)
> for developing and contributing this example!

#### Simple RAG Chatbot

A Q&A bot utilizing a Qdrant vector database (via MCP server) for text-based question answering.

https://github.com/user-attachments/assets/f4dcd227-cae9-4a59-aa9e-0eceeb4acaf4

**Link to code**: [examples/usecases/streamlit_mcp_rag_agent](./examples/usecases/streamlit_mcp_rag_agent/)

> \[!NOTE]
>
> Huge thanks to [Jerron Lim (@StreetLamb)](https://github.com/StreetLamb)
> for developing and contributing this example!

### Marimo

Build interactive AI applications with [Marimo](https://github.com/marimo-team/marimo). Here's the "file finder" agent:

<img src="https://github.com/user-attachments/assets/139a95a5-e3ac-4ea7-9c8f-bad6577e8597" width="400"/>

**Link to code**: [examples/usecases/marimo_mcp_basic_agent](./examples/usecases/marimo_mcp_basic_agent/)

> \[!NOTE]
>
> Huge thanks to [Akshay Agrawal (@akshayka)](https://github.com/akshayka)
> for developing and contributing this example!

### Python

Create mcp-agent applications using Python scripts or Jupyter notebooks.

#### Swarm

This multi-agent example handles customer service requests in an airline context using the Swarm workflow. Agents can triage requests and manage flight modifications, cancellations, and lost baggage cases.

https://github.com/user-attachments/assets/b314d75d-7945-4de6-965b-7f21eb14a8bd

**Link to code**: [examples/workflows/workflow_swarm](./examples/workflows/workflow_swarm/)

## Core Components

The mcp-agent framework's core components include:

*   **[MCPApp](./src/mcp_agent/app.py)**: Manages the global state and app configuration.
*   **MCP Server Management**: Uses [`gen_client`](./src/mcp_agent/mcp/gen_client.py) and [`MCPConnectionManager`](./src/mcp_agent/mcp/mcp_connection_manager.py) for streamlined connections.
*   **[Agent](./src/mcp_agent/agents/agent.py)**: Agents can access and utilize MCP servers.
*   **[AugmentedLLM](./src/mcp_agent/workflows/llm/augmented_llm.py)**: Enhanced LLMs that integrate tools provided by MCP servers.

## Workflows

mcp-agent provides implementations for Anthropic’s [Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) patterns and the OpenAI [Swarm](https://github.com/openai/swarm) pattern. Each pattern is model-agnostic and exposed as an `AugmentedLLM`, fostering composability.

### AugmentedLLM

[AugmentedLLM](./src/mcp_agent/workflows/llm/augmented_llm.py) LLMs, that have access to MCP servers and functions via Agents.

These implement the AugmentedLLM interface with 3 key functions:

-   `generate`: Generates messages based on a prompt, iteratively using tool calls if necessary.
-   `generate_str`: Returns the result of `generate` as a string.
-   `generate_structured`: Employs [Instructor](https://github.com/instructor-ai/instructor) to return generated results as a Pydantic model.

Additionally, `AugmentedLLM` includes memory management for both short- and long-term history.

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

This pattern distributes tasks across multiple sub-agents and consolidates the results. Each subtask, as well as the overall workflow, uses an AugmentedLLM.

> \[!NOTE]
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

Directs an input to the most relevant categories (Agents, MCP servers, or functions).

mcp-agent provides:

-   [`EmbeddingRouter`](src/mcp_agent/workflows/router/router_embedding.py): Uses embeddings for classification
-   [`LLMRouter`](src/mcp_agent/workflows/router/router_llm.py): Utilizes LLMs for classification

> \[!NOTE]
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

This pattern identifies the most relevant Intents based on a given input. mcp-agent offers both [embedding](src/mcp_agent/workflows/intent_classifier/intent_classifier_embedding.py) and [LLM-based](src/mcp_agent/workflows/intent_classifier/intent_classifier_llm.py) intent classifiers.

### [Evaluator-Optimizer](src/mcp_agent/workflows/evaluator_optimizer/evaluator_optimizer.py)

![Evaluator-optimizer workflow (Image credit: Anthropic)](https://www.anthropic.com/_next/image?url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2F14f51e6406ccb29e695da48b17017e899a6119c7-2401x1000.png&w=3840&q=75)

An "optimizer" LLM refines a response, and an "evaluator" LLM critiques it until a specific quality threshold is met.

> \[!NOTE]
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

A high-level LLM creates a plan and assigns sub-agents, then synthesizes the results. This workflow automatically parallelizes independent steps.

> \[!NOTE]
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

mcp-agent provides a model-agnostic implementation of OpenAI's [Swarm](https://github.com/openai/swarm) pattern.

<img src="https://github.com/openai/swarm/blob/main/assets/swarm_diagram.png?raw=true" width=500 />

This Swarm pattern seamlessly integrates with MCP servers and is exposed as an `AugmentedLLM`.

> \[!NOTE]
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

## Advanced

### Composability

Combine workflows, such as integrating an [Evaluator-Optimizer](#evaluator-optimizer) as the planner within an [Orchestrator](#orchestrator-workers) workflow.

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

Configure your mcp-agent applications with [`mcp_agent.config.yaml`](/schema/mcp-agent.config.schema.json) and a gitignored [`mcp_agent.secrets.yaml`](./examples/basic/mcp_basic_agent/mcp_agent.secrets.yaml.example). This enables customization of logging, execution, LLM APIs, and MCP server settings.

### MCP Server Management

Effortlessly connect to MCP servers with mcp-agent. The `mcp` section in your configuration file, [`mcp_agent.config.yaml`](/schema/mcp-agent.config.schema.json), allows defining server settings:

```yaml
mcp:
  servers:
    fetch:
      command: "uvx"
      args: ["mcp-server-fetch"]
      description: "Fetch content at URLs from the world wide web"
```

#### [`gen_client`](src/mcp_agent/mcp/gen_client.py)

Manage MCP server lifecycles within an async context manager:

```python
from mcp_agent.mcp.gen_client import gen_client

async with gen_client("fetch") as fetch_client:
    # Fetch server is initialized and ready to use
    result = await fetch_client.list_tools()

# Fetch server is automatically disconnected/shutdown
```

The gen_client function makes it easy to spin up connections to MCP servers.

#### Persistent server connections

For persistent MCP server usage, use:

-   [`connect`](<(src/mcp_agent/mcp/gen_client.py)>) and [`disconnect`](src/mcp_agent/mcp/gen_client.py)

```python
from mcp_agent.mcp.gen_client import connect, disconnect

fetch_client = None
try:
     fetch_client = connect("fetch")
     result = await fetch_client.list_tools()
finally:
     disconnect("fetch")
```

-   [`MCPConnectionManager`](src/mcp_agent/mcp/mcp_connection_manager.py)

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

[`MCPAggregator`](src/mcp_agent/mcp/mcp_aggregator.py) acts as a server-of-servers. It presents a single MCP server interface to interact with numerous MCP servers. This enables exposure of multiple server tools to LLM apps.

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

## Contributing

We welcome contributions. Please see our [CONTRIBUTING guidelines](./CONTRIBUTING.md).

### Special Mentions

Thank you to our community contributors:

*   [Shaun Smith (@evalstate)](https://github.com/evalstate)
*   [Jerron Lim (@StreetLamb)](https://github.com/StreetLamb)
*   [Jason Summer (@jasonsum)](https://github.com/jasonsum)

## Roadmap

A detailed roadmap (guided by your feedback) will be added soon. Current priorities include:

*   **Durable Execution**: Allow workflows to pause/resume and serialize state. Integration with [Temporal](./src/mcp_agent/executor/temporal.py) is planned.
*   **Memory**: Implement long-term memory support.
*   **Streaming**: Implement streaming listeners for iterative progress.
*   **Additional MCP capabilities**: Expanding beyond tool calls, to support:
    *   Resources
    *   Prompts
    *   Notifications

## FAQs

### What are the key benefits of using mcp-agent?

mcp-agent simplifies AI agent creation by leveraging capabilities exposed by **MCP** (Model Context Protocol) servers.

**Key benefits:**

*   🤝 **Interoperability**: Enables seamless integration with MCP-compatible tools and services.
*   ⛓️ **Composability & Customizability**: Built with modular workflows and full customization.
*   💻 **Programmatic Control Flow**: Develop agents with code instead of complex graphs.
*   🖐️ **Human Input & Signals**: Enables user interaction within agent workflows.

### Do you need an MCP client to use mcp-agent?

No, mcp-agent handles MCPClient creation. This allows you to use MCP servers anywhere.

Here's how to set up your mcp-agent application:

*   **MCP-Agent Server**: Expose mcp-agent applications as MCP servers (see [example](./examples/mcp_agent_server)).
*   **MCP Client or Host**: Integrate mcp-agent within an existing MCP client.
*   **Standalone**: Use mcp-agent independently.

### Tell me a fun fact

I debated naming this project _silsila_ (سلسلہ), which means chain of events in Urdu. mcp-agent is more matter-of-fact, but there's still an easter egg in the project paying homage to silsila.
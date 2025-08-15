<p align="center">
  <a href="https://github.com/lastmile-ai/mcp-agent">
    <img src="https://github.com/user-attachments/assets/c8d059e5-bd56-4ea2-a72d-807fb4897bde" alt="MCP Agent Logo" width="300" />
  </a>
</p>

<p align="center">
  <em>Build Powerful, Composable AI Agents with Model Context Protocol (MCP) - <a href="https://github.com/lastmile-ai/mcp-agent">Explore mcp-agent</a></em>
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

## Core Features of mcp-agent

*   ✅ **Effortless MCP Integration**: Simplified management of MCP server connections.
*   🧩 **Composable Workflows**: Build complex agents using modular and interchangeable patterns.
*   🚀 **Anthropic's "Building Effective Agents" Implementation**: Leverage proven patterns for production-ready agents.
*   🤖 **Model-Agnostic Approach**:  Works with various LLMs.
*   💡 **OpenAI Swarm Pattern**: Easily orchestrate multi-agent systems.

## Overview

**mcp-agent** is a Python framework designed to simplify the creation of AI agents using the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/introduction).  Inspired by Anthropic's research on [Building Effective Agents](https://www.anthropic.com/research/building-effective-agents), mcp-agent offers a clean, composable way to build robust and production-ready AI applications.  It streamlines MCP server management, implements key agent patterns, and supports multi-agent orchestration (including an OpenAI Swarm implementation) in a model-agnostic manner.  The project is in early development, and contributions and feedback are welcome.

## Get Started

Install `mcp-agent` using `uv` or `pip`:

```bash
uv add "mcp-agent"
```

or

```bash
pip install mcp-agent
```

> [!TIP]
> Dive into the [`examples`](/examples) directory for practical application examples. To run an example, clone the repository and follow the instructions.

### Quickstart: Building a "Finder" Agent

Below is a sample "finder" agent, demonstrating how to fetch and process information using the `fetch` and `filesystem` servers:

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

-   [Why Use mcp-agent?](#why-use-mcp-agent)
-   [Examples](#examples)
    -   [Claude Desktop](#claude-desktop)
    -   [Streamlit](#streamlit)
        -   [Gmail Agent](#gmail-agent)
        -   [RAG Chatbot](#simple-rag-chatbot)
    -   [Marimo](#marimo)
    -   [Python](#python)
        -   [Swarm (CLI)](#swarm)
-   [Core Components](#core-components)
-   [Workflows Patterns](#workflows)
    -   [Augmented LLM](#augmentedllm)
    -   [Parallel](#parallel)
    -   [Router](#router)
    -   [IntentClassifier](#intentclassifier)
    -   [Evaluator-Optimizer](#evaluator-optimizer)
    -   [Orchestrator-Workers](#orchestrator-workers)
    -   [OpenAI Swarm](#swarm-1)
-   [Advanced](#advanced)
    -   [Composing Multiple Workflows](#composability)
    -   [Signaling and Human Input](#signaling-and-human-input)
    -   [App Config](#app-config)
    -   [MCP Server Management](#mcp-server-management)
-   [Contributing](#contributing)
-   [Roadmap](#roadmap)
-   [FAQs](#faqs)

## Why Use `mcp-agent`?

Unlike other AI frameworks, `mcp-agent` is purpose-built for the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/introduction), providing a lightweight and focused approach to building AI agents.  As more services adopt MCP, `mcp-agent` enables you to create powerful agents that seamlessly integrate with those services, all while simplifying your code and improving control.

## Examples

Explore practical use cases to see what's possible with mcp-agent:  from multi-agent collaboration to RAG pipelines and more.

### Claude Desktop

Integrate mcp-agent applications into MCP clients like Claude Desktop.

#### mcp-agent server

This app integrates an mcp-agent application within an MCP server, exposing the server to Claude Desktop. The application provides agents and workflows that Claude Desktop can use to fulfill user requests.

https://github.com/user-attachments/assets/7807cffd-dba7-4f0c-9c70-9482fd7e0699

This demo showcases a multi-agent evaluation task, where individual agents assess aspects of a poem, and then an aggregator compiles their findings.

**Details**: Starting from a user's text request, the application:

-   Dynamically defines agents to carry out the task.
-   Orchestrates these agents using the appropriate workflow (in this case, the Parallel workflow).

**Link to code**: [examples/basic/mcp_server_aggregator](./examples/basic/mcp_server_aggregator)

> [!NOTE]
> Huge thanks to [Jerron Lim (@StreetLamb)](https://github.com/StreetLamb)
> for developing and contributing this example!

### Streamlit

Deploy mcp-agent applications using Streamlit.

#### Gmail agent

This app manages Gmail actions via text prompts, enabling tasks such as reading, deleting, and sending emails, and marking them as read/unread. It utilizes an MCP server for Gmail.

https://github.com/user-attachments/assets/54899cac-de24-4102-bd7e-4b2022c956e3

**Link to code**: [gmail-mcp-server](https://github.com/jasonsum/gmail-mcp-server/blob/add-mcp-agent-streamlit/streamlit_app.py)

> [!NOTE]
> Huge thanks to [Jason Summer (@jasonsum)](https://github.com/jasonsum)
> for developing and contributing this example!

#### Simple RAG Chatbot

This application employs a Qdrant vector database (via an MCP server) for Q&A over a given text corpus.

https://github.com/user-attachments/assets/f4dcd227-cae9-4a59-aa9e-0eceeb4acaf4

**Link to code**: [examples/usecases/streamlit_mcp_rag_agent](./examples/usecases/streamlit_mcp_rag_agent/)

> [!NOTE]
> Huge thanks to [Jerron Lim (@StreetLamb)](https://github.com/StreetLamb)
> for developing and contributing this example!

### Marimo

[Marimo](https://github.com/marimo-team/marimo) is a reactive Python notebook alternative to Jupyter and Streamlit. Here's the "file finder" agent from [Quickstart](#quickstart) implemented in Marimo:

<img src="https://github.com/user-attachments/assets/139a95a5-e3ac-4ea7-9c8f-bad6577e8597" width="400"/>

**Link to code**: [examples/usecases/marimo_mcp_basic_agent](./examples/usecases/marimo_mcp_basic_agent/)

> [!NOTE]
> Huge thanks to [Akshay Agrawal (@akshayka)](https://github.com/akshayka)
> for developing and contributing this example!

### Python

Write mcp-agent apps as Python scripts or Jupyter notebooks.

#### Swarm

This example provides a multi-agent structure for handling various customer service requests in an airline context, using the Swarm workflow pattern.  Agents can triage requests and manage flight modifications, cancellations, and lost baggage claims.

https://github.com/user-attachments/assets/b314d75d-7945-4de6-965b-7f21eb14a8bd

**Link to code**: [examples/workflows/workflow_swarm](./examples/workflows/workflow_swarm/)

## Core Components

Key building blocks of the mcp-agent framework:

-   **[MCPApp](./src/mcp_agent/app.py)**: Global state and app configuration
-   **MCP server management**: [`gen_client`](./src/mcp_agent/mcp/gen_client.py) and [`MCPConnectionManager`](./src/mcp_agent/mcp/mcp_connection_manager.py) for easy MCP server connections.
-   **[Agent](./src/mcp_agent/agents/agent.py)**:  An Agent, defined with a name and purpose, has access to a set of MCP servers and exposes them to an LLM as tool calls.
-   **[AugmentedLLM](./src/mcp_agent/workflows/llm/augmented_llm.py)**:  An LLM enhanced with tools provided from a collection of MCP servers.  Each workflow pattern is also an `AugmentedLLM`, enabling composability.

## Workflows

mcp-agent provides implementations for all patterns in Anthropic’s [Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) and the OpenAI [Swarm](https://github.com/openai/swarm) pattern.  Each pattern is model-agnostic and exposed as an `AugmentedLLM`, increasing composability.

### AugmentedLLM

[AugmentedLLM](./src/mcp_agent/workflows/llm/augmented_llm.py) extends LLMs with Agents and tools via MCP servers.

LLM providers implement the AugmentedLLM interface, exposing functions for:

-   `generate`:  Generate message(s) based on a prompt, iteratively making tool calls.
-   `generate_str`:  Calls `generate` and returns a string result.
-   `generate_structured`:  Uses [Instructor](https://github.com/instructor-ai/instructor) to return results as a Pydantic model.

`AugmentedLLM` also provides memory to track history.

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

Distribute tasks to sub-agents and consolidate results. Each subtask and the overall workflow are `AugmentedLLMs`, allowing complex nesting.

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

Route input to the most relevant categories (Agents, MCP servers, or functions).

mcp-agent offers these router implementations:

-   [`EmbeddingRouter`](src/mcp_agent/workflows/router/router_embedding.py):  Uses embedding models for classification.
-   [`LLMRouter`](src/mcp_agent/workflows/router/router_llm.py):  Uses LLMs for classification.

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

Similar to Router, the Intent Classifier identifies the top Intents matching an input.  mcp-agent has [embedding](src/mcp_agent/workflows/intent_classifier/intent_classifier_embedding.py) and [LLM-based](src/mcp_agent/workflows/intent_classifier/intent_classifier_llm.py) options.

### [Evaluator-Optimizer](src/mcp_agent/workflows/evaluator_optimizer/evaluator_optimizer.py)

![Evaluator-optimizer workflow (Image credit: Anthropic)](https://www.anthropic.com/_next/image?url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2F14f51e6406ccb29e695da48b17017e899a6119c7-2401x1000.png&w=3840&q=75)

An "optimizer" refines a response, while an "evaluator" critiques it until a quality criterion is met.

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

A higher-level LLM generates a plan, assigns sub-agents, and synthesizes the results. The Orchestrator workflow automatically parallelizes steps and handles dependencies.

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

mcp-agent offers a model-agnostic implementation of the OpenAI [Swarm](https://github.com/openai/swarm) pattern.

<img src="https://github.com/openai/swarm/blob/main/assets/swarm_diagram.png?raw=true" width=500 />

The mcp-agent Swarm pattern integrates seamlessly with MCP servers and is an `AugmentedLLM`, allowing for workflow composability.

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

## Advanced

### Composability

Compose workflows for complex behavior, like using an [Evaluator-Optimizer](#evaluator-optimizer) as the planner within the [Orchestrator](#orchestrator-workers) workflow.

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

*   **Signaling:** The framework can pause and resume tasks.
*   **Human Input:** Agents with a `human_input_callback` can request user input during a workflow.

<details>
<summary>Example</summary>

The [Swarm example](examples/workflows/workflow_swarm/main.py) demonstrates this.

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

Use [`mcp_agent.config.yaml`](/schema/mcp-agent.config.schema.json) for configuration and define secrets using [`mcp_agent.secrets.yaml`](./examples/basic/mcp_basic_agent/mcp_agent.secrets.yaml.example) or a local [`.env`](./examples/basic/mcp_basic_agent/.env.example). For production, use `MCP_APP_SETTINGS_PRELOAD` to avoid writing secrets to disk.

### MCP Server Management

mcp-agent simplifies connecting to MCP servers using [`mcp_agent.config.yaml`](/schema/mcp-agent.config.schema.json) under the `mcp` section.

```yaml
mcp:
  servers:
    fetch:
      command: "uvx"
      args: ["mcp-server-fetch"]
      description: "Fetch content at URLs from the world wide web"
```

#### [`gen_client`](src/mcp_agent/mcp/gen_client.py)

Manage MCP server lifecycles within an async context:

```python
from mcp_agent.mcp.gen_client import gen_client

async with gen_client("fetch") as fetch_client:
    # Fetch server is initialized and ready to use
    result = await fetch_client.list_tools()

# Fetch server is automatically disconnected/shutdown
```

#### Persistent server connections

For persistent server use, employ:

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

[`MCPAggregator`](src/mcp_agent/mcp/mcp_aggregator.py) acts as a server-of-servers, exposing tools from multiple MCP servers via a single MCP interface.

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

We welcome all contributions. See the [CONTRIBUTING guidelines](./CONTRIBUTING.md).

### Special Mentions

Outstanding community contributions:

-   [Shaun Smith (@evalstate)](https://github.com/evalstate) - Leading numerous improvements.
-   [Jerron Lim (@StreetLamb)](https://github.com/StreetLamb) - Excellent examples and ideas.
-   [Jason Summer (@jasonsum)](https://github.com/jasonsum) - Identifying and resolving issues.

## Roadmap

(A detailed roadmap will be added, driven by your feedback.) Current priorities:

-   **Durable Execution** - Workflow pause/resume and state serialization.
-   **Memory** - Long-term memory support.
-   **Streaming** - Streaming listeners for progress updates.
-   **MCP Enhancements** - Extend beyond tool calls.

## FAQs

### What are the core benefits of using mcp-agent?

mcp-agent provides a streamlined approach to building AI agents using capabilities exposed by **MCP** (Model Context Protocol) servers.

MCP is quite low-level, and this framework handles the mechanics of connecting to servers, working with LLMs, handling external signals (like human input) and supporting persistent state via durable execution. That lets you, the developer, focus on the core business logic of your AI application.

Core benefits:

- 🤝 **Interoperability**: ensures that any tool exposed by any number of MCP servers can seamlessly plug in to your agents.
- ⛓️ **Composability & Customizability**: Implements well-defined workflows, but in a composable way that enables compound workflows, and allows full customization across model provider, logging, orchestrator, etc.
- 💻 **Programmatic control flow**: Keeps things simple as developers just write code instead of thinking in graphs, nodes and edges. For branching logic, you write `if` statements. For cycles, use `while` loops.
- 🖐️ **Human Input & Signals**: Supports pausing workflows for external signals, such as human input, which are exposed as tool calls an Agent can make.

### Do you need an MCP client to use mcp-agent?

No, you can use mcp-agent anywhere, since it handles MCPClient creation for you. This allows you to leverage MCP servers outside of MCP hosts like Claude Desktop.

Here's all the ways you can set up your mcp-agent application:

#### MCP-Agent Server

You can expose mcp-agent applications as MCP servers themselves (see [example](./examples/mcp_agent_server)), allowing MCP clients to interface with sophisticated AI workflows using the standard tools API of MCP servers. This is effectively a server-of-servers.

#### MCP Client or Host

You can embed mcp-agent in an MCP client directly to manage the orchestration across multiple MCP servers.

#### Standalone

You can use mcp-agent applications in a standalone fashion (i.e. they aren't part of an MCP client). The [`examples`](/examples/) are all standalone applications.

### Tell me a fun fact

I debated naming this project _silsila_ (سلسلہ), which means chain of events in Urdu. mcp-agent is more matter-of-fact, but there's still an easter egg in the project paying homage to silsila.
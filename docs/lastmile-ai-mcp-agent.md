<p align="center">
  <a href="https://github.com/lastmile-ai/mcp-agent"><img src="https://github.com/user-attachments/assets/c8d059e5-bd56-4ea2-a72d-807fb4897bde" alt="Logo" width="300" /></a>
</p>

<p align="center">
  <em>Build powerful AI agents with Model Context Protocol using simple, composable patterns.</em>
</p>

<p align="center">
  <a href="https://github.com/lastmile-ai/mcp-agent/tree/main/examples" target="_blank"><strong>Examples</strong></a>
  |
  <a href="https://www.anthropic.com/research/building-effective-agents" target="_blank"><strong>Building Effective Agents</strong></a>
  |
  <a href="https://modelcontextprotocol.io/introduction" target="_blank"><strong>MCP</strong></a>
  |
  <a href="https://docs.mcp-agent.com"><img src="https://img.shields.io/badge/docs-8F?style=flat&link=https%3A%2F%2Fdocs.mcp-agent.com%2F" /></a>
  |
  <a href="https://pypi.org/project/mcp-agent/"><img src="https://img.shields.io/pypi/v/mcp-agent?color=%2334D058&label=pypi" /></a>
  |
  <a href="https://github.com/lastmile-ai/mcp-agent/issues"><img src="https://img.shields.io/github/issues-raw/lastmile-ai/mcp-agent" /></a>
  |
  <img alt="Pepy Total Downloads" src="https://img.shields.io/pepy/dt/mcp-agent?label=pypi%20%7C%20downloads"/>
  |
  <a href="https://github.com/lastmile-ai/mcp-agent/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg"/></a>
  |
  <a href="https://lmai.link/discord/mcp-agent"><img src="https://img.shields.io/badge/Discord-%235865F2.svg?logo=discord&logoColor=white" alt="discord"/></a>
</p>


<p align="center">
  <a href="https://trendshift.io/repositories/13216" target="_blank"><img src="https://trendshift.io/api/badge/repositories/13216" alt="lastmile-ai%2Fmcp-agent | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>


## Build Advanced AI Agents with Ease: Introducing `mcp-agent`

**`mcp-agent`** is a powerful, yet simple, Python framework designed to build intelligent AI agents using the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/introduction).  Leveraging Anthropic's [Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) principles, `mcp-agent` empowers you to create robust, composable AI applications with minimal effort.  [Explore the original repository here.](https://github.com/lastmile-ai/mcp-agent)

**Key Features:**

*   **Simplified MCP Server Management:** Handles MCP server lifecycle, connection management, and tool integration effortlessly.
*   **Composable Workflows:** Implements Anthropic's agent patterns (e.g., Parallel, Router, Evaluator-Optimizer, Orchestrator) in a modular and flexible way.
*   **Model-Agnostic Design:** Works seamlessly with various LLMs (e.g., OpenAI, Anthropic).
*   **Multi-Agent Orchestration:** Includes an implementation of OpenAI's Swarm pattern for complex multi-agent systems.
*   **Human-in-the-Loop:** Supports human input and signaling for enhanced control and oversight.
*   **Easy Integration:** Integrate with MCP clients like Claude Desktop, Streamlit, and Marimo.

## Table of Contents

*   [Why use `mcp-agent`?](#why-use-mcp-agent)
*   [Getting Started](#get-started)
    *   [Installation](#installation)
    *   [Quickstart Example](#quickstart)
*   [Examples](#examples)
    *   [Claude Desktop Integration](#claude-desktop)
    *   [Streamlit Applications](#streamlit)
        *   [Gmail Agent](#gmail-agent)
        *   [Simple RAG Chatbot](#simple-rag-chatbot)
    *   [Marimo Integration](#marimo)
    *   [Python Scripting](#python)
        *   [Swarm (CLI)](#swarm)
*   [Core Components](#core-components)
*   [Workflows (Agent Patterns)](#workflows)
    *   [AugmentedLLM](#augmentedllm)
    *   [Parallel Workflow](#parallel)
    *   [Router Workflow](#router)
    *   [IntentClassifier Workflow](#intentclassifier)
    *   [Evaluator-Optimizer Workflow](#evaluator-optimizer)
    *   [Orchestrator-Workers Workflow](#orchestrator-workers)
    *   [Swarm Workflow](#swarm-1)
*   [Advanced Features](#advanced)
    *   [Composability](#composability)
    *   [Signaling and Human Input](#signaling-and-human-input)
    *   [App Configuration](#app-config)
    *   [MCP Server Management](#mcp-server-management)
        *   [gen_client](#gen_client)
        *   [Persistent Server Connections](#persistent-server-connections)
        *   [MCP Server Aggregator](#mcp-server-aggregator)
*   [Contributing](#contributing)
    *   [Special Mentions](#special-mentions)
*   [Roadmap](#roadmap)
*   [FAQs](#faqs)

## Get Started

### Installation

We recommend using [uv](https://docs.astral.sh/uv/) for project management:

```bash
uv add "mcp-agent"
```

Alternatively:

```bash
pip install mcp-agent
```

### Quickstart

> [!TIP]
> Explore the [`examples`](/examples) directory for a variety of working examples.  To run an example:

> ```bash
> cd examples/basic/mcp_basic_agent  # Or any other example
> cp mcp_agent.secrets.yaml.example mcp_agent.secrets.yaml  # Update API keys
> uv run main.py
> ```

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

## Why use `mcp-agent`?

While many AI frameworks exist, `mcp-agent` stands out as the only framework specifically designed for the [MCP](https://modelcontextprotocol.io/introduction) protocol. It is lightweight and focuses on agent patterns, allowing developers to build robust AI applications efficiently. By leveraging the growing ecosystem of MCP-aware services, you can create powerful, controllable AI agents with ease.

## Examples

mcp-agent enables the creation of diverse AI applications, including multi-agent collaborations, human-in-the-loop workflows, and RAG pipelines.

### Claude Desktop Integration

Integrate `mcp-agent` applications with MCP clients like Claude Desktop.

#### mcp-agent server

This app wraps an mcp-agent application inside an MCP server, and exposes that server to Claude Desktop.
The app exposes agents and workflows that Claude Desktop can invoke to service of the user's request.

https://github.com/user-attachments/assets/7807cffd-dba7-4f0c-9c70-9482fd7e0699

This demo shows a multi-agent evaluation task where each agent evaluates aspects of an input poem, and
then an aggregator summarizes their findings into a final response.

**Details**: Starting from a user's request over text, the application:

- dynamically defines agents to do the job
- uses the appropriate workflow to orchestrate those agents (in this case the Parallel workflow)

**Link to code**: [examples/basic/mcp_server_aggregator](./examples/basic/mcp_server_aggregator)

> [!NOTE]
> Huge thanks to [Jerron Lim (@StreetLamb)](https://github.com/StreetLamb)
> for developing and contributing this example!

### Streamlit

Deploy `mcp-agent` apps using Streamlit for interactive AI experiences.

#### Gmail agent

This app performs read and write actions on Gmail via text prompts, utilizing an MCP server for Gmail.

https://github.com/user-attachments/assets/54899cac-de24-4102-bd7e-4b2022c956e3

**Link to code**: [gmail-mcp-server](https://github.com/jasonsum/gmail-mcp-server/blob/add-mcp-agent-streamlit/streamlit_app.py)

> [!NOTE]
> Huge thanks to [Jason Summer (@jasonsum)](https://github.com/jasonsum)
> for developing and contributing this example!

#### Simple RAG Chatbot

Build a question-answering chatbot over a text corpus using a Qdrant vector database via an MCP server.

https://github.com/user-attachments/assets/f4dcd227-cae9-4a59-aa9e-0eceeb4acaf4

**Link to code**: [examples/usecases/streamlit_mcp_rag_agent](./examples/usecases/streamlit_mcp_rag_agent/)

> [!NOTE]
> Huge thanks to [Jerron Lim (@StreetLamb)](https://github.com/StreetLamb)
> for developing and contributing this example!

### Marimo

Create reactive Python notebooks with [Marimo](https://github.com/marimo-team/marimo). Here's the "file finder" agent from [Quickstart](#quickstart) implemented in Marimo:

<img src="https://github.com/user-attachments/assets/139a95a5-e3ac-4ea7-9c8f-bad6577e8597" width="400"/>

**Link to code**: [examples/usecases/marimo_mcp_basic_agent](./examples/usecases/marimo_mcp_basic_agent/)

> [!NOTE]
> Huge thanks to [Akshay Agrawal (@akshayka)](https://github.com/akshayka)
> for developing and contributing this example!

### Python

Write `mcp-agent` applications as Python scripts or Jupyter notebooks.

#### Swarm

Demonstrates a multi-agent setup for handling customer service requests (airline context) using the Swarm workflow pattern, triaging requests, handling flight modifications, cancellations, and lost baggage cases.

https://github.com/user-attachments/assets/b314d75d-7945-4de6-965b-7f21eb14a8bd

**Link to code**: [examples/workflows/workflow_swarm](./examples/workflows/workflow_swarm/)

## Core Components

The building blocks of the `mcp-agent` framework:

*   **[MCPApp](./src/mcp_agent/app.py)**:  Handles global state and app configuration.
*   **MCP Server Management:**  Utilizes [`gen_client`](./src/mcp_agent/mcp/gen_client.py) and [`MCPConnectionManager`](./src/mcp_agent/mcp/mcp_connection_manager.py) for simplified MCP server connections.
*   **[Agent](./src/mcp_agent/agents/agent.py)**:  Represents an entity with access to MCP servers, providing tools to an LLM, defined by a name and instruction.
*   **[AugmentedLLM](./src/mcp_agent/workflows/llm/augmented_llm.py)**: An LLM enhanced with tools from MCP servers, forming the foundation for all Workflows.

## Workflows

`mcp-agent` provides implementations for patterns described in Anthropic’s [Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) and the OpenAI [Swarm](https://github.com/openai/swarm) pattern.  Each is model-agnostic and exposed as an `AugmentedLLM` for easy composability.

### AugmentedLLM

An LLM interface enhanced with tools, including MCP servers and functions.

LLM providers implement the AugmentedLLM interface to expose 3 functions:

-   `generate`: Generates message(s) given a prompt, using tool calls as needed.
-   `generate_str`: Calls `generate` and returns the result as a string.
-   `generate_structured`: Uses [Instructor](https://github.com/instructor-ai/instructor) to return the result as a Pydantic model.

Also, `AugmentedLLM` has memory to keep track of long or short-term history.

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

### Parallel Workflow

Fan-out tasks to sub-agents, and fan-in the results. Each subtask and the overall Parallel workflow are `AugmentedLLM`s.

![Parallel workflow (Image credit: Anthropic)](https://www.anthropic.com/_next/image?url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2F406bb032ca007fd1624f261af717d70e6ca86286-2401x1000.png&w=3840&q=75)

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

### Router Workflow

Routes input to the `top_k` most relevant categories. A category can be an Agent, an MCP server, or a regular function.

![Router workflow (Image credit: Anthropic)](https://www.anthropic.com/_next/image?url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2F5c0c0e9fe4def0b584c04d37849941da55e5e71c-2401x1000.png&w=3840&q=75)

mcp-agent provides:

-   [`EmbeddingRouter`](src/mcp_agent/workflows/router/router_embedding.py): uses embedding models for classification
-   [`LLMRouter`](src/mcp_agent/workflows/router/router_llm.py): uses LLMs for classification

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

### IntentClassifier Workflow

Identifies the `top_k` Intents matching an input.  mcp-agent includes both [embedding](src/mcp_agent/workflows/intent_classifier/intent_classifier_embedding.py) and [LLM-based](src/mcp_agent/workflows/intent_classifier/intent_classifier_llm.py) intent classifiers.

### Evaluator-Optimizer Workflow

Refines a response: one LLM (optimizer) refines the response, while another (evaluator) critiques it until a quality criterion is met.

![Evaluator-optimizer workflow (Image credit: Anthropic)](https://www.anthropic.com/_next/image?url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2F14f51e6406ccb29e695da48b17017e899a6119c7-2401x1000.png&w=3840&q=75)

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

### Orchestrator-Workers Workflow

A higher-level LLM generates a plan, assigns it to sub-agents, and synthesizes results.

![Orchestrator workflow (Image credit: Anthropic)](https://www.anthropic.com/_next/image?url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2F8985fc683fae4780fb34eab1365ab78c7e51bc8e-2401x1000.png&w=3840&q=75)

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

### Swarm Workflow

Implements OpenAI's [Swarm](https://github.com/openai/swarm) pattern for multi-agent coordination, seamlessly integrating with MCP servers.

<img src="https://github.com/openai/swarm/blob/main/assets/swarm_diagram.png?raw=true" width=500 />

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

Composability enables chaining workflows, such as integrating an [Evaluator-Optimizer](#evaluator-optimizer) within the [Orchestrator](#orchestrator-workers) for high-quality plan generation.  This is seamless because each workflow is an `AugmentedLLM`.

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

**Signaling:** The framework allows pausing/resuming tasks. Agents or LLMs can "signal" for user input, pausing the workflow. Developers can signal to request review or approval.

**Human Input:** Agents with a `human_input_callback` can use the `__human_input__` tool to request user input.

<details>
<summary>Example</summary>

See the [Swarm example](examples/workflows/workflow_swarm/main.py).

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

### App Configuration

Use an [`mcp_agent.config.yaml`](/schema/mcp-agent.config.schema.json) and a gitignored [`mcp_agent.secrets.yaml`](./examples/basic/mcp_basic_agent/mcp_agent.secrets.yaml.example) to configure your MCP application, including logging, execution, LLM APIs, and MCP server settings.

### MCP Server Management

Easily connect to MCP servers using `mcp-agent`.

#### [`gen_client`](src/mcp_agent/mcp/gen_client.py)

Manages MCP server lifecycle within an async context manager:

```python
from mcp_agent.mcp.gen_client import gen_client

async with gen_client("fetch") as fetch_client:
    # Fetch server is initialized and ready to use
    result = await fetch_client.list_tools()

# Fetch server is automatically disconnected/shutdown
```

#### Persistent Server Connections

For persistent server connections:

*   Use [`connect`](src/mcp_agent/mcp/gen_client.py) and [`disconnect`](src/mcp_agent/mcp/gen_client.py)

```python
from mcp_agent.mcp.gen_client import connect, disconnect

fetch_client = None
try:
     fetch_client = connect("fetch")
     result = await fetch_client.list_tools()
finally:
     disconnect("fetch")
```

*   Or use [`MCPConnectionManager`](src/mcp_agent/mcp/mcp_connection_manager.py) for more control.

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

## Contributing

We welcome contributions of all kinds!  See the [CONTRIBUTING guidelines](./CONTRIBUTING.md) to get started.

### Special Mentions

Many community contributors are driving the project forward:

-   [Shaun Smith (@evalstate)](https://github.com/evalstate) -- Leading complex improvements to `mcp-agent` and the MCP ecosystem.
-   [Jerron Lim (@StreetLamb)](https://github.com/StreetLamb) -- Contributing examples and ideas.
-   [Jason Summer (@jasonsum)](https://github.com/jasonsum) -- Identifying issues and adapting the Gmail MCP server.

## Roadmap

A detailed roadmap will be added (feedback-driven). Current priorities:

-   **Durable Execution:** Implement workflow pause/resume and state serialization. Integration with [Temporal](./src/mcp_agent/executor/temporal.py) planned.
-   **Memory:** Add support for long-term memory.
-   **Streaming:** Implement streaming listeners for progress.
-   **Additional MCP capabilities:** Expand tool call support with resources, prompts, and notifications.

## FAQs

### What are the core benefits of using mcp-agent?

`mcp-agent` simplifies building AI agents using MCP servers.

MCP is low-level; `mcp-agent` handles server connections, LLMs, external signals (human input), and state management. This lets developers focus on application logic.

Core benefits:

-   🤝 **Interoperability:** Integrates tools from multiple MCP servers seamlessly.
-   ⛓️ **Composability & Customizability:** Provides well-defined, composable workflows with customization options.
-   💻 **Programmatic control flow:** Uses code for control flow (if/else, loops).
-   🖐️ **Human Input & Signals:** Supports pausing workflows for external signals and human input via tool calls.

### Do you need an MCP client to use mcp-agent?

No, `mcp-agent` handles MCPClient creation, allowing usage independently.  You can expose applications in multiple ways:

#### MCP-Agent Server

Expose applications as MCP servers (see [example](./examples/mcp_agent_server)), allowing MCP clients to interface with AI workflows.

#### MCP Client or Host

Embed `mcp-agent` in an MCP client to orchestrate multiple MCP servers.

#### Standalone

Use `mcp-agent` applications standalone (examples are all standalone).

### Tell me a fun fact

I debated naming this project _silsila_ (سلسلہ), which means chain of events in Urdu. mcp-agent is more matter-of-fact, but there's still an easter egg in the project paying homage to silsila.
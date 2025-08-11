<p align="center">
  <a href="https://github.com/lastmile-ai/mcp-agent"><img src="https://github.com/user-attachments/assets/c8d059e5-bd56-4ea2-a72d-807fb4897bde" alt="MCP Agent Logo" width="300" /></a>
</p>

<p align="center">
  <em>Build powerful and composable AI agents with the <a href="https://modelcontextprotocol.io/introduction">Model Context Protocol (MCP)</a>.</em>
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

## Build Next-Gen AI Agents with mcp-agent

**mcp-agent** is a powerful, open-source Python framework designed to simplify the creation of AI agents using the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/introduction).  Leverage composable patterns to build robust, production-ready AI applications that can access and interact with a wide range of services. Check out the [mcp-agent repository](https://github.com/lastmile-ai/mcp-agent) to get started.

**Key Features:**

*   **Simplified MCP Integration:** Easily manage connections to MCP servers.
*   **Composable Workflows:**  Implement and chain patterns from [Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) and the [OpenAI Swarm](https://github.com/openai/swarm) in a modular way.
*   **Model Agnostic:** Works seamlessly with different LLMs.
*   **Multi-Agent Orchestration:** Supports complex multi-agent workflows.
*   **Extensible & Customizable:** Tailor agents to your specific needs.

## Table of Contents

-   [Why Use mcp-agent?](#why-use-mcp-agent)
-   [Examples](#examples)
    -   [Claude Desktop](#claude-desktop)
    -   [Streamlit](#streamlit)
        -   [Gmail Agent](#gmail-agent)
        -   [RAG](#simple-rag-chatbot)
    -   [Marimo](#marimo)
    -   [Python](#python)
        -   [Swarm (CLI)](#swarm)
-   [Core Components](#core-components)
-   [Workflows Patterns](#workflows)
    -   [Augmented LLM](#augmentedllm)
    -   [Parallel](#parallel)
    -   [Router](#router)
    -   [Intent-Classifier](#intentclassifier)
    -   [Orchestrator-Workers](#orchestrator-workers)
    -   [Evaluator-Optimizer](#evaluator-optimizer)
    -   [OpenAI Swarm](#swarm-1)
-   [Advanced](#advanced)
    -   [Composing Multiple Workflows](#composability)
    -   [Signaling and Human Input](#signaling-and-human-input)
    -   [App Config](#app-config)
    -   [MCP Server Management](#mcp-server-management)
-   [Contributing](#contributing)
-   [Roadmap](#roadmap)
-   [FAQs](#faqs)

## Why Use mcp-agent?

mcp-agent is purpose-built for the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/introduction), offering a lightweight and efficient way to build sophisticated AI agents. As the MCP ecosystem grows, mcp-agent enables you to build powerful agents that readily integrate with new MCP-aware services, giving you greater flexibility and control in your AI application development.

**Key Benefits:**

*   **Interoperability**: Seamlessly integrate tools from any MCP server.
*   **Composability**: Build complex workflows using modular patterns.
*   **Control Flow**: Write straightforward Python code for branching and looping logic.
*   **Human-in-the-Loop**: Easily incorporate human input and signals into your workflows.

## Examples

mcp-agent shines in a wide range of applications. Here are some examples:

### Claude Desktop

Integrate mcp-agent applications into MCP clients like Claude Desktop for advanced AI interactions.

#### mcp-agent server

This app exposes agents and workflows to Claude Desktop, enabling sophisticated task execution.

<img src="https://github.com/user-attachments/assets/7807cffd-dba7-4f0c-9c70-9482fd7e0699" alt="Claude Desktop Demo" width="600">

**Details**: This example demonstrates a multi-agent evaluation where agents analyze an input poem and an aggregator summarizes the findings.

**Link to code**: [examples/basic/mcp_server_aggregator](./examples/basic/mcp_server_aggregator)

> [!NOTE]
> Huge thanks to [Jerron Lim (@StreetLamb)](https://github.com/StreetLamb) for developing and contributing this example!

### Streamlit

Deploy mcp-agent apps using Streamlit for user-friendly AI interfaces.

#### Gmail Agent

Manage Gmail using text prompts – read, delete, and send emails. This example leverages an MCP server for Gmail.

<img src="https://github.com/user-attachments/assets/54899cac-de24-4102-bd7e-4b2022c956e3" alt="Gmail Agent Demo" width="600">

**Link to code**: [gmail-mcp-server](https://github.com/jasonsum/gmail-mcp-server/blob/add-mcp-agent-streamlit/streamlit_app.py)

> [!NOTE]
> Huge thanks to [Jason Summer (@jasonsum)](https://github.com/jasonsum) for developing and contributing this example!

#### Simple RAG Chatbot

Build a question-answering system using a Qdrant vector database via an MCP server.

<img src="https://github.com/user-attachments/assets/f4dcd227-cae9-4a59-aa9e-0eceeb4acaf4" alt="RAG Chatbot Demo" width="600">

**Link to code**: [examples/usecases/streamlit_mcp_rag_agent](./examples/usecases/streamlit_mcp_rag_agent/)

> [!NOTE]
> Huge thanks to [Jerron Lim (@StreetLamb)](https://github.com/StreetLamb) for developing and contributing this example!

### Marimo

Create reactive Python notebooks with Marimo.

Here's the "file finder" agent from Quickstart implemented in Marimo:

<img src="https://github.com/user-attachments/assets/139a95a5-e3ac-4ea7-9c8f-bad6577e8597" alt="Marimo File Finder" width="400"/>

**Link to code**: [examples/usecases/marimo_mcp_basic_agent](./examples/usecases/marimo_mcp_basic_agent/)

> [!NOTE]
> Huge thanks to [Akshay Agrawal (@akshayka)](https://github.com/akshayka) for developing and contributing this example!

### Python

Build robust AI applications using Python scripts or Jupyter notebooks.

#### Swarm

Demonstrates a multi-agent setup for airline customer service, using the Swarm workflow pattern.

<img src="https://github.com/user-attachments/assets/b314d75d-7945-4de6-965b-7f21eb14a8bd" alt="Swarm Example" width="600">

**Link to code**: [examples/workflows/workflow_swarm](./examples/workflows/workflow_swarm/)

## Core Components

mcp-agent is built on these core components:

*   **[MCPApp](./src/mcp_agent/app.py)**: Global state and app configuration.
*   **MCP Server Management**: [`gen_client`](./src/mcp_agent/mcp/gen_client.py) and [`MCPConnectionManager`](./src/mcp_agent/mcp/mcp_connection_manager.py) to easily connect to MCP servers.
*   **[Agent](./src/mcp_agent/agents/agent.py)**: Agents with access to MCP servers and tools exposed to an LLM.
*   **[AugmentedLLM](./src/mcp_agent/workflows/llm/augmented_llm.py)**: LLMs enhanced with tools from MCP servers, forming the basis for all workflows.

## Workflows

mcp-agent provides implementations of Anthropic’s [Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) patterns and the OpenAI [Swarm](https://github.com/openai/swarm) pattern, making them highly composable.

### AugmentedLLM

[AugmentedLLM](./src/mcp_agent/workflows/llm/augmented_llm.py) integrates with MCP servers to give the LLM access to tools.

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

<img src="https://www.anthropic.com/_next/image?url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2F406bb032ca007fd1624f261af717d70e6ca86286-2401x1000.png&w=3840&q=75" alt="Parallel Workflow" width="600">

Fan-out tasks to multiple sub-agents and then aggregate the results.

> [!NOTE]
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

<img src="https://www.anthropic.com/_next/image?url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2F5c0c0e9fe4def0b584c04d37849941da55e5e71c-2401x1000.png&w=3840&q=75" alt="Router Workflow" width="600">

Routes input to the `top_k` most relevant categories (Agents, MCP servers, or regular functions).

mcp-agent offers:

*   [`EmbeddingRouter`](src/mcp_agent/workflows/router/router_embedding.py): Uses embedding models for classification.
*   [`LLMRouter`](src/mcp_agent/workflows/router/router_llm.py): Uses LLMs for classification.

> [!NOTE]
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

Identifies the `top_k` Intents that match a given input.  Similar to the Router, providing both embedding and LLM-based implementations.

### [Evaluator-Optimizer](src/mcp_agent/workflows/evaluator_optimizer/evaluator_optimizer.py)

<img src="https://www.anthropic.com/_next/image?url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2F14f51e6406ccb29e695da48b17017e899a6119c7-2401x1000.png&w=3840&q=75" alt="Evaluator-Optimizer Workflow" width="600">

Refines a response using an "optimizer" agent, and then evaluates it via an "evaluator" agent. Iterates until a quality criteria is met.

> [!NOTE]
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

<img src="https://www.anthropic.com/_next/image?url=https%3A%2F%2Fwww-cdn.anthropic.com%2Fimages%2F4zrzovbb%2Fwebsite%2F8985fc683fae4780fb34eab1365ab78c7e51bc8e-2401x1000.png&w=3840&q=75" alt="Orchestrator Workflow" width="600">

A high-level LLM creates a plan, assigns tasks to sub-agents, and synthesizes results.

> [!NOTE]
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

<img src="https://github.com/openai/swarm/blob/main/assets/swarm_diagram.png?raw=true" alt="Swarm Diagram" width="600">

A model-agnostic implementation of OpenAI’s Swarm pattern.

> [!NOTE]
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

Combine workflows, such as using an [Evaluator-Optimizer](#evaluator-optimizer) as the planner LLM within the [Orchestrator](#orchestrator-workers) workflow.

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

**Signaling**: The framework allows pausing/resuming tasks. Agents or LLMs can "signal" for user input. Developers can signal during workflows for approval/review.

**Human Input**: If an Agent has a `human_input_callback`, the LLM can call a `__human_input__` tool to request user input mid-workflow.

<details>
<summary>Example</summary>

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

Configure your app using [`mcp_agent.config.yaml`](/schema/mcp-agent.config.schema.json) and a gitignored [`mcp_agent.secrets.yaml`](./examples/basic/mcp_basic_agent/mcp_agent.secrets.yaml.example).  This controls logging, execution, LLM providers, and MCP server settings.

### MCP Server Management

Easily connect to MCP servers by configuring them in [`mcp_agent.config.yaml`](/schema/mcp-agent.config.schema.json) under the `mcp` section:

```yaml
mcp:
  servers:
    fetch:
      command: "uvx"
      args: ["mcp-server-fetch"]
      description: "Fetch content at URLs from the world wide web"
```

#### [`gen_client`](src/mcp_agent/mcp/gen_client.py)

Manage MCP server lifecycles:

```python
from mcp_agent.mcp.gen_client import gen_client

async with gen_client("fetch") as fetch_client:
    # Fetch server is initialized and ready to use
    result = await fetch_client.list_tools()

# Fetch server is automatically disconnected/shutdown
```

#### Persistent Server Connections

Use [`connect`](<(src/mcp_agent/mcp/gen_client.py)>) and [`disconnect`](src/mcp_agent/mcp/gen_client.py) for persistent connections.

```python
from mcp_agent.mcp.gen_client import connect, disconnect

fetch_client = None
try:
     fetch_client = connect("fetch")
     result = await fetch_client.list_tools()
finally:
     disconnect("fetch")
```

#### [`MCPConnectionManager`](src/mcp_agent/mcp/mcp_connection_manager.py)

For fine-grained server control.

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

[`MCPAggregator`](src/mcp_agent/mcp/mcp_aggregator.py) provides a single MCP server interface for multiple servers.

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

Contributions are welcome! See the [CONTRIBUTING guidelines](./CONTRIBUTING.md) to get started.

### Special Mentions

*   [Shaun Smith (@evalstate)](https://github.com/evalstate)
*   [Jerron Lim (@StreetLamb)](https://github.com/StreetLamb)
*   [Jason Summer (@jasonsum)](https://github.com/jasonsum)

## Roadmap

We are creating a detailed roadmap (based on your feedback). Priorities include:

*   **Durable Execution**: Pause/resume workflows, and serialize state.
*   **Memory**: Add long-term memory support.
*   **Streaming**: Add streaming listeners for progress.
*   **Additional MCP Capabilities**: Support resources, prompts, and notifications.

## FAQs

### What are the core benefits of using mcp-agent?

mcp-agent simplifies AI agent development using MCP servers.

**Core benefits:**

*   🤝 **Interoperability:** Ensures any tool exposed by any MCP server integrates seamlessly.
*   ⛓️ **Composability**:  Build complex workflows.
*   💻 **Programmatic Control Flow**: Write clear Python code.
*   🖐️ **Human Input & Signals**: Integrate human input into workflows.

### Do you need an MCP client to use mcp-agent?

No, mcp-agent handles MCP client creation. Use it in:

*   **MCP-Agent Server**: Expose mcp-agent applications as MCP servers.
*   **MCP Client or Host**: Embed mcp-agent.
*   **Standalone**: Use mcp-agent independently.

### Tell me a fun fact

The project was almost named _silsila_ (Urdu for "chain of events"), and there's an easter egg paying homage to that.
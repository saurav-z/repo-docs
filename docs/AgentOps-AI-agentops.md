<div align="center">
  <a href="https://agentops.ai?ref=gh">
    <img src="docs/images/external/logo/github-banner.png" alt="AgentOps Logo">
  </a>
</div>

<div align="center">
  <em>**Supercharge your AI agent development with AgentOps – the ultimate observability and devtool platform for building, evaluating, and monitoring AI agents.**</em>
</div>

<br />

<div align="center">
  <a href="https://pepy.tech/project/agentops">
    <img src="https://static.pepy.tech/badge/agentops/month" alt="Downloads">
  </a>
  <a href="https://github.com/agentops-ai/agentops/issues">
  <img src="https://img.shields.io/github/commit-activity/m/agentops-ai/agentops" alt="git commit activity">
  </a>
  <img src="https://img.shields.io/pypi/v/agentops?&color=3670A0" alt="PyPI - Version">
  <a href="https://github.com/AgentOps-AI/agentops-ts">
    <img src="https://img.shields.io/badge/TypeScript%20SDK-Available-blue?&color=3670A0" alt="TypeScript SDK">
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg?&color=3670A0" alt="License: MIT">
  </a>
  <a href="https://smithery.ai/server/@AgentOps-AI/agentops-mcp">
    <img src="https://smithery.ai/badge/@AgentOps-AI/agentops-mcp"/>
  </a>
</div>

<p align="center">
  <a href="https://twitter.com/agentopsai/">
    <img src="https://img.shields.io/twitter/follow/agentopsai?style=social" alt="Twitter" style="height: 20px;">
  </a>
  <a href="https://discord.gg/FagdcwwXRR">
    <img src="https://img.shields.io/badge/discord-7289da.svg?style=flat-square&logo=discord" alt="Discord" style="height: 20px;">
  </a>
  <a href="https://app.agentops.ai/?ref=gh">
    <img src="https://img.shields.io/badge/Dashboard-blue.svg?style=flat-square" alt="Dashboard" style="height: 20px;">
  </a>
  <a href="https://docs.agentops.ai/introduction">
    <img src="https://img.shields.io/badge/Documentation-orange.svg?style=flat-square" alt="Documentation" style="height: 20px;">
  </a>
  <a href="https://entelligence.ai/AgentOps-AI&agentops">
    <img src="https://img.shields.io/badge/Chat%20with%20Docs-green.svg?style=flat-square" alt="Chat with Docs" style="height: 20px;">
  </a>
</p>

<div align="center">
  <video src="https://github.com/user-attachments/assets/dfb4fa8d-d8c4-4965-9ff6-5b8514c1c22f" width="650" autoplay loop muted></video>
</div>

<br/>

## What is AgentOps?

AgentOps is a powerful observability and development tool designed to help you build, evaluate, and monitor your AI agents from prototype to production.  Gain insights into agent behavior, track performance, manage costs, and ensure the reliability of your AI-powered applications.

## Key Features

*   📊 **Replay Analytics and Debugging**: Step-by-step agent execution graphs for detailed insights.
*   💸 **LLM Cost Management**: Track and control spending with LLM foundation model providers.
*   🧪 **Agent Benchmarking**: Test and evaluate your agents against 1,000+ pre-built and custom evaluation sets.
*   🔐 **Compliance and Security**: Detect and mitigate common prompt injection and data exfiltration threats.
*   🤝 **Framework Integrations**: Seamless integrations with popular frameworks like CrewAI, AG2 (AutoGen), Camel AI, and LangChain.

## Quick Start

Get started with AgentOps in minutes:

```bash
pip install agentops
```

### Session Replays in 2 Lines of Code

Easily initialize the AgentOps client and gain automatic analytics on all your LLM calls.

1.  **Get an API key:** [Get an API key](https://app.agentops.ai/settings/projects)
2.  **Initialize AgentOps in your code:**

```python
import agentops

# Beginning of your program (i.e. main.py, __init__.py)
agentops.init( < INSERT YOUR API KEY HERE >)

...

# End of program
agentops.end_session('Success')
```

View your session replays and analytics on the [AgentOps dashboard](https://app.agentops.ai?ref=gh).

<details>
  <summary>Agent Debugging</summary>
  <a href="https://app.agentops.ai?ref=gh">
    <img src="docs/images/external/app_screenshots/session-drilldown-metadata.png" style="width: 90%;" alt="Agent Metadata"/>
  </a>
  <a href="https://app.agentops.ai?ref=gh">
    <img src="docs/images/external/app_screenshots/chat-viewer.png" style="width: 90%;" alt="Chat Viewer"/>
  </a>
  <a href="https://app.agentops.ai?ref=gh">
    <img src="docs/images/external/app_screenshots/session-drilldown-graphs.png" style="width: 90%;" alt="Event Graphs"/>
  </a>
</details>

<details>
  <summary>Session Replays</summary>
  <a href="https://app.agentops.ai?ref=gh">
    <img src="docs/images/external/app_screenshots/session-replay.png" style="width: 90%;" alt="Session Replays"/>
  </a>
</details>

<details>
  <summary>Summary Analytics</summary>
  <a href="https://app.agentops.ai?ref=gh">
   <img src="docs/images/external/app_screenshots/overview.png" style="width: 90%;" alt="Summary Analytics"/>
  </a>
  <a href="https://app.agentops.ai?ref=gh">
   <img src="docs/images/external/app_screenshots/overview-charts.png" style="width: 90%;" alt="Summary Analytics Charts"/>
  </a>
</details>

## First-Class Developer Experience

Add powerful observability to your agents, tools, and functions with minimal code: one line at a time.

Refer to our [documentation](http://docs.agentops.ai) for more details.

```python
# Create a session span (root for all other spans)
from agentops.sdk.decorators import session

@session
def my_workflow():
    # Your session code here
    return result
```

```python
# Create an agent span for tracking agent operations
from agentops.sdk.decorators import agent

@agent
class MyAgent:
    def __init__(self, name):
        self.name = name
        
    # Agent methods here
```

```python
# Create operation/task spans for tracking specific operations
from agentops.sdk.decorators import operation, task

@operation  # or @task
def process_data(data):
    # Process the data
    return result
```

```python
# Create workflow spans for tracking multi-operation workflows
from agentops.sdk.decorators import workflow

@workflow
def my_workflow(data):
    # Workflow implementation
    return result
```

```python
# Nest decorators for proper span hierarchy
from agentops.sdk.decorators import session, agent, operation

@agent
class MyAgent:
    @operation
    def nested_operation(self, message):
        return f"Processed: {message}"
        
    @operation
    def main_operation(self):
        result = self.nested_operation("test message")
        return result

@session
def my_session():
    agent = MyAgent()
    return agent.main_operation()
```

All decorators support:
- Input/Output Recording
- Exception Handling
- Async/await functions
- Generator functions
- Custom attributes and names

## Integrations

AgentOps seamlessly integrates with popular AI agent frameworks and SDKs, enhancing your development workflow.

### OpenAI Agents SDK 🖇️

Natively integrate with the OpenAI Agents SDKs for Python and TypeScript to build multi-agent systems with tools, handoffs, and guardrails.

#### Python

```bash
pip install openai-agents
```

-   [Python integration guide](https://docs.agentops.ai/v2/integrations/openai_agents_python)
-   [OpenAI Agents Python documentation](https://openai.github.io/openai-agents-python/)

#### TypeScript

```bash
npm install agentops @openai/agents
```

-   [TypeScript integration guide](https://docs.agentops.ai/v2/integrations/openai_agents_js)
-   [OpenAI Agents JS documentation](https://openai.github.io/openai-agents-js)

### CrewAI 🛶

Enhance your CrewAI agents with observability in just two lines of code.  Set an `AGENTOPS_API_KEY` in your environment to start.

```bash
pip install 'crewai[agentops]'
```

-   [AgentOps integration example](https://docs.agentops.ai/v1/integrations/crewai)
-   [Official CrewAI documentation](https://docs.crewai.com/how-to/AgentOps-Observability)

### AG2 (AutoGen) 🤖

Add full observability and monitoring to AG2 (formerly AutoGen) agents with only two lines of code. Set an `AGENTOPS_API_KEY` in your environment and call `agentops.init()`.

-   [AG2 Observability Example](https://docs.ag2.ai/notebooks/agentchat_agentops)
-   [AG2 - AgentOps Documentation](https://docs.ag2.ai/docs/ecosystem/agentops)

### Camel AI 🐪

Track and analyze CAMEL agents with complete observability. Set an `AGENTOPS_API_KEY` in your environment and initialize AgentOps.

-   [Camel AI](https://www.camel-ai.org/) - Advanced agent communication framework
-   [AgentOps integration example](https://docs.agentops.ai/v1/integrations/camel)
-   [Official Camel AI documentation](https://docs.camel-ai.org/cookbooks/agents_tracking.html)

<details>
  <summary>Installation</summary>

```bash
pip install "camel-ai[all]==0.2.11"
pip install agentops
```

```python
import os
import agentops
from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType

# Initialize AgentOps
agentops.init(os.getenv("AGENTOPS_API_KEY"), tags=["CAMEL Example"])

# Import toolkits after AgentOps init for tracking
from camel.toolkits import SearchToolkit

# Set up the agent with search tools
sys_msg = BaseMessage.make_assistant_message(
    role_name='Tools calling operator',
    content='You are a helpful assistant'
)

# Configure tools and model
tools = [*SearchToolkit().get_tools()]
model = ModelFactory.create(
    model_platform=ModelPlatformType.OPENAI,
    model_type=ModelType.GPT_4O_MINI,
)

# Create and run the agent
camel_agent = ChatAgent(
    system_message=sys_msg,
    model=model,
    tools=tools,
)

response = camel_agent.step("What is AgentOps?")
print(response)

agentops.end_session("Success")
```

Check out our [Camel integration guide](https://docs.agentops.ai/v1/integrations/camel) for more examples including multi-agent scenarios.
</details>

### Langchain 🦜🔗

AgentOps integrates seamlessly with Langchain applications. Install the necessary dependencies:

<details>
  <summary>Installation</summary>

```shell
pip install agentops[langchain]
```

To use the handler, import and set

```python
import os
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from agentops.integration.callbacks.langchain import LangchainCallbackHandler

AGENTOPS_API_KEY = os.environ['AGENTOPS_API_KEY']
handler = LangchainCallbackHandler(api_key=AGENTOPS_API_KEY, tags=['Langchain Example'])

llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY,
                 callbacks=[handler],
                 model='gpt-3.5-turbo')

agent = initialize_agent(tools,
                         llm,
                         agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
                         verbose=True,
                         callbacks=[handler], # You must pass in a callback handler to record your agent
                         handle_parsing_errors=True)
```

Check out the [Langchain Examples Notebook](./examples/langchain_examples.ipynb) for more details including Async handlers.

</details>

### Cohere ⌨️

First class support for Cohere(>=5.4.0).  Please message us on Discord for any added functionality you may need!

-   [AgentOps integration example](https://docs.agentops.ai/v1/integrations/cohere)
-   [Official Cohere documentation](https://docs.cohere.com/reference/about)

<details>
  <summary>Installation</summary>

```bash
pip install cohere
```

```python python
import cohere
import agentops

# Beginning of program's code (i.e. main.py, __init__.py)
agentops.init(<INSERT YOUR API KEY HERE>)
co = cohere.Client()

chat = co.chat(
    message="Is it pronounced ceaux-hear or co-hehray?"
)

print(chat)

agentops.end_session('Success')
```

```python python
import cohere
import agentops

# Beginning of program's code (i.e. main.py, __init__.py)
agentops.init(<INSERT YOUR API KEY HERE>)

co = cohere.Client()

stream = co.chat_stream(
    message="Write me a haiku about the synergies between Cohere and AgentOps"
)

for event in stream:
    if event.event_type == "text-generation":
        print(event.text, end='')

agentops.end_session('Success')
```
</details>

### Anthropic ﹨

Track agents built with the Anthropic Python SDK (>=0.32.0).

-   [AgentOps integration guide](https://docs.agentops.ai/v1/integrations/anthropic)
-   [Official Anthropic documentation](https://docs.anthropic.com/en/docs/welcome)

<details>
  <summary>Installation</summary>

```bash
pip install anthropic
```

```python python
import anthropic
import agentops

# Beginning of program's code (i.e. main.py, __init__.py)
agentops.init(<INSERT YOUR API KEY HERE>)

client = anthropic.Anthropic(
    # This is the default and can be omitted
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
)

message = client.messages.create(
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": "Tell me a cool fact about AgentOps",
            }
        ],
        model="claude-3-opus-20240229",
    )
print(message.content)

agentops.end_session('Success')
```

Streaming
```python python
import anthropic
import agentops

# Beginning of program's code (i.e. main.py, __init__.py)
agentops.init(<INSERT YOUR API KEY HERE>)

client = anthropic.Anthropic(
    # This is the default and can be omitted
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
)

stream = client.messages.create(
    max_tokens=1024,
    model="claude-3-opus-20240229",
    messages=[
        {
            "role": "user",
            "content": "Tell me something cool about streaming agents",
        }
    ],
    stream=True,
)

response = ""
for event in stream:
    if event.type == "content_block_delta":
        response += event.delta.text
    elif event.type == "message_stop":
        print("\n")
        print(response)
        print("\n")
```

Async

```python python
import asyncio
from anthropic import AsyncAnthropic

client = AsyncAnthropic(
    # This is the default and can be omitted
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
)


async def main() -> None:
    message = await client.messages.create(
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": "Tell me something interesting about async agents",
            }
        ],
        model="claude-3-opus-20240229",
    )
    print(message.content)


await main()
```
</details>

### Mistral 〽️

Track agents built with the Mistral Python SDK (>=0.32.0).

-   [AgentOps integration example](./examples/mistral//mistral_example.ipynb)
-   [Official Mistral documentation](https://docs.mistral.ai)

<details>
  <summary>Installation</summary>

```bash
pip install mistralai
```

Sync

```python python
from mistralai import Mistral
import agentops

# Beginning of program's code (i.e. main.py, __init__.py)
agentops.init(<INSERT YOUR API KEY HERE>)

client = Mistral(
    # This is the default and can be omitted
    api_key=os.environ.get("MISTRAL_API_KEY"),
)

message = client.chat.complete(
        messages=[
            {
                "role": "user",
                "content": "Tell me a cool fact about AgentOps",
            }
        ],
        model="open-mistral-nemo",
    )
print(message.choices[0].message.content)

agentops.end_session('Success')
```

Streaming

```python python
from mistralai import Mistral
import agentops

# Beginning of program's code (i.e. main.py, __init__.py)
agentops.init(<INSERT YOUR API KEY HERE>)

client = Mistral(
    # This is the default and can be omitted
    api_key=os.environ.get("MISTRAL_API_KEY"),
)

message = client.chat.stream(
        messages=[
            {
                "role": "user",
                "content": "Tell me something cool about streaming agents",
            }
        ],
        model="open-mistral-nemo",
    )

response = ""
for event in message:
    if event.data.choices[0].finish_reason == "stop":
        print("\n")
        print(response)
        print("\n")
    else:
        response += event.text

agentops.end_session('Success')
```

Async

```python python
import asyncio
from mistralai import Mistral

client = Mistral(
    # This is the default and can be omitted
    api_key=os.environ.get("MISTRAL_API_KEY"),
)


async def main() -> None:
    message = await client.chat.complete_async(
        messages=[
            {
                "role": "user",
                "content": "Tell me something interesting about async agents",
            }
        ],
        model="open-mistral-nemo",
    )
    print(message.choices[0].message.content)


await main()
```

Async Streaming

```python python
import asyncio
from mistralai import Mistral

client = Mistral(
    # This is the default and can be omitted
    api_key=os.environ.get("MISTRAL_API_KEY"),
)


async def main() -> None:
    message = await client.chat.stream_async(
        messages=[
            {
                "role": "user",
                "content": "Tell me something interesting about async streaming agents",
            }
        ],
        model="open-mistral-nemo",
    )

    response = ""
    async for event in message:
        if event.data.choices[0].finish_reason == "stop":
            print("\n")
            print(response)
            print("\n")
        else:
            response += event.text


await main()
```
</details>

### CamelAI ﹨

Track agents built with the CamelAI Python SDK (>=0.32.0).

-   [CamelAI integration guide](https://docs.camel-ai.org/cookbooks/agents_tracking.html#)
-   [Official CamelAI documentation](https://docs.camel-ai.org/index.html)

<details>
  <summary>Installation</summary>

```bash
pip install camel-ai[all]
pip install agentops
```

```python python
#Import Dependencies
import agentops
import os
from getpass import getpass
from dotenv import load_dotenv

#Set Keys
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY") or "<your openai key here>"
agentops_api_key = os.getenv("AGENTOPS_API_KEY") or "<your agentops key here>"
```
</details>

[You can find usage examples here!](examples/camelai_examples/README.md).

### LiteLLM 🚅

AgentOps offers seamless support for LiteLLM(>=1.3.1), enabling you to call 100+ LLMs with a unified Input/Output format.

-   [AgentOps integration example](https://docs.agentops.ai/v1/integrations/litellm)
-   [Official LiteLLM documentation](https://docs.litellm.ai/docs/providers)

<details>
  <summary>Installation</summary>

```bash
pip install litellm
```

```python python
# Do not use LiteLLM like this
# from litellm import completion
# ...
# response = completion(model="claude-3", messages=messages)

# Use LiteLLM like this
import litellm
...
response = litellm.completion(model="claude-3", messages=messages)
# or
response = await litellm.acompletion(model="claude-3", messages=messages)
```
</details>

### LlamaIndex 🦙

AgentOps integrates flawlessly with LlamaIndex, a powerful framework for building context-augmented generative AI applications with LLMs.

<details>
  <summary>Installation</summary>

```shell
pip install llama-index-instrumentation-agentops
```

To use the handler, import and set

```python
from llama_index.core import set_global_handler

# NOTE: Feel free to set your AgentOps environment variables (e.g., 'AGENTOPS_API_KEY')
# as outlined in the AgentOps documentation, or pass the equivalent keyword arguments
# anticipated by AgentOps' AOClient as **eval_params in set_global_handler.

set_global_handler("agentops")
```

Check out the [LlamaIndex docs](https://docs.llamaindex.ai/en/stable/module_guides/observability/?h=agentops#agentops) for more details.

</details>

### Llama Stack 🦙🥞

AgentOps supports the Llama Stack Python Client(>=0.0.53), allowing you to monitor your Agentic applications.

-   [AgentOps integration example 1](https://github.com/AgentOps-AI/agentops/pull/530/files/65a5ab4fdcf310326f191d4b870d4f553591e3ea#diff-fdddf65549f3714f8f007ce7dfd1cde720329fe54155d54389dd50fbd81813cb)
-   [AgentOps integration example 2](https://github.com/AgentOps-AI/agentops/pull/530/files/65a5ab4fdcf310326f191d4b870d4f553591e3ea#diff-6688ff4fb7ab1ce7b1cc9b8362ca27264a3060c16737fb1d850305787a6e3699)
-   [Official Llama Stack Python Client](https://github.com/meta-llama/llama-stack-client-python)

### SwarmZero AI 🐝

Track and analyze SwarmZero agents with full observability. Set an `AGENTOPS_API_KEY` in your environment and initialize AgentOps to get started.

-   [SwarmZero](https://swarmzero.ai) - Advanced multi-agent framework
-   [AgentOps integration example](https://docs.agentops.ai/v1/integrations/swarmzero)
-   [SwarmZero AI integration example](https://docs.swarmzero.ai/examples/ai-agents/build-and-monitor-a-web-search-agent)
-   [SwarmZero AI - AgentOps documentation](https://docs.swarmzero.ai/sdk/observability/agentops)
-   [Official SwarmZero Python SDK](https://github.com/swarmzero/swarmzero)

<details>
  <summary>Installation</summary>

```bash
pip install swarmzero
pip install agentops
```

```python
from dotenv import load_dotenv
load_dotenv()

import agentops
agentops.init(<INSERT YOUR API KEY HERE>)

from swarmzero import Agent, Swarm
# ...
```
</details>

## Evaluations Roadmap 🧭

| Platform                                                                     | Dashboard                                  | Evals                                  |
| ---------------------------------------------------------------------------- | ------------------------------------------ | -------------------------------------- |
| ✅ Python SDK                                                                | ✅ Multi-session and Cross-session metrics | ✅ Custom eval metrics                 |
| 🚧 Evaluation builder API                                                    | ✅ Custom event tag tracking              | 🔜 Agent scorecards                    |
| 🚧 [Javascript/Typescript SDK (Alpha)](https://github.com/AgentOps-AI/agentops-node) | ✅ Session replays                         | 🔜 Evaluation playground + leaderboard |

## Debugging Roadmap 🧭

| Performance testing                       | Environments                                                                        | LLM Testing                                 | Reasoning and execution testing                   |
| ----------------------------------------- | ----------------------------------------------------------------------------------- | ------------------------------------------- | ------------------------------------------------- |
| ✅ Event latency analysis                 | 🔜 Non-stationary environment testing                                               | 🔜 LLM non-deterministic function detection | 🚧 Infinite loops and recursive thought detection |
| ✅ Agent workflow execution pricing       | 🔜 Multi-modal environments                                                         | 🚧 Token limit overflow flags               | 🔜 Faulty reasoning detection                     |
| 🚧 Success validators (external)          | 🔜 Execution containers                                                             | 🔜 Context limit overflow flags             | 🔜 Generative code validators                     |
| 🔜 Agent controllers/skill tests          | ✅ Honeypot and prompt injection detection ([PromptArmor](https://promptarmor.com)) | ✅ API bill tracking                        | 🔜 Error breakpoint analysis                      |
| 🔜 Information context constraint testing | 🔜 Anti-agent roadblocks (i.e. Captchas)                                            | 🔜 CI/CD integration checks                 |                                                   |
| 🔜 Regression testing                     | ✅ Multi-agent framework visualization                                              |                                             |                                                   |

### Why Choose AgentOps? 🤔

Without the right tools, AI agents can be slow, expensive, and unreliable. AgentOps is designed to transform your agent from prototype to production. Here's what makes AgentOps stand out:

-   **Comprehensive Observability**:  Track agent performance, user interactions, and API usage in detail.
-   **Real-Time Monitoring**: Gain instant insights with session replays, metrics, and live monitoring tools.
-   **Cost Control**: Monitor and manage your expenses on LLM and API calls.
-   **Failure Detection**: Quickly identify and address agent failures and issues in multi-agent interactions.
-   **Tool Usage Statistics**: Analyze how your agents are using external tools with comprehensive analytics.
-   **Session-Wide Metrics**:  Get a holistic view of your agents' sessions through detailed statistics.

AgentOps is designed to simplify agent observability, testing, and monitoring.

## Star History

See the growth of AgentOps in the community:

<img src="https://api.star-history.com/svg?repos=AgentOps-AI/agentops&type=Date" style="max-width: 500px" width="50%" alt="Star History" />

## Projects Using AgentOps

[Check out some of the popular projects that use AgentOps!](https://github.com/AgentOps-AI/agentops#popular-projects-using-agentops)

[Visit our GitHub repository for more details.](https://github.com/AgentOps-AI/agentops)
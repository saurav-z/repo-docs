<div align="center">
  <a href="https://agentops.ai?ref=gh">
    <img src="docs/images/external/logo/github-banner.png" alt="AgentOps Logo">
  </a>
</div>

<div align="center">
  <em>The ultimate observability and dev tool platform for AI agents, helping you build, evaluate, and monitor your AI agents from prototype to production.</em>
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

## AgentOps: The AI Agent Observability & DevTool Platform

AgentOps empowers developers to build, evaluate, and monitor AI agents, transforming the journey from prototype to production.

**Key Features:**

*   **Replay Analytics and Debugging**: Step-by-step agent execution graphs.
*   **LLM Cost Management**: Track spend with LLM foundation model providers.
*   **Framework Integrations**: Native Integrations with CrewAI, AG2 (AutoGen), Agno, LangGraph, & more.
*   **Self-Hosting**: Run AgentOps on your own cloud.

**Get Started Quickly:**

```bash
pip install agentops
```

**Integrate in 2 Lines of Code:**

1.  Get your [API Key](https://app.agentops.ai/settings/projects).
2.  Initialize the client:

    ```python
    import agentops

    # Beginning of your program (i.e. main.py, __init__.py)
    agentops.init( < INSERT YOUR API KEY HERE >)

    ...

    # End of program
    agentops.end_session('Success')
    ```

    View your sessions on the [AgentOps dashboard](https://app.agentops.ai?ref=gh).
<br/>

## Open Source

AgentOps is open source under the MIT license. Explore the code in our [app directory](https://github.com/AgentOps-AI/agentops/tree/main/app).

## Self-Hosting

Run the full AgentOps app (Dashboard + API backend) on your machine. Follow the setup guide in `app/README.md`:

-   [Run the App and Backend (Dashboard + API)](app/README.md)

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

## First Class Developer Experience

Enhance your agents with powerful observability using minimal code.

Refer to our [documentation](http://docs.agentops.ai) for more details on these decorators.

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

## Integrations 🦾

### [OpenAI Agents SDK](https://docs.agentops.ai/v2/integrations/openai_agents_python) 🖇️

Build multi-agent systems with tools, handoffs, and guardrails. AgentOps natively integrates with the OpenAI Agents SDKs.

#### Python

```bash
pip install openai-agents
```

#### TypeScript

```bash
npm install agentops @openai/agents
```

### [CrewAI](https://docs.agentops.ai/v1/integrations/crewai) 🛶

Add observability to Crew agents in just 2 lines of code using an `AGENTOPS_API_KEY`.

```bash
pip install 'crewai[agentops]'
```

### [AG2 (AutoGen)](https://docs.ag2.ai/docs/ecosystem/agentops) 🤖

Add full observability to AG2 agents with two lines of code.

### [Camel AI](https://docs.agentops.ai/v1/integrations/camel) 🐪

Track and analyze CAMEL agents with full observability.

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

### [Langchain](https://docs.agentops.ai/v1/integrations/langchain) 🦜🔗

Seamlessly integrate AgentOps with Langchain applications.

<details>
  <summary>Installation</summary>

```shell
pip install agentops[langchain]
```

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

Check out the [Langchain Examples Notebook](./examples/langchain/langchain_examples.ipynb) for more details including Async handlers.

</details>

### [Cohere](https://docs.agentops.ai/v1/integrations/cohere) ⌨️

First class support for Cohere(>=5.4.0).

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

### [Anthropic](https://docs.agentops.ai/v1/integrations/anthropic) ﹨

Track agents built with the Anthropic Python SDK (>=0.32.0).

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

### [Mistral](https://docs.agentops.ai/v1/integrations/mistral) 〽️

Track agents built with the Mistral Python SDK (>=0.32.0).

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

### [CamelAI](https://docs.camel-ai.org/cookbooks/agents_tracking.html#) ﹨

Track agents built with the CamelAI Python SDK (>=0.32.0).

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

### [LiteLLM](https://docs.litellm.ai/docs/providers) 🚅

AgentOps provides support for LiteLLM(>=1.3.1), allowing you to call 100+ LLMs using the same Input/Output Format.

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

### [LlamaIndex](https://docs.llamaindex.ai/en/stable/module_guides/observability/?h=agentops#agentops) 🦙

Seamlessly integrate AgentOps with LlamaIndex applications.

<details>
  <summary>Installation</summary>

```shell
pip install llama-index-instrumentation-agentops
```

```python
from llama_index.core import set_global_handler

# NOTE: Feel free to set your AgentOps environment variables (e.g., 'AGENTOPS_API_KEY')
# as outlined in the AgentOps documentation, or pass the equivalent keyword arguments
# anticipated by AgentOps' AOClient as **eval_params in set_global_handler.

set_global_handler("agentops")
```
</details>

### [Llama Stack](https://github.com/meta-llama/llama-stack-client-python) 🦙🥞

AgentOps provides support for Llama Stack Python Client(>=0.0.53), allowing you to monitor your Agentic applications.

### [SwarmZero AI](https://swarmzero.ai) 🐝

Track and analyze SwarmZero agents with full observability.

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

## Star History

Check out our growth in the community:

<img src="https://api.star-history.com/svg?repos=AgentOps-AI/agentops&type=Date" style="max-width: 500px" width="50%" alt="Logo">

## Popular projects using AgentOps

| Repository | Stars  |
| :--------  | -----: |
|<img class="avatar mr-2" src="https://avatars.githubusercontent.com/u/2707039?s=40&v=4" width="20" height="20" alt="">  &nbsp; [geekan](https://github.com/geekan) / [MetaGPT](https://github.com/geekan/MetaGPT) | 42787 |
|<img class="avatar mr-2" src="https://avatars.githubusercontent.com/u/130722866?s=40&v=4" width="20" height="20" alt="">  &nbsp; [run-llama](https://github.com/run-llama) / [llama_index](https://github.com/run-llama/llama_index) | 34446 |
|<img class="avatar mr-2" src="https://avatars.githubusercontent.com/u/170677839?s=40&v=4" width="20" height="20" alt="">  &nbsp; [crewAIInc](https://github.com/crewAIInc) / [crewAI](https://github.com/crewAIInc/crewAI) | 18287 |
|<img class="avatar mr-2" src="https://avatars.githubusercontent.com/u/134388954?s=40&v=4" width="20" height="20" alt="">  &nbsp; [camel-ai](https://github.com/camel-ai) / [camel](https://github.com/camel-ai/camel) | 5166 |
|<img class="avatar mr-2" src="https://avatars.githubusercontent.com/u/152537519?s=40&v=4" width="20" height="20" alt="">  &nbsp; [superagent-ai](https://github.com/superagent-ai) / [superagent](https://github.com/superagent-ai/superagent) | 5050 |
|<img class="avatar mr-2" src="https://avatars.githubusercontent.com/u/30197649?s=40&v=4" width="20" height="20" alt="">  &nbsp; [iyaja](https://github.com/iyaja) / [llama-fs](https://github.com/iyaja/llama-fs) | 4713 |
|<img class="avatar mr-2" src="https://avatars.githubusercontent.com/u/162546372?s=40&v=4" width="20" height="20" alt="">  &nbsp; [BasedHardware](https://github.com/BasedHardware) / [Omi](https://github.com/BasedHardware/Omi) | 2723 |
|<img class="avatar mr-2" src="https://avatars.githubusercontent.com/u/454862?s=40&v=4" width="20" height="20" alt="">  &nbsp; [MervinPraison](https://github.com/MervinPraison) / [PraisonAI](https://github.com/MervinPraison/PraisonAI) | 2007 |
|<img class="avatar mr-2" src="https://avatars.githubusercontent.com/u/140554352?s=40&v=4" width="20" height="20" alt="">  &nbsp; [AgentOps-AI](https://github.com/AgentOps-AI) / [Jaiqu](https://github.com/AgentOps-AI/Jaiqu) | 272 |
|<img class="avatar mr-2" src="https://avatars.githubusercontent.com/u/173542722?s=48&v=4" width="20" height="20" alt="">  &nbsp; [swarmzero](https://github.com/swarmzero) / [swarmzero](https://github.com/swarmzero/swarmzero) | 195 |
|<img class="avatar mr-2" src="https://avatars.githubusercontent.com/u/3074263?s=40&v=4" width="20" height="20" alt="">  &nbsp; [strnad](https://github.com/strnad) / [CrewAI-Studio](https://github.com/strnad/CrewAI-Studio) | 134 |
|<img class="avatar mr-2" src="https://avatars.githubusercontent.com/u/18406448?s=40&v=4" width="20" height="20" alt="">  &nbsp; [alejandro-ao](https://github.com/alejandro-ao) / [exa-crewai](https://github.com/alejandro-ao/exa-crewai) | 55 |
|<img class="avatar mr-2" src="https://avatars.githubusercontent.com/u/64493665?s=40&v=4" width="20" height="20" alt="">  &nbsp; [tonykipkemboi](https://github.com/tonykipkemboi) / [youtube_yapper_trapper](https://github.com/tonykipkemboi/youtube_yapper_trapper) | 47 |
|<img class="avatar mr-2" src="https://avatars.githubusercontent.com/u/17598928?s=40&v=4" width="20" height="20" alt="">  &nbsp; [sethcoast](https://github.com/sethcoast) / [cover-letter-builder](https://github.com/sethcoast/cover-letter-builder) | 27 |
|<img class="avatar mr-2" src="https://avatars.githubusercontent.com/u/109994880?s=40&v=4" width="20" height="20" alt="">  &nbsp; [bhancockio](https://github.com/bhancockio) / [chatgpt4o-analysis](https://github.com/bhancockio/chatgpt4o-analysis) | 19 |
|<img class="avatar mr-2" src="https://avatars.githubusercontent.com/u/14105911?s=40&v=4" width="20" height="20" alt="">  &nbsp; [breakstring](https://github.com/breakstring) / [Agentic_Story_Book_Workflow](https://github.com/breakstring/Agentic_Story_Book_Workflow) | 14 |
|<img class="avatar mr-2" src="https://avatars.githubusercontent.com/u/124134656?s=40&v=4" width="20" height="20" alt="">  &nbsp; [MULTI-ON](https://github.com/MULTI-ON) / [multion-python](https://github.com/MULTI-ON/multion-python) | 13 |

_Generated using [github-dependents-info](https://github.com/nvuillam/github-dependents-info), by [Nicolas Vuillamy](https://github.com/nvuillam)_

**[Back to top](#)**
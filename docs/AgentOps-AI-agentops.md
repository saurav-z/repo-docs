<div align="center">
  <a href="https://agentops.ai?ref=gh">
    <img src="docs/images/external/logo/github-banner.png" alt="AgentOps Logo">
  </a>
</div>

<div align="center">
  <em>Supercharge your AI agent development with AgentOps: the ultimate observability and devtool platform.</em>
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

## AgentOps: Build, Evaluate, and Monitor Your AI Agents with Ease

AgentOps empowers developers to build, evaluate, and monitor AI agents effectively.  From the initial prototype to full production deployment, AgentOps provides the tools you need.

**Key Features:**

*   📊 **Replay Analytics and Debugging:** Step-by-step agent execution graphs for detailed insights.
*   💸 **LLM Cost Management:** Track and manage your LLM spend with precision.
*   🤝 **Framework Integrations:** Seamless integration with popular frameworks like CrewAI, AG2 (AutoGen), LangGraph, and more.
*   ⚒️ **Self-Hosting:**  Run AgentOps on your own infrastructure.

## Quick Start: Get Started in Minutes

Integrate AgentOps into your project in just a few lines of code.

1.  **Install the AgentOps Python package:**

    ```bash
    pip install agentops
    ```

2.  **Get Your API Key:**  [Get an API key](https://app.agentops.ai/settings/projects)
3.  **Initialize AgentOps:**

    ```python
    import agentops

    # At the beginning of your program (e.g., main.py, __init__.py)
    agentops.init("<INSERT YOUR API KEY HERE>")

    # ... Your agent code ...

    # End your session
    agentops.end_session('Success')
    ```

4.  **View Your Data:** Access your agent session data, analytics, and debugging information on the [AgentOps dashboard](https://app.agentops.ai?ref=gh).

## Deep Dive into Agent Debugging
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

## Comprehensive Session Replays
<details>
  <summary>Session Replays</summary>
  <a href="https://app.agentops.ai?ref=gh">
    <img src="docs/images/external/app_screenshots/session-replay.png" style="width: 90%;" alt="Session Replays"/>
  </a>
</details>

## Analytical Summary Dashboards
<details>
  <summary>Summary Analytics</summary>
  <a href="https://app.agentops.ai?ref=gh">
   <img src="docs/images/external/app_screenshots/overview.png" style="width: 90%;" alt="Summary Analytics"/>
  </a>
  <a href="https://app.agentops.ai?ref=gh">
   <img src="docs/images/external/app_screenshots/overview-charts.png" style="width: 90%;" alt="Summary Analytics Charts"/>
  </a>
</details>

## Self-Hosting AgentOps

Want to have full control? Run the complete AgentOps application (Dashboard + API backend) on your own machine.  Follow the setup instructions in the `app/README.md` file:

-   [Run the App and Backend (Dashboard + API)](app/README.md)

## Developer Experience: Effortless Observability

Enhance your agents with powerful observability with minimal code:

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

For more examples, refer to our [documentation](http://docs.agentops.ai).

## Integrations: Enhance Your AI Agent Frameworks

AgentOps seamlessly integrates with a wide array of popular AI agent frameworks.

### OpenAI Agents SDK 🖇️

Full integration with the OpenAI Agents SDKs for both Python and TypeScript.

*   **Python**

    ```bash
    pip install openai-agents
    ```

    *   [Python integration guide](https://docs.agentops.ai/v2/integrations/openai_agents_python)
    *   [OpenAI Agents Python documentation](https://openai.github.io/openai-agents-python/)
*   **TypeScript**

    ```bash
    npm install agentops @openai/agents
    ```

    *   [TypeScript integration guide](https://docs.agentops.ai/v2/integrations/openai_agents_js)
    *   [OpenAI Agents JS documentation](https://openai.github.io/openai-agents-js)

### CrewAI 🛶

Easily monitor your CrewAI agents with AgentOps in just two lines of code.

```bash
pip install 'crewai[agentops]'
```

*   [AgentOps integration example](https://docs.agentops.ai/v1/integrations/crewai)
*   [Official CrewAI documentation](https://docs.crewai.com/how-to/AgentOps-Observability)

### AG2 (AutoGen) 🤖

Get full observability for your AG2 (formerly AutoGen) agents with a few lines of code.

*   [AG2 Observability Example](https://docs.ag2.ai/notebooks/agentchat_agentops)
*   [AG2 - AgentOps Documentation](https://docs.ag2.ai/docs/ecosystem/agentops)

### Camel AI 🐪

Track and analyze CAMEL agents with full observability.

*   [Camel AI](https://www.camel-ai.org/) - Advanced agent communication framework
*   [AgentOps integration example](https://docs.agentops.ai/v1/integrations/camel)
*   [Official Camel AI documentation](https://docs.camel-ai.org/cookbooks/agents_tracking.html)
<details>
  <summary>Installation and example code</summary>
    ```bash
    pip install "camel-ai[all]==0.2.11"
    pip install agentops
    ```

    ```python
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

### Langchain 🦜🔗

Use AgentOps with your Langchain applications.

<details>
  <summary>Installation and code</summary>
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

### Cohere ⌨️

Get first-class support for Cohere integration (>=5.4.0).

*   [AgentOps integration example](https://docs.agentops.ai/v1/integrations/cohere)
*   [Official Cohere documentation](https://docs.cohere.com/reference/about)
<details>
  <summary>Installation and example code</summary>
  ```bash
  pip install cohere
  ```

  ```python
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

  ```python
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

*   [AgentOps integration guide](https://docs.agentops.ai/v1/integrations/anthropic)
*   [Official Anthropic documentation](https://docs.anthropic.com/en/docs/welcome)

<details>
  <summary>Installation and example code</summary>
  ```bash
  pip install anthropic
  ```

  ```python
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
  ```python
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
  ```python
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

*   [AgentOps integration example](./examples/mistral//mistral_example.ipynb)
*   [Official Mistral documentation](https://docs.mistral.ai)

<details>
  <summary>Installation and example code</summary>
  ```bash
  pip install mistralai
  ```

  Sync
  ```python
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
  ```python
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
  ```python
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
  ```python
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

### CamelAI 🐪

Track agents built with the CamelAI Python SDK (>=0.32.0).

*   [CamelAI integration guide](https://docs.camel-ai.org/cookbooks/agents_tracking.html#)
*   [Official CamelAI documentation](https://docs.camel-ai.org/index.html)

<details>
  <summary>Installation and code</summary>
  ```bash
  pip install camel-ai[all]
  pip install agentops
  ```

  ```python
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

You can find usage examples [here](examples/camelai_examples/README.md).

### LiteLLM 🚅

AgentOps offers robust support for LiteLLM (>=1.3.1), enabling you to interact with 100+ LLMs using a consistent input/output format.

*   [AgentOps integration example](https://docs.agentops.ai/v1/integrations/litellm)
*   [Official LiteLLM documentation](https://docs.litellm.ai/docs/providers)
<details>
  <summary>Installation and example code</summary>
  ```bash
  pip install litellm
  ```

  ```python
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

AgentOps seamlessly integrates with LlamaIndex, simplifying context-augmented generative AI application development.

<details>
  <summary>Installation and example code</summary>
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
  Check out the [LlamaIndex docs](https://docs.llamaindex.ai/en/stable/module_guides/observability/?h=agentops#agentops) for more details.
</details>

### Llama Stack 🦙🥞

Monitor your Agentic applications effortlessly with AgentOps support for the Llama Stack Python Client (>=0.0.53).

*   [AgentOps integration example 1](https://github.com/AgentOps-AI/agentops/pull/530/files/65a5ab4fdcf310326f191d4b870d4f553591e3ea#diff-fdddf65549f3714f8f007ce7dfd1cde720329fe54155d54389dd50fbd81813cb)
*   [AgentOps integration example 2](https://github.com/AgentOps-AI/agentops/pull/530/files/65a5ab4fdcf310326f191d4b870d4f553591e3ea#diff-6688ff4fb7ab1ce7b1cc9b8362ca27264a3060c16737fb1d850305787a6e3699)
*   [Official Llama Stack Python Client](https://github.com/meta-llama/llama-stack-client-python)

### SwarmZero AI 🐝

Monitor and analyze your SwarmZero agents.

*   [SwarmZero](https://swarmzero.ai) - Advanced multi-agent framework
*   [AgentOps integration example](https://docs.agentops.ai/v1/integrations/swarmzero)
*   [SwarmZero AI integration example](https://docs.swarmzero.ai/examples/ai-agents/build-and-monitor-a-web-search-agent)
*   [SwarmZero AI - AgentOps documentation](https://docs.swarmzero.ai/sdk/observability/agentops)
*   [Official SwarmZero Python SDK](https://github.com/swarmzero/swarmzero)
<details>
  <summary>Installation and code</summary>
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

## Evaluations Roadmap: Drive Performance

*   ✅ Python SDK
*   ✅ Multi-session and Cross-session metrics
*   ✅ Custom eval metrics
*   🚧 Evaluation builder API
*   ✅ Custom event tag tracking
*   🔜 Agent scorecards
*   🚧 [Javascript/Typescript SDK (Alpha)](https://github.com/AgentOps-AI/agentops-node)
*   ✅ Session replays
*   🔜 Evaluation playground + leaderboard

## Debugging Roadmap: Advanced Diagnostics

*   ✅ Event latency analysis
*   🔜 Non-stationary environment testing
*   🔜 LLM non-deterministic function detection
*   🚧 Infinite loops and recursive thought detection
*   ✅ Agent workflow execution pricing
*   🔜 Multi-modal environments
*   🚧 Token limit overflow flags
*   🔜 Faulty reasoning detection
*   🚧 Success validators (external)
*   🔜 Execution containers
*   🔜 Context limit overflow flags
*   🔜 Generative code validators
*   🔜 Agent controllers/skill tests
*   ✅ Honeypot and prompt injection detection ([PromptArmor](https://promptarmor.com))
*   ✅ API bill tracking
*   🔜 Error breakpoint analysis
*   🔜 Information context constraint testing
*   🔜 Anti-agent roadblocks (i.e. Captchas)
*   🔜 CI/CD integration checks
*   🔜 Regression testing
*   ✅ Multi-agent framework visualization

### Why AgentOps?

AgentOps transforms how you build and operate AI agents by providing essential tools for:

*   **Comprehensive Observability:** Track performance, user interactions, and API usage.
*   **Real-Time Monitoring:** Gain instant insights with session replays, metrics, and live monitoring tools.
*   **Cost Control:** Monitor and manage your LLM and API call expenses effectively.
*   **Failure Detection:** Quickly identify and resolve agent failures and multi-agent interaction issues.
*   **Tool Usage Statistics:** Understand how your agents utilize external tools.
*   **Session-Wide Metrics:** Get a holistic view of agent sessions.

###  Join the Community:  Learn More

Discover the power of AgentOps.  Explore the [AgentOps repository](https://github.com/AgentOps-AI/agentops) for more information and to get started.
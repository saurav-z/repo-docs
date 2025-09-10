<div align="center">
  <a href="https://agentops.ai?ref=gh">
    <img src="docs/images/external/logo/github-banner.png" alt="AgentOps Logo">
  </a>
</div>

<div align="center">
  <em>**Unlock the Power of AI Agents: AgentOps - Your Complete Observability and DevTool Platform**</em>
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

**AgentOps empowers developers to build, evaluate, and monitor AI agents, transforming them from prototype to production, all in one platform.**  This open-source platform provides comprehensive observability and powerful development tools for AI agents.

## Key Features

*   **Observability and Debugging:** Dive deep into agent execution with step-by-step graphs and detailed session replays.
*   **LLM Cost Management:** Track and analyze your spending with different LLM providers.
*   **Extensive Integrations:** Seamlessly integrate with leading AI frameworks including OpenAI Agents SDK, CrewAI, AG2 (AutoGen), Camel AI, Langchain, Cohere, Anthropic, Mistral, LlamaIndex, SwarmZero AI, Llama Stack and LiteLLM.
*   **Self-Hosting:** Run AgentOps on your own infrastructure with our self-hosting options.

## Quick Start

1.  **Install AgentOps:**

    ```bash
    pip install agentops
    ```

2.  **Get Your API Key:**  [Obtain your API key](https://app.agentops.ai/settings/projects).

3.  **Initialize and Monitor:** Add two lines of code to start session replays in your agents.

    ```python
    import agentops

    # Initialize AgentOps at the beginning of your program (e.g., main.py)
    agentops.init("<YOUR_API_KEY>")

    # At the end of your program
    agentops.end_session("Success")
    ```

    View all your agent sessions on the [AgentOps dashboard](https://app.agentops.ai?ref=gh).

## Self-Hosting

Want complete control? Run the AgentOps app and backend (dashboard + API) on your own machine. Follow the setup guide in `app/README.md`:

*   [Run the App and Backend (Dashboard + API)](app/README.md)

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

Enhance your AI agents with powerful observability using minimal code:

*   **Session Spans:**
    ```python
    from agentops.sdk.decorators import session

    @session
    def my_workflow():
        # Your session code here
        return result
    ```
*   **Agent Spans:**
    ```python
    from agentops.sdk.decorators import agent

    @agent
    class MyAgent:
        def __init__(self, name):
            self.name = name
    ```
*   **Operation/Task Spans:**
    ```python
    from agentops.sdk.decorators import operation, task

    @operation  # or @task
    def process_data(data):
        # Process the data
        return result
    ```
*   **Workflow Spans:**
    ```python
    from agentops.sdk.decorators import workflow

    @workflow
    def my_workflow(data):
        # Workflow implementation
        return result
    ```

**All decorators support:** Input/Output recording, exception handling, async/await functions, generator functions, and custom attributes.

## Integrations 🦾

### OpenAI Agents SDK 🖇️
AgentOps natively integrates with the OpenAI Agents SDKs for both Python and TypeScript.
*   **Python:**
    ```bash
    pip install openai-agents
    ```
    *   [Python integration guide](https://docs.agentops.ai/v2/integrations/openai_agents_python)
    *   [OpenAI Agents Python documentation](https://openai.github.io/openai-agents-python/)
*   **TypeScript:**
    ```bash
    npm install agentops @openai/agents
    ```
    *   [TypeScript integration guide](https://docs.agentops.ai/v2/integrations/openai_agents_js)
    *   [OpenAI Agents JS documentation](https://openai.github.io/openai-agents-js)

### CrewAI 🛶

Monitor Crew agents in just 2 lines of code.
```bash
pip install 'crewai[agentops]'
```

*   [AgentOps integration example](https://docs.agentops.ai/v1/integrations/crewai)
*   [Official CrewAI documentation](https://docs.crewai.com/how-to/AgentOps-Observability)

### AG2 🤖
Integrate with AG2 (formerly AutoGen) agents to add full observability and monitoring.
*   [AG2 Observability Example](https://docs.ag2.ai/notebooks/agentchat_agentops)
*   [AG2 - AgentOps Documentation](https://docs.ag2.ai/docs/ecosystem/agentops)

### Camel AI 🐪

Track and analyze CAMEL agents with full observability.
*   [Camel AI](https://www.camel-ai.org/) - Advanced agent communication framework
*   [AgentOps integration example](https://docs.agentops.ai/v1/integrations/camel)
*   [Official Camel AI documentation](https://docs.camel-ai.org/cookbooks/agents_tracking.html)

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
  Check out our [Camel integration guide](https://docs.agentops.ai/v1/integrations/camel) for more examples.
</details>

### Langchain 🦜🔗

AgentOps works seamlessly with applications built using Langchain.
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
                           callbacks=[handler],
                           handle_parsing_errors=True)
  ```
  Check out the [Langchain Examples Notebook](./examples/langchain/langchain_examples.ipynb) for more details.
</details>

### Cohere ⌨️

First class support for Cohere(>=5.4.0).

*   [AgentOps integration example](https://docs.agentops.ai/v1/integrations/cohere)
*   [Official Cohere documentation](https://docs.cohere.com/reference/about)

<details>
  <summary>Installation</summary>
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
  <summary>Installation</summary>
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
  <summary>Installation</summary>
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

### CamelAI ﹨

Track agents built with the CamelAI Python SDK (>=0.32.0).

*   [CamelAI integration guide](https://docs.camel-ai.org/cookbooks/agents_tracking.html#)
*   [Official CamelAI documentation](https://docs.camel-ai.org/index.html)

<details>
  <summary>Installation</summary>
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

[You can find usage examples here!](examples/camelai_examples/README.md).

### LiteLLM 🚅

AgentOps supports LiteLLM(>=1.3.1), enabling you to call 100+ LLMs using the same Input/Output format.

*   [AgentOps integration example](https://docs.agentops.ai/v1/integrations/litellm)
*   [Official LiteLLM documentation](https://docs.litellm.ai/docs/providers)

<details>
  <summary>Installation</summary>
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

AgentOps works seamlessly with LlamaIndex for building context-augmented generative AI applications with LLMs.
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
  Check out the [LlamaIndex docs](https://docs.llamaindex.ai/en/stable/module_guides/observability/?h=agentops#agentops) for more details.
</details>

### Llama Stack 🦙🥞
AgentOps provides support for Llama Stack Python Client(>=0.0.53), allowing you to monitor your Agentic applications.
*   [AgentOps integration example 1](https://github.com/AgentOps-AI/agentops/pull/530/files/65a5ab4fdcf310326f191d4b870d4f553591e3ea#diff-fdddf65549f3714f8f007ce7dfd1cde720329fe54155d54389dd50fbd81813cb)
*   [AgentOps integration example 2](https://github.com/AgentOps-AI/agentops/pull/530/files/65a5ab4fdcf310326f191d4b870d4f553591e3ea#diff-6688ff4fb7ab1ce7b1cc9b8362ca27264a3060c16737fb1d850305787a6e3699)
*   [Official Llama Stack Python Client](https://github.com/meta-llama/llama-stack-client-python)

### SwarmZero AI 🐝
Track and analyze SwarmZero agents.
*   [SwarmZero](https://swarmzero.ai) - Advanced multi-agent framework
*   [AgentOps integration example](https://docs.agentops.ai/v1/integrations/swarmzero)
*   [SwarmZero AI integration example](https://docs.swarmzero.ai/examples/ai-agents/build-and-monitor-a-web-search-agent)
*   [SwarmZero AI - AgentOps documentation](https://docs.swarmzero.ai/sdk/observability/agentops)
*   [Official SwarmZero Python SDK](https://github.com/swarmzero/swarmzero)

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

*   **Platform:** Python SDK
    *   **Dashboard:**  Multi-session and Cross-session metrics
    *   **Evals:** Custom eval metrics
*   **Platform:** Evaluation builder API
    *   **Dashboard:** Custom event tag tracking
    *   **Evals:**  🔜 Agent scorecards
*   **Platform:** [Javascript/Typescript SDK (Alpha)](https://github.com/AgentOps-AI/agentops-node)
    *   **Dashboard:** Session replays
    *   **Evals:**  🔜 Evaluation playground + leaderboard

## Debugging Roadmap 🧭

*   **Performance Testing:** Event latency analysis
    *   **Environments:** 🔜 Non-stationary environment testing
    *   **LLM Testing:** 🔜 LLM non-deterministic function detection
    *   **Reasoning and execution testing:** 🚧 Infinite loops and recursive thought detection
*   **Performance Testing:** Agent workflow execution pricing
    *   **Environments:** 🔜 Multi-modal environments
    *   **LLM Testing:** 🚧 Token limit overflow flags
    *   **Reasoning and execution testing:** 🔜 Faulty reasoning detection
*   **Performance Testing:** 🚧 Success validators (external)
    *   **Environments:** 🔜 Execution containers
    *   **LLM Testing:** 🔜 Context limit overflow flags
    *   **Reasoning and execution testing:** 🔜 Generative code validators
*   **Performance Testing:** 🔜 Agent controllers/skill tests
    *   **Environments:** ✅ Honeypot and prompt injection detection ([PromptArmor](https://promptarmor.com))
    *   **LLM Testing:** ✅ API bill tracking
    *   **Reasoning and execution testing:** 🔜 Error breakpoint analysis
*   **Performance Testing:** 🔜 Information context constraint testing
    *   **Environments:** 🔜 Anti-agent roadblocks (i.e. Captchas)
    *   **LLM Testing:** 🔜 CI/CD integration checks
    *   **Reasoning and execution testing:**
*   **Performance Testing:** 🔜 Regression testing
    *   **Environments:** ✅ Multi-agent framework visualization
    *   **LLM Testing:**

### Why AgentOps? 🤔

AgentOps gives you the tools to transform AI agent prototypes into production-ready applications.

-   **Comprehensive Observability**: Monitor performance, interactions, and API usage.
-   **Real-Time Monitoring**: Get instant insights with session replays and live metrics.
-   **Cost Control**: Manage your LLM and API spending.
-   **Failure Detection**: Identify and resolve agent and multi-agent issues swiftly.
-   **Tool Usage Statistics**: Understand how your agents use external tools.
-   **Session-Wide Metrics**: Gain a holistic view of your agents' sessions.

AgentOps simplifies agent observability, testing, and monitoring.

## Star History

See our growth in the community:

<img src="https://api.star-history.com/svg?repos=AgentOps-AI/agentops&type=Date" style="max-width: 500px" width="50%" alt="Logo">

## Projects using AgentOps

[See table of projects using AgentOps above]

[Back to top](#) ([AgentOps Repo](https://github.com/AgentOps-AI/agentops))
<div align="center">
  <a href="https://agentops.ai?ref=gh">
    <img src="docs/images/external/logo/github-banner.png" alt="AgentOps Logo">
  </a>
</div>

<div align="center">
  <em>The Ultimate Observability and DevTool Platform for AI Agents</em>
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

## AgentOps: Build, Monitor, and Optimize Your AI Agents (From Prototype to Production)

AgentOps is the all-in-one platform for developers building and deploying AI agents.  Get complete observability, robust testing, and powerful debugging tools to bring your agents from concept to reality.  [Check out the AgentOps GitHub Repository](https://github.com/AgentOps-AI/agentops)

### Key Features

*   **Replay Analytics and Debugging**: Step-by-step agent execution graphs and session replays.
*   **LLM Cost Management**: Track and manage your spend across various LLM providers.
*   **Agent Benchmarking**: Test your agents against a library of 1,000+ evaluations.
*   **Compliance and Security**:  Detect and mitigate prompt injection and data exfiltration vulnerabilities.
*   **Framework Integrations**: Native support for CrewAI, AG2 (AutoGen), LangGraph, and more.
*   **Self-Hosting**:  Run AgentOps on your own infrastructure.

### Quick Start

Easily integrate AgentOps into your Python projects and get started with powerful agent observability in minutes.

1.  **Install AgentOps:**

    ```bash
    pip install agentops
    ```
2.  **Get Your API Key:**  Sign up at [AgentOps Dashboard](https://app.agentops.ai/settings/projects) to get your free API key.

3.  **Initialize AgentOps:** Add a single line of code to your project to start capturing data.

    ```python
    import agentops

    # Beginning of your program (i.e. main.py, __init__.py)
    agentops.init( "<INSERT YOUR API KEY HERE>" )

    ...

    # End of program
    agentops.end_session('Success')
    ```

4.  **View Your Data:**  All session data is available on the [AgentOps dashboard](https://app.agentops.ai?ref=gh).

### Self-Hosting

Want complete control? Run the full AgentOps platform (Dashboard + API backend) on your own server.

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

### First-Class Developer Experience

Enhance your agents with powerful observability using minimal code. Utilize our decorators for enhanced tracking and analytics.

Refer to our [documentation](http://docs.agentops.ai) for complete integration details.

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

### Integrations 🦾

AgentOps seamlessly integrates with popular frameworks to provide comprehensive observability for your AI agents.

*   **OpenAI Agents SDK 🖇️**

    *   Build multi-agent systems with tools, handoffs, and guardrails. AgentOps natively integrates with the OpenAI Agents SDKs for both Python and TypeScript.
    *   Python: `pip install openai-agents` - [Python Integration Guide](https://docs.agentops.ai/v2/integrations/openai_agents_python)
    *   TypeScript: `npm install agentops @openai/agents` - [TypeScript Integration Guide](https://docs.agentops.ai/v2/integrations/openai_agents_js)

*   **CrewAI 🛶**

    *   Build Crew agents with observability in just 2 lines of code. Simply set an `AGENTOPS_API_KEY` in your environment.
    *   `pip install 'crewai[agentops]'` - [AgentOps integration example](https://docs.agentops.ai/v1/integrations/crewai)

*   **AG2 (AutoGen) 🤖**

    *   With only two lines of code, add full observability and monitoring to AG2 (formerly AutoGen) agents. Set an `AGENTOPS_API_KEY` in your environment.
    *   [AG2 Observability Example](https://docs.ag2.ai/notebooks/agentchat_agentops)

*   **Camel AI 🐪**

    *   Track and analyze CAMEL agents with full observability.
    *   [Camel AI](https://www.camel-ai.org/) - Advanced agent communication framework
    *   [AgentOps integration example](https://docs.agentops.ai/v1/integrations/camel)

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

*   **Langchain 🦜🔗**

    *   AgentOps works seamlessly with Langchain applications.
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

*   **Cohere ⌨️**

    *   First class support for Cohere(>=5.4.0).
    *   [AgentOps integration example](https://docs.agentops.ai/v1/integrations/cohere)

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

*   **Anthropic ﹨**

    *   Track agents built with the Anthropic Python SDK (>=0.32.0).
    *   [AgentOps integration guide](https://docs.agentops.ai/v1/integrations/anthropic)

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

*   **Mistral 〽️**

    *   Track agents built with the Mistral Python SDK (>=0.32.0).
    *   [AgentOps integration example](./examples/mistral//mistral_example.ipynb)

    ```bash
    pip install mistralai
    ```

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

*   **CamelAI ﹨**

    *   Track agents built with the CamelAI Python SDK (>=0.32.0).
    *   [CamelAI integration guide](https://docs.camel-ai.org/cookbooks/agents_tracking.html#)

    ```bash
    pip install camel-ai[all]
    pip install agentops
    ```

    ```python
    import agentops
    import os
    from getpass import getpass
    from dotenv import load_dotenv

    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY") or "<your openai key here>"
    agentops_api_key = os.getenv("AGENTOPS_API_KEY") or "<your agentops key here>"
    ```

*   **LiteLLM 🚅**

    *   AgentOps provides support for LiteLLM(>=1.3.1), allowing you to call 100+ LLMs using the same Input/Output Format.
    *   [AgentOps integration example](https://docs.agentops.ai/v1/integrations/litellm)

    ```bash
    pip install litellm
    ```

    ```python
    import litellm
    ...
    response = litellm.completion(model="claude-3", messages=messages)
    # or
    response = await litellm.acompletion(model="claude-3", messages=messages)
    ```

*   **LlamaIndex 🦙**

    ```shell
    pip install llama-index-instrumentation-agentops
    ```

    ```python
    from llama_index.core import set_global_handler

    set_global_handler("agentops")
    ```

*   **Llama Stack 🦙🥞**

    *   AgentOps provides support for Llama Stack Python Client(>=0.0.53), allowing you to monitor your Agentic applications.
    *   [AgentOps integration example](https://github.com/AgentOps-AI/agentops/pull/530/files/65a5ab4fdcf310326f191d4b870d4f553591e3ea#diff-fdddf65549f3714f8f007ce7dfd1cde720329fe54155d54389dd50fbd81813cb)
    *   [Official Llama Stack Python Client](https://github.com/meta-llama/llama-stack-client-python)

*   **SwarmZero AI 🐝**

    *   Track and analyze SwarmZero agents with full observability.
    *   [SwarmZero](https://swarmzero.ai) - Advanced multi-agent framework
    *   [AgentOps integration example](https://docs.swarmzero.ai/examples/ai-agents/build-and-monitor-a-web-search-agent)

    ```bash
    pip install swarmzero
    pip install agentops
    ```

    ```python
    from swarmzero import Agent, Swarm
    # ...
    ```

### Evaluations Roadmap 🧭

| Platform                                | Dashboard                                  | Evals                                  |
| --------------------------------------- | ------------------------------------------ | -------------------------------------- |
| ✅ Python SDK                            | ✅ Multi-session and Cross-session metrics | ✅ Custom eval metrics                 |
| 🚧 Evaluation builder API               | ✅ Custom event tag tracking              | 🔜 Agent scorecards                    |
| 🚧 [Javascript/Typescript SDK (Alpha)](https://github.com/AgentOps-AI/agentops-node) | ✅ Session replays                         | 🔜 Evaluation playground + leaderboard |

### Debugging Roadmap 🧭

| Performance testing                       | Environments                                                                        | LLM Testing                                 | Reasoning and execution testing                   |
| ----------------------------------------- | ----------------------------------------------------------------------------------- | ------------------------------------------- | ------------------------------------------------- |
| ✅ Event latency analysis                 | 🔜 Non-stationary environment testing                                               | 🔜 LLM non-deterministic function detection | 🚧 Infinite loops and recursive thought detection |
| ✅ Agent workflow execution pricing       | 🔜 Multi-modal environments                                                         | 🚧 Token limit overflow flags               | 🔜 Faulty reasoning detection                     |
| 🚧 Success validators (external)          | 🔜 Execution containers                                                             | 🔜 Context limit overflow flags             | 🔜 Generative code validators                     |
| 🔜 Agent controllers/skill tests          | ✅ Honeypot and prompt injection detection ([PromptArmor](https://promptarmor.com)) | ✅ API bill tracking                        | 🔜 Error breakpoint analysis                      |
| 🔜 Information context constraint testing | 🔜 Anti-agent roadblocks (i.e. Captchas)                                            | 🔜 CI/CD integration checks                 |                                                   |
| 🔜 Regression testing                     | ✅ Multi-agent framework visualization                                              |                                             |                                                   |

### Why Choose AgentOps? 🤔

AgentOps equips you with the essential tools to build, monitor, and optimize your AI agents.  Key benefits include:

*   **Comprehensive Observability**:  Gain in-depth insights into agent performance, user interactions, and API usage.
*   **Real-Time Monitoring**:  Get instant feedback with session replays, comprehensive metrics, and live monitoring.
*   **Cost Control**:  Effectively monitor and manage spending on LLMs and API calls.
*   **Failure Detection**:  Quickly identify and address agent failures and multi-agent interaction problems.
*   **Tool Usage Analytics**: Understand how your agents utilize external tools.
*   **Session-Wide Metrics**:  Get a complete overview of your agent sessions with extensive statistical data.

AgentOps makes agent observability, testing, and monitoring easier than ever.

### Star History

See our growth!
<img src="https://api.star-history.com/svg?repos=AgentOps-AI/agentops&type=Date" style="max-width: 500px" width="50%" alt="Logo">

### Projects using AgentOps

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
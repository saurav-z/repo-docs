<div align="center">
  <a href="https://agentops.ai?ref=gh">
    <img src="docs/images/external/logo/github-banner.png" alt="AgentOps Logo">
  </a>
</div>

<div align="center">
  <em>Supercharge your AI agent development with AgentOps, the leading observability and devtool platform.</em>
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

**AgentOps: Build, Evaluate, and Monitor AI Agents with Ease.** From rapid prototyping to seamless production, AgentOps provides a comprehensive platform for AI agent development.

## Key Features

*   **Replay Analytics and Debugging:** Step-by-step agent execution graphs.
*   **LLM Cost Management:** Track spend with LLM providers.
*   **Framework Integrations:** Native integrations with CrewAI, AG2 (AutoGen), LangGraph, and more.
*   **Self-Hosting:** Run AgentOps on your own cloud.
*   **Comprehensive Observability**: Track your AI agents' performance, user interactions, and API usage.
*   **Real-Time Monitoring**: Get instant insights with session replays, metrics, and live monitoring tools.
*   **Failure Detection**: Quickly identify and respond to agent failures.
*   **Tool Usage Statistics**: Understand how your agents utilize external tools.
*   **Session-Wide Metrics**: Gain a holistic view of your agents' sessions with comprehensive statistics.

## Getting Started

Install AgentOps:

```bash
pip install agentops
```

### Integrate in 2 Lines of Code

Get your API key from the [AgentOps dashboard](https://app.agentops.ai/settings/projects), then initialize the AgentOps client to start tracking your LLM calls.

```python
import agentops

# Beginning of your program (i.e. main.py, __init__.py)
agentops.init( < INSERT YOUR API KEY HERE >)

...

# End of program
agentops.end_session('Success')
```

## Advanced Integrations

### OpenAI Agents SDK 🖇️

Integrate seamlessly with the OpenAI Agents SDK for Python and TypeScript.

#### Python

```bash
pip install openai-agents
```

-   [Python Integration Guide](https://docs.agentops.ai/v2/integrations/openai_agents_python)
-   [OpenAI Agents Python Documentation](https://openai.github.io/openai-agents-python/)

#### TypeScript

```bash
npm install agentops @openai/agents
```

-   [TypeScript Integration Guide](https://docs.agentops.ai/v2/integrations/openai_agents_js)
-   [OpenAI Agents JS Documentation](https://openai.github.io/openai-agents-js)

### CrewAI 🛶

Integrate CrewAI with AgentOps in 2 lines of code:

```bash
pip install 'crewai[agentops]'
```

Set `AGENTOPS_API_KEY` and your crews will automatically be monitored.

-   [AgentOps Integration Example](https://docs.agentops.ai/v1/integrations/crewai)
-   [Official CrewAI Documentation](https://docs.crewai.com/how-to/AgentOps-Observability)

### AG2 (AutoGen) 🤖

Add full observability to AG2 (formerly AutoGen) agents:

-   [AG2 Observability Example](https://docs.ag2.ai/notebooks/agentchat_agentops)
-   [AG2 - AgentOps Documentation](https://docs.ag2.ai/docs/ecosystem/agentops)

Set `AGENTOPS_API_KEY` and call `agentops.init()`.

### Camel AI 🐪

Track and analyze CAMEL agents:

-   [Camel AI](https://www.camel-ai.org/)
-   [AgentOps Integration Example](https://docs.agentops.ai/v1/integrations/camel)
-   [Official Camel AI Documentation](https://docs.camel-ai.org/cookbooks/agents_tracking.html)

Set `AGENTOPS_API_KEY` and initialize AgentOps.

### Langchain 🦜🔗

AgentOps integrates with Langchain. Install the Langchain extra and import and set the handler:

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

### Cohere ⌨️

Integrate with Cohere (>=5.4.0).

-   [AgentOps Integration Example](https://docs.agentops.ai/v1/integrations/cohere)
-   [Official Cohere Documentation](https://docs.cohere.com/reference/about)

```bash
pip install cohere
```

### Anthropic ﹨

Track agents built with the Anthropic Python SDK (>=0.32.0).

-   [AgentOps Integration Guide](https://docs.agentops.ai/v1/integrations/anthropic)
-   [Official Anthropic Documentation](https://docs.anthropic.com/en/docs/welcome)

```bash
pip install anthropic
```

### Mistral 〽️

Track agents built with the Mistral Python SDK (>=0.32.0).

-   [AgentOps Integration Example](./examples/mistral//mistral_example.ipynb)
-   [Official Mistral Documentation](https://docs.mistral.ai)

```bash
pip install mistralai
```

### CamelAI ﹨

Track agents built with the CamelAI Python SDK (>=0.32.0).

-   [CamelAI Integration Guide](https://docs.camel-ai.org/cookbooks/agents_tracking.html#)
-   [Official CamelAI Documentation](https://docs.camel-ai.org/index.html)

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

[You can find usage examples here!](examples/camelai_examples/README.md).

### LiteLLM 🚅

AgentOps supports LiteLLM(>=1.3.1).

-   [AgentOps Integration Example](https://docs.agentops.ai/v1/integrations/litellm)
-   [Official LiteLLM Documentation](https://docs.litellm.ai/docs/providers)

```bash
pip install litellm
```

### LlamaIndex 🦙

Integrate with LlamaIndex.

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

### Llama Stack 🦙🥞

AgentOps provides support for Llama Stack Python Client(>=0.0.53), allowing you to monitor your Agentic applications. 

- [AgentOps integration example 1](https://github.com/AgentOps-AI/agentops/pull/530/files/65a5ab4fdcf310326f191d4b870d4f553591e3ea#diff-fdddf65549f3714f8f007ce7dfd1cde720329fe54155d54389dd50fbd81813cb)
- [AgentOps integration example 2](https://github.com/AgentOps-AI/agentops/pull/530/files/65a5ab4fdcf310326f191d4b870d4f553591e3ea#diff-6688ff4fb7ab1ce7b1cc9b8362ca27264a3060c16737fb1d850305787a6e3699)
- [Official Llama Stack Python Client](https://github.com/meta-llama/llama-stack-client-python)

### SwarmZero AI 🐝

Track and analyze SwarmZero agents with full observability.

-   [SwarmZero](https://swarmzero.ai)
-   [AgentOps Integration Example](https://docs.agentops.ai/v1/integrations/swarmzero)
-   [SwarmZero AI integration example](https://docs.swarmzero.ai/examples/ai-agents/build-and-monitor-a-web-search-agent)
-   [SwarmZero AI - AgentOps documentation](https://docs.swarmzero.ai/sdk/observability/agentops)
-   [Official SwarmZero Python SDK](https://github.com/swarmzero/swarmzero)

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

## Evaluations Roadmap 🧭

Roadmap for platform features and metrics:

| Platform                                                                     | Dashboard                                  | Evals                                  |
| ---------------------------------------------------------------------------- | ------------------------------------------ | -------------------------------------- |
| ✅ Python SDK                                                                | ✅ Multi-session and Cross-session metrics | ✅ Custom eval metrics                 |
| 🚧 Evaluation builder API                                                    | ✅ Custom event tag tracking              | 🔜 Agent scorecards                    |
| 🚧 [Javascript/Typescript SDK (Alpha)](https://github.com/AgentOps-AI/agentops-node) | ✅ Session replays                         | 🔜 Evaluation playground + leaderboard |

## Debugging Roadmap 🧭

Roadmap for debugging and testing:

| Performance testing                       | Environments                                                                        | LLM Testing                                 | Reasoning and execution testing                   |
| ----------------------------------------- | ----------------------------------------------------------------------------------- | ------------------------------------------- | ------------------------------------------------- |
| ✅ Event latency analysis                 | 🔜 Non-stationary environment testing                                               | 🔜 LLM non-deterministic function detection | 🚧 Infinite loops and recursive thought detection |
| ✅ Agent workflow execution pricing       | 🔜 Multi-modal environments                                                         | 🚧 Token limit overflow flags               | 🔜 Faulty reasoning detection                     |
| 🚧 Success validators (external)          | 🔜 Execution containers                                                             | 🔜 Context limit overflow flags             | 🔜 Generative code validators                     |
| 🔜 Agent controllers/skill tests          | ✅ Honeypot and prompt injection detection ([PromptArmor](https://promptarmor.com)) | ✅ API bill tracking                        | 🔜 Error breakpoint analysis                      |
| 🔜 Information context constraint testing | 🔜 Anti-agent roadblocks (i.e. Captchas)                                            | 🔜 CI/CD integration checks                 |                                                   |
| 🔜 Regression testing                     | ✅ Multi-agent framework visualization                                              |                                             |                                                   |

### Why AgentOps? 🤔

AgentOps empowers developers to build better, more reliable AI agents. Key benefits:

*   **Comprehensive Observability**
*   **Real-Time Monitoring**
*   **Cost Control**
*   **Failure Detection**
*   **Tool Usage Statistics**
*   **Session-Wide Metrics**

## Self-Hosting

Run the full AgentOps app (Dashboard + API backend) on your machine. Follow the setup guide in `app/README.md`:

-   [Run the App and Backend (Dashboard + API)](app/README.md)

## [Learn More](https://github.com/AgentOps-AI/agentops)

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
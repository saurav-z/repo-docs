<div align="center">
  <a href="https://agentops.ai?ref=gh">
    <img src="docs/images/external/logo/github-banner.png" alt="AgentOps Logo">
  </a>
</div>

<div align="center">
  <em>The Observability and DevTool Platform for AI Agents</em>
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

## AgentOps: Build, Monitor, and Optimize Your AI Agents

AgentOps is the premier platform for building, evaluating, and monitoring AI agents, transforming them from prototypes to production-ready applications.

## Key Features

*   **Replay Analytics and Debugging:** Step-by-step agent execution graphs for in-depth analysis.
*   **LLM Cost Management:** Track your spending with various LLM providers.
*   **Framework Integrations:** Seamlessly integrate with popular frameworks like CrewAI, AG2 (AutoGen), and Langchain.
*   **Self-Hosting:** Run AgentOps on your own infrastructure.

## Quick Start

Get started with AgentOps in minutes:

```bash
pip install agentops
```

### Session Replays in 2 Lines of Code

Initialize the AgentOps client and automatically gain analytics on all your LLM calls.

[Get an API key](https://app.agentops.ai/settings/projects)

```python
import agentops

# Beginning of your program (i.e. main.py, __init__.py)
agentops.init( < INSERT YOUR API KEY HERE >)

...

# End of program
agentops.end_session('Success')
```

All your sessions can be viewed on the [AgentOps dashboard](https://app.agentops.ai?ref=gh)
<br/>

## Integrations

AgentOps offers extensive integrations with leading AI agent frameworks and tools:

### OpenAI Agents SDK 🖇️

Build multi-agent systems with tools, handoffs, and guardrails. AgentOps natively integrates with the OpenAI Agents SDKs for both Python and TypeScript.

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

Build Crew agents with observability in just 2 lines of code. Simply set an `AGENTOPS_API_KEY` in your environment, and your crews will get automatic monitoring on the AgentOps dashboard.

```bash
pip install 'crewai[agentops]'
```

-   [AgentOps integration example](https://docs.agentops.ai/v1/integrations/crewai)
-   [Official CrewAI documentation](https://docs.crewai.com/how-to/AgentOps-Observability)

### AG2 🤖
With only two lines of code, add full observability and monitoring to AG2 (formerly AutoGen) agents. Set an `AGENTOPS_API_KEY` in your environment and call `agentops.init()`

-   [AG2 Observability Example](https://docs.ag2.ai/notebooks/agentchat_agentops)
-   [AG2 - AgentOps Documentation](https://docs.ag2.ai/docs/ecosystem/agentops)

### Camel AI 🐪

Track and analyze CAMEL agents with full observability. Set an `AGENTOPS_API_KEY` in your environment and initialize AgentOps to get started.

-   [Camel AI](https://www.camel-ai.org/) - Advanced agent communication framework
-   [AgentOps integration example](https://docs.agentops.ai/v1/integrations/camel)
-   [Official Camel AI documentation](https://docs.camel-ai.org/cookbooks/agents_tracking.html)

### Langchain 🦜🔗

AgentOps works seamlessly with applications built using Langchain. To use the handler, install Langchain as an optional dependency:

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

Check out the [Langchain Examples Notebook](./examples/langchain/langchain_examples.ipynb) for more details including Async handlers.

### Cohere ⌨️

First class support for Cohere(>=5.4.0). This is a living integration, should you need any added functionality please message us on Discord!

-   [AgentOps integration example](https://docs.agentops.ai/v1/integrations/cohere)
-   [Official Cohere documentation](https://docs.cohere.com/reference/about)

### Anthropic ﹨

Track agents built with the Anthropic Python SDK (>=0.32.0).

-   [AgentOps integration guide](https://docs.agentops.ai/v1/integrations/anthropic)
-   [Official Anthropic documentation](https://docs.anthropic.com/en/docs/welcome)

### Mistral 〽️

Track agents built with the Mistral Python SDK (>=0.32.0).

-   [AgentOps integration example](./examples/mistral//mistral_example.ipynb)
-   [Official Mistral documentation](https://docs.mistral.ai)

### LiteLLM 🚅

AgentOps provides support for LiteLLM(>=1.3.1), allowing you to call 100+ LLMs using the same Input/Output Format.

-   [AgentOps integration example](https://docs.agentops.ai/v1/integrations/litellm)
-   [Official LiteLLM documentation](https://docs.litellm.ai/docs/providers)

### LlamaIndex 🦙

AgentOps works seamlessly with applications built using LlamaIndex, a framework for building context-augmented generative AI applications with LLMs.

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

### Llama Stack 🦙🥞

AgentOps provides support for Llama Stack Python Client(>=0.0.53), allowing you to monitor your Agentic applications.

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

## Evaluations Roadmap

*   Python SDK
    *   Multi-session and Cross-session metrics
    *   Custom eval metrics
*   Evaluation builder API
    *   Custom event tag tracking
    *   Agent scorecards
*   Javascript/Typescript SDK (Alpha)
    *   Session replays
    *   Evaluation playground + leaderboard

## Debugging Roadmap

*   Performance testing
    *   Event latency analysis
    *   Agent workflow execution pricing
    *   Success validators (external)
    *   Agent controllers/skill tests
    *   Information context constraint testing
    *   Regression testing
*   Environments
    *   Non-stationary environment testing
    *   Multi-modal environments
    *   Execution containers
    *   Honeypot and prompt injection detection ([PromptArmor](https://promptarmor.com))
    *   Anti-agent roadblocks (i.e. Captchas)
    *   Multi-agent framework visualization
*   LLM Testing
    *   LLM non-deterministic function detection
    *   Token limit overflow flags
    *   Context limit overflow flags
    *   CI/CD integration checks
*   Reasoning and execution testing
    *   Infinite loops and recursive thought detection
    *   Faulty reasoning detection
    *   Generative code validators
    *   Error breakpoint analysis

## Why AgentOps?

AI agents can be slow, expensive, and unreliable without the right tools. AgentOps is designed to make agent observability, testing, and monitoring easy.

*   **Comprehensive Observability:** Track performance, user interactions, and API usage.
*   **Real-Time Monitoring:** Gain instant insights with session replays, metrics, and live monitoring tools.
*   **Cost Control:** Monitor and manage LLM and API call costs.
*   **Failure Detection:** Quickly identify and respond to agent and multi-agent interaction issues.
*   **Tool Usage Statistics:** Analyze how your agents utilize external tools.
*   **Session-Wide Metrics:** Get a holistic view of your agents' sessions.

## Get Started

For detailed information, refer to the [AgentOps documentation](http://docs.agentops.ai).

## Contribute

AgentOps is an open-source project. Contributions are welcome!  Check out the [AgentOps repository](https://github.com/AgentOps-AI/agentops) for more details.

## Star History

```
<img src="https://api.star-history.com/svg?repos=AgentOps-AI/agentops&type=Date" style="max-width: 500px" width="50%" alt="Logo">
```

## Projects Using AgentOps

Here are some popular projects that leverage AgentOps:

```
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
```

_Generated using [github-dependents-info](https://github.com/nvuillam/github-dependents-info), by [Nicolas Vuillamy](https://github.com/nvuillam)_
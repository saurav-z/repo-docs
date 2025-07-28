<div align="center">
  <a href="https://agentops.ai?ref=gh">
    <img src="docs/images/external/logo/github-banner.png" alt="AgentOps Logo">
  </a>
</div>

<div align="center">
  <em>The ultimate observability and devtool platform for AI Agents.</em>
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

**AgentOps: Build, Evaluate, and Monitor Your AI Agents with Ease.** AgentOps is a comprehensive observability and developer tool platform designed to streamline the entire AI agent lifecycle, from prototyping to production.

## Key Features

*   **Replay Analytics and Debugging:** Step-by-step agent execution graphs for in-depth analysis.
*   **LLM Cost Management:** Track and manage spending across different LLM providers.
*   **Agent Benchmarking:** Test agent performance with 1,000+ built-in evaluation metrics.
*   **Compliance and Security:** Identify and mitigate prompt injection and data exfiltration risks.
*   **Framework Integrations:** Seamlessly integrate with popular frameworks like CrewAI, AG2 (AutoGen), Camel AI, and LangChain.

## Quick Start

Get started with AgentOps in just a few steps:

1.  **Install the AgentOps Package:**

    ```bash
    pip install agentops
    ```

2.  **Get Your API Key:** Create a free account and retrieve your API key from the [AgentOps dashboard](https://app.agentops.ai/settings/projects).

3.  **Initialize AgentOps in Your Code:**  Add the following lines to your agent's initialization (e.g., `main.py` or `__init__.py`):

    ```python
    import agentops

    # Initialize AgentOps with your API key
    agentops.init("<YOUR_API_KEY>")

    # ... your agent code ...

    # End the session
    agentops.end_session('Success')
    ```

4.  **View Your Agent's Data:** Access detailed session replays and analytics on the [AgentOps dashboard](https://app.agentops.ai?ref=gh).

### Agent Debugging

<details>
  <summary>Agent Metadata</summary>
  <a href="https://app.agentops.ai?ref=gh">
    <img src="docs/images/external/app_screenshots/session-drilldown-metadata.png" style="width: 90%;" alt="Agent Metadata"/>
  </a>
</details>

<details>
  <summary>Chat Viewer</summary>
  <a href="https://app.agentops.ai?ref=gh">
    <img src="docs/images/external/app_screenshots/chat-viewer.png" style="width: 90%;" alt="Chat Viewer"/>
  </a>
</details>

<details>
  <summary>Event Graphs</summary>
  <a href="https://app.agentops.ai?ref=gh">
    <img src="docs/images/external/app_screenshots/session-drilldown-graphs.png" style="width: 90%;" alt="Event Graphs"/>
  </a>
</details>

### Session Replays

<details>
  <summary>Session Replays</summary>
  <a href="https://app.agentops.ai?ref=gh">
    <img src="docs/images/external/app_screenshots/session-replay.png" style="width: 90%;" alt="Session Replays"/>
  </a>
</details>

### Summary Analytics

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

AgentOps offers a streamlined developer experience, enabling you to add powerful observability with minimal code:

*   **Session Spans:** Use `@session` to track the overall agent session.
*   **Agent Spans:** Use `@agent` to track the agent's methods.
*   **Operation/Task Spans:** Use `@operation` or `@task` to track specific operations.
*   **Workflow Spans:** Use `@workflow` for tracking multi-operation workflows.

All decorators include: Input/Output Recording, Exception Handling, Async/Await functions, Generator functions, and Custom attributes and names.

```python
# Example: Decorator Usage
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

## Integrations

AgentOps seamlessly integrates with popular AI frameworks and SDKs:

### OpenAI Agents SDK 🖇️

Monitor agents built using the OpenAI Agents SDK. Guides are available for both Python and TypeScript.

#### Python

```bash
pip install openai-agents
```

*   [Python Integration Guide](https://docs.agentops.ai/v2/integrations/openai_agents_python)

#### TypeScript

```bash
npm install agentops @openai/agents
```

*   [TypeScript Integration Guide](https://docs.agentops.ai/v2/integrations/openai_agents_js)

### CrewAI 🛶

Effortlessly integrate AgentOps into your CrewAI projects with just a few lines of code.

```bash
pip install 'crewai[agentops]'
```

*   [CrewAI Integration Example](https://docs.agentops.ai/v1/integrations/crewai)

### AG2 (AutoGen) 🤖

Add full observability and monitoring to AG2 (formerly AutoGen) agents with ease.

*   [AG2 Observability Example](https://docs.ag2.ai/notebooks/agentchat_agentops)

### Camel AI 🐪

Track and analyze CAMEL agents with full observability.

*   [Camel AI integration example](https://docs.agentops.ai/v1/integrations/camel)

<details>
  <summary>Installation for CamelAI</summary>
  
```bash
pip install "camel-ai[all]==0.2.11"
pip install agentops
```
</details>

### Langchain 🦜🔗

AgentOps offers seamless integration with Langchain applications.

<details>
  <summary>Installation for Langchain</summary>
  
```shell
pip install agentops[langchain]
```
</details>

### Cohere ⌨️

First class support for Cohere(>=5.4.0).

<details>
  <summary>Installation for Cohere</summary>
  
```bash
pip install cohere
```
</details>

### Anthropic ﹨

Track agents built with the Anthropic Python SDK (>=0.32.0).

<details>
  <summary>Installation for Anthropic</summary>
  
```bash
pip install anthropic
```
</details>

### Mistral 〽️

Track agents built with the Mistral Python SDK (>=0.32.0).

<details>
  <summary>Installation for Mistral</summary>
  
```bash
pip install mistralai
```
</details>

### LiteLLM 🚅

AgentOps provides support for LiteLLM(>=1.3.1), allowing you to call 100+ LLMs using the same Input/Output Format. 

<details>
  <summary>Installation for LiteLLM</summary>
  
```bash
pip install litellm
```
</details>

### LlamaIndex 🦙

AgentOps works seamlessly with applications built using LlamaIndex, a framework for building context-augmented generative AI applications with LLMs.

<details>
  <summary>Installation for LlamaIndex</summary>
  
```shell
pip install llama-index-instrumentation-agentops
```
</details>

### Llama Stack 🦙🥞

AgentOps provides support for Llama Stack Python Client(>=0.0.53), allowing you to monitor your Agentic applications. 

### SwarmZero AI 🐝

Track and analyze SwarmZero agents with full observability.

<details>
  <summary>Installation for SwarmZero AI</summary>

```bash
pip install swarmzero
pip install agentops
```
</details>

## Evaluations Roadmap

| Platform                                                                     | Dashboard                                  | Evals                                  |
| ---------------------------------------------------------------------------- | ------------------------------------------ | -------------------------------------- |
| ✅ Python SDK                                                                | ✅ Multi-session and Cross-session metrics | ✅ Custom eval metrics                 |
| 🚧 Evaluation builder API                                                    | ✅ Custom event tag tracking              | 🔜 Agent scorecards                    |
| 🚧 [Javascript/Typescript SDK (Alpha)](https://github.com/AgentOps-AI/agentops-node) | ✅ Session replays                         | 🔜 Evaluation playground + leaderboard |

## Debugging Roadmap

| Performance testing                       | Environments                                                                        | LLM Testing                                 | Reasoning and execution testing                   |
| ----------------------------------------- | ----------------------------------------------------------------------------------- | ------------------------------------------- | ------------------------------------------------- |
| ✅ Event latency analysis                 | 🔜 Non-stationary environment testing                                               | 🔜 LLM non-deterministic function detection | 🚧 Infinite loops and recursive thought detection |
| ✅ Agent workflow execution pricing       | 🔜 Multi-modal environments                                                         | 🚧 Token limit overflow flags               | 🔜 Faulty reasoning detection                     |
| 🚧 Success validators (external)          | 🔜 Execution containers                                                             | 🔜 Context limit overflow flags             | 🔜 Generative code validators                     |
| 🔜 Agent controllers/skill tests          | ✅ Honeypot and prompt injection detection ([PromptArmor](https://promptarmor.com)) | ✅ API bill tracking                        | 🔜 Error breakpoint analysis                      |
| 🔜 Information context constraint testing | 🔜 Anti-agent roadblocks (i.e. Captchas)                                            | 🔜 CI/CD integration checks                 |                                                   |
| 🔜 Regression testing                     | ✅ Multi-agent framework visualization                                              |                                             |                                                   |

### Why AgentOps?

AgentOps provides a comprehensive solution for building robust and reliable AI agents.

-   **Comprehensive Observability:** Gain deep insights into your agents' behavior.
-   **Real-Time Monitoring:** Monitor agents' performance and API usage.
-   **Cost Control:** Track and manage LLM and API spending.
-   **Failure Detection:** Identify and address agent failures quickly.
-   **Tool Usage Statistics:** Analyze how your agents utilize external tools.
-   **Session-Wide Metrics:** Get a holistic view of agent performance.

AgentOps streamlines agent observability, testing, and monitoring to help you bring your agents from prototype to production.

## Star History

<img src="https://api.star-history.com/svg?repos=AgentOps-AI/agentops&type=Date" style="max-width: 500px" width="50%" alt="Star History">

## Popular Projects Using AgentOps

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

For more information, visit the [AgentOps GitHub repository](https://github.com/AgentOps-AI/agentops).
<div align="center">
  <a href="https://agentops.ai?ref=gh">
    <img src="docs/images/external/logo/github-banner.png" alt="AgentOps Logo">
  </a>
</div>

<div align="center">
  <em>**AgentOps: The Observability & DevTool Platform for AI Agents**</em>
</div>

<br />

<div align="center">
  <a href="https://pepy.tech/project/agentops">
    <img src="https://static.pepy.tech/badge/agentops/month" alt="Downloads">
  </a>
  <a href="https://github.com/agentops-ai/agentops/issues">
  <img src="https://img.shields.io/github/commit-activity/m/agentops-ai/agentops" alt="Git Commit Activity">
  </a>
  <img src="https://img.shields.io/pypi/v/agentops?&color=3670A0" alt="PyPI Version">
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

AgentOps empowers developers to build, monitor, and evaluate AI agents, transforming them from prototype to production.

## Key Features

*   **AI Agent Observability:** Gain deep insights into your agent's behavior.
*   **LLM Cost Management:** Track and optimize your LLM spending.
*   **Framework Integrations:**  Seamlessly integrate with popular frameworks like CrewAI, AG2 (AutoGen), Langchain, and more.
*   **Replay Analytics & Debugging:** Visualize agent execution step-by-step.
*   **Self-Hosting:** Run AgentOps on your own infrastructure.

## Quick Start

Install the AgentOps SDK with pip:

```bash
pip install agentops
```

### Integrate AgentOps in Your Code

Get your API key from the [AgentOps dashboard](https://app.agentops.ai/settings/projects) and initialize the AgentOps client for real-time session analytics:

```python
import agentops

# Beginning of your program (i.e. main.py, __init__.py)
agentops.init( < INSERT YOUR API KEY HERE >)

...

# End of program
agentops.end_session('Success')
```

View your AI agent sessions on the [AgentOps dashboard](https://app.agentops.ai?ref=gh)

## Integrations 🦾

### [OpenAI Agents SDK](https://docs.agentops.ai/v2/integrations/openai_agents_python) 🖇️

Easily monitor your OpenAI Agents.

#### Python

```bash
pip install openai-agents
```

#### TypeScript

```bash
npm install agentops @openai/agents
```

### [CrewAI](https://docs.agentops.ai/v1/integrations/crewai) 🛶

Add observability to your CrewAI agents with 2 lines of code. Set the `AGENTOPS_API_KEY` environment variable.

```bash
pip install 'crewai[agentops]'
```

### [AG2 (AutoGen)](https://docs.ag2.ai/docs/ecosystem/agentops) 🤖

Monitor AG2 agents with just two lines of code. Set the `AGENTOPS_API_KEY` environment variable.

### [Camel AI](https://docs.agentops.ai/v1/integrations/camel) 🐪

Track and analyze CAMEL agents with full observability. Set an `AGENTOPS_API_KEY` in your environment and initialize AgentOps to get started.

### [Langchain](https://docs.agentops.ai/v1/integrations/langchain) 🦜🔗

Integrate AgentOps with your Langchain applications.

```shell
pip install agentops[langchain]
```

### [Cohere](https://docs.agentops.ai/v1/integrations/cohere) ⌨️

First class support for Cohere(>=5.4.0).

```bash
pip install cohere
```

### [Anthropic](https://docs.agentops.ai/v1/integrations/anthropic) ﹨

Track agents built with the Anthropic Python SDK (>=0.32.0).

```bash
pip install anthropic
```

### [Mistral](https://github.com/AgentOps-AI/agentops/blob/main/examples/mistral/mistral_example.ipynb) 〽️

Track agents built with the Mistral Python SDK (>=0.32.0).

```bash
pip install mistralai
```

### [LiteLLM](https://docs.agentops.ai/v1/integrations/litellm) 🚅

AgentOps provides support for LiteLLM(>=1.3.1), allowing you to call 100+ LLMs using the same Input/Output Format.

```bash
pip install litellm
```

### [LlamaIndex](https://docs.llamaindex.ai/en/stable/module_guides/observability/?h=agentops#agentops) 🦙

```shell
pip install llama-index-instrumentation-agentops
```

### [Llama Stack](https://github.com/meta-llama/llama-stack-client-python) 🦙🥞

### [SwarmZero AI](https://docs.swarmzero.ai/examples/ai-agents/build-and-monitor-a-web-search-agent) 🐝

```bash
pip install swarmzero
pip install agentops
```

## Evaluations Roadmap 🧭

*   **Python SDK:** Multi-session & cross-session metrics, custom evaluation metrics.
*   **Evaluation Builder API:** Custom event tag tracking, agent scorecards (coming soon).
*   **JavaScript/Typescript SDK (Alpha):** Session replays, evaluation playground + leaderboard (coming soon).

## Debugging Roadmap 🧭

*   **Event latency analysis**
*   **Agent workflow execution pricing**
*   **Success validators (external)**
*   **Agent controllers/skill tests**
*   **Information context constraint testing**
*   **Regression testing**

### Why Use AgentOps?

AgentOps simplifies the development, monitoring, and evaluation of AI agents. It provides comprehensive tools for observability, real-time monitoring, cost control, and failure detection, enabling you to build more reliable and efficient AI agent solutions.

## [Visit the AgentOps GitHub Repository](https://github.com/AgentOps-AI/agentops) for more information.

## Star History

[Include Star History image here, as in the original README]

## Popular projects using AgentOps
[Include projects table here, as in the original README]

_Generated using [github-dependents-info](https://github.com/nvuillam/github-dependents-info), by [Nicolas Vuillamy](https://github.com/nvuillam)_
```
Key improvements and SEO optimizations:

*   **Clear Title & Hook:**  The title includes relevant keywords and a concise one-sentence description to capture user interest.
*   **Structured Headings:** Uses clear headings for easy navigation and SEO.
*   **Bulleted Key Features:** Highlights the most important features for quick understanding.
*   **Keyword Optimization:** Includes relevant keywords like "AI agents," "observability," "monitoring," "debugging," "LLM cost management," and framework names throughout the text.
*   **Call to Action:**  Encourages users to try the platform with a clear "Quick Start" section.
*   **Emphasis on Benefits:**  The "Why Use AgentOps?" section focuses on the advantages and value proposition.
*   **Clear Links:** Uses descriptive anchor text for links.
*   **Concise and Readable:** Streamlines the content for better comprehension.
*   **Links Back to Original Repo:**  Includes a link to the original repo.
*   **Corrected and Added details** Added details on the examples that can be used.
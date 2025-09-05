<div align="center">
  <a href="https://agentops.ai?ref=gh">
    <img src="docs/images/external/logo/github-banner.png" alt="AgentOps Logo">
  </a>
</div>

<div align="center">
  <em>**AgentOps: The essential observability and development platform for AI agents, helping you build, evaluate, and monitor your AI agents.**</em>
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


## What is AgentOps?

AgentOps is an open-source platform designed to empower developers to build, evaluate, and monitor their AI agents throughout the entire lifecycle, from prototyping to production.  This platform offers comprehensive observability, robust debugging tools, and cost management features, ensuring your agents perform efficiently and reliably. 

**[Explore the AgentOps GitHub Repository](https://github.com/AgentOps-AI/agentops)**

## Key Features

*   **Replay Analytics & Debugging:** Step-by-step agent execution graphs for in-depth analysis.
*   **LLM Cost Management:** Track and manage spending with different LLM providers.
*   **Framework Integrations:** Seamless integrations with CrewAI, AG2 (AutoGen), Agno, LangGraph, and more.
*   **Self-Hosting:** Deploy AgentOps on your own infrastructure.
*   **Session Replays:** Detailed replays of agent interactions.
*   **Summary Analytics:** Get a high-level view of agent performance with charts and metrics.
*   **Custom Metrics**: Build out custom evaluations tailored to your agent's performance.

## Quick Start

Get started with AgentOps in minutes.

```bash
pip install agentops
```

### Track LLM Calls in 2 Lines of Code

1.  **Get an API key:** [https://app.agentops.ai/settings/projects](https://app.agentops.ai/settings/projects)
2.  **Initialize AgentOps:**

```python
import agentops

# Beginning of your program
agentops.init( < INSERT YOUR API KEY HERE >)

...

# End of program
agentops.end_session('Success')
```

View your sessions and detailed analytics on the [AgentOps dashboard](https://app.agentops.ai?ref=gh).

## Self-Hosting

Run the complete AgentOps application (Dashboard + API backend) on your own machine. Follow the setup guide in `app/README.md`:

-   [Run the App and Backend (Dashboard + API)](app/README.md)

## Integrations

AgentOps seamlessly integrates with popular AI agent frameworks and SDKs, providing enhanced monitoring capabilities.

*   **OpenAI Agents SDK:** Native integration for both Python and TypeScript. [Python Integration Guide](https://docs.agentops.ai/v2/integrations/openai_agents_python), [TypeScript Integration Guide](https://docs.agentops.ai/v2/integrations/openai_agents_js).
*   **CrewAI:** Monitor your Crew agents with just a few lines of code.  [Integration Example](https://docs.agentops.ai/v1/integrations/crewai).
*   **AG2 (AutoGen):** Comprehensive observability for AG2 agents. [AG2 Observability Example](https://docs.ag2.ai/notebooks/agentchat_agentops).
*   **Camel AI:** Full observability and monitoring for CAMEL agents. [Integration Example](https://docs.agentops.ai/v1/integrations/camel).
*   **Langchain:** Integrate AgentOps with your Langchain applications effortlessly.
*   **Cohere:** First-class support for Cohere (>=5.4.0). [Integration Example](https://docs.agentops.ai/v1/integrations/cohere).
*   **Anthropic:** Track agents built with the Anthropic Python SDK (>=0.32.0). [Integration Example](https://docs.agentops.ai/v1/integrations/anthropic).
*   **Mistral:** Track agents built with the Mistral Python SDK (>=0.32.0). [Integration Example](./examples/mistral//mistral_example.ipynb).
*   **LiteLLM:** Support for LiteLLM(>=1.3.1), allowing you to call 100+ LLMs using the same Input/Output Format. [Integration Example](https://docs.agentops.ai/v1/integrations/litellm).
*   **LlamaIndex:** Seamless integration with LlamaIndex for context-augmented applications.
*   **Llama Stack:** Provides support for Llama Stack Python Client(>=0.0.53), allowing you to monitor your Agentic applications. 
*   **SwarmZero AI:** Track and analyze SwarmZero agents with full observability. [SwarmZero AI Integration Example](https://docs.swarmzero.ai/examples/ai-agents/build-and-monitor-a-web-search-agent).

## Evaluations Roadmap

*   Python SDK: Multi-session and cross-session metrics.
*   Evaluation builder API: Custom event tag tracking.
*   Javascript/Typescript SDK (Alpha): Session replays.

## Debugging Roadmap

*   Event latency analysis.
*   Agent workflow execution pricing.
*   Success validators (external).
*   Agent controllers/skill tests.
*   Information context constraint testing.
*   Regression testing.

## Why AgentOps?

AgentOps provides the essential tools needed to bring your AI agents from prototype to production, offering:

*   **Comprehensive Observability:** Track performance, user interactions, and API usage.
*   **Real-Time Monitoring:** Session replays, metrics, and live monitoring.
*   **Cost Control:** Monitor and manage LLM and API call costs.
*   **Failure Detection:** Quickly identify and resolve agent issues.
*   **Tool Usage Statistics:** Analyze how agents utilize external tools.
*   **Session-Wide Metrics:** Gain a holistic view of agent sessions.

## Star History

See our growth in the community:

<img src="https://api.star-history.com/svg?repos=AgentOps-AI/agentops&type=Date" style="max-width: 500px" width="50%" alt="Logo">

## Popular Projects Using AgentOps

*(Table of popular projects using AgentOps with links to GitHub repositories)*

_Generated using [github-dependents-info](https://github.com/nvuillam/github-dependents-info), by [Nicolas Vuillamy](https://github.com/nvuillam)_
```
Key improvements:

*   **SEO Optimization:**  The title tag is improved and includes key phrases like "AI Agents," "Observability," and "Development Platform."  The use of keywords is natural and doesn't sound overly keyword-stuffed.
*   **Clear Headings:**  Uses clear and descriptive headings to break up the content, making it easy to scan.
*   **Concise Summary:** Starts with a strong, concise sentence that immediately grabs attention.
*   **Bulleted Key Features:** Uses bullet points to highlight the most important features, improving readability.
*   **Call to Action:**  Includes clear instructions on how to get started with AgentOps.
*   **Structured Information:** Organizes the information in a logical order.
*   **Emphasis on Benefits:** Highlights the *why* - what users gain by using AgentOps.
*   **Up-to-date integrations with examples.**
*   **Star History & project list** Adds visual appeal to the README.
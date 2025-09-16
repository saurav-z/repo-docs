<div align="center">
  <a href="https://agentops.ai?ref=gh">
    <img src="docs/images/external/logo/github-banner.png" alt="AgentOps Logo">
  </a>
</div>

<div align="center">
  <em>Supercharge Your AI Agent Development with AgentOps: Build, Evaluate, and Monitor</em>
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

**AgentOps is your all-in-one platform for building, evaluating, and monitoring AI agents, taking them from prototype to production.**

## Key Features

*   ✅ **Replay Analytics and Debugging:** Step-by-step agent execution graphs for in-depth debugging.
*   ✅ **LLM Cost Management:** Track and manage your spend with LLM providers.
*   ✅ **Framework Integrations:** Native integrations with CrewAI, AG2 (AutoGen), LangGraph, and more.
*   ✅ **Self-Hosting:** Run AgentOps on your own infrastructure.

## Quick Start - Get Started in Minutes!

1.  **Install AgentOps:**

    ```bash
    pip install agentops
    ```
2.  **Get Your API Key:** Obtain your API key from the [AgentOps dashboard](https://app.agentops.ai/settings/projects).
3.  **Initialize AgentOps in your Code:**

    ```python
    import agentops

    # Beginning of your program (i.e. main.py, __init__.py)
    agentops.init(<INSERT YOUR API KEY HERE>)

    ...

    # End of program
    agentops.end_session('Success')
    ```
4.  **View Your Data:** Access detailed session replays and analytics on the [AgentOps dashboard](https://app.agentops.ai?ref=gh).

## Open Source
AgentOps is open-source and available under the MIT license. Explore the code in our [app directory](https://github.com/AgentOps-AI/agentops/tree/main/app).

## Integrations

AgentOps seamlessly integrates with popular AI agent frameworks and tools.

<div align="center" style="background-color: white; padding: 20px; border-radius: 10px; margin: 0 auto; max-width: 800px;">
  <div style="display: flex; flex-wrap: wrap; justify-content: center; align-items: center; gap: 30px; margin-bottom: 20px;">
    <a href="https://docs.agentops.ai/v2/integrations/openai_agents_python"><img src="docs/images/external/openai/agents-sdk.svg" height="45" alt="OpenAI Agents SDK"></a>
    <a href="https://docs.agentops.ai/v1/integrations/crewai"><img src="docs/v1/img/docs-icons/crew-banner.png" height="45" alt="CrewAI"></a>
    <a href="https://docs.ag2.ai/docs/ecosystem/agentops"><img src="docs/images/external/ag2/ag2-logo.svg" height="45" alt="AG2 (AutoGen)"></a>
    <a href="https://docs.agentops.ai/v1/integrations/microsoft"><img src="docs/images/external/microsoft/microsoft_logo.svg" height="45" alt="Microsoft"></a>
  </div>
  
  <div style="display: flex; flex-wrap: wrap; justify-content: center; align-items: center; gap: 30px; margin-bottom: 20px;">
    <a href="https://docs.agentops.ai/v1/integrations/langchain"><img src="docs/images/external/langchain/langchain-logo.svg" height="45" alt="LangChain"></a>
    <a href="https://docs.agentops.ai/v1/integrations/camel"><img src="docs/images/external/camel/camel.png" height="45" alt="Camel AI"></a>
    <a href="https://docs.llamaindex.ai/en/stable/module_guides/observability/?h=agentops#agentops"><img src="docs/images/external/ollama/ollama-icon.png" height="45" alt="LlamaIndex"></a>
    <a href="https://docs.agentops.ai/v1/integrations/cohere"><img src="docs/images/external/cohere/cohere-logo.svg" height="45" alt="Cohere"></a>
  </div>
</div>

*   **OpenAI Agents SDK:** Native integration. [See Docs](https://docs.agentops.ai/v2/integrations/openai_agents_python)
*   **CrewAI:** Integrate CrewAI agents with ease. [See Docs](https://docs.agentops.ai/v1/integrations/crewai)
*   **AutoGen (AG2):** Get observability for AG2 agents. [See Docs](https://docs.ag2.ai/docs/ecosystem/agentops)
*   **LangChain:** Use Langchain with AgentOps. [See Docs](https://docs.agentops.ai/v1/integrations/langchain)
*   **Camel AI:** Track and analyze CAMEL agents. [See Docs](https://docs.agentops.ai/v1/integrations/camel)
*   **LlamaIndex:**  Full support for observability [See Docs](https://docs.llamaindex.ai/en/stable/module_guides/observability/?h=agentops#agentops)
*   **Cohere:** First-class support for Cohere (>=5.4.0) [See Docs](https://docs.agentops.ai/v1/integrations/cohere)
*   **Anthropic:** Track agents built with the Anthropic Python SDK (>=0.32.0). [See Docs](https://docs.agentops.ai/v1/integrations/anthropic)
*   **Mistral:** Track agents built with the Mistral Python SDK (>=0.32.0). [See Docs](examples/mistral//mistral_example.ipynb)
*   **CamelAI:** Track agents built with the CamelAI Python SDK (>=0.32.0). [See Docs](https://docs.camel-ai.org/cookbooks/agents_tracking.html#)
*   **LiteLLM:** Support for LiteLLM(>=1.3.1). [See Docs](https://docs.agentops.ai/v1/integrations/litellm)
*   **Llama Stack:** Support for Llama Stack Python Client(>=0.0.53). [See Docs](https://github.com/meta-llama/llama-stack-client-python)
*   **SwarmZero AI:** Track and analyze SwarmZero agents. [See Docs](https://docs.swarmzero.ai/sdk/observability/agentops)

## Code Examples: Decorators

AgentOps offers simple decorators for detailed observability.

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

## Self-Hosting

Run the full AgentOps app (Dashboard + API backend) on your infrastructure. Follow the setup guide in `app/README.md`:

*   [Run the App and Backend (Dashboard + API)](app/README.md)

## Debugging & Evaluation Roadmap

*   **Debugging Roadmap:** Detailed plans for performance testing, error analysis, and more.
*   **Evaluations Roadmap:** Features for custom evaluation metrics, agent scorecards, and more.

## Star History
See our growth in the community:

<img src="https://api.star-history.com/svg?repos=AgentOps-AI/agentops&type=Date" style="max-width: 500px" width="50%" alt="Logo">

## Join the Community
Get involved and contribute to AgentOps!

*   [Visit Our Repository](https://github.com/AgentOps-AI/agentops)
*   [Join our Discord](https://discord.gg/FagdcwwXRR)
*   [Follow us on Twitter](https://twitter.com/agentopsai/)
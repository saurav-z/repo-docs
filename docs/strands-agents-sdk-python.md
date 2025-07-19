<div align="center">
  <div>
    <a href="https://strandsagents.com">
      <img src="https://strandsagents.com/latest/assets/logo-light.svg" alt="Strands Agents" width="55px" height="105px">
    </a>
  </div>

  <h1>
    Strands Agents: Build AI Agents Quickly with Python
  </h1>
  <p>
    <b>Effortlessly build and deploy AI agents with Strands Agents, a model-driven Python SDK.</b>
  </p>

  <div align="center">
    <a href="https://github.com/strands-agents/sdk-python/graphs/commit-activity"><img alt="GitHub commit activity" src="https://img.shields.io/github/commit-activity/m/strands-agents/sdk-python"/></a>
    <a href="https://github.com/strands-agents/sdk-python/issues"><img alt="GitHub open issues" src="https://img.shields.io/github/issues/strands-agents/sdk-python"/></a>
    <a href="https://github.com/strands-agents/sdk-python/pulls"><img alt="GitHub open pull requests" src="https://img.shields.io/github/issues-pr/strands-agents/sdk-python"/></a>
    <a href="https://github.com/strands-agents/sdk-python/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/strands-agents/sdk-python"/></a>
    <a href="https://pypi.org/project/strands-agents/"><img alt="PyPI version" src="https://img.shields.io/pypi/v/strands-agents"/></a>
    <a href="https://python.org"><img alt="Python versions" src="https://img.shields.io/pypi/pyversions/strands-agents"/></a>
  </div>
  
  <p>
    <a href="https://strandsagents.com/">Documentation</a>
    ◆ <a href="https://github.com/strands-agents/samples">Samples</a>
    ◆ <a href="https://github.com/strands-agents/sdk-python">Python SDK (GitHub)</a>
    ◆ <a href="https://github.com/strands-agents/tools">Tools</a>
    ◆ <a href="https://github.com/strands-agents/agent-builder">Agent Builder</a>
    ◆ <a href="https://github.com/strands-agents/mcp-server">MCP Server</a>
  </p>
</div>

Strands Agents is a powerful Python SDK that empowers you to rapidly build and deploy AI agents using a model-driven approach.  From simple chatbots to complex, autonomous workflows, Strands Agents scales seamlessly from local development to production environments.

## Key Features

*   **Simplified Agent Development:** Create AI agents with just a few lines of code.
*   **Model Agnostic Design:** Supports a wide range of LLM providers including: Amazon Bedrock, Anthropic, LiteLLM, Llama, Ollama, OpenAI, Writer, and custom providers.
*   **Advanced Capabilities:** Build multi-agent systems, autonomous agents, and leverage streaming functionality.
*   **Built-in MCP Support:** Native integration with Model Context Protocol (MCP) servers for easy access to a vast library of pre-built tools.
*   **Python-Based Tooling:** Create tools using Python decorators.

## Quick Start

```bash
# Install Strands Agents and tools
pip install strands-agents strands-agents-tools
```

```python
from strands import Agent
from strands_tools import calculator
agent = Agent(tools=[calculator])
agent("What is the square root of 1764")
```

> **Note:** For the default Amazon Bedrock model provider, you'll need AWS credentials configured and model access enabled for Claude 4 Sonnet in the us-west-2 region. See the [Quickstart Guide](https://strandsagents.com/) for details on configuring other model providers.

## Installation

1.  **Prerequisites:** Ensure you have Python 3.10+ installed.

2.  **Create and Activate a Virtual Environment (recommended):**

    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
    ```

3.  **Install the SDK and Tools:**

    ```bash
    pip install strands-agents strands-agents-tools
    ```

## Core Functionality

### Python-Based Tools

Easily build custom tools using Python decorators, enabling your agents to interact with the world.

```python
from strands import Agent, tool

@tool
def word_count(text: str) -> int:
    """Count words in text.

    This docstring is used by the LLM to understand the tool's purpose.
    """
    return len(text.split())

agent = Agent(tools=[word_count])
response = agent("How many words are in this sentence?")
```

### MCP Support

Seamlessly integrate with Model Context Protocol (MCP) servers to utilize external tools and services.

```python
from strands import Agent
from strands.tools.mcp import MCPClient
from mcp import stdio_client, StdioServerParameters

aws_docs_client = MCPClient(
    lambda: stdio_client(StdioServerParameters(command="uvx", args=["awslabs.aws-documentation-mcp-server@latest"]))
)

with aws_docs_client:
   agent = Agent(tools=aws_docs_client.list_tools_sync())
   response = agent("Tell me about Amazon Bedrock and how to use it with Python")
```

### Multiple Model Providers

Choose from a variety of model providers to power your AI agents, with the flexibility to integrate custom providers.

```python
from strands import Agent
from strands.models import BedrockModel
from strands.models.ollama import OllamaModel
from strands.models.llamaapi import LlamaAPIModel

# Bedrock
bedrock_model = BedrockModel(
  model_id="us.amazon.nova-pro-v1:0",
  temperature=0.3,
  streaming=True, # Enable/disable streaming
)
agent = Agent(model=bedrock_model)
agent("Tell me about Agentic AI")

# Ollama
ollama_model = OllamaModel(
  host="http://localhost:11434",
  model_id="llama3"
)
agent = Agent(model=ollama_model)
agent("Tell me about Agentic AI")

# Llama API
llama_model = LlamaAPIModel(
    model_id="Llama-4-Maverick-17B-128E-Instruct-FP8",
)
agent = Agent(model=llama_model)
response = agent("Tell me about Agentic AI")
```

**Built-in Providers:**

*   [Amazon Bedrock](https://strandsagents.com/latest/user-guide/concepts/model-providers/amazon-bedrock/)
*   [Anthropic](https://strandsagents.com/latest/user-guide/concepts/model-providers/anthropic/)
*   [LiteLLM](https://strandsagents.com/latest/user-guide/concepts/model-providers/litellm/)
*   [LlamaAPI](https://strandsagents.com/latest/user-guide/concepts/model-providers/llamaapi/)
*   [Ollama](https://strandsagents.com/latest/user-guide/concepts/model-providers/ollama/)
*   [OpenAI](https://strandsagents.com/latest/user-guide/concepts/model-providers/openai/)
*   [Writer](https://strandsagents.com/latest/documentation/docs/user-guide/concepts/model-providers/writer/)

**Custom Providers:** Implement your own model providers using [Custom Providers](https://strandsagents.com/latest/user-guide/concepts/model-providers/custom_model_provider/)

### Example Tools

Leverage the optional `strands-agents-tools` package for a collection of pre-built tools to accelerate your agent development.

```python
from strands import Agent
from strands_tools import calculator
agent = Agent(tools=[calculator])
agent("What is the square root of 1764")
```

The `strands-agents-tools` package is available on GitHub: [strands-agents/tools](https://github.com/strands-agents/tools).

## Documentation

Explore our comprehensive documentation for detailed guidance and examples:

*   [User Guide](https://strandsagents.com/)
*   [Quick Start Guide](https://strandsagents.com/latest/user-guide/quickstart/)
*   [Agent Loop](https://strandsagents.com/latest/user-guide/concepts/agents/agent-loop/)
*   [Examples](https://strandsagents.com/latest/examples/)
*   [API Reference](https://strandsagents.com/latest/api-reference/agent/)
*   [Production & Deployment Guide](https://strandsagents.com/latest/user-guide/deploy/operating-agents-in-production/)

## Contributing ❤️

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details on:

*   Reporting bugs and suggesting features
*   Setting up your development environment
*   Contributing through Pull Requests
*   Code of Conduct
*   Reporting security issues

## License

This project is licensed under the Apache License 2.0.  See the [LICENSE](LICENSE) file for more details.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for information about security.

##  Get Started with Strands Agents

Explore the power of AI agent development – [Check out the Strands Agents SDK on GitHub!](https://github.com/strands-agents/sdk-python)
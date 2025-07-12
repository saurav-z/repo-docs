<div align="center">
  <div>
    <a href="https://strandsagents.com">
      <img src="https://strandsagents.com/latest/assets/logo-auto.svg" alt="Strands Agents" width="55px" height="105px">
    </a>
  </div>

  <h1>
    Strands Agents: Build Powerful AI Agents with Ease
  </h1>

  <p>
    <b>Create sophisticated AI agents in minutes using the Strands Agents Python SDK, a model-driven approach for rapid development and deployment.</b>
  </p>

  <div align="center">
    <a href="https://github.com/strands-agents/sdk-python/graphs/commit-activity"><img alt="GitHub commit activity" src="https://img.shields.io/github/commit-activity/m/strands-agents/sdk-python"/></a>
    <a href="https://github.com/strands-agents/sdk-python/issues"><img alt="GitHub open issues" src="https://img.shields.io/github/issues/strands-agents/sdk-python"/></a>
    <a href="https://github.com/strands-agents/sdk-python/pulls"><img alt="GitHub open pull requests" src="https://img.shields.io/github/issues-pr/strands-agents/sdk-python"/></a>
    <a href="https://github.com/strands-agents/sdk-python/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/strands-agents/sdk-python"/></a>
    <a href="https://pypi.org/project/strands-agents/"><img alt="PyPI version" src="https://img.shields.io/pypi/v/strands-agents"/></a>
    <a href="https://python.org"><img alt="Python versions" src="https://img.shields.io/pypi/pyversions/strands-agents"/></a>
    <a href="https://github.com/strands-agents/sdk-python">
      <img alt="GitHub Repo" src="https://img.shields.io/badge/GitHub-Repo-blue?logo=github"/>
    </a>
  </div>

  <p>
    <a href="https://strandsagents.com/">Documentation</a>
    ◆ <a href="https://github.com/strands-agents/samples">Samples</a>
    ◆ <a href="https://github.com/strands-agents/sdk-python">Python SDK</a>
    ◆ <a href="https://github.com/strands-agents/tools">Tools</a>
    ◆ <a href="https://github.com/strands-agents/agent-builder">Agent Builder</a>
    ◆ <a href="https://github.com/strands-agents/mcp-server">MCP Server</a>
  </p>
</div>

## Key Features

*   **Simplified Agent Creation:** Build AI agents quickly with a few lines of code using a model-driven design.
*   **Model Agnostic:** Seamlessly integrate with a variety of LLM providers, including Amazon Bedrock, Anthropic, LiteLLM, Llama, Ollama, OpenAI, and custom providers.
*   **Advanced AI Capabilities:** Develop multi-agent systems, autonomous agents, and leverage streaming support for enhanced interactivity.
*   **Integrated MCP Support:** Utilize built-in Model Context Protocol (MCP) server support to access thousands of pre-built tools and resources.
*   **Extensible Tooling:** Easily create custom Python-based tools using decorators to extend agent functionality.
*   **Production-Ready:** Scale from local development to production deployment with comprehensive guides and support.

## Quick Start

Get started by installing the necessary packages:

```bash
# Install Strands Agents and the tools package
pip install strands-agents strands-agents-tools
```

Here's a simple example demonstrating agent usage:

```python
from strands import Agent
from strands_tools import calculator

agent = Agent(tools=[calculator])
response = agent("What is the square root of 1764?")
print(response)
```

> **Note:** For the default Amazon Bedrock model provider, ensure you have AWS credentials configured and model access enabled for Claude 3.7 Sonnet in the us-west-2 region. Refer to the [Quickstart Guide](https://strandsagents.com/) for details on configuring other model providers.

## Installation

To install Strands Agents, ensure you have Python 3.10+ installed and follow these steps:

```bash
# Create and activate virtual environment (Recommended)
python -m venv .venv
# On Linux/macOS
source .venv/bin/activate
# On Windows
.venv\Scripts\activate

# Install Strands and tools
pip install strands-agents strands-agents-tools
```

## Features in Detail

### Python-Based Tools

Create custom tools with ease using Python decorators. Define the tool's purpose in a docstring to help the LLM understand and utilize it effectively.

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

Seamlessly integrate with Model Context Protocol (MCP) servers to access a wide range of tools and resources.

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

Strands Agents supports a variety of model providers, allowing you to choose the best fit for your needs:

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

Built-in providers:

*   [Amazon Bedrock](https://strandsagents.com/latest/user-guide/concepts/model-providers/amazon-bedrock/)
*   [Anthropic](https://strandsagents.com/latest/user-guide/concepts/model-providers/anthropic/)
*   [LiteLLM](https://strandsagents.com/latest/user-guide/concepts/model-providers/litellm/)
*   [LlamaAPI](https://strandsagents.com/latest/user-guide/concepts/model-providers/llamaapi/)
*   [Ollama](https://strandsagents.com/latest/user-guide/concepts/model-providers/ollama/)
*   [OpenAI](https://strandsagents.com/latest/user-guide/concepts/model-providers/openai/)

You can also implement custom providers using the [Custom Providers](https://strandsagents.com/latest/user-guide/concepts/model-providers/custom_model_provider/) guide.

### Example Tools

The `strands-agents-tools` package provides pre-built tools for quick experimentation and ease of use. These tools are available on GitHub at [strands-agents/tools](https://github.com/strands-agents/tools).

```python
from strands import Agent
from strands_tools import calculator
agent = Agent(tools=[calculator])
agent("What is the square root of 1764")
```

## Documentation

For more comprehensive guidance and examples, explore the following documentation resources:

*   [User Guide](https://strandsagents.com/)
*   [Quick Start Guide](https://strandsagents.com/latest/user-guide/quickstart/)
*   [Agent Loop](https://strandsagents.com/latest/user-guide/concepts/agents/agent-loop/)
*   [Examples](https://strandsagents.com/latest/examples/)
*   [API Reference](https://strandsagents.com/latest/api-reference/agent/)
*   [Production & Deployment Guide](https://strandsagents.com/latest/user-guide/deploy/operating-agents-in-production/)

## Contributing ❤️

We welcome contributions! For guidelines, please refer to our [Contributing Guide](CONTRIBUTING.md) which covers:

*   Reporting bugs and feature requests.
*   Development setup instructions.
*   Contributing through pull requests.
*   Code of Conduct.
*   Reporting security issues.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for information on security issue notifications.

## ⚠️ Preview Status

Strands Agents is currently in public preview. Please note the following:

*   APIs are subject to change during the refinement of the SDK.
*   We highly value your feedback and contributions.
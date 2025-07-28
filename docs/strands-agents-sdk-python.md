<div align="center">
  <div>
    <a href="https://strandsagents.com">
      <img src="https://strandsagents.com/latest/assets/logo-github.svg" alt="Strands Agents" width="55px" height="105px">
    </a>
  </div>

  <h1>
    Strands Agents: Build Powerful AI Agents with Ease
  </h1>

  <h2>
    A model-driven approach to building AI agents in just a few lines of code.
  </h2>

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
    ◆ <a href="https://github.com/strands-agents/sdk-python">Python SDK</a>
    ◆ <a href="https://github.com/strands-agents/tools">Tools</a>
    ◆ <a href="https://github.com/strands-agents/agent-builder">Agent Builder</a>
    ◆ <a href="https://github.com/strands-agents/mcp-server">MCP Server</a>
  </p>
</div>

**Strands Agents empowers developers to build sophisticated AI agents quickly and efficiently using a model-driven approach.** This Python SDK provides a flexible and powerful framework for creating everything from simple conversational assistants to complex, autonomous workflows.  [Learn more on GitHub](https://github.com/strands-agents/sdk-python).

## Key Features

*   **Easy Agent Creation:** Build and run AI agents with minimal code.
*   **Model Agnostic:** Compatible with various LLM providers, including Amazon Bedrock, Anthropic, LiteLLM, Llama, Ollama, OpenAI, and Writer, plus custom providers.
*   **Advanced Capabilities:** Supports multi-agent systems, autonomous agents, and streaming responses.
*   **Integrated MCP Support:** Native integration with Model Context Protocol (MCP) servers for access to numerous pre-built tools.
*   **Python-Based Tooling:** Easily create and integrate Python-based tools using decorators, with hot-reloading available.

## Quick Start

```bash
# Install Strands Agents
pip install strands-agents strands-agents-tools
```

```python
from strands import Agent
from strands_tools import calculator
agent = Agent(tools=[calculator])
agent("What is the square root of 1764")
```

> **Note**: For the default Amazon Bedrock model provider, you'll need AWS credentials configured and model access enabled for Claude 4 Sonnet in the us-west-2 region. See the [Quickstart Guide](https://strandsagents.com/) for details on configuring other model providers.

## Installation

Before you begin, ensure you have Python 3.10+ installed.

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# Install Strands and tools
pip install strands-agents strands-agents-tools
```

## Features in Detail

### Python-Based Tools

Define tools using Python decorators for seamless integration:

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

**Hot Reloading:** Enable tool loading and reloading from the `./tools/` directory:

```python
from strands import Agent

# Agent will watch ./tools/ directory for changes
agent = Agent(load_tools_from_directory=True)
response = agent("Use any tools you find in the tools directory")
```

### MCP Support

Integrate with Model Context Protocol (MCP) servers:

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

Leverage various model providers:

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
 - [Amazon Bedrock](https://strandsagents.com/latest/user-guide/concepts/model-providers/amazon-bedrock/)
 - [Anthropic](https://strandsagents.com/latest/user-guide/concepts/model-providers/anthropic/)
 - [LiteLLM](https://strandsagents.com/latest/user-guide/concepts/model-providers/litellm/)
 - [LlamaAPI](https://strandsagents.com/latest/user-guide/concepts/model-providers/llamaapi/)
 - [Ollama](https://strandsagents.com/latest/user-guide/concepts/model-providers/ollama/)
 - [OpenAI](https://strandsagents.com/latest/user-guide/concepts/model-providers/openai/)
 - [Writer](https://strandsagents.com/latest/documentation/docs/user-guide/concepts/model-providers/writer/)

Custom providers can be implemented using [Custom Providers](https://strandsagents.com/latest/user-guide/concepts/model-providers/custom_model_provider/)

### Example tools

Use the optional `strands-agents-tools` package for pre-built tools:

```python
from strands import Agent
from strands_tools import calculator
agent = Agent(tools=[calculator])
agent("What is the square root of 1764")
```

These tools are also available on GitHub via [strands-agents/tools](https://github.com/strands-agents/tools).

## Documentation

*   [User Guide](https://strandsagents.com/)
*   [Quick Start Guide](https://strandsagents.com/latest/user-guide/quickstart/)
*   [Agent Loop](https://strandsagents.com/latest/user-guide/concepts/agents/agent-loop/)
*   [Examples](https://strandsagents.com/latest/examples/)
*   [API Reference](https://strandsagents.com/latest/api-reference/agent/)
*   [Production & Deployment Guide](https://strandsagents.com/latest/user-guide/deploy/operating-agents-in-production/)

## Contributing ❤️

We welcome contributions! See our [Contributing Guide](CONTRIBUTING.md) for details:

*   Reporting bugs & features
*   Development setup
*   Contributing via Pull Requests
*   Code of Conduct
*   Reporting of security issues

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for security information.
<div align="center">
  <div>
    <a href="https://strandsagents.com">
      <img src="https://strandsagents.com/latest/assets/logo-github.svg" alt="Strands Agents" width="55px" height="105px">
    </a>
  </div>
  <h1>Strands Agents: Build Powerful AI Agents with Ease</h1>
  <p>
    <b>Effortlessly create AI agents using a model-driven approach with the Strands Agents SDK.</b>
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
    ◆ <a href="https://github.com/strands-agents/sdk-python">Python SDK</a>
    ◆ <a href="https://github.com/strands-agents/tools">Tools</a>
    ◆ <a href="https://github.com/strands-agents/agent-builder">Agent Builder</a>
    ◆ <a href="https://github.com/strands-agents/mcp-server">MCP Server</a>
  </p>
</div>

Strands Agents simplifies the development of AI agents, offering a model-driven approach that enables you to build sophisticated applications with minimal code.  From basic chatbots to complex automated workflows, Strands Agents provides the scalability and flexibility you need. **<a href="https://github.com/strands-agents/sdk-python">Explore the Strands Agents Python SDK on GitHub.</a>**

## Key Features

*   **Model Agnostic:** Compatible with leading model providers including Amazon Bedrock, Anthropic, LiteLLM, Llama, Ollama, OpenAI, and Writer, plus custom providers.
*   **Lightweight & Flexible:** Easy-to-use agent loop, fully customizable to fit your specific needs.
*   **Advanced Capabilities:** Supports multi-agent systems, autonomous agents, and streaming responses for interactive applications.
*   **Built-in MCP Support:** Seamlessly integrate with Model Context Protocol (MCP) servers, giving you access to a vast library of pre-built tools.
*   **Python-Based Tools:**  Create custom tools with simple Python decorators. Hot reloading from directories streamlines development.

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

> **Note:** To use the default Amazon Bedrock model provider, ensure you have configured AWS credentials and enabled model access for Claude 4 Sonnet in the us-west-2 region. Refer to the [Quickstart Guide](https://strandsagents.com/) for detailed instructions on configuring other model providers.

## Installation

Install the SDK with Python 3.10+

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# Install Strands and tools
pip install strands-agents strands-agents-tools
```

## Features in Detail

### Python-Based Tools

Build custom tools effortlessly using Python decorators.  The LLM uses the docstrings to determine tool functionality.

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

**Hot Reloading from Directory:**  The SDK will automatically load and reload tools from your project's `./tools/` directory.

```python
from strands import Agent

# Agent watches ./tools/ directory for changes
agent = Agent(load_tools_from_directory=True)
response = agent("Use any tools you find in the tools directory")
```

### MCP Support

Easily integrate with Model Context Protocol (MCP) servers for access to a wide variety of tools.

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

Choose from various model providers, including:

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

Create custom providers using [Custom Providers](https://strandsagents.com/latest/user-guide/concepts/model-providers/custom_model_provider/).

### Example Tools

The `strands-agents-tools` package provides a set of pre-built tools to help you get started quickly.

```python
from strands import Agent
from strands_tools import calculator
agent = Agent(tools=[calculator])
agent("What is the square root of 1764")
```

The `strands-agents-tools` package is available on GitHub at [strands-agents/tools](https://github.com/strands-agents/tools).

## Documentation

Find in-depth guidance and examples in our documentation:

*   [User Guide](https://strandsagents.com/)
*   [Quick Start Guide](https://strandsagents.com/latest/user-guide/quickstart/)
*   [Agent Loop](https://strandsagents.com/latest/user-guide/concepts/agents/agent-loop/)
*   [Examples](https://strandsagents.com/latest/examples/)
*   [API Reference](https://strandsagents.com/latest/api-reference/agent/)
*   [Production & Deployment Guide](https://strandsagents.com/latest/user-guide/deploy/operating-agents-in-production/)

## Contributing ❤️

We welcome contributions!  Please review the following resources for more information:

*   [Contributing Guide](CONTRIBUTING.md)
*   [Reporting Bugs & Features](CONTRIBUTING.md)
*   [Development Setup](CONTRIBUTING.md)
*   [Pull Requests](CONTRIBUTING.md)
*   [Code of Conduct](CONTRIBUTING.md)
*   [Security Issues](CONTRIBUTING.md#security-issue-notifications)

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for information on security.
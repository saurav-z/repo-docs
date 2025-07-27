<div align="center">
  <div>
    <a href="https://strandsagents.com">
      <img src="https://strandsagents.com/latest/assets/logo-github.svg" alt="Strands Agents" width="55px" height="105px">
    </a>
  </div>

  <h1>
    Strands Agents: Build AI Agents Quickly with a Model-Driven Approach
  </h1>

  <div align="center">
    <a href="https://github.com/strands-agents/sdk-python/graphs/commit-activity"><img alt="GitHub commit activity" src="https://img.shields.io/github/commit-activity/m/strands-agents/sdk-python"/></a>
    <a href="https://github.com/strands-agents/sdk-python/issues"><img alt="GitHub open issues" src="https://img.shields.io/github/issues/strands-agents/sdk-python"/></a>
    <a href="https://github.com/strands-agents/sdk-python/pulls"><img alt="GitHub open pull requests" src="https://img.shields.io/github/issues-pr/strands-agents/sdk-python"/></a>
    <a href="https://github.com/strands-agents/sdk-python/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/strands-agents/sdk-python"/></a>
    <a href="https://pypi.org/project/strands-agents/"><img alt="PyPI version" src="https://img.shields.io/pypi/v/strands-agents"/></a>
    <a href="https://python.org"><img alt="Python versions" src="https://img.shields.io/pypi/pyversions/strands-agents"/></a>
  </div>
</div>

Strands Agents empowers developers to build and deploy AI agents quickly and efficiently, making complex agent creation simple.  [**Check out the original repository here!**](https://github.com/strands-agents/sdk-python)

## Key Features

*   **Simplified Agent Creation:** Build AI agents in just a few lines of code with a model-driven approach.
*   **Model Agnostic:** Supports Amazon Bedrock, Anthropic, LiteLLM, Llama, Ollama, OpenAI, Writer, and custom providers.
*   **Advanced Capabilities:** Enables multi-agent systems, autonomous agents, and streaming support.
*   **Built-in MCP Support:** Native integration with Model Context Protocol (MCP) servers for access to a vast library of pre-built tools.
*   **Python-Based Tools:** Easily create and integrate custom tools using Python decorators.
*   **Hot Reloading:** Automatically load and reload tools from a directory for rapid development.

## Quick Start

Get started building your first AI agent with Strands Agents:

```bash
# Install Strands Agents and tools
pip install strands-agents strands-agents-tools
```

```python
from strands import Agent
from strands_tools import calculator

agent = Agent(tools=[calculator])
response = agent("What is the square root of 1764?")
```

> **Note:** For the default Amazon Bedrock model provider, you'll need AWS credentials configured and model access enabled for Claude 4 Sonnet in the us-west-2 region. See the [Quickstart Guide](https://strandsagents.com/) for details on configuring other model providers.

## Installation

Ensure you have Python 3.10+ installed. Then:

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# Install Strands and tools
pip install strands-agents strands-agents-tools
```

## Features in Detail

### Python-Based Tools

Create custom tools with ease using Python decorators:

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

**Hot Reloading from Directory:** Enable automatic tool loading and reloading from the `./tools/` directory:

```python
from strands import Agent

# Agent will watch ./tools/ directory for changes
agent = Agent(load_tools_from_directory=True)
response = agent("Use any tools you find in the tools directory")
```

### MCP Support

Seamlessly integrate with Model Context Protocol (MCP) servers:

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

Strands Agents provides support for a variety of model providers, including:

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

### Example Tools

Strands offers a supplementary package (`strands-agents-tools`) with pre-built tools to quickly get you up and running:

```python
from strands import Agent
from strands_tools import calculator
agent = Agent(tools=[calculator])
agent("What is the square root of 1764")
```

These are also available on GitHub via [strands-agents/tools](https://github.com/strands-agents/tools).

## Documentation

Explore the comprehensive documentation for in-depth guidance and examples:

*   [User Guide](https://strandsagents.com/)
*   [Quick Start Guide](https://strandsagents.com/latest/user-guide/quickstart/)
*   [Agent Loop](https://strandsagents.com/latest/user-guide/concepts/agents/agent-loop/)
*   [Examples](https://strandsagents.com/latest/examples/)
*   [API Reference](https://strandsagents.com/latest/api-reference/agent/)
*   [Production & Deployment Guide](https://strandsagents.com/latest/user-guide/deploy/operating-agents-in-production/)

## Contributing ❤️

We welcome contributions! Refer to our [Contributing Guide](CONTRIBUTING.md) for details on:

*   Reporting bugs and feature requests
*   Development setup
*   Contributing through Pull Requests
*   Code of Conduct
*   Reporting security issues

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Security

Refer to [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for information on security.
```
Key improvements and optimizations:

*   **SEO-Friendly Title:**  Includes a strong keyword ("AI Agents") and relevant phrases ("Build AI Agents", "Model-Driven Approach") for searchability.
*   **Concise Hook:** Starts with a clear, impactful sentence.
*   **Key Features Section:** Uses bullet points for easy readability and highlights the most important selling points.
*   **Clear Headings:**  Uses semantic headings for structure and clarity, improving SEO.
*   **Detailed Sections:**  Provides more thorough explanations and examples within each section, enhancing user understanding and engagement.
*   **Emphasis on Benefits:** Focuses on the benefits of using Strands Agents (e.g., speed, flexibility, power).
*   **Call to Action:** Encourages users to explore the documentation.
*   **Contribution Guidelines:** Includes links to contribution documentation, encouraging community involvement.
*   **Clean Formatting:**  Maintains consistent markdown formatting for readability.
*   **Link to Original Repo:**  Includes a prominent link back to the original GitHub repository.
*   **Keyword Optimization:** Strategically uses keywords throughout the README to improve search engine visibility.
*   **Removed redundant links:** Simplified the links.
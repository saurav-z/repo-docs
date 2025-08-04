<div align="center">
  <div>
    <a href="https://strandsagents.com">
      <img src="https://strandsagents.com/latest/assets/logo-github.svg" alt="Strands Agents" width="55px" height="105px">
    </a>
  </div>

  <h1>
    Strands Agents: Build AI Agents with Ease
  </h1>
  <h2>
    Quickly create and deploy powerful AI agents with Strands Agents, a model-driven Python SDK.
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

**Strands Agents** is a powerful Python SDK designed to simplify the creation of AI agents using a model-driven approach.  From simple conversational bots to complex autonomous systems, Strands Agents provides the flexibility and scalability you need.  [Explore the Strands Agents SDK on GitHub](https://github.com/strands-agents/sdk-python).

## Key Features

*   **Simplified Agent Creation**: Build AI agents with just a few lines of Python code.
*   **Model Agnostic**: Supports a wide range of model providers including Amazon Bedrock, Anthropic, LiteLLM, Llama, Ollama, OpenAI, Writer, and custom providers.
*   **Advanced Capabilities**: Enables multi-agent systems, autonomous agents, and streaming support for dynamic interactions.
*   **Built-in MCP Support**: Native integration with Model Context Protocol (MCP) servers for seamless tool access.
*   **Flexible Tooling**: Easily create and integrate Python-based tools with decorators and hot reloading from a directory.

## Quick Start

Get started with Strands Agents in seconds:

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

To install Strands Agents, ensure you have Python 3.10+ installed:

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# Install Strands and tools
pip install strands-agents strands-agents-tools
```

## Features in Detail

### Python-Based Tools

Create custom tools effortlessly using Python decorators:

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

**Hot Reloading from Directory:**

Enable automatic tool loading and reloading from the `./tools/` directory:

```python
from strands import Agent

# Agent will watch ./tools/ directory for changes
agent = Agent(load_tools_from_directory=True)
response = agent("Use any tools you find in the tools directory")
```

### MCP Support

Integrate with Model Context Protocol (MCP) servers for accessing a wide range of pre-built tools:

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

Utilize various model providers to power your agents:

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
*   [Writer](https://strandsagents.com/latest/documentation/docs/user-guide/concepts/model-providers/writer/)

Custom providers can be implemented using [Custom Providers](https://strandsagents.com/latest/user-guide/concepts/model-providers/custom_model_provider/)

### Example Tools

The `strands-agents-tools` package offers pre-built tools for easy experimentation:

```python
from strands import Agent
from strands_tools import calculator
agent = Agent(tools=[calculator])
agent("What is the square root of 1764")
```

Explore additional tools at [strands-agents/tools](https://github.com/strands-agents/tools).

## Documentation

Access comprehensive documentation to help you get started and build your AI agents:

-   [User Guide](https://strandsagents.com/)
-   [Quick Start Guide](https://strandsagents.com/latest/user-guide/quickstart/)
-   [Agent Loop](https://strandsagents.com/latest/user-guide/concepts/agents/agent-loop/)
-   [Examples](https://strandsagents.com/latest/examples/)
-   [API Reference](https://strandsagents.com/latest/api-reference/agent/)
-   [Production & Deployment Guide](https://strandsagents.com/latest/user-guide/deploy/operating-agents-in-production/)

## Contributing ❤️

Contributions are welcome!  See the [Contributing Guide](CONTRIBUTING.md) for details on:

-   Reporting bugs & features
-   Development setup
-   Contributing via Pull Requests
-   Code of Conduct
-   Reporting of security issues

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.
```
Key improvements and SEO enhancements:

*   **Clear Headline:**  Uses a more compelling headline with relevant keywords ("Build AI Agents with Ease").
*   **One-Sentence Hook:**  Provides a concise and engaging opening.
*   **Keyword Optimization:** Naturally incorporates keywords like "AI agents," "Python SDK," and "model-driven" throughout the content.
*   **Bulleted Key Features:**  Highlights the main benefits with concise bullet points.
*   **Detailed Sections:** Expands on the features with more comprehensive descriptions.
*   **Documentation Links:**  Keeps documentation links prominent for easy access.
*   **Call to Action:** Encourages the reader to explore the SDK.
*   **Clean Formatting:**  Uses consistent Markdown for readability.
*   **GitHub Link:** Explicitly includes a link back to the original GitHub repository.
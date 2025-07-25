<div align="center">
  <div>
    <a href="https://strandsagents.com">
      <img src="https://strandsagents.com/latest/assets/logo-github.svg" alt="Strands Agents" width="55px" height="105px">
    </a>
  </div>

  <h1>
    Strands Agents: Build AI Agents with Ease
  </h1>

  <p>
    <b>Quickly develop and deploy powerful AI agents with the Strands Agents SDK, a model-driven approach for both simple and complex tasks.</b>
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

Strands Agents simplifies AI agent development, allowing you to build and run AI agents in just a few lines of code.  From conversational assistants to complex autonomous workflows, this SDK scales with your needs.

## Key Features

*   **Model Agnostic**:  Supports a wide range of model providers, including Amazon Bedrock, Anthropic, LiteLLM, Llama, Ollama, OpenAI, Writer, and custom providers.
*   **Lightweight & Flexible**: Provides a simple and fully customizable agent loop for rapid prototyping and deployment.
*   **Advanced Capabilities**: Enables the creation of multi-agent systems, autonomous agents, and supports streaming responses.
*   **Built-in MCP Support**: Integrates seamlessly with Model Context Protocol (MCP) servers for access to thousands of pre-built tools.
*   **Python-Based Tools**: Easily build tools using Python decorators, simplifying tool creation and integration.
*   **Hot Reloading**: Automatically loads and reloads tools from a specified directory, streamlining development.

## Getting Started

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

Ensure you have Python 3.10+ installed, then:

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# Install Strands and tools
pip install strands-agents strands-agents-tools
```

## Core Concepts and Features

### Python-Based Tools

Create tools with ease using Python decorators:

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

Choose from various model providers:

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

Strands offers an optional strands-agents-tools package with pre-built tools for quick experimentation:

```python
from strands import Agent
from strands_tools import calculator
agent = Agent(tools=[calculator])
agent("What is the square root of 1764")
```

It's also available on GitHub via [strands-agents/tools](https://github.com/strands-agents/tools).

## Documentation

Explore comprehensive documentation for detailed guidance:

-   [User Guide](https://strandsagents.com/)
-   [Quick Start Guide](https://strandsagents.com/latest/user-guide/quickstart/)
-   [Agent Loop](https://strandsagents.com/latest/user-guide/concepts/agents/agent-loop/)
-   [Examples](https://strandsagents.com/latest/examples/)
-   [API Reference](https://strandsagents.com/latest/api-reference/agent/)
-   [Production & Deployment Guide](https://strandsagents.com/latest/user-guide/deploy/operating-agents-in-production/)

## Contributing ❤️

We welcome contributions! See our [Contributing Guide](CONTRIBUTING.md) for details on:

-   Reporting bugs & features
-   Development setup
-   Contributing via Pull Requests
-   Code of Conduct
-   Reporting of security issues

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

[Back to Top](#top) - [Strands Agents SDK Python](https://github.com/strands-agents/sdk-python)
```
Key improvements and SEO considerations:

*   **Concise Hook:**  Uses a clear and compelling opening sentence.
*   **Targeted Keywords:**  Includes relevant keywords like "AI Agents," "SDK," "Model-driven," and mentions specific model providers.
*   **Clear Headings:**  Organized with well-defined headings and subheadings for readability and SEO structure.
*   **Bulleted Key Features:**  Highlights the core benefits in an easy-to-scan format.
*   **Installation and Usage:**  Keeps the essential "Quick Start" and "Installation" sections prominent.
*   **Detailed Feature Sections:** Includes the critical content to demonstrate the flexibility and power of Strands Agents.
*   **Documentation Links:** Provides clear links to important resources.
*   **Call to Action:** Encourages contribution.
*   **SEO-Friendly:** The structure of the document is much more SEO-friendly, allowing search engines to understand the content easily.
*   **Back to Repo Link:** Adds a clear link at the end of the document to the original repo.
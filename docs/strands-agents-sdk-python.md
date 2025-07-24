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
    Quickly build and deploy AI agents with a model-driven approach using the Strands Agents Python SDK.
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

Strands Agents is a Python SDK designed to simplify the creation and deployment of AI agents. This model-driven approach allows you to build everything from simple chatbots to complex autonomous systems, all while scaling seamlessly from local development to production environments.  **Unlock the power of AI agents with the Strands Agents Python SDK, offering a flexible and efficient way to build intelligent applications.**

## Key Features

*   **Simplified Agent Development:** Build AI agents with just a few lines of code.
*   **Model Agnostic:** Supports a wide range of model providers, including Amazon Bedrock, Anthropic, LiteLLM, Llama, Ollama, OpenAI, and Writer, as well as custom providers.
*   **Advanced Agent Capabilities:**  Create multi-agent systems, autonomous agents, and take advantage of streaming support.
*   **Built-in MCP Support:** Integrates seamlessly with Model Context Protocol (MCP) servers for access to numerous pre-built tools.
*   **Python-Based Tooling:** Easily create custom tools using Python decorators.
*   **Hot Reloading:** Enables automatic tool loading and reloading from a specified directory.

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

> **Note:** For the default Amazon Bedrock model provider, you'll need AWS credentials configured and model access enabled for Claude 4 Sonnet in the us-west-2 region. See the [Quickstart Guide](https://strandsagents.com/) for details on configuring other model providers.

## Installation

Ensure you have Python 3.10+ installed, then:

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# Install Strands and tools
pip install strands-agents strands-agents-tools
```

## Core Concepts

### Python-Based Tools

Build tools with ease using Python decorators.  This allows you to define the functionality of your agent's tools using familiar Python syntax:

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

Integrate seamlessly with Model Context Protocol (MCP) servers to utilize a vast library of pre-built tools and enhance your agent's capabilities.

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

Utilize a variety of model providers to power your AI agents:

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

The `strands-agents-tools` package provides pre-built tools for quick experimentation and rapid prototyping:

```python
from strands import Agent
from strands_tools import calculator
agent = Agent(tools=[calculator])
agent("What is the square root of 1764")
```

The tools package is available on GitHub: [strands-agents/tools](https://github.com/strands-agents/tools).

## Resources

*   **[Documentation](https://strandsagents.com/)**: Comprehensive documentation with guides, examples, and an API reference.
*   [Quick Start Guide](https://strandsagents.com/latest/user-guide/quickstart/)
*   [Agent Loop](https://strandsagents.com/latest/user-guide/concepts/agents/agent-loop/)
*   [Examples](https://strandsagents.com/latest/examples/)
*   [API Reference](https://strandsagents.com/latest/api-reference/agent/)
*   [Production & Deployment Guide](https://strandsagents.com/latest/user-guide/deploy/operating-agents-in-production/)

## Contributing ❤️

We welcome contributions! See our [Contributing Guide](CONTRIBUTING.md) for details on how to contribute.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for information on security issues.

## Further Information
Explore the [Strands Agents Python SDK](https://github.com/strands-agents/sdk-python) for more details.
```
Key improvements and SEO considerations:

*   **Clear Headline:** "Strands Agents: Build Powerful AI Agents with Ease" is more keyword-rich.
*   **Concise Hook:** The one-sentence hook immediately describes the SDK's purpose.
*   **Keyword Optimization:**  Uses relevant keywords like "AI agents," "Python SDK," "model-driven," and "build and deploy."
*   **Bulleted Key Features:**  Easily scannable for users.
*   **More Descriptive Headings:** Improved headings for clarity and SEO.
*   **Simplified Language:** More straightforward and user-friendly language.
*   **Calls to Action:** Implicit calls to action (e.g., "Unlock the power of AI agents").
*   **Internal Links:** Use of descriptive anchor text for internal links (e.g., "Quick Start Guide").
*   **Improved Formatting:** Use of Markdown headings and bullet points to improve readability.
*   **`Further Information` section**: Added a section to emphasize linking to original repo.
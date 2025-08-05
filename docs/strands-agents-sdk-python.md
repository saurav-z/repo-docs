<div align="center">
  <div>
    <a href="https://strandsagents.com">
      <img src="https://strandsagents.com/latest/assets/logo-github.svg" alt="Strands Agents" width="55px" height="105px">
    </a>
  </div>

  <h1>
    Strands Agents: Build Powerful AI Agents Quickly
  </h1>

  <p>
    <b>Create sophisticated AI agents with ease using Strands Agents, a model-driven Python SDK.</b>
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

Strands Agents is a powerful Python SDK designed to simplify the creation and deployment of AI agents.  It offers a model-driven approach, allowing you to build everything from simple conversational bots to complex, autonomous workflows with minimal code.  Scale your AI agent projects seamlessly, whether you're developing locally or deploying to production.

## Key Features

*   **Model Agnostic:** Supports a wide range of model providers including Amazon Bedrock, Anthropic, LiteLLM, Llama, Ollama, OpenAI, Writer, and custom providers.
*   **Easy Tool Integration:**  Build and integrate custom tools with Python decorators for enhanced agent capabilities. Hot reloading from a directory for development.
*   **Model Context Protocol (MCP) Support:**  Native support for MCP servers, providing access to a vast library of pre-built tools and knowledge resources.
*   **Built-in features:** Multi-agent systems, autonomous agents, and streaming support

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

> **Note:** Configure AWS credentials and enable model access for Claude 4 Sonnet in the us-west-2 region for the default Amazon Bedrock provider. Refer to the [Quickstart Guide](https://strandsagents.com/) for detailed model provider configuration instructions.

## Installation

To get started with Strands Agents, ensure you have Python 3.10+ installed and follow these steps:

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# Install Strands and tools
pip install strands-agents strands-agents-tools
```

## Features in Detail

### Python-Based Tools

Easily create and integrate custom tools using Python decorators.

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

#### Hot Reloading

Enable automatic tool loading and reloading from the `./tools/` directory:

```python
from strands import Agent

# Agent will watch ./tools/ directory for changes
agent = Agent(load_tools_from_directory=True)
response = agent("Use any tools you find in the tools directory")
```

### Model Context Protocol (MCP) Support

Seamlessly integrate with Model Context Protocol (MCP) servers for access to additional tools and resources.

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

Strands Agents offers broad support for various Large Language Models (LLMs).

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

**Custom Providers:**
Implement custom providers using the [Custom Providers](https://strandsagents.com/latest/user-guide/concepts/model-providers/custom_model_provider/) documentation.

### Example Tools

The `strands-agents-tools` package provides pre-built tools for rapid experimentation:

```python
from strands import Agent
from strands_tools import calculator
agent = Agent(tools=[calculator])
agent("What is the square root of 1764")
```

Explore the `strands-agents-tools` package on GitHub:  [strands-agents/tools](https://github.com/strands-agents/tools).

## Documentation

Find comprehensive documentation and examples to guide your projects:

-   [User Guide](https://strandsagents.com/)
-   [Quick Start Guide](https://strandsagents.com/latest/user-guide/quickstart/)
-   [Agent Loop](https://strandsagents.com/latest/user-guide/concepts/agents/agent-loop/)
-   [Examples](https://strandsagents.com/latest/examples/)
-   [API Reference](https://strandsagents.com/latest/api-reference/agent/)
-   [Production & Deployment Guide](https://strandsagents.com/latest/user-guide/deploy/operating-agents-in-production/)

## Contributing ❤️

We welcome contributions! Learn how to contribute by reviewing our [Contributing Guide](CONTRIBUTING.md), including details on:

-   Reporting bugs & features
-   Development setup
-   Contributing via Pull Requests
-   Code of Conduct
-   Reporting of security issues

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for information on security.

**[Back to Top](#top)**  (Added for better navigation)

**(Original Repo: [https://github.com/strands-agents/sdk-python](https://github.com/strands-agents/sdk-python))**
```
Key improvements and SEO considerations:

*   **Clear Headline:**  The title is optimized with the keyword "AI Agents" to improve search visibility.
*   **One-Sentence Hook:** A concise and compelling introduction to capture attention.
*   **Keyword Optimization:** "AI Agents," "Python SDK," "Model-driven" used strategically.
*   **Bulleted Key Features:** Uses bullet points for scannability and clarity.
*   **Structured Sections:**  Uses headings (H1, H2, H3) for better readability and SEO ranking.
*   **Detailed Explanations:** Expanded explanations of features.
*   **Internal Linking:**  Added a "Back to Top" link for easy navigation and the original repo link.
*   **Stronger Call to Action:** Encourages use with installation and quick start examples.
*   **More Comprehensive Feature Descriptions:** Expand on the benefits of each feature.
*   **Clean Formatting:** Consistent Markdown for readability.
*   **Emphasis on Benefits:** Focuses on what users can *do* with the SDK (e.g., "Create sophisticated AI agents").
*   **Clearer Instructions:** Improves installation and usage guidance.
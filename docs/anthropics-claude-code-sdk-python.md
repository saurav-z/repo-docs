# Claude Code SDK for Python: Interact with Claude Code in Your Python Applications

**Unlock the power of Claude Code directly within your Python projects with the Claude Code SDK!**  This Python SDK simplifies interacting with the powerful Claude Code model, enabling you to seamlessly integrate advanced code generation and analysis capabilities into your applications. [Explore the original repository](https://github.com/anthropics/claude-code-sdk-python).

## Key Features

*   **Easy Integration:**  Simple installation and quick start guide get you up and running fast.
*   **Asynchronous Operations:** Built for asynchronous operations using `async` and `await` for non-blocking interaction.
*   **Flexible Configuration:** Customize your interactions with options like system prompts and maximum turns.
*   **Tool Support:** Enables the use of tools like `Read`, `Write`, and `Bash` within your prompts, expanding the model's capabilities.
*   **In-Process MCP Servers:**  Significantly simplifies tool integration with built-in SDK MCP servers, eliminating subprocess management.

## Installation

Install the SDK using `pip`:

```bash
pip install claude-code-sdk
```

**Prerequisites:**

*   Python 3.10+
*   Node.js
*   Claude Code:  `npm install -g @anthropic-ai/claude-code`

## Quick Start

Get started querying Claude Code with a few lines of code:

```python
import anyio
from claude_code_sdk import query

async def main():
    async for message in query(prompt="What is 2 + 2?"):
        print(message)

anyio.run(main)
```

## Usage

### Basic Querying

Send simple prompts and receive responses:

```python
from claude_code_sdk import query, ClaudeCodeOptions, AssistantMessage, TextBlock

# Simple query
async for message in query(prompt="Hello Claude"):
    if isinstance(message, AssistantMessage):
        for block in message.content:
            if isinstance(block, TextBlock):
                print(block.text)

# With options
options = ClaudeCodeOptions(
    system_prompt="You are a helpful assistant",
    max_turns=1
)

async for message in query(prompt="Tell me a joke", options=options):
    print(message)
```

### Utilizing Tools

Leverage Claude Code's tool capabilities:

```python
options = ClaudeCodeOptions(
    allowed_tools=["Read", "Write", "Bash"],
    permission_mode='acceptEdits'  # auto-accept file edits
)

async for message in query(
    prompt="Create a hello.py file", 
    options=options
):
    # Process tool use and results
    pass
```

### Setting the Working Directory

Configure the working directory for your interactions:

```python
from pathlib import Path

options = ClaudeCodeOptions(
    cwd="/path/to/project"  # or Path("/path/to/project")
)
```

### In-Process SDK MCP Servers (Advanced)

The SDK allows creating and using MCP servers within your Python code, offering advantages like reduced overhead and simpler deployments.

#### Creating a Simple Tool

```python
from claude_code_sdk import tool, create_sdk_mcp_server

# Define a tool using the @tool decorator
@tool("greet", "Greet a user", {"name": str})
async def greet_user(args):
    return {
        "content": [
            {"type": "text", "text": f"Hello, {args['name']}!"}
        ]
    }

# Create an SDK MCP server
server = create_sdk_mcp_server(
    name="my-tools",
    version="1.0.0",
    tools=[greet_user]
)

# Use it with Claude
options = ClaudeCodeOptions(
    mcp_servers={"tools": server}
)

async for message in query(prompt="Greet Alice", options=options):
    print(message)
```

#### Benefits of In-Process Servers

*   No subprocess management
*   Better performance
*   Simpler deployment
*   Easier debugging
*   Type safety

#### Migration from External Servers

Easily switch from external to SDK MCP servers:

```python
# BEFORE: External MCP server (separate process)
options = ClaudeCodeOptions(
    mcp_servers={
        "calculator": {
            "type": "stdio",
            "command": "python",
            "args": ["-m", "calculator_server"]
        }
    }
)

# AFTER: SDK MCP server (in-process)
from my_tools import add, subtract  # Your tool functions

calculator = create_sdk_mcp_server(
    name="calculator",
    tools=[add, subtract]
)

options = ClaudeCodeOptions(
    mcp_servers={"calculator": calculator}
)
```

#### Mixed Server Support

Combine SDK and external MCP servers:

```python
options = ClaudeCodeOptions(
    mcp_servers={
        "internal": sdk_server,      # In-process SDK server
        "external": {                # External subprocess server
            "type": "stdio",
            "command": "external-server"
        }
    }
)
```

## API Reference

### `query(prompt, options=None)`

The core function for interacting with Claude Code.

**Parameters:**

*   `prompt` (str): The text prompt to send to Claude Code.
*   `options` (ClaudeCodeOptions, *optional*): Configuration settings for the query.

**Returns:**  `AsyncIterator[Message]`: An asynchronous iterator that yields response messages as they are received.

### Types

Explore the available types for configuration and message handling: See [src/claude_code_sdk/types.py](src/claude_code_sdk/types.py) for complete type definitions:

*   `ClaudeCodeOptions` - Configuration options
*   `AssistantMessage`, `UserMessage`, `SystemMessage`, `ResultMessage` - Message types
*   `TextBlock`, `ToolUseBlock`, `ToolResultBlock` - Content blocks

## Error Handling

Implement robust error handling to manage potential issues:

```python
from claude_code_sdk import (
    ClaudeSDKError,      # Base error
    CLINotFoundError,    # Claude Code not installed
    CLIConnectionError,  # Connection issues
    ProcessError,        # Process failed
    CLIJSONDecodeError,  # JSON parsing issues
)

try:
    async for message in query(prompt="Hello"):
        pass
except CLINotFoundError:
    print("Please install Claude Code")
except ProcessError as e:
    print(f"Process failed with exit code: {e.exit_code}")
except CLIJSONDecodeError as e:
    print(f"Failed to parse response: {e}")
```

See [src/claude_code_sdk/_errors.py](src/claude_code_sdk/_errors.py) for a complete list of error types.

## Available Tools

Refer to the [Claude Code documentation](https://docs.anthropic.com/en/docs/claude-code/settings#tools-available-to-claude) for a comprehensive list of tools supported by the Claude Code model.

## Examples

Find a complete working example to help you get started: [examples/quick_start.py](examples/quick_start.py).

## License

MIT
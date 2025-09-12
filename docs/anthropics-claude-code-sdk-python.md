# Claude Code SDK for Python: Build Powerful Applications with Anthropic's Code Generation Model

**Harness the power of Claude Code directly within your Python applications with the official Claude Code SDK for Python.**

[Original Repository](https://github.com/anthropics/claude-code-sdk-python)

## Key Features

*   **Seamless Integration:** Easily integrate with Claude Code for code generation, editing, and more.
*   **Asynchronous Operations:** Built for asynchronous execution, enabling efficient and responsive applications.
*   **Flexible Configuration:** Customize Claude Code behavior with a wide range of options.
*   **Tool Support:** Utilize Claude Code's tool capabilities for enhanced functionality and interaction.
*   **In-Process MCP Servers:** Run MCP servers directly within your Python application for improved performance and simplified deployment.
*   **Comprehensive Error Handling:** Robust error handling to ensure your application's stability.

## Installation

Get started quickly with a simple `pip` command:

```bash
pip install claude-code-sdk
```

**Prerequisites:**

*   Python 3.10+
*   Node.js
*   Claude Code CLI: Install globally using npm: `npm install -g @anthropic-ai/claude-code`

## Quick Start

Query Claude Code in just a few lines of code:

```python
import anyio
from claude_code_sdk import query

async def main():
    async for message in query(prompt="What is 2 + 2?"):
        print(message)

anyio.run(main)
```

## Usage

### Basic Query

Send simple prompts and process the responses:

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

### Using Tools

Enable Claude Code to use tools for advanced capabilities:

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

### Working Directory

Specify a working directory for file operations:

```python
from pathlib import Path

options = ClaudeCodeOptions(
    cwd="/path/to/project"  # or Path("/path/to/project")
)
```

### SDK MCP Servers (In-Process)

Improve performance and simplify deployment with in-process MCP servers:

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

#### Benefits Over External MCP Servers

*   No subprocess management
*   Better performance
*   Simpler deployment
*   Easier debugging
*   Type safety

#### Migration from External Servers

Easily migrate from external MCP servers:

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

Combine in-process and external servers:

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

*   `prompt` (str): The input prompt.
*   `options` (ClaudeCodeOptions): Configuration settings.

**Returns:** `AsyncIterator[Message]` - An asynchronous stream of messages.

### Types

*   `ClaudeCodeOptions`: Configuration options.
*   `AssistantMessage`, `UserMessage`, `SystemMessage`, `ResultMessage`: Message types.
*   `TextBlock`, `ToolUseBlock`, `ToolResultBlock`: Content blocks.

For complete type definitions, see: [src/claude\_code\_sdk/types.py](src/claude_code_sdk/types.py)

## Error Handling

Implement robust error handling to catch potential issues:

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

For a full list of error types, see: [src/claude\_code\_sdk/\_errors.py](src/claude_code_sdk/_errors.py)

## Available Tools

Explore the full range of tools available to Claude Code to enhance your applications. For more details, refer to the: [Claude Code documentation](https://docs.anthropic.com/en/docs/claude-code/settings#tools-available-to-claude)

## Examples

Find a complete working example: [examples/quick\_start.py](examples/quick_start.py)

## License

MIT
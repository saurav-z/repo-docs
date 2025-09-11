# Claude Code SDK for Python: Build Powerful Applications with Anthropic's Code Model

**Unleash the power of Claude Code directly within your Python applications using the official Python SDK.**  [Explore the original repository](https://github.com/anthropics/claude-code-sdk-python).

## Key Features:

*   **Seamless Integration:** Easily interact with Claude Code's code generation capabilities.
*   **Asynchronous Querying:** Leverage `async` and `await` for non-blocking operations.
*   **Tool Support:** Integrate Claude Code with external tools for extended functionality.
*   **In-Process MCP Servers:** Run MCP servers directly within your Python application for improved performance and simpler deployment.
*   **Comprehensive Error Handling:** Robust error handling for a stable application.
*   **Type Safety:** Leveraging types for better code quality.

## Installation

Get started by installing the SDK using `pip`:

```bash
pip install claude-code-sdk
```

**Prerequisites:**

*   Python 3.10+
*   Node.js
*   Claude Code CLI: `npm install -g @anthropic-ai/claude-code`

## Quick Start

Here's how to get up and running quickly:

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

Send simple prompts to Claude Code.

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

Enable Claude Code to interact with tools.

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

Specify a working directory for Claude Code.

```python
from pathlib import Path

options = ClaudeCodeOptions(
    cwd="/path/to/project"  # or Path("/path/to/project")
)
```

### SDK MCP Servers (In-Process)

Reduce overhead by running MCP servers directly within your Python application.

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

*   **No subprocess management** - Runs in the same process as your application
*   **Better performance** - No IPC overhead for tool calls
*   **Simpler deployment** - Single Python process instead of multiple
*   **Easier debugging** - All code runs in the same process
*   **Type safety** - Direct Python function calls with type hints

#### Migration from External Servers

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

Combine in-process and external MCP servers for maximum flexibility.

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

The primary function for interacting with Claude Code.

**Parameters:**

*   `prompt` (str): The text prompt for Claude Code.
*   `options` (ClaudeCodeOptions): Optional configuration settings.

**Returns:** AsyncIterator[Message] - A stream of messages from Claude Code.

### Types

Explore the complete type definitions in `src/claude_code_sdk/types.py`, including:

*   `ClaudeCodeOptions`: Configuration options.
*   `AssistantMessage`, `UserMessage`, `SystemMessage`, `ResultMessage`: Message types.
*   `TextBlock`, `ToolUseBlock`, `ToolResultBlock`: Content blocks.

## Error Handling

Implement robust error handling to manage potential issues.

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

See `src/claude_code_sdk/_errors.py` for a complete list of error types.

## Available Tools

Refer to the [Claude Code documentation](https://docs.anthropic.com/en/docs/claude-code/settings#tools-available-to-claude) for a list of available tools.

## Examples

Find complete working examples in `examples/quick_start.py`.

## License

MIT
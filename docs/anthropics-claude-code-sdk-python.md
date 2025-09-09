# Claude Code SDK for Python: Build Powerful Applications with Anthropic's Code Generation Model

**Unlock the power of Anthropic's Claude Code model directly within your Python applications** with the Claude Code SDK for Python!  Get started today and explore how you can leverage this SDK to build intelligent tools and applications.  For more in-depth information, see the [Claude Code SDK documentation](https://docs.anthropic.com/en/docs/claude-code/sdk).

[![GitHub](https://img.shields.io/badge/GitHub-Claude--Code--SDK--Python-blue?style=flat-square&logo=github)](https://github.com/anthropics/claude-code-sdk-python)

## Key Features

*   **Easy Installation:**  Simple `pip install claude-code-sdk` for quick setup.
*   **Asynchronous Querying:** Leverage Python's `asyncio` for efficient, non-blocking interactions with Claude Code.
*   **Flexible Configuration:** Customize your queries with `ClaudeCodeOptions` for system prompts, tool selection, and more.
*   **Tool Integration:** Seamlessly integrate external tools like Read, Write, and Bash to expand Claude Code's capabilities.
*   **In-Process SDK MCP Servers:** Run MCP servers directly within your Python application for improved performance, simpler deployment, and easier debugging.
*   **Comprehensive Error Handling:** Robust error handling for common issues like CLI installation and process failures.

## Installation

1.  **Prerequisites:**
    *   Python 3.10+
    *   Node.js
    *   Claude Code CLI: `npm install -g @anthropic-ai/claude-code`
2.  **Install the SDK:**
    ```bash
    pip install claude-code-sdk
    ```

## Quickstart

Get up and running with a simple query in just a few lines of code:

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
Send a simple query to Claude Code and process the response.

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
Enable Claude Code to interact with external tools to perform actions.

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
Specify a working directory for Claude Code to operate within.

```python
from pathlib import Path

options = ClaudeCodeOptions(
    cwd="/path/to/project"  # or Path("/path/to/project")
)
```

### SDK MCP Servers (In-Process)

**Benefits:**

*   **No subprocess management**
*   **Better performance**
*   **Simpler deployment**
*   **Easier debugging**
*   **Type safety**

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

You can use both SDK and external MCP servers together:

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

**Description:** The primary async function for querying Claude Code.

**Parameters:**

*   `prompt` (str): The text prompt to send to Claude Code.
*   `options` (ClaudeCodeOptions, optional): Configuration options for the query.

**Returns:** `AsyncIterator[Message]` - An asynchronous stream of response messages.

### Types

Detailed type definitions are available in [src/claude_code_sdk/types.py](src/claude_code_sdk/types.py):

*   `ClaudeCodeOptions`
*   `AssistantMessage`, `UserMessage`, `SystemMessage`, `ResultMessage`
*   `TextBlock`, `ToolUseBlock`, `ToolResultBlock`

## Error Handling

Implement robust error handling to gracefully manage potential issues.

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

See [src/claude_code_sdk/\_errors.py](src/claude_code_sdk/_errors.py) for a complete list of error types.

## Available Tools

Explore the [Claude Code documentation](https://docs.anthropic.com/en/docs/claude-code/settings#tools-available-to-claude) to discover the complete list of tools available for integration.

## Examples

For a fully functional example, see [examples/quick_start.py](examples/quick_start.py).

## License

This project is licensed under the MIT License.
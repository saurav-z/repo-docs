# Claude Code SDK for Python: Build Powerful AI-Powered Code Generation Applications

**Unlock the power of Anthropic's Claude Code directly within your Python projects with the official Python SDK.**  This SDK provides a seamless interface to interact with Claude Code, enabling you to build innovative applications for code generation, debugging, and more.  Learn more in the [Claude Code SDK documentation](https://docs.anthropic.com/en/docs/claude-code/sdk).

[Link to Original Repo](https://github.com/anthropics/claude-code-sdk-python)

## Key Features

*   **Simple Integration:** Easy-to-install Python package.
*   **Asynchronous Querying:** Utilize the `query` function to stream responses.
*   **Flexible Configuration:** Customize Claude Code's behavior with `ClaudeCodeOptions`, including system prompts, maximum turns, and tool permissions.
*   **Tool Integration:** Leverage tools like `Read`, `Write`, and `Bash` to extend Claude Code's capabilities.
*   **Working Directory Support:** Define a working directory for file operations.
*   **In-Process SDK MCP Servers:** Enhanced performance, debugging, and deployment with built-in MCP servers.
*   **Mixed Server Support:** Use both SDK and external MCP servers simultaneously.
*   **Comprehensive Error Handling:** Robust error handling to identify and address issues.

## Installation

Get started by installing the SDK using pip:

```bash
pip install claude-code-sdk
```

**Prerequisites:**

*   Python 3.10+
*   Node.js
*   Claude Code CLI: `npm install -g @anthropic-ai/claude-code`

## Quick Start

Here's a quick example to get you up and running:

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

```python
from pathlib import Path

options = ClaudeCodeOptions(
    cwd="/path/to/project"  # or Path("/path/to/project")
)
```

### SDK MCP Servers (In-Process)

The SDK supports in-process MCP servers, providing direct tool calls within your Python application.

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

*   **No subprocess management:** Run directly within your application.
*   **Improved Performance:** No Inter-Process Communication (IPC) overhead.
*   **Simplified Deployment:** Deploy a single Python process.
*   **Easier Debugging:** Streamlined debugging within your application.
*   **Type Safety:** Leverage Python's type hints for tools.

#### Migration from External Servers

Migrate from external servers to improve performance:

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

Combine SDK and external MCP servers in your applications:

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

The core asynchronous function for interacting with Claude Code.

**Parameters:**

*   `prompt` (str): The user's prompt to send to Claude.
*   `options` (ClaudeCodeOptions, optional): Configuration options. Defaults to `None`.

**Returns:** `AsyncIterator[Message]`: A stream of response messages.

### Types

For detailed information on the available types, see [src/claude_code_sdk/types.py](src/claude_code_sdk/types.py):

*   `ClaudeCodeOptions`: Configuration options.
*   `AssistantMessage`, `UserMessage`, `SystemMessage`, `ResultMessage`: Message types.
*   `TextBlock`, `ToolUseBlock`, `ToolResultBlock`: Content blocks.

## Error Handling

Implement error handling for robustness:

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

For a comprehensive list of error types, refer to [src/claude_code_sdk/\_errors.py](src/claude_code_sdk/_errors.py).

## Available Tools

Explore the full suite of available tools in the [Claude Code documentation](https://docs.anthropic.com/en/docs/claude-code/settings#tools-available-to-claude).

## Examples

Get started quickly with a complete working example in [examples/quick_start.py](examples/quick_start.py).

## License

MIT
# Claude Code SDK for Python: Build Powerful AI-Powered Code Interactions

**Unlock the potential of AI-driven code generation and interaction with the Claude Code SDK for Python!**  [View the original repository](https://github.com/anthropics/claude-code-sdk-python).

## Key Features

*   **Easy Integration:** Seamlessly integrate Claude Code into your Python projects.
*   **Asynchronous Querying:** Utilize asynchronous functions for efficient and non-blocking interactions.
*   **Flexible Configuration:** Customize behavior with `ClaudeCodeOptions` including system prompts and turn limits.
*   **Tool Integration:**  Enable Claude Code to use tools such as Read, Write, and Bash to enhance its capabilities.
*   **In-Process MCP Servers:** Run MCP servers directly within your Python application, simplifying deployment and improving performance.
*   **Comprehensive Error Handling:** Robust error handling for common issues like Claude Code not found, connection problems, and process failures.

## Installation

Install the SDK using pip:

```bash
pip install claude-code-sdk
```

**Prerequisites:**

*   Python 3.10+
*   Node.js
*   Claude Code: Install globally using `npm install -g @anthropic-ai/claude-code`

## Quick Start

Get started quickly with a simple query:

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

Enable Claude Code to leverage tools to perform actions:

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

Specify a working directory for your project:

```python
from pathlib import Path

options = ClaudeCodeOptions(
    cwd="/path/to/project"  # or Path("/path/to/project")
)
```

### SDK MCP Servers (In-Process)

Leverage in-process MCP servers for enhanced performance and simplified deployment.

#### Creating a Simple Tool

Define custom tools within your Python code:

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

Easily migrate from external MCP servers to in-process servers:

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

Combine SDK and external MCP servers for flexible configurations:

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

The main asynchronous function for interacting with Claude Code.

**Parameters:**

*   `prompt` (str): The text prompt to send to Claude.
*   `options` (ClaudeCodeOptions): Optional configuration settings.

**Returns:** `AsyncIterator[Message]` -  A stream of response messages.

### Types

Explore the type definitions for deeper understanding: See [src/claude\_code\_sdk/types.py](src/claude_code_sdk/types.py) for complete type definitions:

*   `ClaudeCodeOptions` - Configuration options
*   `AssistantMessage`, `UserMessage`, `SystemMessage`, `ResultMessage` - Message types
*   `TextBlock`, `ToolUseBlock`, `ToolResultBlock` - Content blocks

## Error Handling

Handle potential errors gracefully:

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

See [src/claude_code_sdk/\_errors.py](src/claude_code_sdk/_errors.py) for a full list of error types.

## Available Tools

Refer to the official documentation for available tools: See the [Claude Code documentation](https://docs.anthropic.com/en/docs/claude-code/settings#tools-available-to-claude) for a complete list of available tools.

## Examples

Find working examples to get started quickly: See [examples/quick\_start.py](examples/quick_start.py) for a complete working example.

## License

MIT
# Claude Code SDK for Python: Build Powerful AI-Powered Code Assistants

**Unlock the power of Anthropic's Claude Code with the Python SDK, enabling you to build and integrate intelligent code assistance directly into your applications.**  [Explore the original repository](https://github.com/anthropics/claude-code-sdk-python) for more details.

**Key Features:**

*   **Asynchronous Querying:** Interact with Claude Code using async functions for efficient processing.
*   **Custom Tool Integration:** Easily define and integrate custom Python functions as tools for Claude Code to use, enhancing its capabilities.
*   **In-Process SDK MCP Servers:** Run custom tools in the same process, improving performance, simplifying deployment, and making debugging easier.
*   **Hooks for Control:** Implement hooks to control Claude Code's behavior at specific points, enabling deterministic processing and automated feedback.
*   **Error Handling:** Comprehensive error handling to gracefully manage potential issues.
*   **Type Definitions:** Access to clear type definitions for easy integration and development.

## Installation

Get started quickly by installing the SDK:

```bash
pip install claude-code-sdk
```

**Prerequisites:**

*   Python 3.10+
*   Node.js
*   Claude Code: `npm install -g @anthropic-ai/claude-code`

## Quick Start

Get up and running with a basic example:

```python
import anyio
from claude_code_sdk import query

async def main():
    async for message in query(prompt="What is 2 + 2?"):
        print(message)

anyio.run(main)
```

## Core Functionality

### Basic Usage: `query()`

The `query()` function allows you to send prompts to Claude Code and receive an `AsyncIterator` of response messages.  Refer to `src/claude_code_sdk/query.py` for the source code.

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

Configure and leverage Claude Code's tool capabilities:

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

Set the working directory for Claude Code:

```python
from pathlib import Path

options = ClaudeCodeOptions(
    cwd="/path/to/project"  # or Path("/path/to/project")
)
```

## Advanced Features

### `ClaudeSDKClient` for Interactive Conversations

The `ClaudeSDKClient` enables bidirectional, interactive conversations with Claude Code.  See `src/claude_code_sdk/client.py` for details.

### Custom Tools with SDK MCP Servers

Implement custom tools as Python functions, running directly within your application for enhanced performance and simplified deployment. For an end-to-end example, see [MCP Calculator](examples/mcp_calculator.py).

#### Creating a Simple Tool

```python
from claude_code_sdk import tool, create_sdk_mcp_server, ClaudeCodeOptions, ClaudeSDKClient

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
    mcp_servers={"tools": server},
    allowed_tools=["mcp__tools__greet"]
)

async with ClaudeSDKClient(options=options) as client:
    await client.query("Greet Alice")

    # Extract and print response
    async for msg in client.receive_response():
        print(msg)
```

#### Benefits of SDK MCP Servers

*   No subprocess management
*   Better performance with no IPC overhead
*   Simpler deployment with a single process
*   Easier debugging in the same process
*   Type safety with direct function calls

#### Migration from External Servers

Easily migrate from external MCP servers to in-process SDK servers:

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

### Hooks

Use hooks to customize Claude Code's agent loop and behavior.  See the [Claude Code Hooks Reference](https://docs.anthropic.com/en/docs/claude-code/hooks) for more information.

#### Example

```python
from claude_code_sdk import ClaudeCodeOptions, ClaudeSDKClient, HookMatcher

async def check_bash_command(input_data, tool_use_id, context):
    tool_name = input_data["tool_name"]
    tool_input = input_data["tool_input"]
    if tool_name != "Bash":
        return {}
    command = tool_input.get("command", "")
    block_patterns = ["foo.sh"]
    for pattern in command:
        if pattern in command:
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": f"Command contains invalid pattern: {pattern}",
                }
            }
    return {}

options = ClaudeCodeOptions(
    allowed_tools=["Bash"],
    hooks={
        "PreToolUse": [
            HookMatcher(matcher="Bash", hooks=[check_bash_command]),
        ],
    }
)

async with ClaudeSDKClient(options=options) as client:
    # Test 1: Command with forbidden pattern (will be blocked)
    await client.query("Run the bash command: ./foo.sh --help")
    async for msg in client.receive_response():
        print(msg)

    print("\n" + "=" * 50 + "\n")

    # Test 2: Safe command that should work
    await client.query("Run the bash command: echo 'Hello from hooks example!'")
    async for msg in client.receive_response():
        print(msg)
```

## Types

Explore the available types in `src/claude_code_sdk/types.py`:
*   `ClaudeCodeOptions`
*   `AssistantMessage`, `UserMessage`, `SystemMessage`, `ResultMessage`
*   `TextBlock`, `ToolUseBlock`, `ToolResultBlock`

## Error Handling

Implement robust error handling:

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

Refer to the [Claude Code documentation](https://docs.anthropic.com/en/docs/claude-code/settings#tools-available-to-claude) for a full list of available tools.

## Examples

Explore detailed examples:

*   `examples/quick_start.py`: A complete working example.
*   `examples/streaming_mode.py`: Comprehensive examples involving `ClaudeSDKClient`.  Interactive examples are available in `examples/streaming_mode_ipython.py`.

## License

MIT
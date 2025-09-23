# Claude Code SDK for Python: Build Powerful AI-Powered Applications

**Unleash the power of Claude Code directly within your Python applications with the official Claude Code SDK!** ([See the original repository](https://github.com/anthropics/claude-code-sdk-python))

## Key Features

*   **Seamless Integration:** Easily integrate Claude Code's capabilities into your Python projects.
*   **Asynchronous Querying:** Leverage the `query()` function for efficient, non-blocking interactions with Claude Code.
*   **Bidirectional Conversations:** Utilize `ClaudeSDKClient` for interactive, stateful conversations.
*   **Custom Tool Creation:** Define and integrate custom tools using Python functions, expanding Claude's functionality.
*   **In-Process SDK MCP Servers:** Run custom tools directly in your Python application for improved performance and simpler deployment.
*   **Advanced Hook System:** Implement deterministic logic and automated feedback using hooks to control Claude Code's behavior.
*   **Comprehensive Error Handling:** Robust error handling for common issues like missing installations, connection problems, and JSON parsing errors.
*   **Type Safety:** Pythonic design ensures direct function calls with type hints.
*   **Streaming Support:** Receive responses in real-time with the `ClaudeSDKClient`.

## Installation

Get started with the SDK:

```bash
pip install claude-code-sdk
```

**Prerequisites:**

*   Python 3.10+
*   Node.js
*   Claude Code CLI: `npm install -g @anthropic-ai/claude-code`

## Quick Start

```python
import anyio
from claude_code_sdk import query

async def main():
    async for message in query(prompt="What is 2 + 2?"):
        print(message)

anyio.run(main)
```

## Basic Usage: `query()`

The `query()` function is an asynchronous function used for simple, single-turn interactions with Claude Code. It returns an asynchronous iterator of response messages. 
*(See [src/claude_code_sdk/query.py](src/claude_code_sdk/query.py) for implementation details.)*

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

## `ClaudeSDKClient`: Interactive Conversations

`ClaudeSDKClient` enables bidirectional, interactive conversations with Claude Code.  This includes support for custom tools and hooks, implemented as Python functions.
*(See [src/claude_code_sdk/client.py](src/claude_code_sdk/client.py) for more details.)*

### Custom Tools (as In-Process SDK MCP Servers)

Custom tools are Python functions that Claude can invoke. In-process SDK MCP servers eliminate the need for separate processes, improving performance.  *(See [MCP Calculator](examples/mcp_calculator.py) for a comprehensive example.)*

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

Hooks are Python functions invoked at specific points in the Claude Code agent loop, offering deterministic processing and feedback. *(Read more in [Claude Code Hooks Reference](https://docs.anthropic.com/en/docs/claude-code/hooks).)*

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

Explore type definitions within `src/claude_code_sdk/types.py`:

*   `ClaudeCodeOptions` - Configuration Options
*   `AssistantMessage`, `UserMessage`, `SystemMessage`, `ResultMessage` - Message Types
*   `TextBlock`, `ToolUseBlock`, `ToolResultBlock` - Content Blocks

## Error Handling

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

*(See [src/claude_code_sdk/_errors.py](src/claude_code_sdk/_errors.py) for all error types.)*

## Available Tools

For a comprehensive list of available tools, refer to the official [Claude Code documentation](https://docs.anthropic.com/en/docs/claude-code/settings#tools-available-to-claude).

## Examples

*   Get a complete working example from [examples/quick_start.py](examples/quick_start.py).
*   Discover comprehensive examples using `ClaudeSDKClient` and streaming mode in [examples/streaming_mode.py](examples/streaming_mode.py).  You can also run interactive examples in IPython from [examples/streaming_mode_ipython.py](examples/streaming_mode_ipython.py).

## License

MIT
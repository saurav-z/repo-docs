# Claude Code SDK for Python: Build Intelligent Applications with Anthropic's Code Generation Model

**Unlock the power of Anthropic's Claude Code model to build powerful code generation and analysis tools with the Claude Code SDK for Python!**  Explore the [Claude Code SDK on GitHub](https://github.com/anthropics/claude-code-sdk-python) to get started.

## Key Features

*   **Asynchronous Querying:** Easily query Claude Code with the `query()` function, receiving responses as an asynchronous iterator.
*   **Custom Tool Integration:**  Define and integrate custom tools as in-process MCP servers, enabling Claude to perform specific actions within your application.
*   **Interactive Conversations:**  Use `ClaudeSDKClient` for bidirectional conversations, allowing for more complex interactions with Claude Code.
*   **Hooks for Deterministic Processing:** Implement hooks to provide deterministic processing and automated feedback within the Claude Code agent loop.
*   **Flexible Configuration:**  Customize Claude Code's behavior with `ClaudeAgentOptions`, including system prompts, allowed tools, and working directory settings.
*   **Comprehensive Error Handling:** Robust error handling with specific exception types for common issues like CLI not found, connection errors, and process failures.
*   **Type Definitions:**  Utilize clear type definitions for messages, content blocks, and configuration options, improving code readability and maintainability.

## Installation

```bash
pip install claude-code-sdk
```

**Prerequisites:**

*   Python 3.10+
*   Node.js
*   Claude Code: `npm install -g @anthropic-ai/claude-code`

## Quick Start

```python
import anyio
from claude_code_sdk import query

async def main():
    async for message in query(prompt="What is 2 + 2?"):
        print(message)

anyio.run(main)
```

## Core Functionality

### `query()`: Asynchronous Querying

The `query()` function is an asynchronous generator for querying Claude Code, returning an `AsyncIterator` of response messages. ([src/claude_code_sdk/query.py](src/claude_code_sdk/query.py))

```python
from claude_code_sdk import query, ClaudeAgentOptions, AssistantMessage, TextBlock

# Simple query
async for message in query(prompt="Hello Claude"):
    if isinstance(message, AssistantMessage):
        for block in message.content:
            if isinstance(block, TextBlock):
                print(block.text)

# With options
options = ClaudeAgentOptions(
    system_prompt="You are a helpful assistant",
    max_turns=1
)

async for message in query(prompt="Tell me a joke", options=options):
    print(message)
```

### Using Tools

```python
options = ClaudeAgentOptions(
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

options = ClaudeAgentOptions(
    cwd="/path/to/project"  # or Path("/path/to/project")
)
```

## Advanced Features

### `ClaudeSDKClient`: Interactive Conversations and Customization

`ClaudeSDKClient` provides bidirectional, interactive conversations with Claude Code, with more advanced capabilities than `query()`.  ([src/claude_code_sdk/client.py](src/claude_code_sdk/client.py))

#### Custom Tools (In-Process SDK MCP Servers)

Create custom tools as in-process MCP servers, directly within your Python application. This offers significant advantages over external MCP servers:

*   **No subprocess management** - Runs in the same process as your application
*   **Better performance** - No IPC overhead for tool calls
*   **Simpler deployment** - Single Python process instead of multiple
*   **Easier debugging** - All code runs in the same process
*   **Type safety** - Direct Python function calls with type hints

For a complete example, see [MCP Calculator](examples/mcp_calculator.py).

##### Creating a Simple Tool

```python
from claude_code_sdk import tool, create_sdk_mcp_server, ClaudeAgentOptions, ClaudeSDKClient

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
options = ClaudeAgentOptions(
    mcp_servers={"tools": server},
    allowed_tools=["mcp__tools__greet"]
)

async with ClaudeSDKClient(options=options) as client:
    await client.query("Greet Alice")

    # Extract and print response
    async for msg in client.receive_response():
        print(msg)
```

##### Migration from External Servers

```python
# BEFORE: External MCP server (separate process)
options = ClaudeAgentOptions(
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

options = ClaudeAgentOptions(
    mcp_servers={"calculator": calculator}
)
```

##### Mixed Server Support

Combine SDK and external MCP servers:

```python
options = ClaudeAgentOptions(
    mcp_servers={
        "internal": sdk_server,      # In-process SDK server
        "external": {                # External subprocess server
            "type": "stdio",
            "command": "external-server"
        }
    }
)
```

#### Hooks

Implement hooks to modify Claude Code's behavior at specific points in the agent loop.  Refer to [Claude Code Hooks Reference](https://docs.anthropic.com/en/docs/claude-code/hooks) for more information.  See [examples/hooks.py](examples/hooks.py) for more examples.

##### Example

```python
from claude_code_sdk import ClaudeAgentOptions, ClaudeSDKClient, HookMatcher

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

options = ClaudeAgentOptions(
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

For comprehensive type definitions, see [src/claude_code_sdk/types.py](src/claude_code_sdk/types.py):

*   `ClaudeAgentOptions`: Configuration options
*   `AssistantMessage`, `UserMessage`, `SystemMessage`, `ResultMessage`: Message types
*   `TextBlock`, `ToolUseBlock`, `ToolResultBlock`: Content blocks

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

All error types can be found in [src/claude_code_sdk/_errors.py](src/claude_code_sdk/_errors.py).

## Available Tools

For a complete list of available tools, please consult the [Claude Code documentation](https://docs.anthropic.com/en/docs/claude-code/settings#tools-available-to-claude).

## Examples

*   [examples/quick\_start.py](examples/quick_start.py): A complete working example.
*   [examples/streaming\_mode.py](examples/streaming_mode.py): Comprehensive examples involving `ClaudeSDKClient`.
*   [examples/streaming\_mode\_ipython.py](examples/streaming_mode_ipython.py): Interactive examples in IPython.

## License

MIT
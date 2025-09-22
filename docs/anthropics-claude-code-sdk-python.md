# Claude Code SDK for Python: Build Intelligent Code Assistants

Use the official Python SDK to seamlessly integrate Claude Code's powerful AI capabilities into your Python applications, allowing you to build AI-powered code assistants and automation tools. [See the original repo here](https://github.com/anthropics/claude-code-sdk-python).

**Key Features:**

*   **Easy Installation:** Simple installation via `pip install claude-code-sdk`.
*   **Asynchronous Querying:**  Leverage the `query()` function for easy, asynchronous interactions with Claude Code.
*   **Custom Tool Integration:**  Define and integrate custom tools as Python functions, enabling Claude Code to interact with your specific application logic.
*   **Advanced SDK Client:** Utilize `ClaudeSDKClient` for bidirectional conversations and support for custom tools and hooks.
*   **In-Process SDK MCP Servers:** Implement custom tools as in-process MCP servers, improving performance and simplifying deployment compared to external servers.
*   **Hooks for Enhanced Control:**  Implement hooks to control and automate interactions within Claude Code's agent loop.
*   **Comprehensive Error Handling:** Robust error handling with specific exception types for common issues.
*   **Detailed Documentation & Examples:**  Thorough documentation with code snippets and working examples to get you started quickly.

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

### Querying with `query()`

The `query()` function allows for simple, asynchronous interactions with Claude Code.

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

## Advanced Features

### `ClaudeSDKClient` for Interactive Conversations

`ClaudeSDKClient` enables bidirectional and interactive conversations with Claude Code, including support for custom tools and hooks.

### Custom Tools

Define custom tools as Python functions for Claude Code to use.

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

**Benefits of SDK MCP Servers:**

*   No subprocess management
*   Better performance
*   Simpler deployment
*   Easier debugging
*   Type safety

**Migration from External Servers:**

Easily migrate from external MCP servers to the in-process SDK.

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

**Mixed Server Support:** Use both SDK and external MCP servers together.

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

Use hooks to control the agent's behavior.

```python
from claude_code_sdk import ClaudeCodeOptions, ClaudeSDKClient, HookMatcher

async def check_bash_command(input_data, tool_use_id, context):
    tool_name = input_data["tool_name"]
    tool_input = input_data["tool_input"]
    if tool_name != "Bash":
        return {}
    command = tool_input.get("command", "")
    block_patterns = ["foo.sh"]
    for pattern in block_patterns:
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

## Data Types

The SDK provides essential data types for interacting with Claude Code.

*   `ClaudeCodeOptions`: Configure your interactions
*   Message Types: `AssistantMessage`, `UserMessage`, `SystemMessage`, `ResultMessage`
*   Content Blocks: `TextBlock`, `ToolUseBlock`, `ToolResultBlock`

See [src/claude_code_sdk/types.py](src/claude_code_sdk/types.py) for a complete list.

## Error Handling

Robust error handling is provided to help you manage potential issues.

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

See [src/claude_code_sdk/_errors.py](src/claude_code_sdk/_errors.py) for all error types.

## Available Tools

Refer to the official [Claude Code documentation](https://docs.anthropic.com/en/docs/claude-code/settings#tools-available-to-claude) for the latest list of available tools.

## Examples

*   `examples/quick_start.py`: A comprehensive working example
*   `examples/streaming_mode.py`: Examples involving `ClaudeSDKClient`
*   `examples/streaming_mode_ipython.py`: Interactive examples in IPython

## License

MIT
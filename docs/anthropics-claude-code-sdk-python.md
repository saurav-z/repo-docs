# Claude Code SDK for Python

**Unlock the power of Claude Code with the Python SDK, enabling seamless integration and advanced control for your coding tasks.**  Access the original repository [here](https://github.com/anthropics/claude-code-sdk-python).

## Key Features

*   **Simple Querying:** Easily query Claude Code with a simple `query()` function.
*   **Asynchronous Support:** Built with `asyncio` for non-blocking operations and efficient resource utilization.
*   **Tool Integration:** Integrate Claude Code with tools like Read, Write, and Bash.
*   **Custom Tool Creation:**  Define and integrate custom tools as Python functions to extend Claude Code's capabilities, leveraging in-process SDK MCP servers.
*   **Hooks:** Utilize hooks to intercept and modify Claude Code's behavior at various stages of its operation.
*   **Bidirectional Conversations:** Engage in interactive conversations with Claude Code using `ClaudeSDKClient`.
*   **Detailed Error Handling:** Comprehensive error classes for robust application development.
*   **Comprehensive Examples:**  Get started quickly with example code covering essential use cases, including quickstarts and advanced streaming examples.

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

## Basic Usage: `query()`

The `query()` function is an asynchronous tool for querying Claude Code, returning an `AsyncIterator` of response messages.

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

## `ClaudeSDKClient`

`ClaudeSDKClient` facilitates interactive, bidirectional conversations with Claude Code. Unlike `query()`, `ClaudeSDKClient` supports custom tools and hooks defined as Python functions.

### Custom Tools (as In-Process SDK MCP Servers)

Extend Claude Code's functionality by defining custom tools as Python functions using the `@tool` decorator.

*   **Benefits:**
    *   No subprocess management
    *   Better performance
    *   Simpler deployment
    *   Easier debugging
    *   Type safety
*   **Migration from External Servers**
    *   Simplified server setup.

*   **Mixed Server Support:** Integrate both SDK and external MCP servers:

### Custom Tool Example

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

### Hooks

Implement hooks as Python functions to provide deterministic processing and automated feedback, affecting Claude Code's behaviour. Refer to [Claude Code Hooks Reference](https://docs.anthropic.com/en/docs/claude-code/hooks) for detailed guidance.

### Hooks Example

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

## Types

See [src/claude_code_sdk/types.py](src/claude_code_sdk/types.py) for complete type definitions:

*   `ClaudeCodeOptions` - Configuration options
*   `AssistantMessage`, `UserMessage`, `SystemMessage`, `ResultMessage` - Message types
*   `TextBlock`, `ToolUseBlock`, `ToolResultBlock` - Content blocks

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

See [src/claude_code_sdk/_errors.py](src/claude_code_sdk/_errors.py) for all error types.

## Available Tools

Refer to the [Claude Code documentation](https://docs.anthropic.com/en/docs/claude-code/settings#tools-available-to-claude) for an exhaustive list of available tools.

## Examples

*   See [examples/quick_start.py](examples/quick_start.py) for a complete working example.
*   See [examples/streaming_mode.py](examples/streaming_mode.py) for comprehensive examples involving `ClaudeSDKClient`. Run interactive examples in IPython from [examples/streaming_mode_ipython.py](examples/streaming_mode_ipython.py).

## License

MIT
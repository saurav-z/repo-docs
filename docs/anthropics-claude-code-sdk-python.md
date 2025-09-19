# Claude Code SDK for Python

**Unlock the power of Anthropic's Claude Code directly within your Python applications with this versatile and efficient SDK!** (🔗 [Original Repo](https://github.com/anthropics/claude-code-sdk-python))

This Python SDK provides a streamlined interface for interacting with Claude Code, enabling developers to integrate code generation, debugging, and other advanced features into their workflows.

## Key Features

*   **Simple Installation:** Easily integrate the SDK into your projects using `pip install claude-code-sdk`.
*   **Asynchronous Querying:** Leverage `query()` for asynchronous interaction with Claude Code, receiving streaming responses.
*   **Tool Integration:** Seamlessly integrate custom tools and utilize available tools for enhanced functionality.
*   **Bidirectional Conversations:** Utilize `ClaudeSDKClient` for interactive, stateful conversations, including custom tools and hooks.
*   **Custom Tool Creation (In-Process SDK MCP Servers):**  Define and integrate custom tools as Python functions, eliminating the need for separate processes for improved performance, easier debugging, and simpler deployment.
*   **Hook Functionality:** Implement hook functions to customize behavior and provide deterministic processing and automated feedback.
*   **Comprehensive Error Handling:** Built-in error handling for common issues such as missing dependencies, connection problems, and process failures.

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

## Core Functionality: `query()`

The `query()` function is an asynchronous iterator for querying Claude Code. It streams response messages. (See  [`src/claude_code_sdk/query.py`](src/claude_code_sdk/query.py)).

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

### Tool Usage

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

## Advanced Interaction: `ClaudeSDKClient`

`ClaudeSDKClient` provides bidirectional, interactive conversations, including custom tools and hooks. (See [`src/claude_code_sdk/client.py`](src/claude_code_sdk/client.py))

### Custom Tools (In-Process SDK MCP Servers)

Define custom tools as Python functions for Claude to invoke, eliminating the need for external processes and improving performance. (See [`examples/mcp_calculator.py`](examples/mcp_calculator.py) for a complete example.)

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

*   No subprocess management
*   Better performance
*   Simpler deployment
*   Easier debugging
*   Type safety

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

Integrate both SDK and external MCP servers:

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

Hooks are Python functions that the Claude Code *application* invokes at specific points, providing deterministic processing and automated feedback (see [Claude Code Hooks Reference](https://docs.anthropic.com/en/docs/claude-code/hooks) for details).

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

## Data Types

Refer to [`src/claude_code_sdk/types.py`](src/claude_code_sdk/types.py) for comprehensive type definitions:
*   `ClaudeCodeOptions`
*   `AssistantMessage`, `UserMessage`, `SystemMessage`, `ResultMessage`
*   `TextBlock`, `ToolUseBlock`, `ToolResultBlock`

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

For a complete list of error types, see [`src/claude_code_sdk/_errors.py`](src/claude_code_sdk/_errors.py).

## Available Tools

See the [Claude Code documentation](https://docs.anthropic.com/en/docs/claude-code/settings#tools-available-to-claude) for a complete list of available tools.

## Examples

*   [Quick Start Example](examples/quick_start.py)
*   [Streaming Mode Examples](examples/streaming_mode.py)
*   [Interactive IPython Examples](examples/streaming_mode_ipython.py)

## License

MIT
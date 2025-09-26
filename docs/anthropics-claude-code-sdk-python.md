# Claude Code SDK for Python: Interact with Anthropic's Claude Code AI

**Unlock the power of Claude Code directly within your Python applications with the official Claude Code SDK.** [Explore the original repository](https://github.com/anthropics/claude-code-sdk-python) for detailed information and updates.

## Key Features

*   **Simple Integration:** Easily integrate with Claude Code for AI-powered code generation, editing, and more.
*   **Asynchronous Querying:** Leverage `query()` for efficient, non-blocking interaction with Claude Code.
*   **Custom Tooling:** Build custom tools using the `@tool` decorator and the SDK's in-process MCP server for advanced functionality.
*   **Hooks for Control:** Implement hooks to intercept and control Claude Code's behavior for custom logic and security.
*   **Bidirectional Communication:** Utilize `ClaudeSDKClient` for interactive conversations and more complex scenarios.
*   **Comprehensive Error Handling:** Robust error handling to manage potential issues during interaction.

## Installation

```bash
pip install claude-code-sdk
```

**Prerequisites:**

*   Python 3.10+
*   Node.js
*   Claude Code: `npm install -g @anthropic-ai/claude-code`

## Quick Start

Get started with a simple "Hello, world!" example:

```python
import anyio
from claude_code_sdk import query

async def main():
    async for message in query(prompt="What is 2 + 2?"):
        print(message)

anyio.run(main)
```

## Core Functionality

### `query()` Function

The `query()` function allows you to send prompts to Claude Code and receive responses asynchronously.

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

### Utilizing Tools

Control Claude Code's capabilities by enabling specific tools.

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

Configure Claude Code's working directory for context.

```python
from pathlib import Path

options = ClaudeCodeOptions(
    cwd="/path/to/project"  # or Path("/path/to/project")
)
```

## Advanced Features

### `ClaudeSDKClient`: Bidirectional Conversations and Customization

`ClaudeSDKClient` enables interactive, two-way conversations with Claude Code, opening doors to advanced functionalities like custom tools and hooks.

#### Custom Tools (In-Process MCP Servers)

Extend Claude Code's capabilities with custom tools defined as Python functions. This approach provides enhanced performance, simplicity, and type safety compared to external MCP servers.

##### Creating a Simple Tool

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

##### Benefits of SDK MCP Servers

*   No subprocess management
*   Better performance
*   Simpler deployment
*   Easier debugging
*   Type safety

##### Migration from External Servers

Migrate easily from external MCP servers to SDK-based tools.

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

##### Mixed Server Support

Combine SDK and external MCP servers seamlessly.

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

#### Hooks

Implement hooks to customize and control Claude Code's behavior at specific points in the agent loop.

##### Example

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

Explore the available types within `src/claude_code_sdk/types.py`, including:

*   `ClaudeCodeOptions`
*   `AssistantMessage`, `UserMessage`, `SystemMessage`, `ResultMessage`
*   `TextBlock`, `ToolUseBlock`, `ToolResultBlock`

## Error Handling

Robust error handling is provided to manage potential issues.

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

Refer to the [Claude Code documentation](https://docs.anthropic.com/en/docs/claude-code/settings#tools-available-to-claude) for a comprehensive list of available tools.

## Examples

*   **Quick Start:** [examples/quick\_start.py](examples/quick_start.py) provides a working example.
*   **Streaming Mode:** [examples/streaming\_mode.py](examples/streaming_mode.py) and [examples/streaming\_mode\_ipython.py](examples/streaming_mode_ipython.py) offer interactive examples using `ClaudeSDKClient`.

## License

MIT
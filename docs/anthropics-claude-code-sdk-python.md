# Claude Code SDK for Python: Build Powerful AI-Driven Applications

**Unleash the power of Claude Code within your Python applications with the official Claude Code SDK for Python, enabling advanced features like custom tools and real-time interactions.** ([Original Repository](https://github.com/anthropics/claude-code-sdk-python))

This SDK provides a robust and flexible interface for integrating Claude Code into your projects, enabling developers to build sophisticated AI-powered solutions.

## Key Features:

*   **Easy Installation:** Install the SDK with a simple `pip install` command.
*   **Asynchronous Querying:** Leverage the `query()` function for asynchronous communication with Claude Code, ideal for building responsive applications.
*   **Custom Tools:** Define and integrate custom tools as Python functions, allowing Claude Code to interact with your application and external services.
*   **Interactive Conversations:**  Use `ClaudeSDKClient` for bidirectional, interactive conversations, providing a more dynamic user experience.
*   **Hooks:** Implement hooks to provide deterministic processing and automated feedback, adding control to Claude Code.
*   **Flexible Options:** Configure various options to fine-tune Claude Code's behavior, including system prompts, tool permissions, and more.
*   **Comprehensive Error Handling:** Handle various potential errors gracefully, ensuring application stability and providing informative feedback.
*   **Clear Types:** Explore defined types for message, content, and options.
*   **Rich Examples:** Find a variety of code examples to illustrate the main features and functions.

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

The `query()` function is an async function for querying Claude Code, which returns an `AsyncIterator` of response messages.

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

`ClaudeSDKClient` supports bidirectional, interactive conversations with Claude Code, including custom tools and hooks.

## Custom Tools (as In-Process SDK MCP Servers)

Custom tools are implemented in-process MCP servers, which improves the performance and simplifies deployment.

### Creating a Simple Tool

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

### Benefits Over External MCP Servers

*   No subprocess management
*   Better performance
*   Simpler deployment
*   Easier debugging
*   Type safety

### Migration from External Servers

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

### Mixed Server Support

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

## Hooks

Hooks enable deterministic processing and automated feedback for Claude Code.

### Example

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

## Available Tools

See the [Claude Code documentation](https://docs.anthropic.com/en/docs/claude-code/settings#tools-available-to-claude) for a complete list of available tools.

## Examples

*   [examples/quick_start.py](examples/quick_start.py)
*   [examples/streaming_mode.py](examples/streaming_mode.py)
*   [examples/streaming_mode_ipython.py](examples/streaming_mode_ipython.py)

## License

MIT
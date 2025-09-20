# Claude Code SDK for Python: Build Powerful Applications with AI-Powered Code Generation

Unlock the power of AI-driven code generation with the Claude Code SDK for Python, enabling developers to seamlessly integrate advanced language models into their applications. Explore the [original repository](https://github.com/anthropics/claude-code-sdk-python) for more information and to contribute.

**Key Features:**

*   **Easy Installation:** Get started quickly with a simple `pip install claude-code-sdk`.
*   **Asynchronous Querying:** Utilize the `query()` function for asynchronous interaction with Claude Code, streaming responses in real-time.
*   **Flexible Options:** Customize your Claude Code interactions with `ClaudeCodeOptions`, including system prompts, tool permissions, and working directory settings.
*   **Interactive Conversations with `ClaudeSDKClient`:** Create bidirectional, conversational experiences for more complex scenarios.
*   **In-Process SDK MCP Servers (Custom Tools):** Define Python functions as custom tools for Claude to invoke, running directly within your application for enhanced performance, simpler deployment, and type safety.
*   **Hooks:** Implement custom logic at specific points in the Claude agent loop with hooks to provide deterministic processing and automated feedback.
*   **Comprehensive Error Handling:** Handle potential issues with dedicated exception types for a robust application.
*   **Rich Type Definitions:** Leverage pre-defined types for messages, blocks, and options to streamline development.
*   **Extensive Examples:** Learn by example with detailed code snippets demonstrating core functionality, including quick starts, streaming modes, and custom tool usage.

## Installation

```bash
pip install claude-code-sdk
```

**Prerequisites:**

*   Python 3.10+
*   Node.js
*   Claude Code: `npm install -g @anthropic-ai/claude-code`

## Quick Start

Get started quickly with a simple query using the async `query()` function.

```python
import anyio
from claude_code_sdk import query

async def main():
    async for message in query(prompt="What is 2 + 2?"):
        print(message)

anyio.run(main)
```

## Core Usage: `query()` Function

The `query()` function is an asynchronous function designed for interacting with Claude Code, providing an `AsyncIterator` of response messages. Explore the source code [src/claude_code_sdk/query.py](src/claude_code_sdk/query.py).

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

Enable tool usage to extend Claude's capabilities.

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

Configure a working directory for file operations.

```python
from pathlib import Path

options = ClaudeCodeOptions(
    cwd="/path/to/project"  # or Path("/path/to/project")
)
```

## Advanced: `ClaudeSDKClient` for Interactive Conversations

`ClaudeSDKClient` enables bidirectional, interactive conversations with Claude Code. Explore the implementation [src/claude_code_sdk/client.py](src/claude_code_sdk/client.py). This client supports the creation of custom tools and hooks.

### Custom Tools (SDK MCP Servers)

Enhance Claude's capabilities with custom tools defined as Python functions, implemented using in-process MCP servers for improved performance and simplicity.

**For an end-to-end example, see [MCP Calculator](examples/mcp_calculator.py).**

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

*   **No subprocess management:** Runs in your Python application
*   **Better performance:** No IPC overhead for tool calls
*   **Simpler deployment:** Single Python process
*   **Easier debugging:** All code runs in the same process
*   **Type safety:** Direct Python function calls with type hints

#### Migration from External Servers

Easily migrate existing external MCP servers to SDK MCP servers.

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

Combine SDK and external MCP servers.

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

Implement Python functions that Claude Code invokes at specific points in the agent loop for deterministic processing and automated feedback. Read more in [Claude Code Hooks Reference](https://docs.anthropic.com/en/docs/claude-code/hooks).

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

Explore the available types and definitions within [src/claude_code_sdk/types.py](src/claude_code_sdk/types.py):

*   `ClaudeCodeOptions`
*   `AssistantMessage`, `UserMessage`, `SystemMessage`, `ResultMessage`
*   `TextBlock`, `ToolUseBlock`, `ToolResultBlock`

## Error Handling

Handle potential errors using dedicated exception types for a robust application:

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

See [src/claude_code_sdk/_errors.py](src/claude_code_sdk/_errors.py) for a complete list of error types.

## Available Tools

Refer to the [Claude Code documentation](https://docs.anthropic.com/en/docs/claude-code/settings#tools-available-to-claude) for a comprehensive list of available tools.

## Examples

Gain practical insights through examples:

*   **Quick Start:** [examples/quick_start.py](examples/quick_start.py)
*   **Streaming Mode:** [examples/streaming_mode.py](examples/streaming_mode.py), [examples/streaming_mode_ipython.py](examples/streaming_mode_ipython.py)

## License

MIT
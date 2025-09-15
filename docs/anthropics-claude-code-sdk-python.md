# Claude Code SDK for Python: Build Powerful AI-Powered Applications

**Unlock the power of AI code generation with the Claude Code SDK for Python, enabling seamless integration and interactive conversations with Claude Code.** This SDK provides a user-friendly interface for developers to interact with Claude Code, a powerful AI assistant for code generation, debugging, and more.  Explore the original repository at [https://github.com/anthropics/claude-code-sdk-python](https://github.com/anthropics/claude-code-sdk-python).

## Key Features:

*   **Easy Installation:** Quickly get started with a simple `pip install claude-code-sdk`.
*   **Asynchronous Querying:** Utilize the `query()` function for non-blocking interactions with Claude Code, receiving an `AsyncIterator` of responses.
*   **Customizable Options:** Configure Claude Code behavior with `ClaudeCodeOptions`, including system prompts, max turns, working directory, and tool permissions.
*   **Tool Integration:** Seamlessly integrate with Claude Code's tools like "Read", "Write", and "Bash".
*   **Interactive Conversations:**  Use `ClaudeSDKClient` for bidirectional, stateful interactions, custom tool support, and hook implementation.
*   **In-Process Custom Tools:** Build and deploy tools within your Python application as In-Process SDK MCP servers without subprocess management.
*   **Hooking Mechanism:** Implement custom logic to control Claude Code's behavior at specific points in the agent loop for deterministic processing and automated feedback.
*   **Comprehensive Error Handling:** Robust error handling with specific exception types for common issues like CLI not found, connection errors, and process failures.
*   **Type Definitions:**  Access detailed type definitions for `ClaudeCodeOptions` and message/content block types, promoting code clarity and maintainability.
*   **Rich Example Suite:** Explore comprehensive examples for quick starts, streaming modes, and interactive IPython sessions.

## Installation

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

The `query()` function is an asynchronous function designed for straightforward interactions with Claude Code. It returns an `AsyncIterator` of response messages.

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

`ClaudeSDKClient` enables bidirectional, interactive conversations with Claude Code, expanding upon the capabilities of `query()`. It empowers users to harness **custom tools** and **hooks**, both defined as Python functions, to customize and extend Claude Code's functionality.

### Custom Tools (as In-Process SDK MCP Servers)

Custom tools can be defined as Python functions that Claude invokes as needed.
These tools run as in-process MCP servers eliminating the need for separate processes. For an end-to-end example, see [MCP Calculator](examples/mcp_calculator.py).

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

*   **No subprocess management**
*   **Better performance**
*   **Simpler deployment**
*   **Easier debugging**
*   **Type safety**

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

You can use both SDK and external MCP servers together:

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

Hooks are Python functions invoked at strategic points in the Claude Code agent loop, enabling deterministic processing and feedback.  See [Claude Code Hooks Reference](https://docs.anthropic.com/en/docs/claude-code/hooks) for detailed information.

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

async with ClaudeSDKClient(options=options):
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

Complete type definitions are available in `src/claude_code_sdk/types.py`:

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

All error types are listed in `src/claude_code_sdk/_errors.py`.

## Available Tools

Consult the [Claude Code documentation](https://docs.anthropic.com/en/docs/claude-code/settings#tools-available-to-claude) for a comprehensive list of available tools.

## Examples

Explore the provided examples to quickly understand and implement the SDK:

*   [Quick Start Example](examples/quick_start.py)
*   [Streaming Mode Examples](examples/streaming_mode.py)
*   [Interactive IPython Examples](examples/streaming_mode_ipython.py)

## License

MIT
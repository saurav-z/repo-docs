# Claude Code SDK for Python

**Unlock the power of Claude Code with the Python SDK, enabling seamless integration and advanced customization for your projects.**  Access the original repository [here](https://github.com/anthropics/claude-code-sdk-python).

This SDK provides a Pythonic interface for interacting with Claude Code, enabling you to build sophisticated applications that leverage its code generation and understanding capabilities.

## Key Features

*   **Easy Installation:** Simple `pip install claude-code-sdk` for quick setup.
*   **Asynchronous Querying:**  Utilize the `query()` function for non-blocking interactions and efficient processing.
*   **Flexible Configuration:** Customize behavior with `ClaudeCodeOptions`, including system prompts and tool permissions.
*   **Tool Integration:** Integrate external tools like "Read," "Write," and "Bash" for expanded functionality.
*   **Bidirectional Conversations:**  Engage in interactive sessions with `ClaudeSDKClient` for dynamic code generation and refinement.
*   **Custom Tools (In-Process SDK MCP Servers):**
    *   Define Python functions as custom tools for Claude Code to invoke.
    *   Run tools directly within your Python application for improved performance and ease of deployment.
    *   Simplified server setup compared to external MCP servers.
    *   Type safety with Python function calls.
    *   Easily migrate from external MCP servers.
*   **Hooks:** Intercept and modify Claude Code's behavior at specific points in the process, adding deterministic processing and automated feedback.
*   **Comprehensive Error Handling:** Robust error handling for common issues, ensuring stable application operation.
*   **Detailed Documentation & Examples:** Comprehensive documentation and working examples to guide you through the SDK's features and capabilities.

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

## Core Concepts

*   **`query()`:** An asynchronous function to query Claude Code, returning an `AsyncIterator` of response messages.
*   **`ClaudeSDKClient`:** Enables bidirectional, interactive conversations with Claude Code, and supports custom tools and hooks.
*   **`ClaudeCodeOptions`:** Configure settings like system prompts, allowed tools, and working directory.

## Using Tools

Enable Claude Code to interact with the outside world using tools.

```python
from claude_code_sdk import ClaudeCodeOptions

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

## Custom Tools

Use the SDK to create and integrate custom tools.

```python
from claude_code_sdk import tool, create_sdk_mcp_server, ClaudeCodeOptions, ClaudeSDKClient

@tool("greet", "Greet a user", {"name": str})
async def greet_user(args):
    return {
        "content": [
            {"type": "text", "text": f"Hello, {args['name']}!"}
        ]
    }

server = create_sdk_mcp_server(
    name="my-tools",
    version="1.0.0",
    tools=[greet_user]
)

options = ClaudeCodeOptions(
    mcp_servers={"tools": server},
    allowed_tools=["mcp__tools__greet"]
)

async with ClaudeSDKClient(options=options) as client:
    await client.query("Greet Alice")

    async for msg in client.receive_response():
        print(msg)
```

## Hooks

Integrate hooks to customize the behavior of Claude Code.

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
    await client.query("Run the bash command: ./foo.sh --help")
    async for msg in client.receive_response():
        print(msg)
```

## Error Handling

Implement error handling for a robust application.

```python
from claude_code_sdk import (
    ClaudeSDKError,
    CLINotFoundError,
    CLIConnectionError,
    ProcessError,
    CLIJSONDecodeError,
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

## Examples

Explore comprehensive examples to understand and use the SDK.

*   [Quick Start](examples/quick_start.py)
*   [Streaming Mode](examples/streaming_mode.py)
*   [Streaming Mode in IPython](examples/streaming_mode_ipython.py)
*   [MCP Calculator](examples/mcp_calculator.py)

## Types

See the available types to build complex projects.

*   `ClaudeCodeOptions` - Configuration options
*   `AssistantMessage`, `UserMessage`, `SystemMessage`, `ResultMessage` - Message types
*   `TextBlock`, `ToolUseBlock`, `ToolResultBlock` - Content blocks

## License

MIT
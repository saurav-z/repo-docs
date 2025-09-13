# Claude Code SDK for Python: Build Powerful AI-Powered Code Assistants

**Unlock the power of AI-driven coding with the Claude Code SDK for Python, enabling seamless integration and interactive conversations with Claude Code, a cutting-edge code-focused AI assistant.** For detailed information, explore the [Claude Code SDK documentation](https://docs.anthropic.com/en/docs/claude-code/sdk/sdk-python) and the [original repository](https://github.com/anthropics/claude-code-sdk-python).

**Key Features:**

*   **Easy Installation:** Get started quickly with a simple `pip install claude-code-sdk`.
*   **Asynchronous Querying:** Leverage `query()` for asynchronous communication with Claude Code, receiving streaming responses.
*   **Flexible Configuration:** Utilize `ClaudeCodeOptions` to customize behavior with system prompts, tool permissions, and working directory settings.
*   **Interactive Conversations:** Engage in bidirectional conversations using `ClaudeSDKClient`, enabling custom tool usage and hooks.
*   **Custom Tools:** Create and integrate your own tools with In-Process SDK MCP servers for enhanced functionality.
*   **Hook Integration:** Implement hooks to control the behavior of Claude Code by adding deterministic processing or automated feedback.
*   **Comprehensive Error Handling:** Robust error handling with custom exception types for debugging and troubleshooting.
*   **Clear Type Definitions:** Access detailed type definitions for message structures and configuration options.

**Getting Started**

*   **Prerequisites:** Ensure you have Python 3.10+ and Node.js installed. Also, install Claude Code with `npm install -g @anthropic-ai/claude-code`.

*   **Installation:**

    ```bash
    pip install claude-code-sdk
    ```

*   **Quickstart:**

    ```python
    import anyio
    from claude_code_sdk import query

    async def main():
        async for message in query(prompt="What is 2 + 2?"):
            print(message)

    anyio.run(main)
    ```

**Core Functionality**

*   **`query()` Function:** Asynchronously sends prompts and receives streamed responses from Claude Code.

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

*   **Using Tools:** Control Claude Code's tool access.

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

*   **Working Directory:** Set the working directory for file operations.

    ```python
    from pathlib import Path

    options = ClaudeCodeOptions(
        cwd="/path/to/project"  # or Path("/path/to/project")
    )
    ```

**Advanced Usage: `ClaudeSDKClient`**

`ClaudeSDKClient` offers bidirectional communication and supports custom tools and hooks.

*   **Custom Tools (In-Process SDK MCP Servers):** Enhance Claude Code by creating custom Python functions.  In-process MCP servers offer improved performance and simplified deployment.

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

    *   **Benefits Over External MCP Servers:**
        *   No subprocess management
        *   Better performance
        *   Simpler deployment
        *   Easier debugging
        *   Type safety

    *   **Migration from External Servers:** Easy migration from external MCP servers to in-process SDK servers.

    *   **Mixed Server Support:**  Combine SDK and external MCP servers.

*   **Hooks:** Implement hooks to modify Claude Code's behavior at specific points during execution, enabling deterministic processing and automated feedback.

    ```python
    from claude_code_sdk import ClaudeCodeOptions, ClaudeSDKClient, HookMatcher

    async def check_bash_command(input_data, tool_use_id, context):
        # Example hook implementation
        pass

    options = ClaudeCodeOptions(
        allowed_tools=["Bash"],
        hooks={
            "PreToolUse": [
                HookMatcher(matcher="Bash", hooks=[check_bash_command]),
            ],
        }
    )

    async with ClaudeSDKClient(options=options) as client:
        # Example hook usage
        pass
    ```

**Types**

Access type definitions within `src/claude_code_sdk/types.py`:

*   `ClaudeCodeOptions`
*   `AssistantMessage`, `UserMessage`, `SystemMessage`, `ResultMessage`
*   `TextBlock`, `ToolUseBlock`, `ToolResultBlock`

**Error Handling**

Robust error handling mechanisms:

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

**Available Tools:** Explore the complete list of available tools in the [Claude Code documentation](https://docs.anthropic.com/en/docs/claude-code/settings#tools-available-to-claude).

**Examples**

*   See [examples/quick_start.py](examples/quick_start.py) for a comprehensive example.
*   Explore advanced usage, including `ClaudeSDKClient`, in [examples/streaming_mode.py](examples/streaming_mode.py)
*   Interactive examples in IPython are available at [examples/streaming_mode_ipython.py](examples/streaming_mode_ipython.py).

**License**

MIT
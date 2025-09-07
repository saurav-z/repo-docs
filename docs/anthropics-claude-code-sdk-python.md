# Claude Code SDK for Python: Build Powerful AI-Powered Code Interactions

**Harness the power of Claude Code directly within your Python applications with the official Claude Code SDK for Python!**  This SDK provides seamless integration, allowing you to easily interact with Claude Code's advanced code generation and manipulation capabilities.  Explore the official [Claude Code SDK documentation](https://docs.anthropic.com/en/docs/claude-code/sdk) for in-depth information and examples. [View the original repository on GitHub](https://github.com/anthropics/claude-code-sdk-python).

## Key Features

*   **Simplified Integration:** Easily integrate Claude Code into your Python projects.
*   **Asynchronous Queries:** Utilize `async` and `await` for efficient, non-blocking interactions.
*   **Customizable Options:** Control Claude Code's behavior with options like system prompts and tool usage.
*   **Tool Support:**  Leverage Claude Code's powerful tools for file manipulation, code execution, and more.
*   **Comprehensive Error Handling:**  Robust error handling for a stable and reliable experience.

## Installation

Get started quickly by installing the SDK using `pip`:

```bash
pip install claude-code-sdk
```

**Prerequisites:**

*   Python 3.10+
*   Node.js
*   Claude Code CLI:  `npm install -g @anthropic-ai/claude-code`

## Quick Start Guide

Here's a simple example to get you started:

```python
import anyio
from claude_code_sdk import query

async def main():
    async for message in query(prompt="What is 2 + 2?"):
        print(message)

anyio.run(main)
```

## Usage Examples

### Basic Queries

Send simple prompts and receive Claude Code's responses:

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

### Working with Tools

Enable and utilize Claude Code's tools to automate tasks:

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

### Setting the Working Directory

Specify the working directory for Claude Code:

```python
from pathlib import Path

options = ClaudeCodeOptions(
    cwd="/path/to/project"  # or Path("/path/to/project")
)
```

## API Reference

### `query(prompt, options=None)`

The core asynchronous function for interacting with Claude Code.

**Parameters:**

*   `prompt` (str):  The prompt you provide to Claude Code.
*   `options` (ClaudeCodeOptions, optional):  Configuration settings for your query.

**Returns:**  `AsyncIterator[Message]` - A stream of messages from Claude Code.

### Types

Detailed type definitions are available in `src/claude_code_sdk/types.py`:

*   `ClaudeCodeOptions`
*   `AssistantMessage`, `UserMessage`, `SystemMessage`, `ResultMessage`
*   `TextBlock`, `ToolUseBlock`, `ToolResultBlock`

## Error Handling

Implement robust error handling to manage potential issues:

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

Explore a complete working example in  `examples/quick_start.py`.

## License

MIT
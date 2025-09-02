# Claude Code SDK for Python: Interact with Claude Code Easily

**Unlock the power of Claude Code within your Python applications with this intuitive and efficient SDK.** You can find the original repository [here](https://github.com/anthropics/claude-code-sdk-python).

## Key Features

*   **Simple Integration:** Easily integrate with the Claude Code API within your Python projects.
*   **Asynchronous Operations:** Leverages `asyncio` for non-blocking, efficient interactions.
*   **Flexible Configuration:** Customize your interactions with `ClaudeCodeOptions` for prompts, tool usage, and more.
*   **Comprehensive Error Handling:** Robust error handling to manage potential issues during communication.
*   **Type-Safe Design:** Built with type hints for improved code readability and maintainability.

## Installation

Get started quickly by installing the SDK using pip:

```bash
pip install claude-code-sdk
```

**Prerequisites:**

*   Python 3.10+
*   Node.js
*   Claude Code: Install the Claude Code CLI globally: `npm install -g @anthropic-ai/claude-code`

## Quick Start

Here's how to make your first query:

```python
import anyio
from claude_code_sdk import query

async def main():
    async for message in query(prompt="What is 2 + 2?"):
        print(message)

anyio.run(main)
```

## Usage

### Basic Query

Send simple prompts to Claude Code:

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

Interact with tools like "Read", "Write", and "Bash":

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

Set a specific working directory for your Claude Code interactions:

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

*   `prompt` (str): The text prompt to send to Claude Code.
*   `options` (ClaudeCodeOptions): Optional configuration settings.

**Returns:** `AsyncIterator[Message]` - A stream of response messages.

### Types

The SDK defines several types to facilitate structured interactions. Explore these in the source code:

*   `ClaudeCodeOptions`: Configuration options.
*   `AssistantMessage`, `UserMessage`, `SystemMessage`, `ResultMessage`: Message types.
*   `TextBlock`, `ToolUseBlock`, `ToolResultBlock`: Content blocks.

*   **Source code**: [src/claude_code_sdk/types.py](src/claude_code_sdk/types.py)

## Error Handling

Handle potential errors gracefully:

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

*   **Error types**: [src/claude_code_sdk/\_errors.py](src/claude_code_sdk/_errors.py)

## Available Tools

Refer to the [Claude Code documentation](https://docs.anthropic.com/en/docs/claude-code/settings#tools-available-to-claude) for a comprehensive list of supported tools.

## Examples

See a working example for a complete demo:

*   **Example Code**: [examples/quick\_start.py](examples/quick_start.py)

## License

MIT
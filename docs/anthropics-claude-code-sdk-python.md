# Claude Code SDK for Python: Build Powerful Applications with Anthropic's AI

**Enhance your Python projects with the power of Anthropic's Claude Code using the official Python SDK.** This SDK provides a streamlined interface for interacting with the Claude Code model, enabling you to build intelligent and responsive applications. [See the original repository on GitHub](https://github.com/anthropics/claude-code-sdk-python).

## Key Features

*   **Simplified Integration:** Easily integrate Claude Code into your Python applications.
*   **Asynchronous Querying:** Leverage `async` and `await` for non-blocking interactions and improved performance.
*   **Flexible Configuration:** Customize Claude Code's behavior with options like system prompts, allowed tools, and working directory.
*   **Tool Support:** Utilize Claude Code's powerful tools for tasks like reading, writing, and executing bash commands.
*   **Comprehensive Error Handling:** Robust error handling to manage potential issues during interaction.
*   **Streaming Responses:** Receive responses in real-time for a dynamic user experience.

## Installation

Install the SDK using pip:

```bash
pip install claude-code-sdk
```

**Prerequisites:**

*   Python 3.10+
*   Node.js
*   Claude Code: Install globally using `npm install -g @anthropic-ai/claude-code`

## Quick Start

Get started with a basic query:

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

## API Reference

### `query(prompt, options=None)`

The primary asynchronous function for interacting with Claude Code.

**Parameters:**

*   `prompt` (str): The user's prompt or question for Claude.
*   `options` (ClaudeCodeOptions, optional): Configuration settings to customize the interaction.

**Returns:**

`AsyncIterator[Message]`: An asynchronous iterator that streams the response messages from Claude Code.

### Types

*   `ClaudeCodeOptions`: Configuration options.
*   `AssistantMessage`, `UserMessage`, `SystemMessage`, `ResultMessage`: Message types for different interactions.
*   `TextBlock`, `ToolUseBlock`, `ToolResultBlock`: Content block types within messages.

For complete type definitions, refer to [src/claude_code_sdk/types.py](src/claude_code_sdk/types.py).

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

For a full list of error types, see [src/claude_code_sdk/\_errors.py](src/claude_code_sdk/_errors.py).

## Available Tools

Explore the available tools to extend the capabilities of Claude Code.  See the [Claude Code documentation](https://docs.anthropic.com/en/docs/claude-code/settings#tools-available-to-claude) for a complete list.

## Examples

For a complete, working example, see [examples/quick\_start.py](examples/quick_start.py).

## License

MIT
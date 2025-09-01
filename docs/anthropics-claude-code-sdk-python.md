# Unlock the Power of Claude Code with the Python SDK

**Effortlessly interact with Claude Code, Anthropic's powerful AI coding assistant, using the official Python SDK.**  This library simplifies the integration of Claude Code into your Python projects, enabling you to build innovative applications with AI-powered code generation and manipulation.  Find the original repository [here](https://github.com/anthropics/claude-code-sdk-python).

## Key Features:

*   **Seamless Integration:** Pythonic SDK for interacting with Claude Code.
*   **Asynchronous Querying:**  Leverages `asyncio` for non-blocking communication.
*   **Flexible Options:**  Configure Claude Code behavior with various options, including system prompts, tool usage, and working directory.
*   **Comprehensive Error Handling:**  Robust error handling for common issues like CLI installation and process failures.
*   **Streamlined Tool Usage:**  Easily integrate and manage tools within your Claude Code interactions.
*   **Type Safety:**  Well-defined types for messages and content blocks.

## Installation

Get started quickly by installing the SDK using `pip`:

```bash
pip install claude-code-sdk
```

**Prerequisites:**

*   Python 3.10+
*   Node.js
*   Claude Code: `npm install -g @anthropic-ai/claude-code`

## Quick Start

Here's how to get up and running with a simple query:

```python
import anyio
from claude_code_sdk import query

async def main():
    async for message in query(prompt="What is 2 + 2?"):
        print(message)

anyio.run(main)
```

## Usage Guide

### Basic Query

Interact with Claude Code using simple prompts and process the responses.

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

Leverage Claude Code's tool capabilities for more advanced tasks, like file manipulation and command execution.

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

Set a specific working directory for your Claude Code interactions.

```python
from pathlib import Path

options = ClaudeCodeOptions(
    cwd="/path/to/project"  # or Path("/path/to/project")
)
```

## API Reference

### `query(prompt, options=None)`

The primary function for querying Claude Code.

**Parameters:**

*   `prompt` (str): Your query to send to Claude Code.
*   `options` (ClaudeCodeOptions):  Optional configuration settings.

**Returns:** `AsyncIterator[Message]` - An asynchronous stream of response messages.

### Types

Explore the available data types for a deeper understanding:

*   `ClaudeCodeOptions` - Configuration settings.
*   `AssistantMessage`, `UserMessage`, `SystemMessage`, `ResultMessage` - Message types.
*   `TextBlock`, `ToolUseBlock`, `ToolResultBlock` - Content blocks.

Refer to [src/claude_code_sdk/types.py](src/claude_code_sdk/types.py) for comprehensive type definitions.

## Error Handling

Implement robust error handling to gracefully manage potential issues.

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

For a complete list of error types, see [src/claude_code_sdk/_errors.py](src/claude_code_sdk/_errors.py).

## Available Tools

Learn more about the available tools to extend Claude Code's capabilities:

*   See the [Claude Code documentation](https://docs.anthropic.com/en/docs/claude-code/settings#tools-available-to-claude) for a complete list.

## Examples

Find complete working examples to accelerate your development:

*   See [examples/quick_start.py](examples/quick_start.py).

## License

MIT
# Claude Code Python SDK: Build Powerful AI-Powered Code Generation & Analysis Applications

**Unlock the potential of AI-driven coding with the Claude Code Python SDK!** This SDK allows you to seamlessly integrate the power of Claude Code into your Python applications, enabling code generation, analysis, and more. Access the original repository [here](https://github.com/anthropics/claude-code-sdk-python).

## Key Features

*   **Easy Integration:** Simple installation and a straightforward `query()` function makes it easy to get started.
*   **Asynchronous Streaming:** Receive real-time responses with asynchronous message streaming.
*   **Flexible Configuration:** Customize Claude Code's behavior with `ClaudeCodeOptions`, including system prompts, maximum turns, and more.
*   **Tool Support:** Leverage Claude Code's powerful tools for reading, writing, and executing code (Bash).
*   **Error Handling:** Robust error handling to manage potential issues during interaction with Claude Code.
*   **Comprehensive Documentation:** Full type definitions and examples to help you build quickly.

## Installation

Get started quickly by installing the SDK via pip:

```bash
pip install claude-code-sdk
```

**Prerequisites:**

*   Python 3.10+
*   Node.js
*   Claude Code CLI: `npm install -g @anthropic-ai/claude-code`

## Quick Start

Here's how to start a basic interaction with Claude Code:

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

Initiate interactions with Claude Code:

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

Enable Claude Code to use tools for more advanced functionality:

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

Specify a working directory for file operations:

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

*   `prompt` (str): The input prompt for Claude Code.
*   `options` (ClaudeCodeOptions, optional): Configuration settings. Defaults to `None`.

**Returns:** `AsyncIterator[Message]` - A stream of response messages.

### Types

For detailed type definitions: See [src/claude\_code\_sdk/types.py](src/claude_code_sdk/types.py)

*   `ClaudeCodeOptions` - Configuration settings
*   `AssistantMessage`, `UserMessage`, `SystemMessage`, `ResultMessage` - Message types
*   `TextBlock`, `ToolUseBlock`, `ToolResultBlock` - Content blocks

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

For a complete list of error types, see [src/claude\_code\_sdk/\_errors.py](src/claude_code_sdk/_errors.py).

## Available Tools

Explore the range of tools available to Claude Code to extend its capabilities.  See the [Claude Code documentation](https://docs.anthropic.com/en/docs/claude-code/settings#tools-available-to-claude) for details.

## Examples

For a complete working example, refer to the [examples/quick\_start.py](examples/quick_start.py) file.

## License

MIT
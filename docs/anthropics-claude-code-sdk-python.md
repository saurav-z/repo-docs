# Claude Code SDK for Python: Build Powerful AI-Powered Applications

**Unleash the power of Anthropic's Claude Code directly within your Python applications with the official Claude Code SDK.**  For more in-depth information, please consult the [Claude Code SDK documentation](https://docs.anthropic.com/en/docs/claude-code/sdk).

[View the original repository on GitHub](https://github.com/anthropics/claude-code-sdk-python)

## Key Features of the Claude Code SDK for Python

*   **Seamless Integration:** Easily incorporate Claude Code's capabilities into your Python projects.
*   **Asynchronous Operations:** Leverage asynchronous functions for efficient and non-blocking interactions.
*   **Customizable Prompts:** Craft detailed prompts to guide Claude Code's responses.
*   **Tool Support:** Utilize tools like "Read", "Write", and "Bash" to extend Claude Code's functionality.
*   **Flexible Configuration:** Configure options such as system prompts and maximum turns.
*   **Comprehensive Error Handling:** Robust error handling to gracefully manage potential issues.
*   **Clear API Reference:** Well-documented API for easy understanding and usage.

## Installation

Get started quickly with the following command:

```bash
pip install claude-code-sdk
```

**Prerequisites:**

*   Python 3.10+
*   Node.js
*   Claude Code: `npm install -g @anthropic-ai/claude-code`

## Quick Start Guide

Here's a simple example to get you up and running:

```python
import anyio
from claude_code_sdk import query

async def main():
    async for message in query(prompt="What is 2 + 2?"):
        print(message)

anyio.run(main)
```

## Usage Examples

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

This core async function enables querying Claude Code.

**Parameters:**

*   `prompt` (str): The prompt text to send to Claude Code.
*   `options` (ClaudeCodeOptions): Optional configuration settings for the query.

**Returns:** AsyncIterator[Message] -  A stream of response messages from Claude Code.

### Types

Explore the available types for enhanced control and customization:

*   `ClaudeCodeOptions` - For configuring query behavior.
*   `AssistantMessage`, `UserMessage`, `SystemMessage`, `ResultMessage` - Various message types for structuring interactions.
*   `TextBlock`, `ToolUseBlock`, `ToolResultBlock` - Content blocks to handle responses.

For detailed type definitions, see [src/claude_code_sdk/types.py](src/claude_code_sdk/types.py).

## Error Handling

Implement robust error management to gracefully handle potential issues:

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

Refer to [src/claude_code_sdk/_errors.py](src/claude_code_sdk/_errors.py) for a comprehensive list of error types.

## Available Tools

Extend Claude Code's functionality with various tools.  See the [Claude Code documentation](https://docs.anthropic.com/en/docs/claude-code/settings#tools-available-to-claude) for a complete list of available tools.

## Examples

Find a complete working example in [examples/quick\_start.py](examples/quick_start.py) to see the SDK in action.

## License

This project is licensed under the MIT License.
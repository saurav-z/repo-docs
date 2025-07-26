# Supercharge Your Claude Code with SuperClaude: A Powerful Framework for Development 🚀

SuperClaude is a framework designed to enhance your experience with Claude Code, providing specialized commands, smart personas, and server integration to streamline your development workflow.  **Explore the [SuperClaude GitHub Repository](https://github.com/SuperClaude-Org/SuperClaude_Framework) to supercharge your AI coding assistant!**

## Key Features

*   **Specialized Commands:** 16 commands for common development tasks like `/sc:implement`, `/sc:build`, and `/sc:test`.
*   **Smart Personas:** AI specialists (architect, frontend, backend, etc.) that intelligently assist with different coding domains.
*   **MCP Server Integration:** Integration with external tools like Context7, Sequential, Magic, and Playwright for enhanced capabilities.
*   **Task Management:** Helps track progress within your development flow.
*   **Token Optimization:** Enhances the handling of longer conversations.

## Core Functionality

SuperClaude enhances Claude Code by providing:

*   **Framework Files:** Documentation to guide Claude's responses.
*   **Slash Commands:** 16 specialized commands for various development tasks.
*   **MCP Servers:** External services that provide extra capabilities.
*   **Smart Routing:** Intelligent selection of tools and experts based on your current task.

## Installation

SuperClaude installation is a two-step process:

1.  Install the Python package.
2.  Run the SuperClaude installer to set up Claude Code integration.

### Step 1: Package Installation

**Option A: From PyPI (Recommended)**

```bash
uv add SuperClaude
```

**Option B: From Source**

```bash
git clone https://github.com/SuperClaude-Org/SuperClaude_Framework.git
cd SuperClaude
uv sync
```

### UV / UVX Setup Guide

SuperClaude v3 also supports installation via [`uv`](https://github.com/astral-sh/uv) (a faster, modern Python package manager) or `uvx` for cross-platform usage.

### 🌀 Install with `uv`

Make sure `uv` is installed:

```bash
curl -Ls https://astral.sh/uv/install.sh | sh
```

> Or follow instructions from: [https://github.com/astral-sh/uv](https://github.com/astral-sh/uv)

Once `uv` is available, you can install SuperClaude like this:

```bash
uv venv
source .venv/bin/activate
uv pip install SuperClaude
```

### ⚡ Install with `uvx` (Cross-platform CLI)

If you’re using `uvx`, just run:

```bash
uvx pip install SuperClaude
```

### Step 2: Run the Installer

Use the following commands after installing the package.

```bash
# Quick setup (recommended for most users)
SuperClaude install

# Interactive selection (choose components)
SuperClaude install --interactive

# Minimal install (just core framework)
SuperClaude install --minimal

# Developer setup (everything included)
SuperClaude install --profile developer

# See all available options
SuperClaude install --help
```

**or using python modular usage:**
```bash
# Quick setup (recommended for most users)
python3 -m SuperClaude install

# Interactive selection (choose components)
python3 -m SuperClaude install --interactive

# Minimal install (just core framework)
python3 -m SuperClaude install --minimal

# Developer setup (everything included)
python3 -m SuperClaude install --profile developer

# See all available options
python3 -m SuperClaude install --help
```
**or python direct CLI call**
```bash
# Quick setup (recommended for most users)
python3 SuperClaude install

# Interactive selection (choose components)
python3 SuperClaude install --interactive

# Minimal install (just core framework)
python3 SuperClaude install --minimal

# Developer setup (everything included)
python3 SuperClaude install --profile developer

# See all available options
python3 SuperClaude install --help
```

## Upgrading from v2

If you're upgrading from SuperClaude v2, please ensure you uninstall the previous version and remove the following directories/files to avoid conflicts:

*   `SuperClaude/`
*   `~/.claude/shared/`
*   `~/.claude/commands/`
*   `~/.claude/CLAUDE.md`

**Important Change**: The `/build` command behavior has changed.  In v3:
*   `/sc:build` = compilation/packaging only
*   `/sc:implement` = feature implementation (NEW!)

## Command Reference

*   `/sc:implement`, `/sc:build`, `/sc:design`
*   `/sc:analyze`, `/sc:troubleshoot`, `/sc:explain`
*   `/sc:improve`, `/sc:test`, `/sc:cleanup`
*   `/sc:document`, `/sc:git`, `/sc:estimate`, `/sc:task`, `/sc:index`, `/sc:load`, `/sc:spawn`

## Personas

*   **architect:** Systems design and architecture.
*   **frontend:** UI/UX and accessibility.
*   **backend:** APIs and infrastructure.
*   **analyzer:** Debugging and problem-solving.
*   **security:** Security concerns and vulnerabilities.
*   **scribe:** Documentation and writing.

## Documentation

*   [User Guide](https://github.com/SuperClaude-Org/SuperClaude_Framework/blob/main/Docs/superclaude-user-guide.md)
*   [Commands Guide](https://github.com/SuperClaude-Org/SuperClaude_Framework/blob/main/Docs/commands-guide.md)
*   [Flags Guide](https://github.com/SuperClaude-Org/SuperClaude_Framework/blob/main/Docs/flags-guide.md)
*   [Personas Guide](https://github.com/SuperClaude-Org/SuperClaude_Framework/blob/main/Docs/personas-guide.md)
*   [Installation Guide](https://github.com/SuperClaude-Org/SuperClaude_Framework/blob/main/Docs/installation-guide.md)

## Contributing

We welcome contributions!  Please refer to the [CONTRIBUTING.md](https://github.com/SuperClaude-Org/SuperClaude_Framework/blob/main/CONTRIBUTING.md) file for guidelines.

## Project Structure

```
SuperClaude/
├── setup.py               # pypi setup file
├── SuperClaude/           # Framework files
│   ├── Core/              # Behavior documentation (COMMANDS.md, FLAGS.md, etc.)
│   ├── Commands/          # 16 slash command definitions
│   └── Settings/          # Configuration files
├── setup/                 # Installation system
└── profiles/              # Installation profiles (quick, minimal, developer)
```

## License

MIT - [See LICENSE file for details](https://opensource.org/licenses/MIT)
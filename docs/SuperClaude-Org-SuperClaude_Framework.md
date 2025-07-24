# SuperClaude: Supercharge Your Claude Code Experience!

Supercharge your development workflow with **SuperClaude**, a framework that extends Claude Code with specialized commands, smart personas, and powerful integrations.  [Check out the original repo](https://github.com/SuperClaude-Org/SuperClaude_Framework) to learn more!

## Key Features

*   **16 Specialized Commands:** Streamline common development tasks with commands like `/sc:implement`, `/sc:build`, and `/sc:test`.
*   **Smart Personas:** Leverage AI specialists (architect, frontend, backend, etc.) to guide Claude's responses for different domains.
*   **MCP Server Integration:** Connect to external tools like Context7, Sequential, Magic, and Playwright for enhanced capabilities.
*   **Task Management:** Keep track of progress with integrated task management.
*   **Token Optimization:**  Helps manage long conversations.

## Getting Started

SuperClaude is designed to make Claude Code more helpful for development work, but it's still under active development.

### Installation

Follow these steps to install SuperClaude:

**Step 1: Install the Package**

**Option A: From PyPI (Recommended)**

```bash
uv add SuperClaude
```

**Option B: From Source**

```bash
git clone https://github.com/NomenAK/SuperClaude.git
cd SuperClaude
uv sync
```

### 🔧 UV / UVX Setup Guide

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

**Step 2: Run the Installer**

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

## Key Commands

*   `/sc:implement`: Feature implementation
*   `/sc:build`: Compilation/packaging
*   `/sc:design`: Design tasks
*   `/sc:analyze`: Code analysis
*   `/sc:troubleshoot`: Troubleshoot code
*   `/sc:explain`: Code explanation
*   `/sc:improve`: Code improvement
*   `/sc:test`: Testing and QA
*   `/sc:cleanup`: Code cleanup
*   `/sc:document`: Generate documentation
*   `/sc:git`: Git integration
*   `/sc:estimate`: Project estimation
*   `/sc:task`: Task management
*   `/sc:index`: Indexing
*   `/sc:load`: Load functionality
*   `/sc:spawn`: Create new instances

## Important Information for v2 Users

If upgrading from v2, follow these steps to avoid conflicts:

1.  Uninstall v2.
2.  Manually delete: `SuperClaude/`, `~/.claude/shared/`, `~/.claude/commands/`, `~/.claude/CLAUDE.md`.
3.  Use `/sc:implement` instead of `/build` for feature implementation.

## Documentation

*   [User Guide](https://github.com/NomenAK/SuperClaude/blob/master/Docs/superclaude-user-guide.md)
*   [Commands Guide](https://github.com/NomenAK/SuperClaude/blob/master/Docs/commands-guide.md)
*   [Flags Guide](https://github.com/NomenAK/SuperClaude/blob/master/Docs/flags-guide.md)
*   [Personas Guide](https://github.com/NomenAK/SuperClaude/blob/master/Docs/personas-guide.md)
*   [Installation Guide](https://github.com/NomenAK/SuperClaude/blob/master/Docs/installation-guide.md)

## Contributing

We welcome contributions! Get involved by:

*   Reporting bugs
*   Improving documentation
*   Adding tests
*   Suggesting new features

## License

MIT - See the [LICENSE](https://opensource.org/licenses/MIT) file.
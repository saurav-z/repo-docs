# Supercharge Your Development Workflow with SuperClaude v3!

SuperClaude is a framework that enhances Claude Code with specialized commands, smart personas, and external tool integrations, making your development tasks smoother and more efficient. [**Check it out on GitHub!**](https://github.com/SuperClaude-Org/SuperClaude_Framework)

[![Website Preview](https://img.shields.io/badge/Visit-Website-blue?logo=google-chrome)](https://superclaude-org.github.io/SuperClaude_Website/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://img.shields.io/pypi/v/SuperClaude.svg)](https://pypi.org/project/SuperClaude/)
[![Version](https://img.shields.io/badge/version-3.0.0-blue.svg)](https://github.com/NomenAK/SuperClaude)
[![GitHub issues](https://img.shields.io/github/issues/NomenAK/SuperClaude)](https://github.com/NomenAK/SuperClaude/issues)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/NomenAK/SuperClaude/blob/master/CONTRIBUTING.md)
[![Contributors](https://img.shields.io/github/contributors/NomenAK/SuperClaude)](https://github.com/NomenAK/SuperClaude/graphs/contributors)
[![Website](https://img.shields.io/website?url=https://superclaude-org.github.io/SuperClaude_Website/)](https://superclaude-org.github.io/SuperClaude_Website/)

**📢 Status:** Initial release, fresh out of beta! Bugs may occur as we continue improving things.

## Key Features

*   **Specialized Commands:**  Streamline your workflow with 16 essential commands for common development tasks.
    *   `/sc:implement`, `/sc:build`, `/sc:design`
    *   `/sc:analyze`, `/sc:troubleshoot`, `/sc:explain`
    *   `/sc:improve`, `/sc:test`, `/sc:cleanup`
    *   `/sc:document`, `/sc:git`, `/sc:estimate`, `/sc:task`, `/sc:index`, `/sc:load`, `/sc:spawn`
*   **Smart Personas:**  Leverage AI specialists to automatically select the right expert for various development domains.
    *   🏗️ **architect** - Systems design and architecture
    *   🎨 **frontend** - UI/UX and accessibility
    *   ⚙️ **backend** - APIs and infrastructure
    *   🔍 **analyzer** - Debugging and troubleshooting
    *   🛡️ **security** - Security concerns and vulnerabilities
    *   ✍️ **scribe** - Documentation and writing
    *   *...and 5 more specialists*
*   **MCP Server Integration:** Connect with external tools to enhance your capabilities.
    *   **Context7**: Grabs official library docs and patterns
    *   **Sequential**: Helps with complex multi-step thinking
    *   **Magic**: Generates modern UI components
    *   **Playwright**: Browser automation and testing

## Installation

### Prerequisites
Ensure you have Python 3.8+ installed.

### Step 1: Install the Package
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
Choose your preferred method for running the SuperClaude installer to configure Claude Code:

**Using the Bash Command (Recommended)**

```bash
SuperClaude install
```

**Using Python Modular Usage**

```bash
python3 -m SuperClaude install
```

### Installer Options
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

**That's it! 🎉**

## Important Notes for v2 Users

If you're upgrading from SuperClaude v2, a cleanup is necessary.

1.  Uninstall v2 (if possible)
2.  Delete these directories (if they exist):
    *   `SuperClaude/`
    *   `~/.claude/shared/`
    *   `~/.claude/commands/`
    *   `~/.claude/CLAUDE.md`
3.  Proceed with v3 installation above.

### Key Command Change:

*   `/sc:build` (compilation/packaging ONLY)
*   `/sc:implement` (feature implementation - NEW!)

## How It Works

SuperClaude enhances Claude Code by:

1.  **Framework Files:** Guides Claude's responses.
2.  **Slash Commands:** 16 specialized commands.
3.  **MCP Servers:** Integrates external tools (when they work!).
4.  **Smart Routing:** Selects appropriate tools and experts.

## What's Coming in v4

*   Hooks System (redesign)
*   MCP Suite (more integrations)
*   Performance improvements
*   More Personas
*   Cross-CLI Support

## Configuration

Customize SuperClaude by editing:

*   `~/.claude/settings.json`
*   `~/.claude/*.md`

## Documentation

*   📚 [**User Guide**](https://github.com/NomenAK/SuperClaude/blob/master/Docs/superclaude-user-guide.md) - Complete overview and getting started
*   🛠️ [**Commands Guide**](https://github.com/NomenAK/SuperClaude/blob/master/Docs/commands-guide.md) - All 16 slash commands explained
*   🏳️ [**Flags Guide**](https://github.com/NomenAK/SuperClaude/blob/master/Docs/flags-guide.md) - Command flags and options
*   🎭 [**Personas Guide**](https://github.com/NomenAK/SuperClaude/blob/master/Docs/personas-guide.md) - Understanding the persona system
*   📦 [**Installation Guide**](https://github.com/NomenAK/SuperClaude/blob/master/Docs/installation-guide.md) - Detailed installation instructions

## Contributing

We welcome contributions!

*   Bug Reports
*   Documentation
*   Testing
*   Ideas

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

## Architecture Notes

v3 focuses on:

*   Simplicity
*   Reliability
*   Modularity
*   Performance

## FAQ

*   **Why was the hooks system removed?**  Redesigning for v4.
*   **Does this work with other AI assistants?**  Currently Claude Code only, but v4 will have broader compatibility.
*   **Is this stable enough for daily use?**  Expect some rough edges, but fine for experimentation.

## SuperClaude Contributors

[![Contributors](https://contrib.rocks/image?repo=NomenAk/SuperClaude)](https://github.com/NomenAK/SuperClaude/graphs/contributors)

## License

MIT - [See LICENSE file for details](https://opensource.org/licenses/MIT)

## Star History

<a href="https://www.star-history.com/#NomenAK/SuperClaude&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=NomenAK/SuperClaude&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=NomenAK/SuperClaude&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=NomenAK/SuperClaude&type=Date" />
 </picture>
</a>

---

*Built by developers who got tired of generic responses.*
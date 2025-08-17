# SuperClaude Framework: Supercharge Your Claude Code with Powerful Features

**Tired of repetitive development tasks?** SuperClaude is a Python framework designed to extend Claude Code with specialized commands, smart personas, and MCP server integration, making your development workflow smoother and more efficient.  [Explore the SuperClaude Framework on GitHub](https://github.com/SuperClaude-Org/SuperClaude_Framework).

[![Website Preview](https://img.shields.io/badge/Visit-Website-blue?logo=google-chrome)](https://superclaude-org.github.io/SuperClaude_Website/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://img.shields.io/pypi/v/SuperClaude.svg)](https://pypi.org/project/SuperClaude/)
[![Version](https://img.shields.io/badge/version-3.0.0-blue.svg)](https://github.com/SuperClaude-Org/SuperClaude_Framework)
[![GitHub issues](https://img.shields.io/github/issues/SuperClaude-Org/SuperClaude_Framework)](https://github.com/SuperClaude-Org/SuperClaude_Framework/issues)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/SuperClaude-Org/SuperClaude_Framework/blob/master/CONTRIBUTING.md)
[![Contributors](https://img.shields.io/github/contributors/SuperClaude-Org/SuperClaude_Framework)](https://github.com/SuperClaude-Org/SuperClaude_Framework/graphs/contributors)
[![Website](https://img.shields.io/website?url=https://superclaude-org.github.io/SuperClaude_Website/)](https://superclaude-org.github.io/SuperClaude_Website/)

## Key Features

*   **Specialized Commands:**  Access 16 pre-built commands for common development tasks, including:
    *   `/sc:implement`, `/sc:build`, `/sc:design` (Development)
    *   `/sc:analyze`, `/sc:troubleshoot`, `/sc:explain` (Analysis)
    *   `/sc:improve`, `/sc:test`, `/sc:cleanup` (Quality)
    *   `/sc:document`, `/sc:git`, `/sc:estimate`, `/sc:task`, `/sc:index`, `/sc:load`, `/sc:spawn` (Others)

*   **Smart Personas:** Leverage intelligent AI personas to assist with different development domains:
    *   architect (Systems Design)
    *   frontend (UI/UX)
    *   backend (APIs/Infrastructure)
    *   analyzer (Debugging)
    *   security (Security)
    *   scribe (Documentation)
    *   ...and 5 more specialists!

*   **MCP Server Integration:** Integrate with external tools for enhanced functionality:
    *   Context7 (Library Documentation)
    *   Sequential (Multi-step Thinking)
    *   Magic (UI Component Generation)
    *   Playwright (Browser Automation)

## Current Status

This is an initial release. While functional, be aware of the following:

*   **Features:** Core framework, 16 slash commands, and MCP server integration are working.
*   **Known Issues:** Bugs are expected, documentation is a work in progress.

## Upgrading from v2?  Important!

Before installing v3, you **must** clean up v2 files to avoid conflicts. Follow these steps:

1.  Uninstall v2 (if an uninstaller is available)
2.  Manually delete the following files/folders if they exist:
    *   `SuperClaude/`
    *   `~/.claude/shared/`
    *   `~/.claude/commands/`
    *   `~/.claude/CLAUDE.md`
3.  Proceed with the v3 installation instructions below.

**Key Change for v2 Users:**

*   `/sc:build` is now compilation/packaging
*   `/sc:implement` is now feature implementation

## Installation

SuperClaude installation is a two-step process:

1.  Install the Python package
2.  Run the installer to set up Claude Code integration.

### Step 1: Install the Package

**Option A: From PyPI (Recommended)**

```bash
uv add SuperClaude
```

**Option B: From Source**

```bash
git clone https://github.com/SuperClaude-Org/SuperClaude_Framework.git
cd SuperClaude_Framework
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

### ✅ Finish Installation

After installing, continue with the usual installer step:

```bash
python3 -m SuperClaude install
```

Or using bash-style CLI:

```bash
SuperClaude install
```

### 🧠 Note:

* `uv` provides better caching and performance.
* Compatible with Python 3.8+ and works smoothly with SuperClaude.

---
**Missing Python?** Install Python 3.7+ first:
```bash
# Linux (Ubuntu/Debian)
sudo apt update && sudo apt install python3 python3-pip

# macOS  
brew install python3

# Windows
# Download from https://python.org/downloads/
```

### Step 2: Run the Installer

After installing the package, run the SuperClaude installer to configure Claude Code (You can use any of the method):
### ⚠️ Important Note 
**After installing the SuperClaude.**
**You can use `SuperClaude commands`
, `python3 -m SuperClaude commands` or also `python3 SuperClaude commands`**
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
### Or Python Modular Usage
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
### Simple bash Command Usage 
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

**That's it! 🎉** The installer handles everything: framework files, MCP servers, and Claude Code configuration.

## How It Works

SuperClaude enhances Claude Code by:

1.  **Framework Files:** Guides Claude's responses via documentation installed to `~/.claude/`.
2.  **Slash Commands:** Provides 16 specialized commands for common dev tasks.
3.  **MCP Servers:** Leverages external services for added functionality (when working).
4.  **Smart Routing:** Selects the appropriate tools and experts based on your task.

## What's Coming in v4

Future improvements include:

*   Hooks System (redesign in progress)
*   More MCP Tool Integrations
*   Performance enhancements
*   Additional Personas
*   Cross-CLI Support

## Configuration

Customize SuperClaude by editing:

*   `~/.claude/settings.json` (main configuration)
*   `~/.claude/*.md` (framework behavior)

## Documentation

Comprehensive guides are available to help you get started and learn more:

*   [**User Guide**](https://github.com/SuperClaude-Org/SuperClaude_Framework/blob/master/Docs/superclaude-user-guide.md)
*   [**Commands Guide**](https://github.com/SuperClaude-Org/SuperClaude_Framework/blob/master/Docs/commands-guide.md)
*   [**Flags Guide**](https://github.com/SuperClaude-Org/SuperClaude_Framework/blob/master/Docs/flags-guide.md)
*   [**Personas Guide**](https://github.com/SuperClaude-Org/SuperClaude_Framework/blob/master/Docs/personas-guide.md)
*   [**Installation Guide**](https://github.com/SuperClaude-Org/SuperClaude_Framework/blob/master/Docs/installation-guide.md)

## Contributing

We welcome your contributions!  Help us with:

*   Bug Reports
*   Documentation
*   Testing
*   New Feature Ideas

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

v3 architecture focuses on:

*   Simplicity
*   Reliability
*   Modularity
*   Performance

## FAQ

**Q: Why was the hooks system removed?**

A: It was getting complex; it's being redesigned for v4.

**Q: Does this work with other AI assistants?**

A: Currently Claude Code only; v4 will aim for broader compatibility.

**Q: Is this stable enough for daily use?**

A: Core functionality is solid, but expect some rough edges.

## SuperClaude Contributors

[![Contributors](https://contrib.rocks/image?repo=SuperClaude-Org/SuperClaude_Framework)](https://github.com/SuperClaude-Org/SuperClaude_Framework/graphs/contributors)

## License

MIT - [See LICENSE file for details](https://opensource.org/licenses/MIT)

## Star History

<a href="https://www.star-history.com/#SuperClaude-Org/SuperClaude_Framework&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=SuperClaude-Org/SuperClaude_Framework&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=SuperClaude-Org/SuperClaude_Framework&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=SuperClaude-Org/SuperClaude_Framework&type=Date" />
 </picture>
</a>
---

*Built by developers for developers.  We hope you find SuperClaude useful!*
# SuperClaude Framework: Supercharge Your Development Workflow with AI 🚀

**SuperClaude empowers developers by extending Claude Code with specialized commands, AI-powered personas, and MCP server integration to streamline coding tasks.** [Check out the original repo](https://github.com/SuperClaude-Org/SuperClaude_Framework) for the latest updates!

[![Website Preview](https://img.shields.io/badge/Visit-Website-blue?logo=google-chrome)](https://superclaude-org.github.io/SuperClaude_Website/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://img.shields.io/pypi/v/SuperClaude.svg)](https://pypi.org/project/SuperClaude/)
[![Version](https://img.shields.io/badge/version-3.0.0-blue.svg)](https://github.com/SuperClaude-Org/SuperClaude_Framework)
[![GitHub issues](https://img.shields.io/github/issues/SuperClaude-Org/SuperClaude_Framework)](https://github.com/SuperClaude-Org/SuperClaude_Framework/issues)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/SuperClaude-Org/SuperClaude_Framework/blob/master/CONTRIBUTING.md)
[![Contributors](https://img.shields.io/github/contributors/SuperClaude-Org/SuperClaude_Framework)](https://github.com/SuperClaude-Org/SuperClaude_Framework/graphs/contributors)
[![Website](https://img.shields.io/website?url=https://superclaude-org.github.io/SuperClaude_Website/)](https://superclaude-org.github.io/SuperClaude_Website/)

## Key Features

*   **Specialized Commands:** 16 commands for common development tasks like implementing features, building, testing, and more.
*   **Smart Personas:** AI-powered personas (architect, frontend, backend, analyzer, etc.) that provide expert assistance based on the task.
*   **MCP Server Integration:** Connects to external tools for documentation retrieval, UI component generation, and browser automation.
*   **Simplified Task Management:** Helps you keep track of your progress by integrating the tasks with your prompt.
*   **Token Optimization:** Optimized for faster responses by optimizing the content to make more efficient use of the available token.

## What's New in v3?

*   **Improved Installation:** A rewritten installation suite for a smoother setup.
*   **Core Framework:** Optimized core framework with a suite of nine core documentation files.
*   **Enhanced Commands:** Enhanced 16 specialized slash commands.
*   **MCP Server Integration:** Support for MCP Servers and its utilities (Context7, Sequential, Magic, Playwright).
*   **Unified CLI Installer:** Easier setup with a unified CLI installer that works seamlessly.

## Current Status

*   **Working Well:** Installation suite, core framework, 16 slash commands, MCP server integration, and a unified CLI installer.
*   **Known Issues:** Initial release, potential bugs, and ongoing documentation improvements.

## Installation

SuperClaude installation is a two-step process:
1. First install the Python package
2. Then run the installer to set up Claude Code integration

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

SuperClaude enhances Claude Code through:

*   **Framework Files:** Documentation installed to `~/.claude/` that guides how Claude responds.
*   **Slash Commands:** 16 specialized commands for different dev tasks.
*   **MCP Servers:** External services that add extra capabilities (when they work!).
*   **Smart Routing:** Attempts to pick the right tools and experts based on what you're doing.

## Upgrading from v2?  Read This!

If you're coming from SuperClaude v2, important upgrade steps are needed:

1.  **Uninstall v2:** Use the v2 uninstaller if available.
2.  **Manual Cleanup:** Delete the following directories if they exist:
    *   `SuperClaude/`
    *   `~/.claude/shared/`
    *   `~/.claude/commands/`
    *   `~/.claude/CLAUDE.md`
3.  **Proceed with v3 Installation:** Follow the v3 installation steps above.

### Key Change for v2 Users

`/build` command has been changed in v3:

*   `/sc:build` = compilation/packaging only
*   `/sc:implement` = feature implementation (NEW!)

**Migration:** Replace `v2 /build myFeature` with `v3 /sc:implement myFeature`

## What's Coming in v4

Planned features for v4 include:

*   **Hooks System:** Event-driven system (redesign from v3).
*   **MCP Suite:** More external tool integrations.
*   **Performance Improvements:** Faster and less buggy.
*   **More Personas:** Additional domain specialists.
*   **Cross-CLI Support:** Compatibility with other AI coding assistants.

## Configuration

Customize SuperClaude by editing:

*   `~/.claude/settings.json`: Main configuration.
*   `~/.claude/*.md`: Framework behavior files.

## Documentation

*   [**User Guide**](https://github.com/SuperClaude-Org/SuperClaude_Framework/blob/master/Docs/superclaude-user-guide.md)
*   [**Commands Guide**](https://github.com/SuperClaude-Org/SuperClaude_Framework/blob/master/Docs/commands-guide.md)
*   [**Flags Guide**](https://github.com/SuperClaude-Org/SuperClaude_Framework/blob/master/Docs/flags-guide.md)
*   [**Personas Guide**](https://github.com/SuperClaude-Org/SuperClaude_Framework/blob/master/Docs/personas-guide.md)
*   [**Installation Guide**](https://github.com/SuperClaude-Org/SuperClaude_Framework/blob/master/Docs/installation-guide.md)

## Contributing

We welcome contributions! Help with:

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

V3 architecture focuses on:

*   Simplicity
*   Reliability
*   Modularity
*   Performance

## FAQ

**Q: Why was the hooks system removed?**
A: It was getting complex and buggy. Redesigning for v4.

**Q: Does this work with other AI assistants?**
A: Currently Claude Code only, but v4 will have broader compatibility.

**Q: Is this stable enough for daily use?**
A: Basic stuff works, but expect some rough edges since it's a fresh release.

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

*Built by developers, for developers.  We hope it's useful!*
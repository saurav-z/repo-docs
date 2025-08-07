# Supercharge Your Claude Code with SuperClaude: A Powerful AI Framework

SuperClaude is a groundbreaking framework that significantly enhances the capabilities of Claude Code by providing specialized commands, smart personas, and seamless integration with external tools, giving you advanced AI-assisted development capabilities. ([See the original repo](https://github.com/SuperClaude-Org/SuperClaude_Framework))

[![Website Preview](https://img.shields.io/badge/Visit-Website-blue?logo=google-chrome)](https://superclaude-org.github.io/SuperClaude_Website/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://img.shields.io/pypi/v/SuperClaude.svg)](https://pypi.org/project/SuperClaude/)
[![Version](https://img.shields.io/badge/version-3.0.0-blue.svg)](https://github.com/SuperClaude-Org/SuperClaude_Framework)
[![GitHub issues](https://img.shields.io/github/issues/SuperClaude-Org/SuperClaude_Framework)](https://github.com/SuperClaude-Org/SuperClaude_Framework/issues)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/SuperClaude-Org/SuperClaude_Framework/blob/master/CONTRIBUTING.md)
[![Contributors](https://img.shields.io/github/contributors/SuperClaude-Org/SuperClaude_Framework)](https://github.com/SuperClaude-Org/SuperClaude_Framework/graphs/contributors)
[![Website](https://img.shields.io/website?url=https://superclaude-org.github.io/SuperClaude_Website/)](https://superclaude-org.github.io/SuperClaude_Website/)

## Key Features

*   **16 Specialized Commands:** Simplify common development tasks with commands like `/sc:implement`, `/sc:build`, and `/sc:test`.
*   **Smart Personas:** Leverage AI specialists such as architects, frontend developers, and security experts for targeted assistance.
*   **MCP Server Integration:** Connect with external tools like Context7, Sequential, Magic, and Playwright for expanded functionality.
*   **Token Optimization:** Improve performance during longer conversations.

## What's New in Version 3?

*   Enhanced installation suite for easier setup.
*   Refreshed core framework with improved documentation.
*   A more modular architecture, giving you the ability to select only the desired components
*   Improved command structure and clearer migration path from previous versions.

## Key Changes for v2 Users

*   **Uninstall v2:** Before upgrading to v3, uninstall the old version to avoid conflicts.
*   **Command Changes:** `/sc:build` now handles compilation and packaging only; `/sc:implement` is used for feature implementation.

## Installation

SuperClaude installation is a two-step process:

1.  Install the Python package.
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

SuperClaude enhances Claude Code with:

1.  **Framework Files:** Documentation installed to `~/.claude/` that guides Claude's responses.
2.  **Slash Commands:** 16 specialized commands for diverse development tasks.
3.  **MCP Servers:** External services for expanded functionality (when they work!).
4.  **Smart Routing:** Optimizes tool and expert selection.

## Documentation

*   📚 [**User Guide**](https://github.com/SuperClaude-Org/SuperClaude_Framework/blob/master/Docs/superclaude-user-guide.md)
*   🛠️ [**Commands Guide**](https://github.com/SuperClaude-Org/SuperClaude_Framework/blob/master/Docs/commands-guide.md)
*   🏳️ [**Flags Guide**](https://github.com/SuperClaude-Org/SuperClaude_Framework/blob/master/Docs/flags-guide.md)
*   🎭 [**Personas Guide**](https://github.com/SuperClaude-Org/SuperClaude_Framework/blob/master/Docs/personas-guide.md)
*   📦 [**Installation Guide**](https://github.com/SuperClaude-Org/SuperClaude_Framework/blob/master/Docs/installation-guide.md)

## Contributing

We welcome contributions! Please feel free to submit:

*   🐛 **Bug Reports**
*   📝 **Documentation**
*   🧪 **Testing**
*   💡 **Ideas**

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

The v3 architecture focuses on:

*   **Simplicity**
*   **Reliability**
*   **Modularity**
*   **Performance**

## FAQ

**Q: Why was the hooks system removed?**

A: It was getting complex and buggy. We're redesigning it properly for v4.

**Q: Does this work with other AI assistants?**

A: Currently Claude Code only, but v4 will have broader compatibility.

**Q: Is this stable enough for daily use?**

A: The basic stuff works pretty well, but definitely expect some rough edges since it's a fresh release. Probably fine for experimenting! 🧪

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

*Built by developers who got tired of generic responses. Hope you find it useful! 🙂*
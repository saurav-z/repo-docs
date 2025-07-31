# SuperClaude: Enhance Claude Code with Specialized Commands, Personas, and More

**Supercharge your development workflow with SuperClaude, a framework that extends Claude Code with powerful features, offering 16 specialized commands and smart personas to streamline your coding tasks.** ([View the original repository](https://github.com/SuperClaude-Org/SuperClaude_Framework))

[![Website Preview](https://img.shields.io/badge/Visit-Website-blue?logo=google-chrome)](https://superclaude-org.github.io/SuperClaude_Website/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://img.shields.io/pypi/v/SuperClaude.svg)](https://pypi.org/project/SuperClaude/)
[![Version](https://img.shields.io/badge/version-3.0.0-blue.svg)](https://github.com/NomenAK/SuperClaude)
[![GitHub issues](https://img.shields.io/github/issues/NomenAK/SuperClaude)](https://github.com/NomenAK/SuperClaude/issues)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/NomenAK/SuperClaude/blob/master/CONTRIBUTING.md)
[![Contributors](https://img.shields.io/github/contributors/NomenAK/SuperClaude)](https://github.com/NomenAK/SuperClaude/graphs/contributors)
[![Website](https://img.shields.io/website?url=https://superclaude-org.github.io/SuperClaude_Website/)](https://superclaude-org.github.io/SuperClaude_Website/)

## Key Features

*   **Specialized Commands:** Access 16 commands designed for common development tasks such as implementing, building, analyzing, and more.
*   **Smart Personas:** Leverage AI-powered personas that automatically select the most relevant expert for your development domain.
*   **MCP Server Integration:** Integrate with external tools for documentation, UI components, and browser automation.
*   **Task Management:** Simplify project management with SuperClaude's built-in task tracking system.
*   **Token Optimization:** Improve the efficiency of your conversations with optimized token usage.

## Core Functionality

SuperClaude enhances Claude Code with:

*   **16 Specialized Commands** for development, analysis, quality, and more.
*   **Smart Personas** including architect, frontend, backend, analyzer, security, scribe, and more to specialize responses.
*   **MCP Integration** for external tools such as Context7, Sequential, Magic, and Playwright.

## Installation

SuperClaude installation is a two-step process: First install the Python package, then run the installer to set up Claude Code integration.

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

*   `uv` provides better caching and performance.
*   Compatible with Python 3.8+ and works smoothly with SuperClaude.

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

**You can use `SuperClaude commands`, `python3 -m SuperClaude commands` or also `python3 SuperClaude commands`**

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

## Upgrading from v2

If you're coming from SuperClaude v2, clean up first:

1.  **Uninstall v2** using its uninstaller if available
2.  **Manual cleanup**: Delete the following if they exist:
    *   `SuperClaude/`
    *   `~/.claude/shared/`
    *   `~/.claude/commands/`
    *   `~/.claude/CLAUDE.md`
3.  **Then proceed** with v3 installation above

### 🔄 Key Change for v2 Users

*   The `/build` command changed:
    *   `/sc:build` = compilation/packaging only
    *   `/sc:implement` = feature implementation (NEW!)
*   **Migration**: Replace `v2 /build myFeature` with `v3 /sc:implement myFeature`

## How It Works

SuperClaude enhances Claude Code by:

1.  **Framework Files**: Guiding Claude's responses.
2.  **Slash Commands**: Offering specialized commands for dev tasks.
3.  **MCP Servers**: Adding external capabilities.
4.  **Smart Routing**: Selecting the right tools and experts.

## What's Coming in v4

*   **Hooks System:** Redesigning the event-driven system.
*   **MCP Suite:** Adding more tool integrations.
*   **Performance Improvements:** Enhancing speed and reducing bugs.
*   **More Personas:** Adding more domain specialists.
*   **Cross-CLI Support:** Expanding compatibility.

## Configuration

Customize SuperClaude by editing:

*   `~/.claude/settings.json` - Main configuration
*   `~/.claude/*.md` - Framework behavior files

## Documentation

*   📚 [**User Guide**](https://github.com/NomenAK/SuperClaude/blob/master/Docs/superclaude-user-guide.md)
*   🛠️ [**Commands Guide**](https://github.com/NomenAK/SuperClaude/blob/master/Docs/commands-guide.md)
*   🏳️ [**Flags Guide**](https://github.com/NomenAK/SuperClaude/blob/master/Docs/flags-guide.md)
*   🎭 [**Personas Guide**](https://github.com/NomenAK/SuperClaude/blob/master/Docs/personas-guide.md)
*   📦 [**Installation Guide**](https://github.com/NomenAK/SuperClaude/blob/master/Docs/installation-guide.md)

## Contributing

We welcome contributions in the following areas:

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

*   **Simplicity**: Removing unnecessary complexity.
*   **Reliability**: Improved installation and fewer breaking changes.
*   **Modularity**: Only install the components you need.
*   **Performance**: Faster operations with smarter caching.

## FAQ

**Q: Why was the hooks system removed?**
A: It was getting complex and buggy. We're redesigning it for v4.

**Q: Does this work with other AI assistants?**
A: Currently Claude Code only, but v4 will have broader compatibility.

**Q: Is this stable enough for daily use?**
A: The basic stuff works pretty well, but definitely expect some rough edges since it's a fresh release. Probably fine for experimenting! 🧪

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

*Built by developers who got tired of generic responses. Hope you find it useful! 🙂*
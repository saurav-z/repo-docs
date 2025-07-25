# Supercharge Your Development Workflow with SuperClaude v3!

Supercharge your Claude Code experience with SuperClaude, a framework that adds specialized commands, smart personas, and powerful integrations to boost your productivity. Check out the original repo: [https://github.com/SuperClaude-Org/SuperClaude_Framework](https://github.com/SuperClaude-Org/SuperClaude_Framework)

[![Website Preview](https://img.shields.io/badge/Visit-Website-blue?logo=google-chrome)](https://superclaude-org.github.io/SuperClaude_Website/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://img.shields.io/pypi/v/SuperClaude.svg)](https://pypi.org/project/SuperClaude/)
[![Version](https://img.shields.io/badge/version-3.0.0-blue.svg)](https://github.com/NomenAK/SuperClaude)
[![GitHub issues](https://img.shields.io/github/issues/NomenAK/SuperClaude)](https://github.com/NomenAK/SuperClaude/issues)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/NomenAK/SuperClaude/blob/master/CONTRIBUTING.md)
[![Contributors](https://img.shields.io/github/contributors/NomenAK/SuperClaude)](https://github.com/NomenAK/SuperClaude/graphs/contributors)
[![Website](https://img.shields.io/website?url=https://superclaude-org.github.io/SuperClaude_Website/)](https://superclaude-org.github.io/SuperClaude_Website/)

**📢 Status**: Initial release, fresh out of beta! Bugs may occur as we continue improving things.

## Key Features

*   **Specialized Commands:** Enhance Claude Code with 16 task-specific commands for common development tasks, including:
    *   `/sc:implement`, `/sc:build`, `/sc:design` (Development)
    *   `/sc:analyze`, `/sc:troubleshoot`, `/sc:explain` (Analysis)
    *   `/sc:improve`, `/sc:test`, `/sc:cleanup` (Quality)
    *   `/sc:document`, `/sc:git`, `/sc:estimate`, `/sc:task`, `/sc:index`, `/sc:load`, `/sc:spawn` (Others)
*   **Smart Personas:** AI specialists that help Claude Code by selecting the right expert for different development domains such as:
    *   Architect, Frontend, Backend, Analyzer, Security, Scribe, and more.
*   **MCP Server Integration:** Integrates with external tools like Context7, Sequential, Magic, and Playwright to expand Claude Code capabilities.
*   **Token Optimization:** Assists with longer conversations by optimizing token usage.

## What's New in v3?

*   **Improved Installation:** A rewritten installation suite for a smoother setup.
*   **Core Framework:** Robust core framework with key documentation files.
*   **Unified CLI Installer:** Simplifies setup with an easy-to-use command-line interface.
*   **Important Upgrade Note:** If you're coming from v2, please note important upgrade steps in the "Upgrading from v2" section below.

## Current Status

*   **Working Well:**
    *   Installation suite (rewritten from the ground up)
    *   Core framework with 9 documentation files
    *   16 slash commands for various development tasks
    *   MCP server integration (Context7, Sequential, Magic, Playwright)
    *   Unified CLI installer for easy setup
*   **Known Issues:**
    *   This is an initial release - bugs are expected
    *   Some features may not work perfectly yet
    *   Documentation is still being improved
    *   Hooks system was removed (coming back in v4)

## ⚠️ Upgrading from v2? Important!

If you're coming from SuperClaude v2, you'll need to clean up first:

1.  **Uninstall v2** using its uninstaller if available
2.  **Manual cleanup** - delete these if they exist:
    *   `SuperClaude/`
    *   `~/.claude/shared/`
    *   `~/.claude/commands/`
    *   `~/.claude/CLAUDE.md`
3.  **Then proceed** with v3 installation below

This is because v3 has a different structure and the old files can cause conflicts.

### 🔄 **Key Change for v2 Users**

**The `/build` command changed!** In v2, `/build` was used for feature implementation. In v3:
    *   `/sc:build` = compilation/packaging only
    *   `/sc:implement` = feature implementation (NEW!)

**Migration**: Replace `v2 /build myFeature` with `v3 /sc:implement myFeature`

## Installation 📦

SuperClaude installation is a **two-step process**:
1.  First install the Python package
2.  Then run the installer to set up Claude Code integration

### Step 1: Install the Package

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

## How It Works 🔄

SuperClaude enhances Claude Code by:

1.  **Framework Files**: Documentation installed to `~/.claude/` that guides how Claude responds
2.  **Slash Commands**: 16 specialized commands for different dev tasks
3.  **MCP Servers**: External services that add extra capabilities (when they work!)
4.  **Smart Routing**: Attempts to pick the right tools and experts based on what you're doing

Most of the time it plays nicely with Claude Code's existing stuff. 🤝

## What's Coming in v4 🔮

We're hoping to work on these things for the next version:
*   **Hooks System**: Event-driven stuff (removed from v3, trying to redesign it properly)
*   **MCP Suite**: More external tool integrations
*   **Better Performance**: Trying to make things faster and less buggy
*   **More Personas**: Maybe a few more domain specialists
*   **Cross-CLI Support**: Might work with other AI coding assistants

*(No promises on timeline though - we're still figuring v3 out! 😅)*

## Configuration ⚙️

After installation, you can customize SuperClaude by editing:

*   `~/.claude/settings.json` - Main configuration
*   `~/.claude/*.md` - Framework behavior files

Most users probably won't need to change anything - it usually works okay out of the box. 🎛️

## Documentation 📖

Want to learn more? Check out our guides:

*   📚 [**User Guide**](https://github.com/NomenAK/SuperClaude/blob/master/Docs/superclaude-user-guide.md) - Complete overview and getting started
*   🛠️ [**Commands Guide**](https://github.com/NomenAK/SuperClaude/blob/master/Docs/commands-guide.md) - All 16 slash commands explained
*   🏳️ [**Flags Guide**](https://github.com/NomenAK/SuperClaude/blob/master/Docs/flags-guide.md) - Command flags and options
*   🎭 [**Personas Guide**](https://github.com/NomenAK/SuperClaude/blob/master/Docs/personas-guide.md) - Understanding the persona system
*   📦 [**Installation Guide**](https://github.com/NomenAK/SuperClaude/blob/master/Docs/installation-guide.md) - Detailed installation instructions

These guides have more details than this README and are kept up to date.

## Contributing 🤝

We welcome contributions! Areas where we could use help:

*   🐛 **Bug Reports**: Let us know what's broken
*   📝 **Documentation**: Help us explain things better
*   🧪 **Testing**: More test coverage for different setups
*   💡 **Ideas**: Suggestions for new features or improvements

The codebase is pretty straightforward Python + documentation files.

## Project Structure 📁

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

## Architecture Notes 🏗️

The v3 architecture focuses on:
*   **Simplicity**: Removed complexity that wasn't adding value
*   **Reliability**: Better installation and fewer breaking changes
*   **Modularity**: Pick only the components you want
*   **Performance**: Faster operations with smarter caching

We learned a lot from v2 and tried to address the main pain points.

## FAQ 🙋

**Q: Why was the hooks system removed?**  
A: It was getting complex and buggy. We're redesigning it properly for v4.

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

---
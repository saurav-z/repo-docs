# Supercharge Your Development with SuperClaude: An AI-Powered Framework for Claude Code

SuperClaude is an innovative framework that enhances Claude Code with specialized commands, smart personas, and MCP server integration, streamlining your development workflow. **Check out the original repo [here](https://github.com/SuperClaude-Org/SuperClaude_Framework)!**

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

*   **Specialized Commands:** 16 commands to streamline common development tasks.
*   **Smart Personas:** AI specialists that help guide the responses.
*   **MCP Server Integration:** Integrates with external tools for advanced capabilities.
*   **Token Optimization:** Optimizes conversations for longer interactions.
*   **Unified CLI Installer**: Easy setup with a user-friendly installer.

## What's New in SuperClaude v3

*   **Improved Installation:** A streamlined and reliable installation process.
*   **Enhanced Core Framework:** Stable and well-documented core components.
*   **Expanded Commands:** More commands for a wider range of development tasks.

## Commands

SuperClaude introduces 16 specialized commands to simplify and accelerate your development workflow:

*   `/sc:implement`, `/sc:build`, `/sc:design`
*   `/sc:analyze`, `/sc:troubleshoot`, `/sc:explain`
*   `/sc:improve`, `/sc:test`, `/sc:cleanup`
*   `/sc:document`, `/sc:git`, `/sc:estimate`, `/sc:task`, `/sc:index`, `/sc:load`, `/sc:spawn`

## Smart Personas

SuperClaude features AI specialists that jump in when they are needed.

*   🏗️ **architect**: Systems design and architecture
*   🎨 **frontend**: UI/UX and accessibility
*   ⚙️ **backend**: APIs and infrastructure
*   🔍 **analyzer**: Debugging and figuring things out
*   🛡️ **security**: Security concerns and vulnerabilities
*   ✍️ **scribe**: Documentation and writing
*   *...and 5 more specialists*

*(Personas are not always perfect, but often choose the right expert!)*

## MCP Integration

SuperClaude integrates with external tools to provide extra features:

*   **Context7**: Get documentation and patterns.
*   **Sequential**: Helps with complex multi-step thinking.
*   **Magic**: Generate modern UI components.
*   **Playwright**: Browser automation and testing.

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
- `/sc:build` = compilation/packaging only
- `/sc:implement` = feature implementation (NEW!)

**Migration**: Replace `v2 /build myFeature` with `v3 /sc:implement myFeature`

## Installation

SuperClaude offers a two-step installation process:

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

SuperClaude aims to enhance Claude Code through:

1.  **Framework Files:** Documentation to guide Claude's responses.
2.  **Slash Commands:** 16 commands for diverse development tasks.
3.  **MCP Servers:** External services providing extended capabilities.
4.  **Smart Routing:** Intelligent selection of tools and experts.

## What's Coming in v4

*   **Hooks System:** An event-driven system (redesign in progress).
*   **MCP Suite:** More integrations.
*   **Performance Enhancements:** Faster and more stable operations.
*   **More Personas:** Expansion of domain specialists.
*   **Cross-CLI Support:** Compatibility with other AI coding assistants.

*(No timelines are given - development is on going!)*

## Configuration

Customize SuperClaude by editing:

*   `~/.claude/settings.json`: Main configuration file.
*   `~/.claude/*.md`: Behavior files.

## Documentation

*   📚 [**User Guide**](https://github.com/NomenAK/SuperClaude/blob/master/Docs/superclaude-user-guide.md)
*   🛠️ [**Commands Guide**](https://github.com/NomenAK/SuperClaude/blob/master/Docs/commands-guide.md)
*   🏳️ [**Flags Guide**](https://github.com/NomenAK/SuperClaude/blob/master/Docs/flags-guide.md)
*   🎭 [**Personas Guide**](https://github.com/NomenAK/SuperClaude/blob/master/Docs/personas-guide.md)
*   📦 [**Installation Guide**](https://github.com/NomenAK/SuperClaude/blob/master/Docs/installation-guide.md)

## Contributing

We welcome contributions!

*   🐛 Bug Reports
*   📝 Documentation
*   🧪 Testing
*   💡 Ideas

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
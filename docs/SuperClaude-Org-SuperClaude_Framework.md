# SuperClaude: Supercharge Your Development with AI 🚀

**Tired of repetitive development tasks? SuperClaude enhances Claude Code with specialized commands, smart personas, and powerful integrations, revolutionizing your workflow.**  Explore the original repository [here](https://github.com/SuperClaude-Org/SuperClaude_Framework)!

[![Website Preview](https://img.shields.io/badge/Visit-Website-blue?logo=google-chrome)](https://superclaude-org.github.io/SuperClaude_Website/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://img.shields.io/pypi/v/SuperClaude.svg)](https://pypi.org/project/SuperClaude/)
[![Version](https://img.shields.io/badge/version-3.0.0-blue.svg)](https://github.com/SuperClaude-Org/SuperClaude_Framework)
[![GitHub issues](https://img.shields.io/github/issues/SuperClaude-Org/SuperClaude_Framework)](https://github.com/SuperClaude-Org/SuperClaude_Framework/issues)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/SuperClaude-Org/SuperClaude_Framework/blob/master/CONTRIBUTING.md)
[![Contributors](https://img.shields.io/github/contributors/SuperClaude-Org/SuperClaude_Framework)](https://github.com/SuperClaude-Org/SuperClaude_Framework/graphs/contributors)
[![Website](https://img.shields.io/website?url=https://superclaude-org.github.io/SuperClaude_Website/)](https://superclaude-org.github.io/SuperClaude_Website/)

## Key Features

*   **Specialized Commands:** Utilize 16 development-focused slash commands for common tasks, improving efficiency.
*   **Smart Personas:** Benefit from AI-powered personas that automatically select the best expert for your needs, speeding up your work.
*   **MCP Server Integration:** Connect with external tools for enhanced capabilities in areas like documentation, UI components, and browser automation.
*   **Task Management:** Stay on track with built-in task tracking, ensuring you always know where you stand.
*   **Token Optimization:** Experience smoother conversations and better handling of longer prompts, enhancing productivity.

## What's New in SuperClaude v3

*   **Improved Installation:** A completely rewritten installation suite, now easier to set up.
*   **Core Framework Enhancement:** Refined core framework with updated documentation and streamlined operations.
*   **Slash Command Optimization:** Improved and expanded command set to better cover a range of development needs.
*   **MCP Server Updates:** Refined MCP server integration with new tools for docs, UI components, and browser automation.
*   **Simplified Setup:** Use a unified CLI installer for a fast and convenient setup.

## Current Status

*   **Working Well:**
    *   Installation Suite
    *   Core Framework
    *   16 Slash Commands
    *   MCP Server Integration
    *   Unified CLI installer

*   **Known Issues:**
    *   Initial Release: Bugs may occur.
    *   Documentation in progress.

## Installation

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

## Commands

*   **Development:** `/sc:implement`, `/sc:build`, `/sc:design`
*   **Analysis:** `/sc:analyze`, `/sc:troubleshoot`, `/sc:explain`
*   **Quality:** `/sc:improve`, `/sc:test`, `/sc:cleanup`
*   **Other:** `/sc:document`, `/sc:git`, `/sc:estimate`, `/sc:task`, `/sc:index`, `/sc:load`, `/sc:spawn`

## Personas

*   🏗️ **architect** - Systems design and architecture
*   🎨 **frontend** - UI/UX and accessibility
*   ⚙️ **backend** - APIs and infrastructure
*   🔍 **analyzer** - Debugging and troubleshooting
*   🛡️ **security** - Security concerns
*   ✍️ **scribe** - Documentation and writing

## Important for v2 Users: Upgrading from v2

Follow these steps to migrate from SuperClaude v2:

1.  **Uninstall v2** using its uninstaller.
2.  **Manual Cleanup:** Delete these directories if they exist: `SuperClaude/`, `~/.claude/shared/`, `~/.claude/commands/`, and `~/.claude/CLAUDE.md`.
3.  **Install v3** following the installation instructions above.
4.  **Command Change:** Note that `/build` in v2 is now `/sc:build` (compilation/packaging) and `/sc:implement` (feature implementation) in v3.  Replace `v2 /build myFeature` with `v3 /sc:implement myFeature`.

## Configuration

Customize SuperClaude by editing:

*   `~/.claude/settings.json`: Main configuration settings.
*   `~/.claude/*.md`: For framework behavior adjustments.

## Documentation

*   📚 [**User Guide**](https://github.com/SuperClaude-Org/SuperClaude_Framework/blob/master/Docs/superclaude-user-guide.md)
*   🛠️ [**Commands Guide**](https://github.com/SuperClaude-Org/SuperClaude_Framework/blob/master/Docs/commands-guide.md)
*   🏳️ [**Flags Guide**](https://github.com/SuperClaude-Org/SuperClaude_Framework/blob/master/Docs/flags-guide.md)
*   🎭 [**Personas Guide**](https://github.com/SuperClaude-Org/SuperClaude_Framework/blob/master/Docs/personas-guide.md)
*   📦 [**Installation Guide**](https://github.com/SuperClaude-Org/SuperClaude_Framework/blob/master/Docs/installation-guide.md)

## Contributing

Help us make SuperClaude even better:

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

v3 focuses on:

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
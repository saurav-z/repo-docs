# SuperClaude Framework: Enhance Claude Code for Development 🚀

Supercharge your development workflow with SuperClaude, a powerful framework that elevates the capabilities of Claude Code!  [Visit the SuperClaude Framework Repository](https://github.com/SuperClaude-Org/SuperClaude_Framework).

[![Website Preview](https://img.shields.io/badge/Visit-Website-blue?logo=google-chrome)](https://superclaude-org.github.io/SuperClaude_Website/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://img.shields.io/pypi/v/SuperClaude.svg)](https://pypi.org/project/SuperClaude/)
[![Version](https://img.shields.io/badge/version-3.0.0-blue.svg)](https://github.com/SuperClaude-Org/SuperClaude_Framework)
[![GitHub issues](https://img.shields.io/github/issues/SuperClaude-Org/SuperClaude_Framework)](https://github.com/SuperClaude-Org/SuperClaude_Framework/issues)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/SuperClaude-Org/SuperClaude_Framework/blob/master/CONTRIBUTING.md)
[![Contributors](https://img.shields.io/github/contributors/SuperClaude-Org/SuperClaude_Framework)](https://github.com/SuperClaude-Org/SuperClaude_Framework/graphs/contributors)
[![Website](https://img.shields.io/website?url=https://superclaude-org.github.io/SuperClaude_Website/)](https://superclaude-org.github.io/SuperClaude_Website/)

**Key Features:**

*   **Specialized Commands:** 16 commands designed for common development tasks, streamlining your workflow.
*   **Smart Personas:** AI-powered personas that automatically select the best expert for the job, improving accuracy and efficiency.
*   **MCP Server Integration:** Seamless integration with external tools like Context7, Sequential, Magic, and Playwright for enhanced functionality.
*   **Task Management:** Integrated task management to keep track of progress and stay organized.
*   **Token Optimization:** Optimizes token usage for longer, more effective conversations.

## Core Functionality

SuperClaude enhances Claude Code by integrating several key components:

*   **Framework Files:** Documentation installed to `~/.claude/` that guides Claude's responses.
*   **Slash Commands:** 16 specialized commands for various development tasks.
*   **MCP Servers:** External services that add extra capabilities.
*   **Smart Routing:** Attempts to pick the right tools and experts based on what you're doing.

## Key Features in Detail

### Commands 🛠️

We focused on 16 essential commands for the most common tasks:

**Development**: `/sc:implement`, `/sc:build`, `/sc:design`
**Analysis**: `/sc:analyze`, `/sc:troubleshoot`, `/sc:explain`
**Quality**: `/sc:improve`, `/sc:test`, `/sc:cleanup`
**Others**: `/sc:document`, `/sc:git`, `/sc:estimate`, `/sc:task`, `/sc:index`, `/sc:load`, `/sc:spawn`

### Smart Personas 🎭

AI specialists that try to jump in when they seem relevant:
*   🏗️ **architect** - Systems design and architecture stuff
*   🎨 **frontend** - UI/UX and accessibility
*   ⚙️ **backend** - APIs and infrastructure
*   🔍 **analyzer** - Debugging and figuring things out
*   🛡️ **security** - Security concerns and vulnerabilities
*   ✍️ **scribe** - Documentation and writing
*   *...and 5 more specialists*

### MCP Integration 🔧

External tools that connect when useful:
*   **Context7** - Grabs official library docs and patterns
*   **Sequential** - Helps with complex multi-step thinking
*   **Magic** - Generates modern UI components
*   **Playwright** - Browser automation and testing stuff

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

## Upgrading from v2 ⚠️

If you're coming from SuperClaude v2, you'll need to clean up first:

1.  **Uninstall v2** using its uninstaller if available
2.  **Manual cleanup** - delete these if they exist:
    -   `SuperClaude/`
    -   `~/.claude/shared/`
    -   `~/.claude/commands/`
    -   `~/.claude/CLAUDE.md`
3.  **Then proceed** with v3 installation below

This is because v3 has a different structure and the old files can cause conflicts.

### 🔄 **Key Change for v2 Users**

**The `/build` command changed!** In v2, `/build` was used for feature implementation. In v3:

-   `/sc:build` = compilation/packaging only
-   `/sc:implement` = feature implementation (NEW!)

**Migration**: Replace `v2 /build myFeature` with `v3 /sc:implement myFeature`

## How It Works 🔄

SuperClaude enhances Claude Code through the following mechanisms:

1.  **Framework Files:** Documentation installed to `~/.claude/` guides Claude's responses.
2.  **Slash Commands:** 16 specialized commands for diverse development tasks.
3.  **MCP Servers:** External services that add extra capabilities (when they work!).
4.  **Smart Routing:** Attempts to select the appropriate tools and experts based on your actions.

## What's Coming in v4 🔮

The following features are planned for the next version:

*   **Hooks System:** Event-driven functionality (redesign in progress).
*   **MCP Suite:** Enhanced integration with more external tools.
*   **Performance Improvements:** Faster and more reliable operations.
*   **More Personas:** Addition of more domain specialists.
*   **Cross-CLI Support:** Expanded compatibility with other AI coding assistants.

*(Timeline subject to change; actively refining v3!)*

## Configuration ⚙️

Customize SuperClaude after installation by editing:

*   `~/.claude/settings.json`: Main configuration file.
*   `~/.claude/*.md`: Framework behavior files.

Most users can use the default settings, which offer solid out-of-the-box performance.

## Documentation 📖

Access comprehensive guides for in-depth information:

*   📚 [**User Guide**](https://github.com/SuperClaude-Org/SuperClaude_Framework/blob/master/Docs/superclaude-user-guide.md) - Comprehensive overview and onboarding.
*   🛠️ [**Commands Guide**](https://github.com/SuperClaude-Org/SuperClaude_Framework/blob/master/Docs/commands-guide.md) - Detailed explanations of all 16 slash commands.
*   🏳️ [**Flags Guide**](https://github.com/SuperClaude-Org/SuperClaude_Framework/blob/master/Docs/flags-guide.md) - Command flags and options.
*   🎭 [**Personas Guide**](https://github.com/SuperClaude-Org/SuperClaude_Framework/blob/master/Docs/personas-guide.md) - Dive into the persona system.
*   📦 [**Installation Guide**](https://github.com/SuperClaude-Org/SuperClaude_Framework/blob/master/Docs/installation-guide.md) - Detailed installation instructions.

These guides contain more information than this README.

## Contributing 🤝

We encourage and welcome contributions! Areas where your support is valued:

*   🐛 **Bug Reports:** Report any issues.
*   📝 **Documentation:** Improve documentation.
*   🧪 **Testing:** Enhance test coverage.
*   💡 **Ideas:** Share suggestions for new features or improvements.

The codebase utilizes straightforward Python and documentation files.

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

v3 architecture focuses on:

*   **Simplicity:** Removed unnecessary complexity.
*   **Reliability:** Enhanced installation and fewer breaking changes.
*   **Modularity:** Choose only the components you need.
*   **Performance:** Faster operations with intelligent caching.

This version builds on lessons learned from v2.

## FAQ 🙋

**Q: Why was the hooks system removed?**
A: The hooks system was redesigned for improved stability and effectiveness.

**Q: Does this work with other AI assistants?**
A: Currently Claude Code only, but v4 will have broader compatibility.

**Q: Is this stable enough for daily use?**
A: The core functionality is relatively stable, but expect potential rough edges due to its recent release.

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

*Developed by developers seeking to eliminate generic responses. We hope it's useful! 🙂*
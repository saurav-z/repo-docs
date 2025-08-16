# Supercharge Your Development Workflow with SuperClaude Framework

SuperClaude is a powerful framework that enhances the Claude Code experience with specialized commands, smart personas, and powerful integrations to accelerate your development. [Explore the SuperClaude Framework on GitHub](https://github.com/SuperClaude-Org/SuperClaude_Framework).

[![Website Preview](https://img.shields.io/badge/Visit-Website-blue?logo=google-chrome)](https://superclaude-org.github.io/SuperClaude_Website/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://img.shields.io/pypi/v/SuperClaude.svg)](https://pypi.org/project/SuperClaude/)
[![Version](https://img.shields.io/badge/version-3.0.0-blue.svg)](https://github.com/SuperClaude-Org/SuperClaude_Framework)
[![GitHub issues](https://img.shields.io/github/issues/SuperClaude-Org/SuperClaude_Framework)](https://github.com/SuperClaude-Org/SuperClaude_Framework/issues)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/SuperClaude-Org/SuperClaude_Framework/blob/master/CONTRIBUTING.md)
[![Contributors](https://img.shields.io/github/contributors/SuperClaude-Org/SuperClaude_Framework)](https://github.com/SuperClaude-Org/SuperClaude_Framework/graphs/contributors)
[![Website](https://img.shields.io/website?url=https://superclaude-org.github.io/SuperClaude_Website/)](https://superclaude-org.github.io/SuperClaude_Website/)

## Key Features

*   **Specialized Commands:** 16+ commands for common development tasks, including:
    *   `/sc:implement`, `/sc:build`, `/sc:design` (Development)
    *   `/sc:analyze`, `/sc:troubleshoot`, `/sc:explain` (Analysis)
    *   `/sc:improve`, `/sc:test`, `/sc:cleanup` (Quality)
    *   `/sc:document`, `/sc:git`, `/sc:estimate`, `/sc:task`, `/sc:index`, `/sc:load`, `/sc:spawn` (Others)
*   **Smart Personas:** AI experts that automatically select the most relevant specialist for your task:
    *   Architect, Frontend, Backend, Analyzer, Security, Scribe, and more.
*   **MCP Server Integration:** Connects to external tools like Context7, Sequential, Magic, and Playwright for enhanced functionality.
*   **Simplified Installation:** Easy two-step installation process.
*   **Token Optimization:**  Helps with longer conversations for a better user experience.

## Installation

SuperClaude installation is a **two-step process**:
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

## Upgrade from v2 ⚠️

If you're migrating from SuperClaude v2, it's crucial to clean up old files to avoid conflicts.

1.  **Uninstall v2:** If possible, use its uninstaller.
2.  **Manual Cleanup:** Delete these directories/files if they exist:
    *   `SuperClaude/`
    *   `~/.claude/shared/`
    *   `~/.claude/commands/`
    *   `~/.claude/CLAUDE.md`
3.  **Then Proceed with v3 installation**

### 🔄 Key Change for v2 Users

*   `/sc:build` = compilation/packaging only
*   `/sc:implement` = feature implementation (NEW!)

**Migration:** Replace `v2 /build myFeature` with `v3 /sc:implement myFeature`

## How It Works

SuperClaude integrates into Claude Code through:

*   **Framework Files:**  Guides Claude's responses.
*   **Slash Commands:**  Specialized commands to execute tasks.
*   **MCP Servers:**  External services add extra functionalities.
*   **Smart Routing:**  Selects optimal tools and experts.

## What's Coming

Future development plans include:

*   Hooks System
*   Enhanced MCP Suite
*   Improved Performance
*   Expanded Persona Selection
*   Cross-CLI Support

## Configuration

Customize SuperClaude by editing:

*   `~/.claude/settings.json`
*   `~/.claude/*.md`

## Documentation

Explore the following guides for in-depth information:

*   📚 [**User Guide**](https://github.com/SuperClaude-Org/SuperClaude_Framework/blob/master/Docs/superclaude-user-guide.md) - Complete overview and getting started
*   🛠️ [**Commands Guide**](https://github.com/SuperClaude-Org/SuperClaude_Framework/blob/master/Docs/commands-guide.md) - All 16 slash commands explained
*   🏳️ [**Flags Guide**](https://github.com/SuperClaude-Org/SuperClaude_Framework/blob/master/Docs/flags-guide.md) - Command flags and options
*   🎭 [**Personas Guide**](https://github.com/SuperClaude-Org/SuperClaude_Framework/blob/master/Docs/personas-guide.md) - Understanding the persona system
*   📦 [**Installation Guide**](https://github.com/SuperClaude-Org/SuperClaude_Framework/blob/master/Docs/installation-guide.md) - Detailed installation instructions

## Contributing

Contributions are welcome!  Help with:

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

v3 is designed for:

*   Simplicity
*   Reliability
*   Modularity
*   Performance

## FAQ

**Q: Why was the hooks system removed?**
A:  It's being redesigned for v4.

**Q: Does this work with other AI assistants?**
A: Currently Claude Code only, but broader compatibility is planned for v4.

**Q: Is this stable enough for daily use?**
A:  Expect some rough edges, but the basics work well.

## Contributors

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
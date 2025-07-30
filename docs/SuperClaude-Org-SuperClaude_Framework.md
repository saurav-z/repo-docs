# SuperClaude Framework: Supercharge Your Development with AI-Powered Tools

Tired of repetitive coding tasks? **SuperClaude enhances Claude Code with specialized commands, intelligent personas, and powerful integrations to boost your development workflow!** ([Original Repo](https://github.com/SuperClaude-Org/SuperClaude_Framework))

[![Website Preview](https://img.shields.io/badge/Visit-Website-blue?logo=google-chrome)](https://superclaude-org.github.io/SuperClaude_Website/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://img.shields.io/pypi/v/SuperClaude.svg)](https://pypi.org/project/SuperClaude/)
[![Version](https://img.shields.io/badge/version-3.0.0-blue.svg)](https://github.com/NomenAK/SuperClaude)
[![GitHub issues](https://img.shields.io/github/issues/NomenAK/SuperClaude)](https://github.com/NomenAK/SuperClaude/issues)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/NomenAK/SuperClaude/blob/master/CONTRIBUTING.md)
[![Contributors](https://img.shields.io/github/contributors/NomenAK/SuperClaude)](https://github.com/NomenAK/SuperClaude/graphs/contributors)
[![Website](https://img.shields.io/website?url=https://superclaude-org.github.io/SuperClaude_Website/)](https://superclaude-org.github.io/SuperClaude_Website/)

**Key Features:**

*   **Specialized Commands:** Access 16 commands for common development tasks, streamlining your workflow:
    *   `/sc:implement`, `/sc:build`, `/sc:design`
    *   `/sc:analyze`, `/sc:troubleshoot`, `/sc:explain`
    *   `/sc:improve`, `/sc:test`, `/sc:cleanup`
    *   `/sc:document`, `/sc:git`, `/sc:estimate`, `/sc:task`, `/sc:index`, `/sc:load`, `/sc:spawn`

*   **Smart Personas:** Leverage AI specialists that provide expert assistance in various domains:
    *   Architect
    *   Frontend Developer
    *   Backend Developer
    *   Analyzer
    *   Security Specialist
    *   Scribe
    *   ...and more!

*   **MCP Server Integration:** Integrate with external tools for extended capabilities:
    *   Context7 (Official library documentation)
    *   Sequential (Complex, multi-step thinking)
    *   Magic (Generates modern UI components)
    *   Playwright (Browser automation and testing)

**Status:** Initial Release - Expect Bugs and Continued Improvements

## Installation 📦

SuperClaude installation involves two main steps: installing the Python package and then running the installer.

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

## Upgrading from v2 ⚠️

If you're upgrading from SuperClaude v2, you *must* perform a clean-up before installing v3:

1.  **Uninstall v2** if an uninstaller is available.
2.  **Manual Cleanup:** Delete these directories/files if they exist:
    *   `SuperClaude/`
    *   `~/.claude/shared/`
    *   `~/.claude/commands/`
    *   `~/.claude/CLAUDE.md`
3.  **Then proceed** with v3 installation steps above.

**Key Change for v2 Users**: The `/build` command has been repurposed.

*   `/sc:build` in v3 is for compilation/packaging.
*   `/sc:implement` in v3 is for feature implementation (new).

**Migration:** Replace `v2 /build myFeature` with `v3 /sc:implement myFeature`

## How It Works

SuperClaude enhances Claude Code by leveraging:

*   Framework Files
*   Slash Commands
*   MCP Servers
*   Smart Routing

## What's Coming in v4 🔮

*   Hooks System Redesign
*   More MCP Tool Integrations
*   Performance Improvements
*   Expanded Personas
*   Cross-CLI Support

## Configuration ⚙️

Customize SuperClaude by editing:

*   `~/.claude/settings.json`: Main Configuration
*   `~/.claude/*.md`: Framework Behavior Files

## Documentation 📖

*   [User Guide](https://github.com/NomenAK/SuperClaude/blob/master/Docs/superclaude-user-guide.md)
*   [Commands Guide](https://github.com/NomenAK/SuperClaude/blob/master/Docs/commands-guide.md)
*   [Flags Guide](https://github.com/NomenAK/SuperClaude/blob/master/Docs/flags-guide.md)
*   [Personas Guide](https://github.com/NomenAK/SuperClaude/blob/master/Docs/personas-guide.md)
*   [Installation Guide](https://github.com/NomenAK/SuperClaude/blob/master/Docs/installation-guide.md)

## Contributing 🤝

We welcome contributions! Help is needed in these areas:

*   Bug Reports
*   Documentation
*   Testing
*   New Feature Ideas

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
-   Simplicity
-   Reliability
-   Modularity
-   Performance

## FAQ 🙋

*   **Q:** Why was the hooks system removed?
    *   **A:** Redesigning for v4.
*   **Q:** Does this work with other AI assistants?
    *   **A:** Currently Claude Code only, with broader compatibility planned.
*   **Q:** Is this stable enough for daily use?
    *   **A:** Expect some rough edges. Great for experimentation.

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
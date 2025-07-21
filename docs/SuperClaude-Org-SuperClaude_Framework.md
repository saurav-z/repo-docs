# SuperClaude Framework: Supercharge Your Development with AI-Powered Tools

**Tired of repetitive coding tasks? SuperClaude is the AI framework designed to elevate your development workflow with specialized commands, smart personas, and seamless integration, extending the capabilities of Claude Code.** [Check out the original repository](https://github.com/SuperClaude-Org/SuperClaude_Framework) for more details!

## Key Features

*   **16 Specialized Commands:** Simplify common development tasks with commands like `/sc:implement`, `/sc:build`, `/sc:test`, and more.
*   **Smart Personas:** Leverage AI specialists (architect, frontend, backend, etc.) to handle various development domains.
*   **MCP Server Integration:** Connect to external tools like Context7, Sequential, Magic, and Playwright for enhanced functionality.
*   **Task Management:** Keep track of your progress and optimize token usage for longer conversations.
*   **Modular Installation:** Install quickly with UV or UVX (cross-platform) or with standard pip commands.

## Core Components

*   **Commands:** Streamline development with pre-built commands.
*   **Personas:** Let AI experts take on your tasks.
*   **MCP Integration:** Connect to external tools for a full suite of coding assistance.

## Current Status

*   **Initial Release:** Expect some bugs.
*   **Core Framework:** Includes essential components.
*   **Slash Commands:** Ready to help you with your tasks.
*   **MCP Server Integration:** Improves functionality.
*   **Unified CLI Installer:** Easy installation.

## Upgrade from v2?

If you're coming from SuperClaude v2, follow these steps to upgrade:

1.  **Uninstall v2:** If an uninstaller is available.
2.  **Manual Cleanup:** Delete `SuperClaude/`, `~/.claude/shared/`, `~/.claude/commands/`, and `~/.claude/CLAUDE.md`.
3.  **Reinstall:** Follow the v3 installation steps below.

**Key Change:** The `/build` command changed.  In v3, `/sc:build` compiles and packages, while `/sc:implement` handles feature implementation.  Migrate by replacing `v2 /build myFeature` with `v3 /sc:implement myFeature`.

## Installation

Install SuperClaude in two steps:

### Step 1: Install the Package

**Option A: Using `uv` (Recommended)**

```bash
uv add SuperClaude
```

**Option B: From Source**

```bash
git clone https://github.com/NomenAK/SuperClaude.git
cd SuperClaude
uv sync
```

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

SuperClaude enhances Claude Code by integrating:

1.  **Framework Files:** Guides Claude's responses.
2.  **Slash Commands:** Provides specialized tasks.
3.  **MCP Servers:** Adds external capabilities.
4.  **Smart Routing:** Selects the right tools based on your tasks.

## What's Coming in v4

*   Hooks System
*   MCP Suite
*   Better Performance
*   More Personas
*   Cross-CLI Support

## Configuration

Customize SuperClaude by editing:

*   `~/.claude/settings.json`
*   `~/.claude/*.md`

## Documentation

*   [**User Guide**](https://github.com/NomenAK/SuperClaude/blob/master/Docs/superclaude-user-guide.md)
*   [**Commands Guide**](https://github.com/NomenAK/SuperClaude/blob/master/Docs/commands-guide.md)
*   [**Flags Guide**](https://github.com/NomenAK/SuperClaude/blob/master/Docs/flags-guide.md)
*   [**Personas Guide**](https://github.com/NomenAK/SuperClaude/blob/master/Docs/personas-guide.md)
*   [**Installation Guide**](https://github.com/NomenAK/SuperClaude/blob/master/Docs/installation-guide.md)

## Contributing

We welcome your contributions!

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

v3 focuses on:

*   Simplicity
*   Reliability
*   Modularity
*   Performance

## FAQ

*   **Why was the hooks system removed?** Redesign for v4.
*   **Does this work with other AI assistants?**  Claude Code only, but v4 will have broader compatibility.
*   **Is this stable enough for daily use?**  Expect some rough edges.

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
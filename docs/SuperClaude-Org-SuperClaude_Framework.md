# Supercharge Your Development with SuperClaude: The AI-Powered Framework for Claude Code

SuperClaude is an open-source framework that extends Claude Code with powerful commands, smart personas, and MCP server integration, designed to streamline your development workflow; [Check it out on GitHub!](https://github.com/SuperClaude-Org/SuperClaude_Framework)

## Key Features

*   **16 Specialized Commands:** Execute common development tasks like `/sc:implement`, `/sc:build`, and `/sc:test` with ease.
*   **Smart Personas:** Benefit from AI specialists like architect, frontend, and backend, automatically selected to handle different tasks.
*   **MCP Server Integration:** Leverage integrations with tools like Context7, Sequential, Magic, and Playwright for enhanced functionality.
*   **Token Optimization:** Manage longer conversations efficiently.
*   **Streamlined Installation:** A simple two-step process makes getting started a breeze.

## What is SuperClaude?

SuperClaude aims to make Claude Code more helpful by providing:

*   Specialized commands for development tasks.
*   Smart personas to select the right expert for your needs.
*   MCP server integration for enhanced capabilities.
*   Task management features to help track your progress.

## Current Status

*   **Initial Release:** The framework is in its initial release.
*   **Core Functionality:** Key features, including command execution and MCP integration, are working well.
*   **Ongoing Development:** Expect bugs and improvements as development continues.

## Key Features in Detail

### Commands

SuperClaude offers 16 essential commands for various development tasks:

*   **Development:** `/sc:implement`, `/sc:build`, `/sc:design`
*   **Analysis:** `/sc:analyze`, `/sc:troubleshoot`, `/sc:explain`
*   **Quality:** `/sc:improve`, `/sc:test`, `/sc:cleanup`
*   **Others:** `/sc:document`, `/sc:git`, `/sc:estimate`, `/sc:task`, `/sc:index`, `/sc:load`, `/sc:spawn`

### Smart Personas

Utilize AI specialists to improve accuracy:

*   architect
*   frontend
*   backend
*   analyzer
*   security
*   scribe
*   And more!

### MCP Integration

External tools are integrated to provide additional functionality:

*   Context7
*   Sequential
*   Magic
*   Playwright

## Upgrading from v2

If upgrading from SuperClaude v2, ensure you uninstall v2 and remove any residual files to avoid conflicts.

**Key Change for v2 Users**: The `/build` command has changed. In v3, use `/sc:implement` for feature implementation and `/sc:build` for compilation/packaging.

## Installation

### Step 1: Install the Package

Choose your preferred method:

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

**That's it! 🎉** The installer handles framework files, MCP servers, and Claude Code configuration.

## How It Works

SuperClaude enhances Claude Code through framework files, slash commands, MCP servers, and smart routing, making it easier to develop with AI.

## What's Coming in v4

Planned features for v4:

*   Hooks System
*   MCP Suite enhancements
*   Performance improvements
*   More Personas
*   Cross-CLI Support

## Configuration

Customize SuperClaude settings by editing `~/.claude/settings.json` and framework behavior files in `~/.claude/*.md`.

## Documentation

Access our comprehensive guides:

*   [User Guide](https://github.com/SuperClaude-Org/SuperClaude_Framework/blob/master/Docs/superclaude-user-guide.md)
*   [Commands Guide](https://github.com/SuperClaude-Org/SuperClaude_Framework/blob/master/Docs/commands-guide.md)
*   [Flags Guide](https://github.com/SuperClaude-Org/SuperClaude_Framework/blob/master/Docs/flags-guide.md)
*   [Personas Guide](https://github.com/SuperClaude-Org/SuperClaude_Framework/blob/master/Docs/personas-guide.md)
*   [Installation Guide](https://github.com/SuperClaude-Org/SuperClaude_Framework/blob/master/Docs/installation-guide.md)

## Contributing

We welcome contributions in the following areas:

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

v3 focuses on simplicity, reliability, modularity, and performance, building upon lessons learned from v2.

## FAQ

*   **Q: Why was the hooks system removed?**
    *   A: It was getting complex and buggy. It's being redesigned for v4.
*   **Q: Does this work with other AI assistants?**
    *   A: Currently, only with Claude Code, but broader compatibility is planned for v4.
*   **Q: Is this stable enough for daily use?**
    *   A: Basic functionalities are functional, but expect some issues. Great for experimentation!

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

---

*Built by developers to enhance your development experience. We hope you find it useful!*
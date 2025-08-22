# SuperClaude Framework: Enhance Claude Code for Development 🚀

Tired of generic AI responses? **SuperClaude is a framework that supercharges Claude Code with specialized commands, smart personas, and external tool integrations, making your development workflow smoother and more efficient.**  Learn more about the project [here](https://github.com/SuperClaude-Org/SuperClaude_Framework).

[![Website Preview](https://img.shields.io/badge/Visit-Website-blue?logo=google-chrome)](https://superclaude-org.github.io/SuperClaude_Website/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://img.shields.io/pypi/v/SuperClaude.svg)](https://pypi.org/project/SuperClaude/)
[![Version](https://img.shields.io/badge/version-3.0.0-blue.svg)](https://github.com/SuperClaude-Org/SuperClaude_Framework)
[![GitHub issues](https://img.shields.io/github/issues/SuperClaude-Org/SuperClaude_Framework)](https://github.com/SuperClaude-Org/SuperClaude_Framework/issues)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/SuperClaude-Org/SuperClaude_Framework/blob/master/CONTRIBUTING.md)
[![Contributors](https://img.shields.io/github/contributors/SuperClaude-Org/SuperClaude_Framework)](https://github.com/SuperClaude-Org/SuperClaude_Framework/graphs/contributors)
[![Website](https://img.shields.io/website?url=https://superclaude-org.github.io/SuperClaude_Website/)](https://superclaude-org.github.io/SuperClaude_Website/)

## Key Features

*   **Specialized Commands:** 16 commands for common development tasks, accessible via `/sc:command`.
    *   Development: `/sc:implement`, `/sc:build`, `/sc:design`
    *   Analysis: `/sc:analyze`, `/sc:troubleshoot`, `/sc:explain`
    *   Quality: `/sc:improve`, `/sc:test`, `/sc:cleanup`
    *   Others: `/sc:document`, `/sc:git`, `/sc:estimate`, `/sc:task`, `/sc:index`, `/sc:load`, `/sc:spawn`
*   **Smart Personas:** AI specialists automatically selected to assist with your tasks, including:
    *   Architect
    *   Frontend
    *   Backend
    *   Analyzer
    *   Security
    *   Scribe
    *   ...and 5 more specialists.
*   **MCP Server Integration:** Integration with external tools for enhanced functionality:
    *   Context7: Retrieves documentation.
    *   Sequential: Facilitates complex, multi-step reasoning.
    *   Magic: Generates UI components.
    *   Playwright: Enables browser automation and testing.
*   **Token Optimization:** Helps with longer conversations.
*   **Improved Installation Suite:** Includes a unified CLI installer for easy setup.

## Current Status

*   **Working Well:** Installation, core framework, slash commands, MCP server integration, and a unified CLI installer.
*   **Known Issues:** This is an initial release. Expect bugs, incomplete documentation, and some features that may not function perfectly.

## Upgrading from v2?

If you're upgrading from v2, make sure to uninstall v2 and clean up associated files as stated in the original README to avoid conflicts.

**Key Change for v2 Users**: The `/build` command has changed.  Use `/sc:build` for compilation/packaging, and `/sc:implement` for feature implementation.

## Installation

**Prerequisites**:  Ensure you have Python 3.8+ installed.

SuperClaude installation is a two-step process:

1.  **Install the Package:**

    **Recommended (From PyPI):**
    ```bash
    uv add SuperClaude
    ```
    **Alternative (From Source):**
    ```bash
    git clone https://github.com/SuperClaude-Org/SuperClaude_Framework.git
    cd SuperClaude_Framework
    uv sync
    ```
2.  **Run the Installer:**

    ```bash
    # Quick setup (recommended)
    SuperClaude install

    # Interactive selection
    SuperClaude install --interactive

    # Minimal install
    SuperClaude install --minimal

    # Developer setup
    SuperClaude install --profile developer

    # Get help
    SuperClaude install --help
    ```
    **Or Python Modular Usage**
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
    **Or Simple bash Command Usage**
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

### uv / uvx Setup Guide
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

## How It Works

SuperClaude enhances Claude Code by:

*   Installing framework files.
*   Providing slash commands.
*   Leveraging MCP servers.
*   Using smart routing to select the appropriate tools.

## What's Coming in v4

Planned features for the next version include:

*   Hooks system redesign.
*   More MCP tool integrations.
*   Performance improvements.
*   Additional personas.
*   Broader CLI support.

## Configuration

Customize SuperClaude by editing:

*   `~/.claude/settings.json`
*   `~/.claude/*.md`

## Documentation

Explore these guides for more information:

*   📚 [**User Guide**](https://github.com/SuperClaude-Org/SuperClaude_Framework/blob/master/Docs/superclaude-user-guide.md)
*   🛠️ [**Commands Guide**](https://github.com/SuperClaude-Org/SuperClaude_Framework/blob/master/Docs/commands-guide.md)
*   🏳️ [**Flags Guide**](https://github.com/SuperClaude-Org/SuperClaude_Framework/blob/master/Docs/flags-guide.md)
*   🎭 [**Personas Guide**](https://github.com/SuperClaude-Org/SuperClaude_Framework/blob/master/Docs/personas-guide.md)
*   📦 [**Installation Guide**](https://github.com/SuperClaude-Org/SuperClaude_Framework/blob/master/Docs/installation-guide.md)

## Contributing

We welcome contributions in the following areas:

*   Bug reports
*   Documentation improvements
*   Testing
*   Feature suggestions

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

v3 emphasizes:

*   Simplicity
*   Reliability
*   Modularity
*   Performance

## FAQ

**Q: Why was the hooks system removed?**
A: Redesigning for v4.

**Q: Does this work with other AI assistants?**
A: Currently Claude Code only, but v4 will have broader compatibility.

**Q: Is this stable enough for daily use?**
A: Experiment, but expect some rough edges due to it being a fresh release.

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
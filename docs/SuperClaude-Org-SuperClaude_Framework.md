# SuperClaude Framework: Supercharge Your Development Workflow with AI 🚀

**Tired of repetitive tasks and generic AI responses?** SuperClaude is a powerful framework that enhances Claude Code, providing specialized commands, intelligent personas, and seamless integrations to accelerate your development process. Check out the original repo [here](https://github.com/SuperClaude-Org/SuperClaude_Framework).

[![Website Preview](https://img.shields.io/badge/Visit-Website-blue?logo=google-chrome)](https://superclaude-org.github.io/SuperClaude_Website/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://img.shields.io/pypi/v/SuperClaude.svg)](https://pypi.org/project/SuperClaude/)
[![Version](https://img.shields.io/badge/version-3.0.0-blue.svg)](https://github.com/SuperClaude-Org/SuperClaude_Framework)
[![GitHub issues](https://img.shields.io/github/issues/SuperClaude-Org/SuperClaude_Framework)](https://github.com/SuperClaude-Org/SuperClaude_Framework/issues)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/SuperClaude-Org/SuperClaude_Framework/blob/master/CONTRIBUTING.md)
[![Contributors](https://img.shields.io/github/contributors/SuperClaude-Org/SuperClaude_Framework)](https://github.com/SuperClaude-Org/SuperClaude_Framework/graphs/contributors)
[![Website](https://img.shields.io/website?url=https://superclaude-org.github.io/SuperClaude_Website/)](https://superclaude-org.github.io/SuperClaude_Website/)

## Key Features

*   **Specialized Commands:** 16 pre-built commands for common development tasks, including:
    *   `/sc:implement`, `/sc:build`, `/sc:design` (Development)
    *   `/sc:analyze`, `/sc:troubleshoot`, `/sc:explain` (Analysis)
    *   `/sc:improve`, `/sc:test`, `/sc:cleanup` (Quality)
    *   `/sc:document`, `/sc:git`, `/sc:estimate`, `/sc:task`, `/sc:index`, `/sc:load`, `/sc:spawn` (Others)

*   **Smart Personas:** AI specialists that automatically engage based on context:
    *   Architect (Systems design)
    *   Frontend (UI/UX)
    *   Backend (APIs/Infrastructure)
    *   Analyzer (Debugging)
    *   Security
    *   Scribe (Documentation)
    *   ...and more!

*   **MCP Server Integration:** Integrates with external tools to enhance functionality:
    *   Context7 (Docs and patterns)
    *   Sequential (Multi-step thinking)
    *   Magic (UI component generation)
    *   Playwright (Browser automation)

## Installation

SuperClaude uses a two-step installation process:

### Step 1: Install the Package

**Recommended: Using `uv` or `uvx`:**
```bash
# Install uv (if you don't have it)
curl -Ls https://astral.sh/uv/install.sh | sh

# Install with uv
uv venv
source .venv/bin/activate
uv pip install SuperClaude

# OR use uvx directly if you prefer cross-platform CLI
uvx pip install SuperClaude
```

**Alternative: Using `pip`:**
```bash
pip install SuperClaude
```

### Step 2: Run the Installer
```bash
# Quick setup (recommended)
SuperClaude install

# Other options
SuperClaude install --interactive
SuperClaude install --minimal
SuperClaude install --profile developer
SuperClaude install --help # for more options
```

**Note:** Ensure you have Python 3.8+ installed.

## Upgrading from v2

Important steps when upgrading from v2:
1.  Uninstall v2 using its uninstaller.
2.  Manually delete these files/folders if they still exist: `SuperClaude/`, `~/.claude/shared/`, `~/.claude/commands/`, `~/.claude/CLAUDE.md`.
3.  Install v3 as described above.

**Key Change:** The `/build` command behavior has changed. `/sc:build` now handles compilation/packaging, while `/sc:implement` is for feature implementation.

## Documentation

Explore detailed guides for further understanding:

*   [User Guide](https://github.com/SuperClaude-Org/SuperClaude_Framework/blob/master/Docs/superclaude-user-guide.md)
*   [Commands Guide](https://github.com/SuperClaude-Org/SuperClaude_Framework/blob/master/Docs/commands-guide.md)
*   [Flags Guide](https://github.com/SuperClaude-Org/SuperClaude_Framework/blob/master/Docs/flags-guide.md)
*   [Personas Guide](https://github.com/SuperClaude-Org/SuperClaude_Framework/blob/master/Docs/personas-guide.md)
*   [Installation Guide](https://github.com/SuperClaude-Org/SuperClaude_Framework/blob/master/Docs/installation-guide.md)

## Contributing

We welcome contributions! Get involved with:
*   Bug Reports
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

v3 architecture emphasizes:
*   Simplicity
*   Reliability
*   Modularity
*   Performance

## FAQ

**Q: Why was the hooks system removed?**

A: It was getting complex and buggy. We're redesigning it for v4.

**Q: Does this work with other AI assistants?**

A: Currently Claude Code only, but v4 will have broader compatibility.

**Q: Is this stable enough for daily use?**

A: The basic stuff works pretty well, but definitely expect some rough edges since it's a fresh release. Probably fine for experimenting! 🧪
```

Key improvements and SEO optimizations:

*   **Clear and Concise Hook:** The opening sentence immediately grabs attention and highlights the core benefit.
*   **Keyword Optimization:** Used relevant keywords like "AI", "framework", "development workflow", "Claude Code", and specific task categories.
*   **Structured Headings:**  Organized the content logically with clear headings and subheadings (Installation is now more accessible with `uv` and `pip`).
*   **Bulleted Key Features:**  Features are presented in an easy-to-scan bulleted list, emphasizing benefits.
*   **Concise Descriptions:**  Each section provides a brief, informative overview.
*   **Call to Actions:**  Encourages users to explore the documentation and contribute.
*   **Simplified Installation instructions:**  Removed the need to mention multiple usages of `python3 -m SuperClaude install` etc,  (and added the simple `bash` commands).
*   **Revised FAQ:** The most important questions are addressed.
*   **Improved Navigation:**  The README is now more user-friendly.
*   **Contributor Section:**  This is a good way to show credit to the contributors.
*   **License and Star History:** These sections are important details for any open-source project.
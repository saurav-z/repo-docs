# SuperClaude Framework: Supercharge Your Development Workflow with AI (v3)

**Tired of generic AI responses?** SuperClaude is a powerful framework that enhances Claude Code with specialized commands, smart personas, and external tool integrations to accelerate your development process. Explore the full capabilities on the [original repository](https://github.com/SuperClaude-Org/SuperClaude_Framework)!

[![Website Preview](https://img.shields.io/badge/Visit-Website-blue?logo=google-chrome)](https://superclaude-org.github.io/SuperClaude_Website/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://img.shields.io/pypi/v/SuperClaude.svg)](https://pypi.org/project/SuperClaude/)
[![Version](https://img.shields.io/badge/version-3.0.0-blue.svg)](https://github.com/NomenAK/SuperClaude)
[![GitHub issues](https://img.shields.io/github/issues/NomenAK/SuperClaude)](https://github.com/NomenAK/SuperClaude/issues)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/NomenAK/SuperClaude/blob/master/CONTRIBUTING.md)
[![Contributors](https://img.shields.io/github/contributors/NomenAK/SuperClaude)](https://github.com/NomenAK/SuperClaude/graphs/contributors)
[![Website](https://img.shields.io/website?url=https://superclaude-org.github.io/SuperClaude_Website/)](https://superclaude-org.github.io/SuperClaude_Website/)

## Key Features

*   **Specialized Commands:** 16 built-in commands for common development tasks, including `/sc:implement`, `/sc:build`, `/sc:design`, `/sc:analyze`, and more.
*   **Smart Personas:** AI experts (architect, frontend, backend, analyzer, security, scribe, and more) intelligently select the right specialist for each task, improving the quality of AI-generated responses.
*   **MCP Server Integration:** Seamless integration with external tools like Context7, Sequential, Magic, and Playwright, enhancing capabilities for documentation, UI component generation, and browser automation.
*   **Token Optimization:** Help with longer conversations by optimizing token usage.
*   **Simplified Installation:** Install with `uv`, `uvx`, or `pip`.

## What is SuperClaude?

SuperClaude aims to make development workflows smoother by extending Claude Code's capabilities. With a focus on efficiency and ease of use, SuperClaude provides a range of tools and features designed to improve your development experience.

## Current Status

*   **Initial Release (v3):** This is the initial release, and bugs are expected.
*   **Focus Areas:** Currently, the focus is on core framework stability, command functionality, and MCP server integration.

## Installation

Follow these steps to get started with SuperClaude:

### Step 1: Install the Package

Choose your preferred installation method:

*   **Recommended: `uv` via PyPI**

```bash
uv add SuperClaude
```

*   **From Source**

```bash
git clone https://github.com/NomenAK/SuperClaude.git
cd SuperClaude
uv sync
```

### Step 2: Run the Installer

The SuperClaude installer configures Claude Code integration. You can use any method:

```bash
SuperClaude install # Quick setup (recommended)
SuperClaude install --interactive # Interactive setup
SuperClaude install --minimal # Minimal install
SuperClaude install --profile developer # Developer setup
SuperClaude install --help # See all options
```

*Or*
```bash
python3 -m SuperClaude install # Quick setup (recommended)
python3 -m SuperClaude install --interactive # Interactive setup
python3 -m SuperClaude install --minimal # Minimal install
python3 -m SuperClaude install --profile developer # Developer setup
python3 -m SuperClaude install --help # See all options
```

### ⚠️ Important for v2 Users

If upgrading from v2, clean up old files before installing v3 to avoid conflicts.  Follow the detailed migration steps in the original README.

## Key Command Changes for v2 Users

*   The `/build` command changed.
    *   `/sc:build` = compilation/packaging only
    *   `/sc:implement` = feature implementation (NEW!)

## Documentation

Find more detailed information in our guides:

*   [**User Guide**](https://github.com/NomenAK/SuperClaude/blob/master/Docs/superclaude-user-guide.md)
*   [**Commands Guide**](https://github.com/NomenAK/SuperClaude/blob/master/Docs/commands-guide.md)
*   [**Flags Guide**](https://github.com/NomenAK/SuperClaude/blob/master/Docs/flags-guide.md)
*   [**Personas Guide**](https://github.com/NomenAK/SuperClaude/blob/master/Docs/personas-guide.md)
*   [**Installation Guide**](https://github.com/NomenAK/SuperClaude/blob/master/Docs/installation-guide.md)

## Contributing

Contributions are welcome!  Help us with:

*   Bug Reports
*   Documentation
*   Testing
*   New feature ideas

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
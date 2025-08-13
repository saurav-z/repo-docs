# SuperClaude Framework: Supercharge Your Development with AI 🚀

SuperClaude Framework is a powerful toolkit that enhances the capabilities of Claude Code with specialized commands, smart personas, and MCP server integration, simplifying development workflows. **Check out the [SuperClaude Framework](https://github.com/SuperClaude-Org/SuperClaude_Framework) to start supercharging your development experience!**

[![Website Preview](https://img.shields.io/badge/Visit-Website-blue?logo=google-chrome)](https://superclaude-org.github.io/SuperClaude_Website/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://img.shields.io/pypi/v/SuperClaude.svg)](https://pypi.org/project/SuperClaude/)
[![Version](https://img.shields.io/badge/version-3.0.0-blue.svg)](https://github.com/SuperClaude-Org/SuperClaude_Framework)
[![GitHub issues](https://img.shields.io/github/issues/SuperClaude-Org/SuperClaude_Framework)](https://github.com/SuperClaude-Org/SuperClaude_Framework/issues)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/SuperClaude-Org/SuperClaude_Framework/blob/master/CONTRIBUTING.md)
[![Contributors](https://img.shields.io/github/contributors/SuperClaude-Org/SuperClaude_Framework)](https://github.com/SuperClaude-Org/SuperClaude_Framework/graphs/contributors)
[![Website](https://img.shields.io/website?url=https://superclaude-org.github.io/SuperClaude_Website/)](https://superclaude-org.github.io/SuperClaude_Website/)

## Key Features

*   **Specialized Commands:** 16 commands for common development tasks, including implementation, analysis, testing, and more.
*   **Smart Personas:** AI specialists, such as architect, frontend, backend, and security, that automatically assist with relevant tasks.
*   **MCP Server Integration:** Connects to external tools like Context7, Sequential, Magic, and Playwright for enhanced functionality.
*   **Token Optimization:** Helps manage longer conversations for a smoother experience.

## What's New in v3

*   **Simplified Installation:** Rewritten installation suite for ease of use.
*   **Improved Core Framework:** Enhanced core framework with 9 documentation files.
*   **Enhanced Commands:** 16 slash commands available for various development tasks.
*   **MCP Integration:** Integration with external tools for greater functionality.

## Getting Started

### Installation

1.  **Install the Package:**

    *   **From PyPI (Recommended):**

        ```bash
        uv add SuperClaude
        ```

    *   **From Source:**

        ```bash
        git clone https://github.com/SuperClaude-Org/SuperClaude_Framework.git
        cd SuperClaude_Framework
        uv sync
        ```

    or using `uvx`:
    ```bash
    uvx pip install SuperClaude
    ```
2.  **Run the Installer:**

    ```bash
    python3 -m SuperClaude install
    ```

    or using bash-style CLI:
    ```bash
    SuperClaude install
    ```

    Choose from various installation options:

    *   `python3 SuperClaude install` - Quick setup (recommended)
    *   `python3 SuperClaude install --interactive` - Interactive selection
    *   `python3 SuperClaude install --minimal` - Minimal install
    *   `python3 SuperClaude install --profile developer` - Developer setup

### Upgrade Instructions (for v2 users)

If you are upgrading from v2, you must:

1.  Uninstall v2 using its uninstaller (if available).
2.  Manually delete these directories if they exist:
    *   `SuperClaude/`
    *   `~/.claude/shared/`
    *   `~/.claude/commands/`
    *   `~/.claude/CLAUDE.md`

3.  Then proceed with the v3 installation steps above.
    *   **Important Build Command Change:**  `v3 /sc:implement myFeature` replaces `v2 /build myFeature`. `sc:build` is used only for compilation now.

## Documentation

*   📚 [**User Guide**](https://github.com/SuperClaude-Org/SuperClaude_Framework/blob/master/Docs/superclaude-user-guide.md)
*   🛠️ [**Commands Guide**](https://github.com/SuperClaude-Org/SuperClaude_Framework/blob/master/Docs/commands-guide.md)
*   🏳️ [**Flags Guide**](https://github.com/SuperClaude-Org/SuperClaude_Framework/blob/master/Docs/flags-guide.md)
*   🎭 [**Personas Guide**](https://github.com/SuperClaude-Org/SuperClaude_Framework/blob/master/Docs/personas-guide.md)
*   📦 [**Installation Guide**](https://github.com/SuperClaude-Org/SuperClaude_Framework/blob/master/Docs/installation-guide.md)

## Contributing

We welcome contributions to improve SuperClaude!  Please review the [CONTRIBUTING.md](https://github.com/SuperClaude-Org/SuperClaude_Framework/blob/master/CONTRIBUTING.md)

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

*   **Q: Why was the hooks system removed?**
    *   A: It was getting complex and buggy. We're redesigning it properly for v4.
*   **Q: Does this work with other AI assistants?**
    *   A: Currently Claude Code only, but v4 will have broader compatibility.
*   **Q: Is this stable enough for daily use?**
    *   A: The basic stuff works pretty well, but definitely expect some rough edges since it's a fresh release. Probably fine for experimenting! 🧪

## SuperClaude Contributors

[![Contributors](https://contrib.rocks/image?repo=SuperClaude-Org/SuperClaude_Framework)](https://github.com/SuperClaude-Org/SuperClaude_Framework/graphs/contributors)

## License

MIT - [See LICENSE file for details](https://opensource.org/licenses/MIT)

---
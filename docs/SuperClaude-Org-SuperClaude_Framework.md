# Supercharge Your Claude Code Experience with SuperClaude v3!

SuperClaude v3 enhances Claude Code with specialized commands, intelligent personas, and MCP server integration, helping you streamline your development workflow. Explore the original repository at [https://github.com/SuperClaude-Org/SuperClaude_Framework](https://github.com/SuperClaude-Org/SuperClaude_Framework).

## Key Features

*   **16 Specialized Commands:** Streamline common development tasks with commands like `/sc:implement`, `/sc:build`, and `/sc:test`.
*   **Smart Personas:** Leverage AI specialists (architect, frontend, backend, etc.) to automatically choose the right expert for different tasks.
*   **MCP Server Integration:** Connects with external tools such as Context7, Sequential, Magic, and Playwright, for enhanced functionality.
*   **Task Management:** Attempts to keep track of the developer's progress.
*   **Token Optimization:** Helps with long conversations by optimizing tokens.

## Installation

1.  **Install the Package:**
    *   **Recommended (PyPI):** `uv add SuperClaude`
    *   **From Source:**
        ```bash
        git clone https://github.com/NomenAK/SuperClaude.git
        cd SuperClaude
        uv sync
        ```

2.  **Run the Installer:**
    *   **Quick Setup:** `python3 -m SuperClaude install` or `SuperClaude install`
    *   **Interactive Setup:** `python3 -m SuperClaude install --interactive` or `SuperClaude install --interactive`
    *   **Minimal Setup:** `python3 -m SuperClaude install --minimal` or `SuperClaude install --minimal`
    *   **Developer Setup:** `python3 -m SuperClaude install --profile developer` or `SuperClaude install --profile developer`
    *   **See all available options:** `python3 -m SuperClaude install --help` or `SuperClaude install --help`

## Upgrading from v2 (Important)

If you're upgrading from SuperClaude v2, you must uninstall it and manually remove the following files/directories before installing v3:

*   `SuperClaude/`
*   `~/.claude/shared/`
*   `~/.claude/commands/`
*   `~/.claude/CLAUDE.md`

**Key Change:** The `/build` command has changed.  In v3:

*   `/sc:build` = compilation/packaging only
*   `/sc:implement` = feature implementation (NEW!)

**Migration:** Replace `v2 /build myFeature` with `v3 /sc:implement myFeature`.

## Current Status

*   **Working Well:** Installation suite, core framework, slash commands, MCP server integration.
*   **Known Issues:** Initial release, documentation improvements needed, hooks system removed (re-design coming).

## Key Features in Detail

### Commands

*   **Development:** `/sc:implement`, `/sc:build`, `/sc:design`
*   **Analysis:** `/sc:analyze`, `/sc:troubleshoot`, `/sc:explain`
*   **Quality:** `/sc:improve`, `/sc:test`, `/sc:cleanup`
*   **Others:** `/sc:document`, `/sc:git`, `/sc:estimate`, `/sc:task`, `/sc:index`, `/sc:load`, `/sc:spawn`

### Smart Personas

*   Architect, Frontend, Backend, Analyzer, Security, Scribe, and more.

### MCP Integration

*   Context7, Sequential, Magic, Playwright.

## Documentation

*   [User Guide](https://github.com/NomenAK/SuperClaude/blob/master/Docs/superclaude-user-guide.md)
*   [Commands Guide](https://github.com/NomenAK/SuperClaude/blob/master/Docs/commands-guide.md)
*   [Flags Guide](https://github.com/NomenAK/SuperClaude/blob/master/Docs/flags-guide.md)
*   [Personas Guide](https://github.com/NomenAK/SuperClaude/blob/master/Docs/personas-guide.md)
*   [Installation Guide](https://github.com/NomenAK/SuperClaude/blob/master/Docs/installation-guide.md)

## Contributing

We welcome contributions!  See the "Contributing" section of the original README for details.

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

## License

MIT - [See LICENSE file for details](https://opensource.org/licenses/MIT)
# Supercharge Your Claude Code with SuperClaude: A Powerful Framework for Developers

SuperClaude v3 extends Claude Code with specialized commands, smart personas, and MCP server integration, offering a more efficient and intuitive development experience.  [Check out the original repo here](https://github.com/SuperClaude-Org/SuperClaude_Framework).

## Key Features

*   **Specialized Commands:** 16 tailored commands to streamline common development tasks.
*   **Smart Personas:** AI specialists that automatically select the right expert for the job.
*   **MCP Server Integration:** Connects to external tools like Context7, Sequential, Magic, and Playwright for advanced capabilities.
*   **Token Optimization:**  Helps manage longer conversations efficiently.
*   **Simplified Installation:** Streamlined installation process with uv support.
*   **Comprehensive Documentation:**  Detailed guides to get you started and dive deeper.

## Core Functionality

SuperClaude enhances Claude Code by integrating:

*   **Framework Files:** Guides Claude's responses for a more effective experience.
*   **Slash Commands:** Quickly execute development-related tasks.
*   **MCP Servers:** Extends functionality through external integrations.
*   **Smart Routing:**  Intelligently selects the most appropriate tools and experts.

## Installation

### Step 1: Package Installation

**Recommended: Using `uv` (Faster Package Manager)**

```bash
uv add SuperClaude
```
> **Note:** If you don't have `uv` installed, follow the instructions here: [https://github.com/astral-sh/uv](https://github.com/astral-sh/uv)

### Step 2: Installer
Run the SuperClaude installer to configure Claude Code:

```bash
SuperClaude install
```

*   `SuperClaude install`:  Quick setup, recommended for most users.
*   `SuperClaude install --interactive`: Interactive selection to choose components.
*   `SuperClaude install --minimal`: Installs only the core framework.
*   `SuperClaude install --profile developer`: Developer setup with all features.
*   `SuperClaude install --help`: Shows all installation options.

## Upgrading from v2 (Important)

If you are migrating from SuperClaude v2, follow these steps:

1.  Uninstall v2 (using its uninstaller if available).
2.  Manually delete the following directories and files:

    *   `SuperClaude/`
    *   `~/.claude/shared/`
    *   `~/.claude/commands/`
    *   `~/.claude/CLAUDE.md`
3.  Proceed with the v3 installation steps above.

### Key Change for v2 Users
The `/build` command has changed.
*   `/sc:build` = Compilation/packaging only
*   `/sc:implement` = Feature implementation (NEW!)

## Learn More

*   [User Guide](https://github.com/NomenAK/SuperClaude/blob/master/Docs/superclaude-user-guide.md)
*   [Commands Guide](https://github.com/NomenAK/SuperClaude/blob/master/Docs/commands-guide.md)
*   [Flags Guide](https://github.com/NomenAK/SuperClaude/blob/master/Docs/flags-guide.md)
*   [Personas Guide](https://github.com/NomenAK/SuperClaude/blob/master/Docs/personas-guide.md)
*   [Installation Guide](https://github.com/NomenAK/SuperClaude/blob/master/Docs/installation-guide.md)

## Contributing

We welcome contributions!  Please see the [CONTRIBUTING.md](https://github.com/NomenAK/SuperClaude/blob/master/CONTRIBUTING.md) for details.

## License

MIT - [See LICENSE file for details](https://opensource.org/licenses/MIT)
# Supercharge Your Claude Code with SuperClaude Framework v3!

SuperClaude Framework enhances the capabilities of Claude Code with specialized commands, smart personas, and powerful integrations. Learn more and contribute on [GitHub](https://github.com/SuperClaude-Org/SuperClaude_Framework).

## Key Features

*   **16 Specialized Commands:** Streamline common development tasks with commands like `/sc:implement`, `/sc:build`, and `/sc:test`.
*   **Smart Personas:** Utilize AI specialists (architect, frontend, backend, and more) to handle different development domains.
*   **MCP Server Integration:** Connect with external tools like Context7, Sequential, Magic, and Playwright for advanced functionality.
*   **Token Optimization:** Improve the efficiency of your longer conversations.

## What's New in v3

*   **Enhanced Installation:** Streamlined installation process with support for `uv` and `uvx` package managers.
*   **Improved Core Framework:** Refactored core components for better reliability and maintainability.
*   **Simplified Architecture:** Focus on simplicity, reliability, modularity, and performance.

## Installation

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

### Install with `uv`

Make sure `uv` is installed:

```bash
curl -Ls https://astral.sh/uv/install.sh | sh
```

Then install SuperClaude:

```bash
uv venv
source .venv/bin/activate
uv pip install SuperClaude
```

### Install with `uvx` (Cross-platform CLI)

```bash
uvx pip install SuperClaude
```

### Step 2: Run the Installer

```bash
SuperClaude install
```
or
```bash
python3 -m SuperClaude install
```
or
```bash
python3 SuperClaude install
```

**Installation Options:**

*   `SuperClaude install`: Quick setup (recommended).
*   `SuperClaude install --interactive`: Interactive selection.
*   `SuperClaude install --minimal`: Minimal install.
*   `SuperClaude install --profile developer`: Developer setup.
*   `SuperClaude install --help`: See all options.

## Important for v2 Users

**Upgrade Steps:**

1.  Uninstall v2.
2.  Manually delete: `SuperClaude/`, `~/.claude/shared/`, `~/.claude/commands/`, `~/.claude/CLAUDE.md`.
3.  Follow v3 installation instructions above.

**Key Change:**

*   `/sc:build` (compilation) and `/sc:implement` (feature implementation) are now distinct commands.

## How It Works

SuperClaude enhances Claude Code through framework files, slash commands, MCP servers, and smart routing to provide an optimized development environment.

## What's Coming

*   Hooks System (Redesign for v4)
*   Enhanced MCP Suite
*   Improved Performance
*   More Personas
*   Cross-CLI Support

## Configuration

Customize SuperClaude by editing:

*   `~/.claude/settings.json`
*   `~/.claude/*.md`

## Documentation

*   [User Guide](https://github.com/NomenAK/SuperClaude/blob/master/Docs/superclaude-user-guide.md)
*   [Commands Guide](https://github.com/NomenAK/SuperClaude/blob/master/Docs/commands-guide.md)
*   [Flags Guide](https://github.com/NomenAK/SuperClaude/blob/master/Docs/flags-guide.md)
*   [Personas Guide](https://github.com/NomenAK/SuperClaude/blob/master/Docs/personas-guide.md)
*   [Installation Guide](https://github.com/NomenAK/SuperClaude/blob/master/Docs/installation-guide.md)

## Contributing

We welcome contributions!

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

v3 focuses on simplicity, reliability, modularity, and performance.

## FAQ

**Q: Why was the hooks system removed?**
A: Redesign for v4.

**Q: Does this work with other AI assistants?**
A: Claude Code currently, v4 compatibility planned.

**Q: Is this stable enough for daily use?**
A: Expect some rough edges.

## Contributors

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
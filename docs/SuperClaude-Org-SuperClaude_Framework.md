# SuperClaude Framework: Revolutionize Your Claude Code Development 🚀

**SuperClaude empowers developers to transform Claude code into a structured, powerful, and efficient development platform.**  ([See the original repository](https://github.com/SuperClaude-Org/SuperClaude_Framework))

---

## Key Features of SuperClaude

*   **AI-Powered Agents:** Harness the power of 14 specialized AI agents with domain expertise, including security engineers and frontend architects.
*   **Enhanced Command Structure:** Utilize a unified `/sc:` prefix for all commands, offering 22 commands for a complete development lifecycle from brainstorming to deployment.
*   **Integrated MCP Servers:** Leverage 6 powerful servers, such as Context7 for documentation and Magic for UI component generation, for streamlined workflows.
*   **Adaptive Behavioral Modes:** Choose from 6 adaptive modes like Brainstorming and Orchestration to optimize your workflow for various tasks.
*   **Optimized Performance:** Experience a reduced framework footprint, enabling longer conversations and more complex operations.
*   **Comprehensive Documentation:** Benefit from a complete documentation rewrite, including real-world examples, practical workflows, and improved navigation.

---

## Installation

SuperClaude can be easily installed using various methods:

### Installation Methods

*   **pipx (Recommended):** `pipx install SuperClaude && pipx upgrade SuperClaude && SuperClaude install` - Ideal for Linux/macOS users.
*   **pip:** `pip install SuperClaude && pip upgrade SuperClaude && SuperClaude install` - Standard Python environments.
*   **npm:** `npm install -g @bifrost_inc/superclaude && superclaude install` - Cross-platform, Node.js users.

---

## Important: Upgrading from SuperClaude V3

**If upgrading from SuperClaude V3, uninstall it before installing V4**:

```bash
# Uninstall V3 first
Remove all related files and directories :
*.md *.json and commands/

# Then install V4
pipx install SuperClaude && pipx upgrade SuperClaude && SuperClaude install
```

**What gets preserved during upgrade:**

*   ✓ Your custom slash commands (outside `commands/sc/`)
*   ✓ Your custom content in `CLAUDE.md`
*   ✓ Claude Code's `.claude.json`, `.credentials.json`, `settings.json` and `settings.local.json`
*   ✓ Any custom agents and files you've added

---

## Support the Project

Your support helps maintain and improve SuperClaude:

*   **Ko-fi:** [Support on Ko-fi](https://ko-fi.com/superclaude) - One-time contributions.
*   **Patreon:** [Become a Patron](https://patreon.com/superclaude) - Monthly support.
*   **GitHub Sponsors:** [GitHub Sponsor](https://github.com/sponsors/SuperClaude-Org) - Flexible tiers.

---

## Documentation

Comprehensive documentation is available to help you get started:

### Getting Started

*   [Quick Start Guide](Docs/Getting-Started/quick-start.md)
*   [Installation Guide](Docs/Getting-Started/installation.md)

### User Guides

*   [Commands Reference](Docs/User-Guide/commands.md)
*   [Agents Guide](Docs/User-Guide/agents.md)
*   [Behavioral Modes](Docs/User-Guide/modes.md)
*   [Flags Guide](Docs/User-Guide/flags.md)
*   [MCP Servers](Docs/User-Guide/mcp-servers.md)
*   [Session Management](Docs/User-Guide/session-management.md)

### Developer Resources

*   [Technical Architecture](Docs/Developer-Guide/technical-architecture.md)
*   [Contributing Code](Docs/Developer-Guide/contributing-code.md)
*   [Testing & Debugging](Docs/Developer-Guide/testing-debugging.md)

### Reference

*   [Best Practices](Docs/Reference/quick-start-practices.md)
*   [Examples Cookbook](Docs/Reference/examples-cookbook.md)
*   [Troubleshooting](Docs/Reference/troubleshooting.md)

---

## Contribute

Help build the SuperClaude community:

*   **Documentation:** Improve guides, add examples, and fix typos.
*   **MCP Integration:** Add server configurations and test integrations.
*   **Workflows:** Create command patterns and recipes.
*   **Testing:** Add tests and validate features.
*   **i18n:** Translate docs to other languages.

*   [Contributing Guide](CONTRIBUTING.md)
*   [View All Contributors](https://github.com/SuperClaude-Org/SuperClaude_Framework/graphs/contributors)

---

## License

This project is licensed under the MIT License.  See the [LICENSE](LICENSE) file for details.

---

## Star History

<a href="https://www.star-history.com/#SuperClaude-Org/SuperClaude_Framework&Timeline">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=SuperClaude-Org/SuperClaude_Framework&type=Timeline&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=SuperClaude-Org/SuperClaude_Framework&type=Timeline" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=SuperClaude-Org/SuperClaude_Framework&type=Timeline" />
 </picture>
</a>
```
Key improvements and SEO considerations:

*   **Concise Hook:** A compelling one-sentence opener.
*   **Clear Headings:** Improved structure for readability.
*   **Keyword Optimization:** Used relevant keywords like "Claude code," "development platform," "AI agents," etc.
*   **Bulleted Lists:** Easy-to-scan key features.
*   **Stronger Call to Action:** Encourages contribution and support.
*   **Internal Linking:** Promotes navigation within the README.
*   **External Linking:** Maintains links to the original repo and support channels.
*   **Focus on Benefits:** Highlights *what* the framework *does* for the user.
*   **Concise Language:** Streamlined and improved wording throughout.
*   **Table Formatting:** Replaced many divs with Markdown tables for better readability and accessibility.
# SuperClaude Framework: Transform Your Code with AI 🚀

**Supercharge your development workflow with SuperClaude, a cutting-edge framework that leverages AI to revolutionize how you build, test, and deploy your projects.**  For more details, visit the original repository at [https://github.com/SuperClaude-Org/SuperClaude_Framework](https://github.com/SuperClaude-Org/SuperClaude_Framework).

[![Version](https://img.shields.io/badge/version-4.0.8-blue)](https://github.com/SuperClaude-Org/SuperClaude_Framework)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/SuperClaude-Org/SuperClaude_Framework/blob/main/LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/SuperClaude-Org/SuperClaude_Framework)

**Key Features:**

*   **AI-Powered Agents:** Leverage 14 specialized AI agents with domain expertise for tasks like security analysis, UI pattern recognition, and more.
*   **Enhanced Command Structure:** Streamlined command structure using the `/sc:` prefix, offering 22 commands for a complete development lifecycle.
*   **Powerful MCP Server Integrations:** Seamlessly integrate with 6 powerful servers, including Context7, Sequential, Magic, Playwright, Morphllm, and Serena.
*   **Adaptive Behavioral Modes:** Utilize 6 adaptive modes for different contexts, such as brainstorming, strategic analysis, and token-efficient operations.
*   **Optimized Performance:** Benefit from a smaller framework footprint, enabling longer conversations and complex operations.
*   **Comprehensive Documentation:** Access a complete rewrite of documentation with real-world examples, practical workflows, and improved navigation.

**Quick Installation:**

Choose your preferred installation method:

*   **pipx (Recommended - Linux/macOS):**
    ```bash
    pipx install SuperClaude && pipx upgrade SuperClaude && SuperClaude install
    ```
*   **pip (Traditional Python environments):**
    ```bash
    pip install SuperClaude && pip upgrade SuperClaude && SuperClaude install
    ```
*   **npm (Cross-platform, Node.js users):**
    ```bash
    npm install -g @bifrost_inc/superclaude && superclaude install
    ```

<details>
<summary><b>⚠️ IMPORTANT: Upgrading from SuperClaude V3</b></summary>

**If you have SuperClaude V3 installed, you SHOULD uninstall it before installing V4:**

```bash
# Uninstall V3 first
Remove all related files and directories :
*.md *.json and commands/

# Then install V4
pipx install SuperClaude && pipx upgrade SuperClaude && SuperClaude install
```

**✅ What gets preserved during upgrade:**
- ✓ Your custom slash commands (outside `commands/sc/`)
- ✓ Your custom content in `CLAUDE.md` 
- ✓ Claude Code's `.claude.json`, `.credentials.json`, `settings.json` and `settings.local.json`
- ✓ Any custom agents and files you've added

**⚠️ Note:** Other SuperClaude-related `.json` files from V3 may cause conflicts and should be removed.

</details>

<details>
<summary><b>💡 Troubleshooting PEP 668 Errors</b></summary>

```bash
# Option 1: Use pipx (Recommended)
pipx install SuperClaude

# Option 2: User installation
pip install --user SuperClaude

# Option 3: Force installation (use with caution)
pip install --break-system-packages SuperClaude
```
</details>

**Support the Project:**

Your support helps maintain and develop SuperClaude. Consider contributing via:

*   [Ko-fi](https://ko-fi.com/superclaude) - One-time contributions
*   [Patreon](https://patreon.com/superclaude) - Monthly support
*   [GitHub Sponsors](https://github.com/sponsors/SuperClaude-Org) - Flexible tiers

**What's New in V4:**

*   **Smarter Agent System:** 14 specialized agents with domain expertise
*   **Improved Namespace:** `/sc:` prefix for all commands
*   **MCP Server Integration:** 6 powerful server integrations
*   **Behavioral Modes:** 6 adaptive modes for different contexts
*   **Optimized Performance:** Smaller framework, bigger projects
*   **Documentation Overhaul:** Complete rewrite for developers

**Documentation:**

Explore the comprehensive documentation:

*   **Getting Started:**
    *   [Quick Start Guide](Docs/Getting-Started/quick-start.md)
    *   [Installation Guide](Docs/Getting-Started/installation.md)
*   **User Guides:**
    *   [Commands Reference](Docs/User-Guide/commands.md)
    *   [Agents Guide](Docs/User-Guide/agents.md)
    *   [Behavioral Modes](Docs/User-Guide/modes.md)
    *   [Flags Guide](Docs/User-Guide/flags.md)
    *   [MCP Servers](Docs/User-Guide/mcp-servers.md)
    *   [Session Management](Docs/User-Guide/session-management.md)
*   **Developer Resources:**
    *   [Technical Architecture](Docs/Developer-Guide/technical-architecture.md)
    *   [Contributing Code](Docs/Developer-Guide/contributing-code.md)
    *   [Testing & Debugging](Docs/Developer-Guide/testing-debugging.md)
*   **Reference:**
    *   [Best Practices](Docs/Reference/quick-start-practices.md)
    *   [Examples Cookbook](Docs/Reference/examples-cookbook.md)
    *   [Troubleshooting](Docs/Reference/troubleshooting.md)

**Contributing:**

Join the SuperClaude community and contribute!

*   **High Priority:** Documentation, MCP Integration
*   **Medium Priority:** Workflows, Testing
*   **Low Priority:** i18n (Translation)

[Contributing Guide](CONTRIBUTING.md) | [View All Contributors](https://github.com/SuperClaude-Org/SuperClaude_Framework/graphs/contributors)

**License:**

This project is licensed under the MIT License. [LICENSE](LICENSE)

**Star History:**

[Star History Chart](https://www.star-history.com/#SuperClaude-Org/SuperClaude_Framework&Timeline)

**Built with passion by the SuperClaude community.**
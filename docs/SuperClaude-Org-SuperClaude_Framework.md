# SuperClaude Framework: Unlock AI-Powered Development with a Structured Platform

**Supercharge your Claude code and transform it into a robust development platform with the SuperClaude Framework.**  ([Original Repo](https://github.com/SuperClaude-Org/SuperClaude_Framework))

[![Version](https://img.shields.io/badge/version-4.0.8-blue)](https://github.com/SuperClaude-Org/SuperClaude_Framework/releases)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/SuperClaude-Org/SuperClaude_Framework/pulls)
[![Website](https://img.shields.io/badge/%F0%9F%8C%90_Visit_Website-blue)](https://superclaude.netlify.app/)
[![PyPI](https://img.shields.io/pypi/v/SuperClaude.svg?)](https://pypi.org/project/SuperClaude/)
[![npm](https://img.shields.io/npm/v/@bifrost_inc/superclaude.svg)](https://www.npmjs.com/package/@bifrost_inc/superclaude)
[![English](https://img.shields.io/badge/%F0%9F%87%BA%F0%9F%87%B8_English-blue)](README.md)
[![中文](https://img.shields.io/badge/%F0%9F%87%A8%F0%9F%87%B3_中文-red)](README-zh.md)
[![日本語](https://img.shields.io/badge/%F0%9F%8A%AC_日本語-green)](README-ja.md)

---

## Key Features

*   **Intelligent Agents:** Leverage 14 specialized AI agents for domain-specific expertise in security, UI/UX, and more.
*   **Structured Commands:** Utilize 22 slash commands, all starting with `/sc:`, for a clean and organized workflow from brainstorming to deployment.
*   **Powerful Server Integrations:** Access 6 integrated MCP servers for advanced capabilities such as documentation, complex analysis, UI component generation, browser testing, and session persistence.
*   **Adaptive Behavioral Modes:** Choose from 6 behavioral modes, including Brainstorming, Business Panel, and Task Management, to tailor your development experience.
*   **Optimized Performance:** Benefit from a reduced framework footprint, enabling longer conversations and complex operations.
*   **Comprehensive Documentation:** Explore a completely rewritten documentation, offering real-world examples, practical workflows, and clear navigation.

---

## Quick Installation

Get started with SuperClaude quickly using your preferred method:

### Installation Methods

| Method       | Command                                                                   | Best For                                       |
| :----------- | :------------------------------------------------------------------------ | :--------------------------------------------- |
| **pipx**     | `pipx install SuperClaude && pipx upgrade SuperClaude && SuperClaude install` | **Recommended** - Linux/macOS                  |
| **pip**      | `pip install SuperClaude && pip upgrade SuperClaude && SuperClaude install` | Traditional Python environments               |
| **npm**      | `npm install -g @bifrost_inc/superclaude && superclaude install`            | Cross-platform, Node.js users                 |

---

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

---

## Support the Project

> Maintaining SuperClaude requires time and resources.  Consider supporting the project if you find it valuable.

You can support the project through:

*   [**Ko-fi**](https://ko-fi.com/superclaude) (One-time contributions)
*   [**Patreon**](https://patreon.com/superclaude) (Monthly support)
*   [**GitHub Sponsors**](https://github.com/sponsors/SuperClaude-Org) (Flexible tiers)

Your support enables:

*   Claude Max Testing
*   Feature Development
*   Documentation
*   Community Support
*   MCP Integration
*   Infrastructure

---

## Documentation

### SuperClaude Documentation

*   **Getting Started:**
    *   [Quick Start Guide](Docs/Getting-Started/quick-start.md) - Get up and running fast.
    *   [Installation Guide](Docs/Getting-Started/installation.md) - Detailed setup instructions.
*   **User Guides:**
    *   [Commands Reference](Docs/User-Guide/commands.md) - All 22 slash commands.
    *   [Agents Guide](Docs/User-Guide/agents.md) - 14 specialized agents.
    *   [Behavioral Modes](Docs/User-Guide/modes.md) - 5 adaptive modes.
    *   [Flags Guide](Docs/User-Guide/flags.md) - Control behaviors.
    *   [MCP Servers](Docs/User-Guide/mcp-servers.md) - 6 server integrations.
    *   [Session Management](Docs/User-Guide/session-management.md) - Save & restore state.
*   **Developer Resources:**
    *   [Technical Architecture](Docs/Developer-Guide/technical-architecture.md) - System design details.
    *   [Contributing Code](Docs/Developer-Guide/contributing-code.md) - Development workflow.
    *   [Testing & Debugging](Docs/Developer-Guide/testing-debugging.md) - Quality assurance.
*   **Reference:**
    *   [Best Practices](Docs/Reference/quick-start-practices.md) - Pro tips & patterns.
    *   [Examples Cookbook](Docs/Reference/examples-cookbook.md) - Real-world recipes.
    *   [Troubleshooting](Docs/Reference/troubleshooting.md) - Common issues & fixes.

---

## Contributing

### Join the SuperClaude Community

We welcome contributions!  Areas to contribute include:

| Priority | Area           | Description                            |
| :------- | :------------- | :------------------------------------- |
| High     | Documentation  | Improve guides, add examples, fix typos |
| High     | MCP Integration | Add server configs, test integrations  |
| Medium   | Workflows      | Create command patterns & recipes      |
| Medium   | Testing        | Add tests, validate features           |
| Low      | i18n           | Translate docs to other languages      |

*   [Contributing Guide](CONTRIBUTING.md)
*   [View All Contributors](https://github.com/SuperClaude-Org/SuperClaude_Framework/graphs/contributors)

---

## License

This project is licensed under the MIT License.  See the [LICENSE](LICENSE) file for details.

---

## ⭐ Star History

[View Star History Chart](https://www.star-history.com/#SuperClaude-Org/SuperClaude_Framework&Timeline)

---

### Built with passion by the SuperClaude community.

Made with ❤️ for developers who push boundaries.

[Back to Top ↑](#superclaude-framework)
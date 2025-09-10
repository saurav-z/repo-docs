# SuperClaude Framework: Transform Claude Code into a Powerful Development Platform

**Supercharge your development workflow with SuperClaude, a meta-programming framework that unlocks the full potential of Claude code.**  [Explore the original repository](https://github.com/SuperClaude-Org/SuperClaude_Framework).

[![Version](https://img.shields.io/badge/version-4.0.9-blue)](https://github.com/SuperClaude-Org/SuperClaude_Framework)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/SuperClaude-Org/SuperClaude_Framework/blob/main/LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/SuperClaude-Org/SuperClaude_Framework/pulls)
[![Website](https://img.shields.io/badge/%F0%9F%8C%90_Visit_Website-blue)](https://superclaude.netlify.app/)
[![PyPI](https://img.shields.io/pypi/v/SuperClaude.svg?)](https://pypi.org/project/SuperClaude/)
[![npm](https://img.shields.io/npm/v/@bifrost_inc/superclaude.svg)](https://www.npmjs.com/package/@bifrost_inc/superclaude)
[![English](https://img.shields.io/badge/%F0%9F%87%BA%F0%9F%87%B8_English-blue)](https://github.com/SuperClaude-Org/SuperClaude_Framework/blob/main/README.md)
[![中文](https://img.shields.io/badge/%F0%9F%87%A8%F0%9F%87%B3_中文-red)](https://github.com/SuperClaude-Org/SuperClaude_Framework/blob/main/README-zh.md)
[![日本語](https://img.shields.io/badge/%F0%9F%8A%AC_日本語-green)](https://github.com/SuperClaude-Org/SuperClaude_Framework/blob/main/README-ja.md)

---

## Key Features

*   **Extensive Command Set:** Execute with 23 slash commands, streamlining your development process.
*   **Intelligent Agents:** Leverage 14 specialized AI agents for domain-specific expertise and automation.
*   **Adaptive Behavioral Modes:** Choose from 6 modes, tailored to different tasks like brainstorming and task management.
*   **Powerful MCP Server Integration:** Seamlessly integrates with 6 servers for advanced functionalities, including documentation retrieval and browser testing.
*   **Optimized Performance:** Reduced framework footprint for more efficient context handling.
*   **Comprehensive Documentation:**  Benefit from a complete rewrite of documentation, with real-world examples.

---

## Quick Installation

Choose your preferred method:

*   **pipx (Recommended):** `pipx install SuperClaude && pipx upgrade SuperClaude && SuperClaude install`
*   **pip:** `pip install SuperClaude && pip upgrade SuperClaude && SuperClaude install`
*   **npm:** `npm install -g @bifrost_inc/superclaude && superclaude install`

---

<details>
<summary><b>Important: Upgrading from SuperClaude V3</b></summary>

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

---

## Support the Project

Your support helps maintain and grow SuperClaude. Consider contributing via:

*   **Ko-fi:** [One-time contributions](https://ko-fi.com/superclaude)
*   **Patreon:** [Monthly support](https://patreon.com/superclaude)
*   **GitHub Sponsors:** [Flexible tiers](https://github.com/sponsors/SuperClaude-Org)

Your support enables:
* Claude Max Testing
* Feature Development
* Comprehensive Documentation
* Community Support
* MCP Integration
* Infrastructure

---

## What's New in V4

*   **Smarter Agent System:** 14 specialized agents with domain expertise.
*   **Improved Namespace:** `/sc:` prefix for streamlined command execution.
*   **MCP Server Integration:** 6 powerful servers working together.
*   **Behavioral Modes:** 6 adaptive modes for different contexts.
*   **Optimized Performance:** Smaller framework, bigger projects.
*   **Documentation Overhaul:** Complete rewrite for developers.

---

## Documentation

Access comprehensive guides and resources:

*   **Getting Started:** Quick Start Guide, Installation Guide
*   **User Guides:** Commands Reference, Agents Guide, Behavioral Modes, Flags Guide, MCP Servers, Session Management
*   **Developer Resources:** Technical Architecture, Contributing Code, Testing & Debugging
*   **Reference:** Examples Cookbook, Troubleshooting

---

## Contributing

Join the SuperClaude community!  We welcome contributions in:

*   Documentation
*   MCP Integration
*   Workflows
*   Testing
*   i18n

[Read the Contributing Guide](CONTRIBUTING.md) | [View All Contributors](https://github.com/SuperClaude-Org/SuperClaude_Framework/graphs/contributors)

---

## License

This project is licensed under the MIT License.

---

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=SuperClaude-Org/SuperClaude_Framework&type=Timeline)](https://www.star-history.com/#SuperClaude-Org/SuperClaude_Framework&Timeline)

---

### Built with passion by the SuperClaude community

<sub>Made with ❤️ for developers who push boundaries</sub>
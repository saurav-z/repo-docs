# SuperClaude Framework: Transform Claude Code into a Structured Development Platform

**Supercharge your Claude code and streamline your development workflow with the SuperClaude Framework.** [Explore the SuperClaude Framework on GitHub](https://github.com/SuperClaude-Org/SuperClaude_Framework).

<p align="center">
  <img src="https://img.shields.io/badge/version-4.0.4-blue?style=for-the-badge" alt="Version">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge" alt="License">
  <img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=for-the-badge" alt="PRs Welcome">
</p>

<p align="center">
  <a href="https://superclaude-org.github.io/SuperClaude_Website/">
    <img src="https://img.shields.io/badge/🌐_Visit_Website-blue?style=for-the-badge" alt="Website">
  </a>
  <a href="https://pypi.org/project/SuperClaude/">
    <img src="https://img.shields.io/pypi/v/SuperClaude.svg?style=for-the-badge&label=PyPI" alt="PyPI">
  </a>
  <a href="https://www.npmjs.com/package/@bifrost_inc/superclaude">
    <img src="https://img.shields.io/npm/v/@bifrost_inc/superclaude.svg?style=for-the-badge&label=npm" alt="npm">
  </a>
</p>

<p align="center">
  <a href="#quick-installation">Quick Start</a> •
  <a href="#key-features">Key Features</a> •
  <a href="#support-the-project">Support</a> •
  <a href="#whats-new-in-v4">What's New in V4</a> •
  <a href="#documentation">Documentation</a> •
  <a href="#contributing">Contributing</a>
</p>

---

## Key Features

*   **Structured Development Platform:**  Transforms raw Claude code into a structured, manageable platform.
*   **Intelligent Agents:** Utilizes 14 specialized AI agents for various development tasks, including security, UI, and code coordination.
*   **Powerful Commands:** Offers 21 slash commands for the complete development lifecycle, from brainstorming to deployment.
*   **Advanced Modes:** Provides 5 behavioral modes for optimized efficiency, including Brainstorming, Orchestration, and Token-Efficiency.
*   **MCP Server Integration:** Seamlessly integrates with 6 powerful servers, like Context7, Sequential, and Magic, for enhanced functionality.
*   **Optimized Performance:** Reduces framework footprint for more context in your code.

---

## Quick Installation

Choose your preferred installation method:

| Method | Command | Best For |
|:------:|---------|----------|
| **🐍 pipx** | `pipx install SuperClaude && SuperClaude install` | **✅ Recommended** - Linux/macOS |
| **📦 pip** | `pip install SuperClaude && SuperClaude install` | Traditional Python environments |
| **🌐 npm** | `npm install -g @bifrost_inc/superclaude && superclaude install` | Cross-platform, Node.js users |

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

Your support helps maintain SuperClaude, covering Claude Max subscription costs, feature development, documentation, and community support.  Consider supporting the project through:

*   **Ko-fi:**  [![Ko-fi](https://img.shields.io/badge/Support_on-Ko--fi-ff5e5b?style=for-the-badge&logo=ko-fi)](https://ko-fi.com/superclaude)
*   **Patreon:** [![Patreon](https://img.shields.io/badge/Become_a-Patron-f96854?style=for-the-badge&logo=patreon)](https://patreon.com/superclaude)
*   **GitHub Sponsors:** [![GitHub Sponsors](https://img.shields.io/badge/GitHub-Sponsor-30363D?style=for-the-badge&logo=github-sponsors)](https://github.com/sponsors/SuperClaude-Org)

---

## What's New in V4

*   **Smarter Agent System:** 14 specialized agents with domain expertise.
*   **Improved Namespace:** `/sc:` prefix for command organization.
*   **MCP Server Integration:** 6 powerful servers working together.
*   **Behavioral Modes:** 5 adaptive modes for different contexts.
*   **Optimized Performance:** Reduced framework footprint.
*   **Documentation Overhaul:** Complete rewrite with practical examples.

---

## Documentation

Comprehensive guides and resources to get you started:

<table>
<tr>
<th align="center">🚀 Getting Started</th>
<th align="center">📖 User Guides</th>
<th align="center">🛠️ Developer Resources</th>
<th align="center">📋 Reference</th>
</tr>
<tr>
<td valign="top">

- 📝 [**Quick Start Guide**](Docs/Getting-Started/quick-start.md)  
  *Get up and running fast*

- 💾 [**Installation Guide**](Docs/Getting-Started/installation.md)  
  *Detailed setup instructions*

</td>
<td valign="top">

- 🎯 [**Commands Reference**](Docs/User-Guide/commands.md)  
  *All 21 slash commands*

- 🤖 [**Agents Guide**](Docs/User-Guide/agents.md)  
  *14 specialized agents*

- 🎨 [**Behavioral Modes**](Docs/User-Guide/modes.md)  
  *6 adaptive modes*

- 🚩 [**Flags Guide**](Docs/User-Guide/flags.md)  
  *Control behaviors*

- 🔧 [**MCP Servers**](Docs/User-Guide/mcp-servers.md)  
  *6 server integrations*

- 💼 [**Session Management**](Docs/User-Guide/session-management.md)  
  *Save & restore state*

</td>
<td valign="top">

- 🏗️ [**Technical Architecture**](Docs/Developer-Guide/technical-architecture.md)  
  *System design details*

- 💻 [**Contributing Code**](Docs/Developer-Guide/contributing-code.md)  
  *Development workflow*

- 🧪 [**Testing & Debugging**](Docs/Developer-Guide/testing-debugging.md)  
  *Quality assurance*

</td>
<td valign="top">

- ✨ [**Best Practices**](Docs/Reference/quick-start-practices.md)  
  *Pro tips & patterns*

- 📓 [**Examples Cookbook**](Docs/Reference/examples-cookbook.md)  
  *Real-world recipes*

- 🔍 [**Troubleshooting**](Docs/Reference/troubleshooting.md)  
  *Common issues & fixes*

</td>
</tr>
</table>

---

## Contributing

We welcome contributions!

*   **Documentation:** Improve guides, add examples, and fix typos.
*   **MCP Integration:** Add server configurations and test integrations.
*   **Workflows:** Create command patterns and recipes.
*   **Testing:** Add tests and validate features.
*   **i18n:** Translate docs to other languages.

<p align="center">
  <a href="CONTRIBUTING.md">
    <img src="https://img.shields.io/badge/📖_Read-Contributing_Guide-blue?style=for-the-badge" alt="Contributing Guide">
  </a>
  <a href="https://github.com/SuperClaude-Org/SuperClaude_Framework/graphs/contributors">
    <img src="https://img.shields.io/badge/👥_View-All_Contributors-green?style=for-the-badge" alt="Contributors">
  </a>
</p>

---

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

<p align="center">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge" alt="MIT License">
</p>

---

## ⭐ Star History

<a href="https://www.star-history.com/#SuperClaude-Org/SuperClaude_Framework&Date">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=SuperClaude-Org/SuperClaude_Framework&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=SuperClaude-Org/SuperClaude_Framework&type=Date" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=SuperClaude-Org/SuperClaude_Framework&type=Date" />
  </picture>
</a>

---

### **🚀 Built with passion by the SuperClaude community**

<p align="center">
  <sub>Made with ❤️ for developers who push boundaries</sub>
</p>

<p align="center">
  <a href="#superclaude-framework">Back to Top ↑</a>
</p>
```
Key improvements and explanations:

*   **SEO-Optimized Title & Hook:**  The title is more descriptive, including relevant keywords (e.g., "Claude code," "development platform"). The one-sentence hook immediately highlights the key benefit.
*   **Clear Headings:**  Uses clear, concise headings and subheadings for better readability and SEO.
*   **Bulleted Key Features:**  Uses bullet points to make the key features easily scannable.  This is critical for attracting users.
*   **Concise Summaries:** The overview and "What's New" sections are summarized for clarity.
*   **Call to Action:**  Encourages users to "Explore the SuperClaude Framework" with a direct link.
*   **Keyword Integration:**  Incorporates relevant keywords naturally throughout the text (e.g., "Claude code," "development workflow," "intelligent agents").  These keywords help with search engine visibility.
*   **Improved Structure and Readability:** The formatting uses markdown more effectively, making the content easier to read and digest.
*   **Concise and Direct Language:** The language is more direct and focuses on the benefits of using the framework.
*   **Consistent Formatting:** Maintains consistent formatting throughout, making the document more professional.
*   **Added a "Key Features" section:**  This is crucial for quickly conveying the value proposition of the framework.  It helps users determine if it's the right tool for them.
*   **Included a "Contributing" section:** This encourages community involvement, which benefits the project and its long-term sustainability.
*   **Improved "Support the Project" Section**: Summarized to a concise paragraph instead of just a wall of text.
*   **Added "What's New in V4"**: To highlight recent features and updates.
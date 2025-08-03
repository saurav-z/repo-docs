<!-- SEO-optimized README for CCPlugins -->

# CCPlugins: Supercharge Your Claude Code CLI for Faster Development 🚀

Tired of repetitive development tasks? **CCPlugins enhances your Claude Code CLI with powerful, professional commands, saving you hours each week.** [Check out the original repo](https://github.com/brennercruvinel/CCPlugins) to get started!

[![GitHub Repo stars](https://img.shields.io/github/stars/brennercruvinel/CCPlugins?style=social)](https://github.com/brennercruvinel/CCPlugins)
[![Version](https://img.shields.io/badge/version-2.5.2-blue.svg)](https://github.com/brennercruvinel/CCPlugins)
[![Claude Code CLI](https://img.shields.io/badge/for-Claude%20Code%20CLI-purple.svg)](https://docs.anthropic.com/en/docs/claude-code)
[![Tested on](https://img.shields.io/badge/tested%20on-Opus%204%20%26%20Sonnet%204-orange.svg)](https://claude.ai)
[![Also works with](https://img.shields.io/badge/also%20works%20with-Kimi%20K2-1783ff.svg)](https://github.com/MoonshotAI/Kimi-K2)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/brennercruvinel/CCPlugins/blob/main/CONTRIBUTING.md)

## Key Features

*   **Automated Workflows:** Streamline development tasks with smart commands.
*   **Code Quality & Security:** Enhance code with built-in review, security scans, and proactive issue prediction.
*   **Intelligent Analysis:** Deep project analysis and senior-level code explanations.
*   **Session & Project Management:** Simplify documentation, track progress, and convert TODOs into issues.
*   **Git Integration:** Benefit from secure Git operations and prevent AI attribution.

## Core Commands

CCPlugins offers a comprehensive suite of commands to accelerate your development workflow:

*   **/cleanproject**: Remove debug artifacts and safely prepare your project.
*   **/commit**: Create intelligent and conventional commits with analysis.
*   **/format**: Automatically apply project formatting with auto-detection.
*   **/scaffold feature-name**: Generate complete features from pre-defined patterns.
*   **/test**: Run tests with insightful failure analysis.
*   **/implement url/path/feature**: Import and adapt code, now with validation.
*   **/refactor**: Restructure code intelligently with mapping and validation.
*   **/review**: Analyze with multi-agent analysis (security, performance, quality, architecture)
*   **/security-scan**: Conduct vulnerability analysis with improved remediation.
*   **/predict-issues**: Proactively identify potential issues with time estimates.
*   **/remove-comments**: Clean comments while preserving valuable docs.
*   **/fix-imports**: Repair broken imports with enhanced refactoring support.
*   **/find-todos**: Locate and organize development tasks effectively.
*   **/create-todos**: Add contextual TODO comments from analysis results.
*   **/fix-todos**: Intelligently implement and resolve TODO fixes.
*   **/understand**: Deeply analyze project architecture and patterns.
*   **/explain-like-senior**: Get senior-level code explanations with context.
*   **/contributing**: Analyze your contribution readiness.
*   **/make-it-pretty**: Improve readability without code changes.
*   **/session-start**: Begin documented sessions integrated with CLAUDE.md
*   **/session-end**: Summarize and preserve your session context.
*   **/docs**: Smart documentation management and updates.
*   **/todos-to-issues**: Convert code TODOs to GitHub issues seamlessly.
*   **/undo**: Safe and reversible rollback with git checkpoint restore.

## Installation

### Quick Install

**Mac/Linux:**

```bash
curl -sSL https://raw.githubusercontent.com/brennercruvinel/CCPlugins/main/install.sh | bash
```

**Windows/Cross-platform:**

```bash
python install.py
```

### Manual Install

```bash
git clone https://github.com/brennercruvinel/CCPlugins.git
cd CCPlugins
python install.py
```

### Uninstall

```bash
# Mac/Linux
./uninstall.sh

# Windows/Cross-platform
python uninstall.py
```

## Why Use CCPlugins?

CCPlugins provides a complete development solution, resulting in an average savings of 4-5 hours per week by automating security analysis, code reviews, feature scaffolding, and Git commits.

## How It Works

CCPlugins transforms the Claude Code CLI into an intelligent assistant through these key components:

*   **Intelligent Instructions:** first-person conversational design activates collaborative reasoning.
*   **Native Tool Integration:** Uses Claude Code CLI's native capabilities (grep, glob, read, write)
*   **Safety-First Design:** Automated Git checkpoints before operations, ensuring rollback capabilities.
*   **Universal Compatibility:** Adapts to your project's conventions and patterns.

## Advanced Features

*   **Validation & Refinement**: Validate tasks, now available for /refactor and /implement.
*   **Extended Thinking**: Refactor, Security analysis, performance improvements, and architecture reviews.
*   **Pragmatic Command Integration**: Get suggestions post-major changes.
*   **Session Continuity**: Commands such as /implement and /refactor maintain state across sessions.
*   **Multi-Agent Architecture**: Complex tasks orchestrated through specialized agents (Security, Architecture, etc.)

## Requirements

*   Claude Code CLI
*   Python 3.6+ (for installer)
*   Git (for version control commands)

## Contributing

We welcome contributions. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

**Last Updated:** August 2, 2025 (Based on Anthropic Claude Code CLI documentation v2025.08.01)
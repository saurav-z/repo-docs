# CCPlugins: Supercharge Your Claude Code CLI Workflow (Save Hours Weekly!)

**Tired of repetitive coding tasks and struggling to get Claude Code CLI to deliver?** CCPlugins transforms your coding experience with a suite of powerful commands designed to automate development workflows, improve code quality, and boost your productivity. [Explore CCPlugins on GitHub](https://github.com/brennercruvinel/CCPlugins).

[![GitHub Repo stars](https://img.shields.io/github/stars/brennercruvinel/CCPlugins?style=social)](https://github.com/brennercruvinel/CCPlugins)
[![Version](https://img.shields.io/badge/version-2.5.2-blue.svg)](https://github.com/brennercruvinel/CCPlugins)
[![Claude Code CLI](https://img.shields.io/badge/for-Claude%20Code%20CLI-purple.svg)](https://docs.anthropic.com/en/docs/claude-code)
[![Tested on](https://img.shields.io/badge/tested%20on-Opus%204%20%26%20Sonnet%204-orange.svg)](https://claude.ai)
[![Also works with](https://img.shields.io/badge/also%20works%20with-Kimi%20K2-1783ff.svg)](https://github.com/MoonshotAI/Kimi-K2)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/brennercruvinel/CCPlugins/blob/main/CONTRIBUTING.md)

## Key Features

*   **Automated Workflows:** Streamline your development process with commands for common tasks like formatting, testing, and code reviews.
*   **Enhanced Code Quality:** Improve code maintainability and security with features like vulnerability scanning, comment removal, and import fixing.
*   **Intelligent Analysis:** Leverage advanced analysis for refactoring, security, architecture reviews, and proactive issue prediction.
*   **Time Savings:** Reduce repetitive tasks and free up your time with command-based automation, potentially saving up to 4-5 hours per week.
*   **Seamless Integration:** Designed to work directly with Claude Code CLI, optimizing for Opus 4, Sonnet 4 and Kimi K2 models.
*   **Safe and Reliable:** Ensures code safety with Git checkpoints, session persistence, and rollback capabilities.

## Core Commands - A Developer's Toolkit

*   **/cleanproject**: Remove debug artifacts with git safety
*   **/commit**: Smart conventional commits with analysis
*   **/format**: Auto-detect and apply project formatter
*   **/scaffold feature-name**: Generate complete features from patterns
*   **/test**: Run tests with intelligent failure analysis
*   **/implement url/path/feature**: Import and adapt code from any source with validation phase
*   **/refactor**: Intelligent code restructuring with validation & de-para mapping
*   **/review**: Multi-agent analysis (security, performance, quality, architecture)
*   **/security-scan**: Vulnerability analysis with extended thinking & remediation tracking
*   **/predict-issues**: Proactive problem detection with timeline estimates
*   **/remove-comments**: Clean obvious comments, preserve valuable docs
*   **/fix-imports**: Repair broken imports after refactoring
*   **/find-todos**: Locate and organize development tasks
*   **/create-todos**: Add contextual TODO comments based on analysis results
*   **/fix-todos**: Intelligently implement TODO fixes with context
*   **/understand**: Analyze entire project architecture and patterns
*   **/explain-like-senior**: Senior-level code explanations with context
*   **/contributing**: Complete contribution readiness analysis
*   **/make-it-pretty**: Improve readability without functional changes
*   **/session-start**: Begin documented sessions with CLAUDE.md integration
*   **/session-end**: Summarize and preserve session context
*   **/docs**: Smart documentation management and updates
*   **/todos-to-issues**: Convert code TODOs to GitHub issues
*   **/undo**: Safe rollback with git checkpoint restore

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

## How CCPlugins Works

CCPlugins operates as an intelligent layer on top of Claude Code CLI, turning it into a powerful development assistant. Commands are defined using a clear architecture:

1.  **Command Invocation:** You use the `/command` syntax.
2.  **Command Loading:** Claude reads the markdown definition.
3.  **Contextual Analysis:** The plugin analyzes project structure.
4.  **Intelligent Planning:** Execution strategies are created.
5.  **Safe Execution:** Actions are executed with validation, including Git checkpoints.
6.  **Clear Feedback:** Provides results and insights.

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) file for details.

---
**Last Updated:** August 2, 2025 (Based on Anthropic Claude Code CLI documentation v2025.08.01)
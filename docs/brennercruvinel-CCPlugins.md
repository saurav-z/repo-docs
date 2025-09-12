# CCPlugins: Supercharge Your Claude Code CLI with Enterprise-Grade Development Workflows

**Tired of repetitive coding tasks?** CCPlugins extends Claude Code CLI with professional commands, saving developers hours each week. [Explore CCPlugins on GitHub](https://github.com/brennercruvinel/CCPlugins)

[![GitHub Repo stars](https://img.shields.io/github/stars/brennercruvinel/CCPlugins?style=social)](https://github.com/brennercruvinel/CCPlugins)
[![Version](https://img.shields.io/badge/version-2.5.2-blue.svg)](https://github.com/brennercruvinel/CCPlugins)
[![Claude Code CLI](https://img.shields.io/badge/for-Claude%20Code%20CLI-purple.svg)](https://docs.anthropic.com/en/docs/claude-code)
[![Tested on](https://img.shields.io/badge/tested%20on-Opus%204%20%26%20Sonnet%204-orange.svg)](https://claude.ai)
[![Also works with](https://img.shields.io/badge/also%20works%20with-Kimi%20K2-1783ff.svg)](https://github.com/MoonshotAI/Kimi-K2)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/brennercruvinel/CCPlugins/blob/main/CONTRIBUTING.md)

## Key Features:

*   **Automated Development Workflow:** Streamline your coding process with commands like `/cleanproject`, `/commit`, `/format`, `/scaffold`, `/test`, `/implement`, and `/refactor`.
*   **Enhanced Code Quality & Security:** Proactively identify and resolve issues with commands like `/review`, `/security-scan`, `/predict-issues`, `/remove-comments`, `/fix-imports`, `/find-todos`, `/create-todos`, and `/fix-todos`.
*   **Advanced Analysis:** Gain deeper insights into your codebase using commands like `/understand`, `/explain-like-senior`, `/contributing`, and `/make-it-pretty`.
*   **Project & Session Management:**  Improve workflow efficiency with commands like `/session-start`, `/session-end`, `/docs`, `/todos-to-issues`, and `/undo`.
*   **Validation & Refinement:** Ensure the quality and completeness of complex commands with validation phases built-in.
*   **Real-World Benefits:**  Save 4-5 hours per week on professional-grade analysis.

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

## Commands

### 🚀 Development Workflow

*   `/cleanproject` - Remove debug artifacts with git safety
*   `/commit` - Smart conventional commits with analysis
*   `/format` - Auto-detect and apply project formatter
*   `/scaffold feature-name` - Generate complete features from patterns
*   `/test` - Run tests with intelligent failure analysis
*   `/implement url/path/feature` - Import and adapt code with validation
*   `/refactor` - Intelligent code restructuring with validation

### 🛡️ Code Quality & Security

*   `/review` - Multi-agent analysis (security, performance, quality, architecture)
*   `/security-scan` - Vulnerability analysis with extended thinking & remediation tracking
*   `/predict-issues` - Proactive problem detection with timeline estimates
*   `/remove-comments` - Clean obvious comments, preserve valuable docs
*   `/fix-imports` - Repair broken imports after refactoring
*   `/find-todos` - Locate and organize development tasks
*   `/create-todos` - Add contextual TODO comments
*   `/fix-todos` - Intelligently implement TODO fixes

### 🔍 Advanced Analysis

*   `/understand` - Analyze entire project architecture and patterns
*   `/explain-like-senior` - Senior-level code explanations with context
*   `/contributing` - Complete contribution readiness analysis
*   `/make-it-pretty` - Improve readability without functional changes

### 📋 Session & Project Management

*   `/session-start` - Begin documented sessions with CLAUDE.md integration
*   `/session-end` - Summarize and preserve session context
*   `/docs` - Smart documentation management and updates
*   `/todos-to-issues` - Convert code TODOs to GitHub issues
*   `/undo` - Safe rollback with git checkpoint restore

##  How It Works

CCPlugins enhances Claude Code CLI by providing intelligent commands that automate repetitive development tasks. This is achieved through:

*   **Intelligent Instructions:** First-person language and context-aware adaptations.
*   **Native Tool Integration:** Leveraging Claude Code CLI's native capabilities.
*   **Safety-First Design:** Automatic git checkpoints before destructive operations.
*   **Framework Agnostic:** Works with any programming language or stack.

##  Requirements

*   Claude Code CLI
*   Python 3.6+ (for installer)
*   Git (for version control commands)

##  Advanced Usage

*   **Creating Custom Commands:**  Create your own commands by adding markdown files to `~/.claude/commands/`.
*   **Using Arguments:** Commands support arguments via `$ARGUMENTS`.
*   **CI/CD Integration:** Integrate commands into automated workflows.
*   **Manual Workflow Integration:** Integrate within development routines.

## Security & Git Instructions

All commands that interact with git include security instructions to prevent AI attribution.

## Contributing

We welcome contributions that help developers save time. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) file for details.
*Built by a developer tired of typing "please act like a senior engineer" in every conversation.*

---

**Last Updated:** August 2, 2025 (Based on Anthropic Claude Code CLI documentation v2025.08.01)
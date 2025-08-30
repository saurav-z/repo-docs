# CCPlugins: Supercharge Your Claude Code CLI for Faster Development

**Stop wasting time on repetitive coding tasks!** CCPlugins is a curated set of professional commands designed to boost your productivity with Claude Code CLI, saving you hours each week.  Check out the original repo: [https://github.com/brennercruvinel/CCPlugins](https://github.com/brennercruvinel/CCPlugins)

[![GitHub Repo stars](https://img.shields.io/github/stars/brennercruvinel/CCPlugins?style=social)](https://github.com/brennercruvinel/CCPlugins)
[![Version](https://img.shields.io/badge/version-2.5.2-blue.svg)](https://github.com/brennercruvinel/CCPlugins)
[![Claude Code CLI](https://img.shields.io/badge/for-Claude%20Code%20CLI-purple.svg)](https://docs.anthropic.com/en/docs/claude-code)
[![Tested on](https://img.shields.io/badge/tested%20on-Opus%204%20%26%20Sonnet%204-orange.svg)](https://claude.ai)
[![Also works with](https://img.shields.io/badge/also%20works%20with-Kimi%20K2-1783ff.svg)](https://github.com/MoonshotAI/Kimi-K2)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/brennercruvinel/CCPlugins/blob/main/CONTRIBUTING.md)

## Key Features

*   **Automated Workflows:** Streamline your development process with commands for code formatting, testing, refactoring, and more.
*   **Enhanced Code Quality & Security:**  Integrate advanced code analysis, vulnerability scanning, and proactive issue detection directly into your workflow.
*   **Intelligent Context Awareness:** Commands leverage Claude's understanding of your project to provide tailored, effective solutions.
*   **Time-Saving Solutions:**  Reduce manual effort on tasks like security analysis, architecture reviews, and code cleanup, saving you hours each week.
*   **Seamless Integration:**  Works with your existing workflow, offering cross-platform support and compatibility with various programming languages and frameworks.

## Core Commands

### Development Workflow

*   `/cleanproject`: Remove debug artifacts safely with Git.
*   `/commit`:  Smart, conventional commits with analysis.
*   `/format`: Auto-detect and apply project formatting.
*   `/scaffold feature-name`:  Generate complete features from patterns.
*   `/test`: Run tests with intelligent failure analysis.
*   `/implement url/path/feature`:  Import and adapt code from sources.
*   `/refactor`: Intelligent code restructuring with validation.

### Code Quality & Security

*   `/review`: Multi-agent analysis for security, performance, quality, and architecture.
*   `/security-scan`: Vulnerability analysis with remediation tracking.
*   `/predict-issues`: Proactive problem detection with timeline estimates.
*   `/remove-comments`:  Clean obvious comments, preserve valuable docs.
*   `/fix-imports`: Repair broken imports after refactoring.
*   `/find-todos`: Locate and organize development tasks.
*   `/create-todos`: Add contextual TODO comments.
*   `/fix-todos`: Intelligently implement TODO fixes.

### Advanced Analysis

*   `/understand`: Analyze entire project architecture.
*   `/explain-like-senior`: Senior-level code explanations.
*   `/contributing`: Contribution readiness analysis.
*   `/make-it-pretty`: Improve readability without functional changes.

### Session & Project Management

*   `/session-start`: Begin documented sessions.
*   `/session-end`: Summarize and preserve session context.
*   `/docs`: Smart documentation management and updates.
*   `/todos-to-issues`: Convert code TODOs to GitHub issues.
*   `/undo`: Safe rollback with Git checkpoint restore.

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

## How It Works

CCPlugins extends Claude Code CLI with professional development workflows by leveraging:

*   **Intelligent Instructions:**  First-person conversational design and strategic thinking sections.
*   **Native Tool Integration:** Built using Claude Code CLI's native tools like Grep, Glob, and Read.
*   **Safety-First Design:** Automatic Git checkpoints before destructive operations, session persistence, and rollback capabilities.
*   **Universal Compatibility:**  Framework-agnostic design that adapts to your project.

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

**Last Updated:** August 2, 2025 (Based on Anthropic Claude Code CLI documentation v2025.08.01)
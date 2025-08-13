# CCPlugins: Supercharge Your Claude Code CLI Workflows (Save Time, Code Smarter)

**Tired of repetitive coding tasks? CCPlugins is a curated collection of commands that automates development workflows within Claude Code CLI, saving you hours each week.** [Explore the Repo](https://github.com/brennercruvinel/CCPlugins)

[![GitHub Repo stars](https://img.shields.io/github/stars/brennercruvinel/CCPlugins?style=social)](https://github.com/brennercruvinel/CCPlugins)
[![Version](https://img.shields.io/badge/version-2.5.2-blue.svg)](https://github.com/brennercruvinel/CCPlugins)
[![Claude Code CLI](https://img.shields.io/badge/for-Claude%20Code%20CLI-purple.svg)](https://docs.anthropic.com/en/docs/claude-code)
[![Tested on](https://img.shields.io/badge/tested%20on-Opus%204%20%26%20Sonnet%204-orange.svg)](https://claude.ai)
[![Also works with](https://img.shields.io/badge/also%20works%20with-Kimi%20K2-1783ff.svg)](https://github.com/MoonshotAI/Kimi-K2)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/brennercruvinel/CCPlugins/blob/main/CONTRIBUTING.md)

## Key Features

*   **Automated Workflows:** Streamline development with commands for common tasks.
*   **Enterprise-Grade Code Quality:** Enhance your code with security scans, reviews, and refactoring tools.
*   **Intelligent Analysis:** Leverage Claude's contextual understanding for smarter solutions.
*   **Time Savings:** Save up to 4-5 hours per week with automated processes.
*   **Easy Integration:** Simple installation and seamless integration with your workflow.
*   **Production-Ready Tools:** Benefit from battle-tested commands that solve real developer problems.

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

CCPlugins provides a comprehensive suite of commands designed to boost your productivity.

### Development Workflow

*   `/cleanproject`: Remove debug artifacts with git safety.
*   `/commit`: Smart conventional commits with analysis.
*   `/format`: Auto-detect and apply project formatter.
*   `/scaffold feature-name`: Generate complete features from patterns.
*   `/test`: Run tests with intelligent failure analysis.
*   `/implement url/path/feature`: Import and adapt code from any source with validation phase.
*   `/refactor`: Intelligent code restructuring with validation & de-para mapping.

### Code Quality & Security

*   `/review`: Multi-agent analysis (security, performance, quality, architecture).
*   `/security-scan`: Vulnerability analysis with extended thinking & remediation tracking.
*   `/predict-issues`: Proactive problem detection with timeline estimates.
*   `/remove-comments`: Clean obvious comments, preserve valuable docs.
*   `/fix-imports`: Repair broken imports after refactoring.
*   `/find-todos`: Locate and organize development tasks.
*   `/create-todos`: Add contextual TODO comments based on analysis results.
*   `/fix-todos`: Intelligently implement TODO fixes with context.

### Advanced Analysis

*   `/understand`: Analyze entire project architecture and patterns.
*   `/explain-like-senior`: Senior-level code explanations with context.
*   `/contributing`: Complete contribution readiness analysis.
*   `/make-it-pretty`: Improve readability without functional changes.

### Session & Project Management

*   `/session-start`: Begin documented sessions with CLAUDE.md integration.
*   `/session-end`: Summarize and preserve session context.
*   `/docs`: Smart documentation management and updates.
*   `/todos-to-issues`: Convert code TODOs to GitHub issues.
*   `/undo`: Safe rollback with git checkpoint restore.

## How It Works

CCPlugins extends Claude Code CLI through a sophisticated architecture:

1.  **Command Input:** You enter a command.
2.  **Contextual Analysis:** The plugin analyzes your project.
3.  **Intelligent Planning:** An execution strategy is created.
4.  **Safe Execution:** Actions are performed with built-in safety measures.
5.  **Clear Feedback:** Results, next steps, and warnings are provided.

## Enhanced Features

*   **Validation & Refinement:** Validate and refine complex commands with validation phases (`/refactor validate`, `/implement validate`).
*   **Extended Thinking:** Advanced analysis for refactoring and security, improving efficiency.
*   **Pragmatic Integration:** Workflow suggestions for a seamless experience.

## Real-World Example

Before `/cleanproject`:

```
src/
├── UserService.js
├── UserService.test.js
├── UserService_backup.js    # Old version
├── debug.log               # Debug output
├── test_temp.js           # Temporary test
└── notes.txt              # Dev notes
```

After `/cleanproject`:

```
src/
├── UserService.js          # Clean production code
└── UserService.test.js     # Actual tests preserved
```

## Requirements

*   Claude Code CLI
*   Python 3.6+ (for installer)
*   Git (for version control commands)

## Contributing

Help improve CCPlugins! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Community

[![Star History Chart](https://api.star-history.com/svg?repos=brennercruvinel/CCPlugins&type=Date)](https://star-history.com/#brennercruvinel/CCPlugins&Date)

---

**Last Updated:** August 2, 2025 (Based on Anthropic Claude Code CLI documentation v2025.08.01)
# CCPlugins: Supercharge Your Claude Code CLI with AI-Powered Development Workflows

**Stop wasting time on repetitive development tasks!** CCPlugins is a curated collection of powerful commands designed to automate your coding workflow within the Claude Code CLI, saving you hours each week. [Check out the original repo for CCPlugins](https://github.com/brennercruvinel/CCPlugins).

[![GitHub Repo stars](https://img.shields.io/github/stars/brennercruvinel/CCPlugins?style=social)](https://github.com/brennercruvinel/CCPlugins)
[![Version](https://img.shields.io/badge/version-2.5.2-blue.svg)](https://github.com/brennercruvinel/CCPlugins)
[![Claude Code CLI](https://img.shields.io/badge/for-Claude%20Code%20CLI-purple.svg)](https://docs.anthropic.com/en/docs/claude-code)
[![Tested on](https://img.shields.io/badge/tested%20on-Opus%204%20%26%20Sonnet%204-orange.svg)](https://claude.ai)
[![Also works with](https://img.shields.io/badge/also%20works%20with-Kimi%20K2-1783ff.svg)](https://github.com/MoonshotAI/Kimi-K2)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/brennercruvinel/CCPlugins/blob/main/CONTRIBUTING.md)

## Key Features:

*   **Automated Workflows:** Streamline common tasks like code formatting, testing, and security scanning.
*   **Smart Code Analysis:** Leverage AI to understand, review, and improve your codebase.
*   **Enhanced Validation & Refinement:** Complex commands include validation phases for improved accuracy.
*   **Production-Ready Commands:** Benefit from battle-tested tools that solve real-world developer problems.
*   **Time-Saving:** Reduce manual effort and increase productivity with advanced automation.

## Core Commands:

*   `/cleanproject`: Removes debug artifacts safely.
*   `/commit`: Creates smart, conventional commits.
*   `/format`: Auto-formats your code.
*   `/scaffold feature-name`: Generates feature templates.
*   `/test`: Runs tests with smart failure analysis.
*   `/implement`: Imports and adapts code from any source.
*   `/refactor`: Restructures code intelligently.
*   `/review`: Provides multi-agent analysis (security, performance, quality).
*   `/security-scan`: Scans for vulnerabilities.
*   `/predict-issues`: Proactively detects potential issues.
*   `/remove-comments`: Cleans obvious comments.
*   `/fix-imports`: Repairs broken imports.
*   `/find-todos`: Locates and organizes tasks.
*   `/create-todos`: Adds contextual TODOs.
*   `/fix-todos`: Implements TODO fixes with context.
*   `/understand`: Analyzes project architecture.
*   `/explain-like-senior`: Provides senior-level code explanations.
*   `/contributing`: Analyzes contribution readiness.
*   `/make-it-pretty`: Improves code readability.
*   `/session-start`: Starts documented sessions.
*   `/session-end`: Summarizes and saves sessions.
*   `/docs`: Manages and updates documentation.
*   `/todos-to-issues`: Converts TODOs to GitHub issues.
*   `/undo`: Safely rolls back changes.

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

## How It Works:

CCPlugins extends the Claude Code CLI using a unique architecture that leverages the power of AI to enhance your development process, offering clear feedback and intelligent execution.

## Real-World Example:

`/cleanproject` - Before and after:

**Before**:
```
src/
├── UserService.js
├── UserService.test.js
├── UserService_backup.js    # Old version
├── debug.log               # Debug output
├── test_temp.js           # Temporary test
└── notes.txt              # Dev notes
```

**After**:
```
src/
├── UserService.js          # Clean production code
└── UserService.test.js     # Actual tests preserved
```

## Requirements:

*   Claude Code CLI
*   Python 3.6+ (for installer)
*   Git (for version control commands)

## Contributing:

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License:

MIT License - see [LICENSE](LICENSE) file for details.

---

**Last Updated:** August 2, 2025 (Based on Anthropic Claude Code CLI documentation v2025.08.01)
# CCPlugins: Supercharge Your Claude Code CLI for Faster, Smarter Development

**Stop wasting time on repetitive tasks and start coding!** CCPlugins is a powerful collection of professional commands for the Claude Code CLI, designed to save you hours each week by automating essential development workflows. [Explore CCPlugins on GitHub](https://github.com/brennercruvinel/CCPlugins).

[![GitHub Repo stars](https://img.shields.io/github/stars/brennercruvinel/CCPlugins?style=social)]
[![Version](https://img.shields.io/badge/version-2.5.2-blue.svg)](https://github.com/brennercruvinel/CCPlugins)
[![Claude Code CLI](https://img.shields.io/badge/for-Claude%20Code%20CLI-purple.svg)](https://docs.anthropic.com/en/docs/claude-code)
[![Tested on](https://img.shields.io/badge/tested%20on-Opus%204%20%26%20Sonnet%204-orange.svg)](https://claude.ai)
[![Also works with](https://img.shields.io/badge/also%20works%20with-Kimi%20K2-1783ff.svg)](https://github.com/MoonshotAI/Kimi-K2)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/brennercruvinel/CCPlugins/blob/main/CONTRIBUTING.md)

## Key Features

*   **Automated Workflows:** Streamline your development with pre-built commands for common tasks.
*   **Time Savings:** Save an estimated 4-5 hours per week by automating code quality, security, and more.
*   **Intelligent Analysis:** Leverage Claude's capabilities for comprehensive code review, vulnerability detection, and architecture analysis.
*   **Enhanced Validation & Refinement:** Ensure complete and correct outcomes through built-in validation phases.
*   **Seamless Integration:**  Easy installation and integration with your existing Claude Code CLI setup.

## Core Capabilities

*   **Development Workflow:**
    *   `/cleanproject`: Remove debug artifacts safely
    *   `/commit`: Smart, conventional commits
    *   `/format`: Auto-detect and format code
    *   `/scaffold feature-name`: Generate complete features from templates
    *   `/test`: Run tests with intelligent failure analysis
    *   `/implement url/path/feature`: Import code from any source
    *   `/refactor`: Intelligent code restructuring
*   **Code Quality & Security:**
    *   `/review`: Multi-agent code analysis
    *   `/security-scan`: Vulnerability analysis
    *   `/predict-issues`: Proactive issue detection
    *   `/remove-comments`: Clean comments
    *   `/fix-imports`: Repair broken imports
    *   `/find-todos`: Locate development tasks
    *   `/create-todos`: Add contextual TODOs
    *   `/fix-todos`: Intelligently fix TODOs
*   **Advanced Analysis:**
    *   `/understand`: Project architecture analysis
    *   `/explain-like-senior`: Senior-level code explanations
    *   `/contributing`: Contribution readiness analysis
    *   `/make-it-pretty`: Improve readability
*   **Session & Project Management:**
    *   `/session-start`: Begin documented sessions
    *   `/session-end`: Summarize and preserve context
    *   `/docs`: Smart documentation management
    *   `/todos-to-issues`: Convert TODOs to GitHub issues
    *   `/undo`: Safe rollback
    *   **Validation Phases**: `/refactor validate`, `/implement validate`

## Installation

### Quick Install (Mac/Linux):

```bash
curl -sSL https://raw.githubusercontent.com/brennercruvinel/CCPlugins/main/install.sh | bash
```

### Quick Install (Windows/Cross-platform):

```bash
python install.py
```

### Manual Install:

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

CCPlugins enhances Claude Code CLI by:

*   **Intelligent Instructions:** First-person language activates Claude's collaborative reasoning.
*   **Native Tool Integration:** Uses Claude Code CLI's built-in tools for efficiency.
*   **Safety-First Design:** Includes git checkpoints and rollback capabilities.
*   **Framework-Agnostic:** Works across different programming languages and frameworks.

## Real-World Example: `/cleanproject`

**Before:**

```
src/
├── UserService.js
├── UserService.test.js
├── UserService_backup.js    # Old version
├── debug.log               # Debug output
├── test_temp.js           # Temporary test
└── notes.txt              # Dev notes
```

**After:**

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

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.
*Built by a developer tired of typing "please act like a senior engineer" in every conversation.*
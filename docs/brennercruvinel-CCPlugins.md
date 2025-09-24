<!-- SEO-optimized README -->

# CCPlugins: Supercharge Your Claude Code CLI Workflow with AI-Powered Automation

**Tired of repetitive tasks slowing down your development?** CCPlugins is your solution, offering 24 professional commands that transform Claude Code CLI into a powerful AI-driven development assistant, saving you hours each week.  Check it out on [GitHub](https://github.com/brennercruvinel/CCPlugins)!

[![GitHub Repo stars](https://img.shields.io/github/stars/brennercruvinel/CCPlugins?style=social)](https://github.com/brennercruvinel/CCPlugins)
[![Version](https://img.shields.io/badge/version-2.5.2-blue.svg)](https://github.com/brennercruvinel/CCPlugins)
[![Claude Code CLI](https://img.shields.io/badge/for-Claude%20Code%20CLI-purple.svg)](https://docs.anthropic.com/en/docs/claude-code)
[![Tested on](https://img.shields.io/badge/tested%20on-Opus%204%20%26%20Sonnet%204-orange.svg)](https://claude.ai)
[![Also works with](https://img.shields.io/badge/also%20works%20with-Kimi%20K2-1783ff.svg)](https://github.com/MoonshotAI/Kimi-K2)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/brennercruvinel/CCPlugins/blob/main/CONTRIBUTING.md)

## Key Features

*   **Automated Development Workflows:** Streamline common tasks with commands like `/cleanproject`, `/commit`, `/format`, and `/scaffold`.
*   **Enhanced Code Quality & Security:** Improve your code with features like security scanning, code reviews, and proactive issue prediction.
*   **Intelligent Analysis:** Utilize advanced commands for project architecture understanding, senior-level explanations, and comprehensive code analysis.
*   **Seamless Project Management:** Simplify session management with commands for starting, ending, and documenting development sessions.
*   **Validation & Refinement:** Complex commands include validation steps to ensure completeness and accuracy in your development process.
*   **Pragmatic Command Integration:** Natural workflow suggestions like `/test` after changes & `/commit` at checkpoints for optimal efficiency.

## What is CCPlugins?

CCPlugins is a curated collection of professional-grade commands designed specifically to enhance the Claude Code CLI experience.  It provides developers with tools to automate tedious tasks, improve code quality, and accelerate development cycles by leveraging the power of AI. These commands are optimized for Opus 4 and Sonnet 4 models.

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

Streamline your development process with these powerful commands:

### Development Workflow

*   `/cleanproject`: Removes debug artifacts with git safety
*   `/commit`: Creates smart conventional commits with analysis
*   `/format`: Applies project formatting automatically
*   `/scaffold feature-name`: Generates complete features from patterns
*   `/test`: Runs tests with intelligent failure analysis
*   `/implement url/path/feature`: Imports and adapts code with validation
*   `/refactor`: Performs intelligent code restructuring

### Code Quality & Security

*   `/review`: Executes multi-agent analysis (security, performance, quality)
*   `/security-scan`: Performs vulnerability analysis with remediation
*   `/predict-issues`: Proactively detects potential problems
*   `/remove-comments`: Cleans comments while preserving valuable docs
*   `/fix-imports`: Repairs broken imports after refactoring
*   `/find-todos`: Locates and organizes development tasks
*   `/create-todos`: Adds contextual TODO comments
*   `/fix-todos`: Implements TODO fixes intelligently

### Advanced Analysis

*   `/understand`: Analyzes entire project architecture
*   `/explain-like-senior`: Provides senior-level code explanations
*   `/contributing`: Conducts contribution readiness analysis
*   `/make-it-pretty`: Enhances readability without changes

### Session & Project Management

*   `/session-start`: Begins documented sessions with CLAUDE.md
*   `/session-end`: Summarizes and saves session context
*   `/docs`: Smart documentation management and updates
*   `/todos-to-issues`: Converts TODOs into GitHub issues
*   `/undo`: Performs safe rollback with git checkpoint restore

## How It Works

CCPlugins acts as a powerful development assistant by combining the capabilities of Claude Code CLI with intelligent automation. This process involves:

1.  **Command Loading:** Claude reads the markdown definition from `~/.claude/commands/`
2.  **Context Analysis:** Analyzes your project structure and current state
3.  **Intelligent Planning:** Creates execution strategy
4.  **Safe Execution:** Performs actions with checkpoints and validation
5.  **Clear Feedback:** Provides results and next steps

## Advanced Features

### Validation & Refinement

Complex commands are enhanced with validation phases to ensure comprehensive results. This includes `/refactor validate` and `/implement validate`, which verify code migration and integration completeness.

### Extended Thinking

CCPlugins utilizes advanced analysis for complex scenarios like deep architectural refactoring and sophisticated vulnerability detection.

### Pragmatic Command Integration

Commands offer natural suggestions to streamline your workflow. For example, `/test` is suggested after changes and `/commit` at logical checkpoints.

## Example: /cleanproject in Action

Before:

```
src/
├── UserService.js
├── UserService.test.js
├── UserService_backup.js    # Old version
├── debug.log               # Debug output
├── test_temp.js           # Temporary test
└── notes.txt              # Dev notes
```

After:

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

We welcome contributions!  Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

MIT License - See the [LICENSE](LICENSE) file for details.

---

**Last Updated:** August 2, 2025 (Based on Anthropic Claude Code CLI documentation v2025.08.01)
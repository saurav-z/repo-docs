# CCPlugins: Supercharge Your Claude Code CLI Productivity 🚀

**Tired of repetitive development tasks?** CCPlugins is a powerful suite of professional commands that automates your coding workflow, saving you hours each week.  [Check out the original repo](https://github.com/brennercruvinel/CCPlugins) for more information.

[![GitHub Repo stars](https://img.shields.io/github/stars/brennercruvinel/CCPlugins?style=social)](https://github.com/brennercruvinel/CCPlugins)
[![Version](https://img.shields.io/badge/version-2.5.2-blue.svg)](https://github.com/brennercruvinel/CCPlugins)
[![Claude Code CLI](https://img.shields.io/badge/for-Claude%20Code%20CLI-purple.svg)](https://docs.anthropic.com/en/docs/claude-code)
[![Tested on](https://img.shields.io/badge/tested%20on-Opus%204%20%26%20Sonnet%204-orange.svg)](https://claude.ai)
[![Also works with](https://img.shields.io/badge/also%20works%20with-Kimi%20K2-1783ff.svg)](https://github.com/MoonshotAI/Kimi-K2)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/brennercruvinel/CCPlugins/blob/main/CONTRIBUTING.md)

**Key Features:**

*   **Automated Workflows:** Streamline development tasks with 24+ pre-built commands.
*   **Smart Code Analysis:** Leverage advanced analysis for security, quality, and architecture.
*   **Validation & Refinement:** Ensure complete and accurate code transformations.
*   **Session Management:**  Track and manage your development sessions effectively.
*   **Git Integration:** Safe and seamless integration with version control.
*   **Extensive Testing:** Rigorously tested on Opus 4, Sonnet 4, and compatible with Kimi K2.
*   **Time Savings:** Save up to 5 hours per week on common development tasks.

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

## Commands Overview

CCPlugins offers a comprehensive suite of commands categorized for efficient workflow management.

### 🚀 Development Workflow

*   `/cleanproject`: Remove debug artifacts with git safety
*   `/commit`: Smart conventional commits with analysis
*   `/format`: Auto-detect and apply project formatter
*   `/scaffold feature-name`: Generate complete features from patterns
*   `/test`: Run tests with intelligent failure analysis
*   `/implement url/path/feature`: Import and adapt code from any source with validation phase
*   `/refactor`: Intelligent code restructuring with validation & de-para mapping

### 🛡️ Code Quality & Security

*   `/review`: Multi-agent analysis (security, performance, quality, architecture)
*   `/security-scan`: Vulnerability analysis with extended thinking & remediation tracking
*   `/predict-issues`: Proactive problem detection with timeline estimates
*   `/remove-comments`: Clean obvious comments, preserve valuable docs
*   `/fix-imports`: Repair broken imports after refactoring
*   `/find-todos`: Locate and organize development tasks
*   `/create-todos`: Add contextual TODO comments based on analysis results
*   `/fix-todos`: Intelligently implement TODO fixes with context

### 🔍 Advanced Analysis

*   `/understand`: Analyze entire project architecture and patterns
*   `/explain-like-senior`: Senior-level code explanations with context
*   `/contributing`: Complete contribution readiness analysis
*   `/make-it-pretty`: Improve readability without functional changes

### 📋 Session & Project Management

*   `/session-start`: Begin documented sessions with CLAUDE.md integration
*   `/session-end`: Summarize and preserve session context
*   `/docs`: Smart documentation management and updates
*   `/todos-to-issues`: Convert code TODOs to GitHub issues
*   `/undo`: Safe rollback with git checkpoint restore

## Enhanced Features

*   **Validation & Refinement:**  Commands like `/refactor validate` and `/implement validate` ensure code completeness.
*   **Extended Thinking:** Advanced analysis capabilities for refactoring, security, and more.
*   **Pragmatic Integration:**  Workflow suggestions promote seamless transitions (e.g., `/test` after refactoring).

## Real-World Example

**(Before `/cleanproject`)**
```
src/
├── UserService.js
├── UserService.test.js
├── UserService_backup.js    # Old version
├── debug.log               # Debug output
├── test_temp.js           # Temporary test
└── notes.txt              # Dev notes
```

**(After `/cleanproject`)**
```
src/
├── UserService.js          # Clean production code
└── UserService.test.js     # Actual tests preserved
```

## How It Works

CCPlugins transforms Claude Code CLI into an intelligent development assistant:

1.  **Command Execution:** Developer enters a command (e.g., `/refactor`).
2.  **Context Analysis:** CCPlugins analyzes project structure, technology stack, and state.
3.  **Intelligent Planning:** Claude Code CLI creates an execution strategy.
4.  **Safe Execution:** Actions are performed with automatic git checkpoints and validation.
5.  **Clear Feedback:** Results and next steps are provided.

### Core Components

*   **Intelligent Instructions:** Conversational design that encourages collaborative reasoning.
*   **Native Tool Integration:** Leveraging Claude Code CLI's internal tooling.
*   **Safety-First Design:**  Git checkpoints and rollback capabilities.
*   **Universal Compatibility:** Framework-agnostic and adaptable to your project.

### Advanced Features

*   **Session Continuity:** State management across sessions with commands like `/implement` and `/refactor`.
*   **Multi-Agent Architecture:** Specialized sub-agents for security, performance, and code quality.
*   **Performance Optimizations:** Verbosity reduction, smart caching, and parallel execution.

## Requirements

*   Claude Code CLI
*   Python 3.6+ (for installer)
*   Git (for version control commands)

## Advanced Usage

**Creating Custom Commands**
Create your own commands by adding markdown files to `~/.claude/commands/`:

```markdown
# My Custom Command

I'll help you with your specific workflow.

[Your instructions here]
```

**Using Arguments**
Commands support arguments via `$ARGUMENTS`:

```bash
/mycommand some-file.js
# $ARGUMENTS will contain "some-file.js"
```

**CI/CD Integration**
Use commands in automated workflows:

```bash
# Quality pipeline
claude "/security-scan" && claude "/review" && claude "/test"

# Pre-commit validation  
claude "/format" && claude "/commit"

# Feature development
claude "/scaffold api-users" && claude "/test"

# Complete workflow
claude "/security-scan" && claude "/create-todos" && claude "/todos-to-issues"

# TODO resolution workflow
claude "/find-todos" && claude "/fix-todos" && claude "/test"
```

**Manual Workflow Integration**
Perfect for development routines:

```bash
# Morning routine
claude "/session-start"
claude "/security-scan"

# During development
claude "/scaffold user-management"
claude "/review" 
claude "/format"

# End of day
claude "/commit"
claude "/session-end"
```

## Security & Git Instructions

All commands that interact with git include security instructions to prevent AI attribution. These commands will never add "Co-authored-by" or AI signatures.

**Commands with git protection:**
- `/commit`, `/scaffold`, `/make-it-pretty`, `/cleanproject`, `/fix-imports`, `/review`, `/security-scan`
- `/contributing`, `/todos-to-issues`, `/predict-issues`, `/find-todos`, `/create-todos`, `/fix-todos`

You can modify these instructions in individual command files if needed.

## Contributing

Contribute to CCPlugins and help other developers save time.  See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) file for details.
*Built by a developer tired of typing "please act like a senior engineer" in every conversation.*

## Community

[![Star History Chart](https://api.star-history.com/svg?repos=brennercruvinel/CCPlugins&type=Date)](https://star-history.com/#brennercruvinel/CCPlugins&Date)

---

**Last Updated:** August 2, 2025 (Based on Anthropic Claude Code CLI documentation v2025.08.01)
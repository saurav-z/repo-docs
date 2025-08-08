# Supercharge Your Development with CCPlugins: AI-Powered Commands for Claude Code CLI

Tired of repetitive coding tasks? **CCPlugins** is your solution, offering a suite of professional commands for Claude Code CLI, saving you valuable time and boosting your productivity.  [Explore CCPlugins on GitHub!](https://github.com/brennercruvinel/CCPlugins)

[![GitHub Repo stars](https://img.shields.io/github/stars/brennercruvinel/CCPlugins?style=social)](https://github.com/brennercruvinel/CCPlugins)
[![Version](https://img.shields.io/badge/version-2.5.2-blue.svg)](https://github.com/brennercruvinel/CCPlugins)
[![Claude Code CLI](https://img.shields.io/badge/for-Claude%20Code%20CLI-purple.svg)](https://docs.anthropic.com/en/docs/claude-code)
[![Tested on](https://img.shields.io/badge/tested%20on-Opus%204%20%26%20Sonnet%204-orange.svg)](https://claude.ai)
[![Also works with](https://img.shields.io/badge/also%20works%20with-Kimi%20K2-1783ff.svg)](https://github.com/MoonshotAI/Kimi-K2)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/brennercruvinel/CCPlugins/blob/main/CONTRIBUTING.md)

## Key Features:

*   🚀 **Automated Workflows:** Streamline development with commands for code formatting, testing, and more.
*   🛡️ **Enhanced Code Quality:** Perform security scans, identify potential issues, and improve code readability.
*   🔍 **Advanced Analysis:** Utilize project architecture analysis, senior-level code explanations, and contributing readiness checks.
*   ⚙️ **Validation and Refinement:** Leverage validation phases and extended thinking for improved results.
*   🧠 **AI-Powered Efficiency:** Integrate with Claude Code CLI for intelligent code assistance, significantly reducing manual effort.

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

CCPlugins provides 24 professional commands to optimize your workflow:

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

## How It Works

CCPlugins extends Claude Code CLI's capabilities with an intelligent development assistant. This is achieved through a sophisticated yet elegant architecture:

1.  **Command Loading:** Claude reads the markdown definition from `~/.claude/commands/`.
2.  **Context Analysis:** Analyzes your project structure, technology stack, and current state.
3.  **Intelligent Planning:** Creates an execution strategy based on your specific situation.
4.  **Safe Execution:** Performs actions with automatic checkpoints and validation.
5.  **Clear Feedback:** Provides results, next steps, and any warnings.

## Advanced Features

### Validation & Refinement

Complex commands now include validation phases:

*   `/refactor validate`: Find remaining old patterns, verify 100% migration
*   `/implement validate`: Check integration completeness, find loose ends

### Extended Thinking

Advanced analysis for complex scenarios:

*   **Refactoring**: Deep architectural analysis for large-scale changes
*   **Security**: Sophisticated vulnerability detection with chain analysis

### Pragmatic Command Integration

Natural workflow suggestions without over-engineering:

*   Suggests `/test` after major changes
*   Recommends `/commit` at logical checkpoints
*   Maintains user control, no automatic execution

## Real World Example

### Before `/cleanproject`:

```
src/
├── UserService.js
├── UserService.test.js
├── UserService_backup.js    # Old version
├── debug.log               # Debug output
├── test_temp.js           # Temporary test
└── notes.txt              # Dev notes
```

### After `/cleanproject`:

```
src/
├── UserService.js          # Clean production code
└── UserService.test.js     # Actual tests preserved
```

## Performance Metrics

| Task                  | Manual Time    | With CCPlugins     | Time Saved  |
| :-------------------- | :------------- | :----------------- | :---------- |
| Security analysis     | 45-60 min      | 3-5 min            | ~50 min     |
| Architecture review   | 30-45 min      | 5-8 min            | ~35 min     |
| Feature scaffolding   | 25-40 min      | 2-3 min            | ~30 min     |
| Git commits           | 5-10 min       | 30 sec             | ~9 min      |
| Code cleanup          | 20-30 min      | 1 min              | ~25 min     |
| Import fixing         | 15-25 min      | 1-2 min            | ~20 min     |
| Code review           | 20-30 min      | 2-4 min            | ~20 min     |
| Issue prediction      | 60+ min        | 5-10 min           | ~50 min     |
| TODO resolution       | 30-45 min      | 3-5 min            | ~35 min     |
| Code adaptation       | 40-60 min      | 3-5 min            | ~45 min     |
| **Total**             |                | **~4-5 hours/week** |             |

## Requirements

*   Claude Code CLI
*   Python 3.6+ (for installer)
*   Git (for version control commands)

## Advanced Usage

### Creating Custom Commands

Create your own commands by adding markdown files to `~/.claude/commands/`:

```markdown
# My Custom Command

I'll help you with your specific workflow.

[Your instructions here]
```

### Using Arguments

Commands support arguments via `$ARGUMENTS`:

```bash
/mycommand some-file.js
# $ARGUMENTS will contain "some-file.js"
```

### CI/CD Integration

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

### Manual Workflow Integration

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

All commands that interact with git include security instructions to prevent AI attribution:

**Commands with git protection:**

*   `/commit`, `/scaffold`, `/make-it-pretty`, `/cleanproject`, `/fix-imports`, `/review`, `/security-scan`
*   `/contributing`, `/todos-to-issues`, `/predict-issues`, `/find-todos`, `/create-todos`, `/fix-todos`

These commands will NEVER:

*   Add "Co-authored-by" or AI signatures
*   Include "Generated with Claude Code" messages
*   Modify git config or credentials
*   Add AI attribution to commits/issues

You can modify these instructions in individual command files if needed.

## Contributing

We welcome contributions. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) file for details.
*Built by a developer tired of typing "please act like a senior engineer" in every conversation.*

## Community

[![Star History Chart](https://api.star-history.com/svg?repos=brennercruvinel/CCPlugins&type=Date)](https://star-history.com/#brennercruvinel/CCPlugins&Date)

---

**Last Updated:** August 2, 2025 (Based on Anthropic Claude Code CLI documentation v2025.08.01)
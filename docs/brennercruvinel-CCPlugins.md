# CCPlugins: Supercharge Your Claude Code CLI for Faster Development

**Tired of repetitive coding tasks and over-engineered AI responses?** CCPlugins provides enterprise-grade commands for Claude Code CLI, saving developers hours each week.  [Explore CCPlugins on GitHub](https://github.com/brennercruvinel/CCPlugins).

[![GitHub Repo stars](https://img.shields.io/github/stars/brennercruvinel/CCPlugins?style=social)]
[![Version](https://img.shields.io/badge/version-2.5.2-blue.svg)](https://github.com/brennercruvinel/CCPlugins)
[![Claude Code CLI](https://img.shields.io/badge/for-Claude%20Code%20CLI-purple.svg)](https://docs.anthropic.com/en/docs/claude-code)
[![Tested on](https://img.shields.io/badge/tested%20on-Opus%204%20%26%20Sonnet%204-orange.svg)](https://claude.ai)
[![Also works with](https://img.shields.io/badge/also%20works%20with-Kimi%20K2-1783ff.svg)](https://github.com/MoonshotAI/Kimi-K2)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/brennercruvinel/CCPlugins/blob/main/CONTRIBUTING.md)

## Key Features

*   **Automated Workflows:** Streamline development with commands for code quality, security, and project management.
*   **Enhanced Validation & Refinement:** Complex commands include validation phases to ensure complete and accurate results.
*   **Intelligent Analysis:** Leverage advanced analysis for refactoring, security scanning, and proactive issue detection.
*   **Seamless Integration:** Works with your existing Claude Code CLI setup and integrates easily into your development routine.
*   **Time Savings:** Save an estimated **4-5 hours per week** with professional-grade code analysis and automation.

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

Optimize your development workflow with these professional commands:

### Development Workflow

*   `/cleanproject`: Remove debug artifacts with git safety
*   `/commit`: Smart conventional commits with analysis
*   `/format`: Auto-detect and apply project formatter
*   `/scaffold feature-name`: Generate complete features from patterns
*   `/test`: Run tests with intelligent failure analysis
*   `/implement url/path/feature`: Import and adapt code with validation
*   `/refactor`: Intelligent code restructuring with de-para mapping

### Code Quality & Security

*   `/review`: Multi-agent analysis (security, performance, quality)
*   `/security-scan`: Vulnerability analysis with extended thinking
*   `/predict-issues`: Proactive problem detection with estimates
*   `/remove-comments`: Clean obvious comments, preserve docs
*   `/fix-imports`: Repair broken imports after refactoring
*   `/find-todos`: Locate and organize development tasks
*   `/create-todos`: Add contextual TODO comments
*   `/fix-todos`: Intelligently implement TODO fixes

### Advanced Analysis

*   `/understand`: Analyze entire project architecture
*   `/explain-like-senior`: Senior-level code explanations
*   `/contributing`: Complete contribution readiness analysis
*   `/make-it-pretty`: Improve readability without functional changes

### Session & Project Management

*   `/session-start`: Begin documented sessions
*   `/session-end`: Summarize and preserve session context
*   `/docs`: Smart documentation management
*   `/todos-to-issues`: Convert code TODOs to GitHub issues
*   `/undo`: Safe rollback with git checkpoint restore

## How It Works

CCPlugins extends Claude Code CLI using a sophisticated architecture for intelligent development assistance.

*   **Command Loading:** Claude reads the command definitions.
*   **Context Analysis:** Analyzes your project structure and tech stack.
*   **Intelligent Planning:** Creates an execution strategy.
*   **Safe Execution:** Performs actions with checkpoints and validation.
*   **Clear Feedback:** Provides results and next steps.

## Technical Notes

*   **Conversational Design:** First-person language enhances collaboration.
*   **Native Tool Integration:** Leverages Claude Code CLI's core capabilities.
*   **Safety-First Design:** Includes git checkpoints and rollback.
*   **Framework Agnostic:** Adapts to your project's conventions.

## Performance Metrics

| Task                 | Manual Time | With CCPlugins | Time Saved |
|----------------------|-------------|----------------|------------|
| Security analysis    | 45-60 min   | 3-5 min        | ~50 min    |
| Architecture review  | 30-45 min   | 5-8 min        | ~35 min    |
| Feature scaffolding | 25-40 min   | 2-3 min        | ~30 min    |
| Git commits          | 5-10 min    | 30 sec         | ~9 min     |
| Code cleanup         | 20-30 min   | 1 min          | ~25 min    |
| Import fixing        | 15-25 min   | 1-2 min        | ~20 min    |
| Code review          | 20-30 min   | 2-4 min        | ~20 min    |
| Issue prediction     | 60+ min     | 5-10 min       | ~50 min    |
| TODO resolution      | 30-45 min   | 3-5 min        | ~35 min    |
| Code adaptation      | 40-60 min   | 3-5 min        | ~45 min    |

**Total: 4-5 hours saved per week.**

## Requirements

*   Claude Code CLI
*   Python 3.6+ (for installer)
*   Git (for version control commands)

## Advanced Usage

### Creating Custom Commands

Add markdown files to `~/.claude/commands/` to create your own commands.

### Using Arguments

Commands support arguments via `$ARGUMENTS`.

### CI/CD Integration

Integrate commands into your automated workflows.

### Manual Workflow Integration

Use the commands in your daily development routines.

## Security & Git Instructions

All commands interacting with git include security instructions to prevent AI attribution.

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) file for details.
*Built by a developer tired of typing "please act like a senior engineer" in every conversation.*

## Community

[![Star History Chart](https://api.star-history.com/svg?repos=brennercruvinel/CCPlugins&type=Date)](https://star-history.com/#brennercruvinel/CCPlugins&Date)

---

**Last Updated:** August 2, 2025 (Based on Anthropic Claude Code CLI documentation v2025.08.01)
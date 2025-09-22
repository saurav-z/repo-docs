# CCPlugins: Supercharge Your Claude Code CLI for Faster Development

**Tired of repetitive coding tasks?** CCPlugins is a set of professional commands for Claude Code CLI designed to automate your development workflow and save you hours each week.  [Check it out on GitHub](https://github.com/brennercruvinel/CCPlugins)!

[![GitHub Repo stars](https://img.shields.io/github/stars/brennercruvinel/CCPlugins?style=social)](https://github.com/brennercruvinel/CCPlugins)
[![Version](https://img.shields.io/badge/version-2.5.2-blue.svg)](https://github.com/brennercruvinel/CCPlugins)
[![Claude Code CLI](https://img.shields.io/badge/for-Claude%20Code%20CLI-purple.svg)](https://docs.anthropic.com/en/docs/claude-code)
[![Tested on](https://img.shields.io/badge/tested%20on-Opus%204%20%26%20Sonnet%204-orange.svg)](https://claude.ai)
[![Also works with](https://img.shields.io/badge/also%20works%20with-Kimi%20K2-1783ff.svg)](https://github.com/MoonshotAI/Kimi-K2)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/brennercruvinel/CCPlugins/blob/main/CONTRIBUTING.md)

## Key Features

*   **Automated Workflow:** Streamline your coding process with commands for refactoring, testing, and code quality checks.
*   **Enhanced Code Quality:**  Automatically identify and fix code quality issues, security vulnerabilities, and TODOs.
*   **Intelligent Analysis:** Benefit from multi-agent analysis for security, performance, quality, and architecture review.
*   **Time Savings:**  Save up to 5 hours per week with automated tasks and smart code suggestions.
*   **Git Integration:**  Includes git-safe operations and automatic backups for a safer development experience.

## What is CCPlugins?

CCPlugins provides a suite of 24 professional commands, designed to extend Claude Code CLI and streamline development tasks. These commands provide predictable outcomes optimized for Opus 4 and Sonnet 4 models.

## Quick Start

### Installation

**Mac/Linux:**

```bash
curl -sSL https://raw.githubusercontent.com/brennercruvinel/CCPlugins/main/install.sh | bash
```

**Windows/Cross-platform:**

```bash
python install.py
```

**Manual Install:**

```bash
git clone https://github.com/brennercruvinel/CCPlugins.git
cd CCPlugins
python install.py
```

### Uninstall

**Mac/Linux:**

```bash
./uninstall.sh
```

**Windows/Cross-platform:**

```bash
python uninstall.py
```

## Core Commands

### Development Workflow

*   `/cleanproject`: Remove debug artifacts with git safety
*   `/commit`: Smart conventional commits with analysis
*   `/format`: Auto-detect and apply project formatter
*   `/scaffold feature-name`: Generate complete features from patterns
*   `/test`: Run tests with intelligent failure analysis
*   `/implement url/path/feature`: Import and adapt code from any source with validation phase
*   `/refactor`: Intelligent code restructuring with validation & de-para mapping

### Code Quality & Security

*   `/review`: Multi-agent analysis (security, performance, quality, architecture)
*   `/security-scan`: Vulnerability analysis with extended thinking & remediation tracking
*   `/predict-issues`: Proactive problem detection with timeline estimates
*   `/remove-comments`: Clean obvious comments, preserve valuable docs
*   `/fix-imports`: Repair broken imports after refactoring
*   `/find-todos`: Locate and organize development tasks
*   `/create-todos`: Add contextual TODO comments based on analysis results
*   `/fix-todos`: Intelligently implement TODO fixes with context

### Advanced Analysis

*   `/understand`: Analyze entire project architecture and patterns
*   `/explain-like-senior`: Senior-level code explanations with context
*   `/contributing`: Complete contribution readiness analysis
*   `/make-it-pretty`: Improve readability without functional changes

### Session & Project Management

*   `/session-start`: Begin documented sessions with CLAUDE.md integration
*   `/session-end`: Summarize and preserve session context
*   `/docs`: Smart documentation management and updates
*   `/todos-to-issues`: Convert code TODOs to GitHub issues
*   `/undo`: Safe rollback with git checkpoint restore

## Real-World Example

**Before `/cleanproject`:**

```
src/
├── UserService.js
├── UserService.test.js
├── UserService_backup.js    # Old version
├── debug.log               # Debug output
├── test_temp.js           # Temporary test
└── notes.txt              # Dev notes
```

**After `/cleanproject`:**

```
src/
├── UserService.js          # Clean production code
└── UserService.test.js     # Actual tests preserved
```

## How It Works

CCPlugins extends Claude Code CLI with intelligent commands for development workflows. The architecture involves a three step approach;
1.  Command Loading
2.  Context Analysis
3.  Intelligent Planning

## Technical Details

### Design Philosophy

The Conversational Commands, and Build-Agnostic Instructions makes the plugin compatible with all languages and stacks.
### Architecture

All commands leverage Claude Code CLI's native capabilities. Safety-First Design, Conversational Interface, and Framework Agnostic principles contribute to the plugin's functionality.

### Advanced Features

**Session Continuity**
Commands such as `/implement` and `/refactor` maintain state across Claude sessions:

**Multi-Agent Architecture**
Complex commands orchestrate specialized sub-agents:
- Security analysis agent for vulnerability detection
- Performance optimization agent for bottleneck identification
- Architecture review agent for design pattern analysis
- Code quality agent for maintainability assessment

**Performance Optimizations**
- Reduced verbosity for senior developer efficiency
- Smart caching of project analysis results
- Incremental processing for large codebases
- Parallel execution of independent tasks

## Performance Metrics

| Task | Manual Time | With CCPlugins | Time Saved |
|------|-------------|----------------|------------|
| Security analysis | 45-60 min | 3-5 min | ~50 min |
| Architecture review | 30-45 min | 5-8 min | ~35 min |
| Feature scaffolding | 25-40 min | 2-3 min | ~30 min |
| Git commits | 5-10 min | 30 sec | ~9 min |
| Code cleanup | 20-30 min | 1 min | ~25 min |
| Import fixing | 15-25 min | 1-2 min | ~20 min |
| Code review | 20-30 min | 2-4 min | ~20 min |
| Issue prediction | 60+ min | 5-10 min | ~50 min |
| TODO resolution | 30-45 min | 3-5 min | ~35 min |
| Code adaptation | 40-60 min | 3-5 min | ~45 min |

**Total: 4-5 hours saved per week with professional-grade analysis**

## Requirements

*   Claude Code CLI
*   Python 3.6+ (for installer)
*   Git (for version control commands)

## Advanced Usage

*   **Creating Custom Commands:**  Extend CCPlugins with your own workflows by adding markdown files to `~/.claude/commands/`.
*   **Using Arguments:** Pass arguments to commands using `$ARGUMENTS`.
*   **CI/CD Integration:** Automate your pipelines with CCPlugins commands.
*   **Manual Workflow Integration:** Integrate into your daily development routine.

## Security & Git Instructions

All commands that interact with git include security instructions to prevent AI attribution, ensuring your code remains yours.
The `/commit` command will NEVER:
- Add "Co-authored-by" or AI signatures
- Include "Generated with Claude Code" messages
- Modify git config or credentials
- Add AI attribution to commits/issues

## Contributing

We welcome contributions!  See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

MIT License - see [LICENSE](LICENSE) file for details.
*Built by a developer tired of typing "please act like a senior engineer" in every conversation.*

## Community

[![Star History Chart](https://api.star-history.com/svg?repos=brennercruvinel/CCPlugins&type=Date)](https://star-history.com/#brennercruvinel/CCPlugins&Date)

---

**Last Updated:** August 2, 2025 (Based on Anthropic Claude Code CLI documentation v2025.08.01)
# CCPlugins: Supercharge Your Claude Code CLI Workflow with AI-Powered Automation

**Tired of repetitive development tasks?** CCPlugins extends your Claude Code CLI with powerful, AI-driven commands that automate common workflows and boost your productivity. [Explore the CCPlugins Repository](https://github.com/brennercruvinel/CCPlugins)

[![GitHub Repo stars](https://img.shields.io/github/stars/brennercruvinel/CCPlugins?style=social)](https://github.com/brennercruvinel/CCPlugins)
[![Version](https://img.shields.io/badge/version-2.5.2-blue.svg)](https://github.com/brennercruvinel/CCPlugins)
[![Claude Code CLI](https://img.shields.io/badge/for-Claude%20Code%20CLI-purple.svg)](https://docs.anthropic.com/en/docs/claude-code)
[![Tested on](https://img.shields.io/badge/tested%20on-Opus%204%20%26%20Sonnet%204-orange.svg)](https://claude.ai)
[![Also works with](https://img.shields.io/badge/also%20works%20with-Kimi%20K2-1783ff.svg)](https://github.com/MoonshotAI/Kimi-K2)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/brennercruvinel/CCPlugins/blob/main/CONTRIBUTING.md)

CCPlugins provides a curated set of professional commands that integrate seamlessly with Claude Code CLI, saving developers hours each week on tedious tasks and streamlining the development lifecycle. These commands are optimized for the Anthropic models Opus 4 and Sonnet 4, and work with Kimi K2, ensuring structured, predictable, and production-ready outcomes.

## Key Features

*   **Automated Workflows:** Automate routine tasks like code formatting, testing, and commit messages.
*   **Code Quality & Security:**  Enhance code quality and improve security through automated scans, review, and vulnerability analysis.
*   **Intelligent Analysis:** Deep code analysis for refactoring, architecture review, and proactive issue detection.
*   **Session Management:** Simplify project management with session start/end features, documentation, and task management.
*   **Validation & Refinement:** Complex commands include validation phases to ensure completeness.
*   **Extended Thinking:** Advanced analysis capabilities for complex scenarios, like security and refactoring.
*   **Pragmatic Command Integration:** Suggestions for workflow improvements, maintaining user control.

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

CCPlugins offers 24 professional commands to optimize your Claude Code CLI experience.

### Development Workflow

```bash
/cleanproject                    # Remove debug artifacts with git safety
/commit                          # Smart conventional commits with analysis
/format                          # Auto-detect and apply project formatter
/scaffold feature-name           # Generate complete features from patterns
/test                            # Run tests with intelligent failure analysis
/implement url/path/feature      # Import and adapt code from any source with validation phase
/refactor                        # Intelligent code restructuring with validation & de-para mapping
```

### Code Quality & Security

```bash
/review                # Multi-agent analysis (security, performance, quality, architecture)
/security-scan         # Vulnerability analysis with extended thinking & remediation tracking
/predict-issues        # Proactive problem detection with timeline estimates
/remove-comments       # Clean obvious comments, preserve valuable docs
/fix-imports           # Repair broken imports after refactoring
/find-todos            # Locate and organize development tasks
/create-todos          # Add contextual TODO comments based on analysis results
/fix-todos             # Intelligently implement TODO fixes with context
```

### Advanced Analysis

```bash
/understand            # Analyze entire project architecture and patterns
/explain-like-senior   # Senior-level code explanations with context
/contributing          # Complete contribution readiness analysis
/make-it-pretty        # Improve readability without functional changes
```

### Session & Project Management

```bash
/session-start         # Begin documented sessions with CLAUDE.md integration
/session-end           # Summarize and preserve session context
/docs                  # Smart documentation management and updates
/todos-to-issues       # Convert code TODOs to GitHub issues
/undo                  # Safe rollback with git checkpoint restore
```

## How It Works

CCPlugins empowers your Claude Code CLI experience through:

*   **Intelligent Instructions:** First-person conversational design, builds on collaborative reasoning, context-aware adaptations.
*   **Native Tool Integration:** Leverages Claude Code's built-in tools for efficient operations.
*   **Safety-First Design:** Automatic git checkpoints and rollback capabilities.
*   **Framework Agnostic:** Adapts to your project's conventions and patterns.

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
- `/commit`, `/scaffold`, `/make-it-pretty`, `/cleanproject`, `/fix-imports`, `/review`, `/security-scan`
- `/contributing`, `/todos-to-issues`, `/predict-issues`, `/find-todos`, `/create-todos`, `/fix-todos`

These commands will NEVER:
- Add "Co-authored-by" or AI signatures
- Include "Generated with Claude Code" messages
- Modify git config or credentials
- Add AI attribution to commits/issues

You can modify these instructions in individual command files if needed.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.
*Built by a developer tired of typing "please act like a senior engineer" in every conversation.*

## Community

[![Star History Chart](https://api.star-history.com/svg?repos=brennercruvinel/CCPlugins&type=Date)](https://star-history.com/#brennercruvinel/CCPlugins&Date)

---

**Last Updated:** August 2, 2025 (Based on Anthropic Claude Code CLI documentation v2025.08.01)
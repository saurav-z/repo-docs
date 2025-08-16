# CCPlugins: Supercharge Your Claude Code CLI for Faster Development

**Tired of repetitive coding tasks?** CCPlugins is a curated set of 24 professional commands that extends Claude Code CLI, saving developers hours each week on common development workflows. [See the original repo](https://github.com/brennercruvinel/CCPlugins).

[![GitHub Repo stars](https://img.shields.io/github/stars/brennercruvinel/CCPlugins?style=social)](https://github.com/brennercruvinel/CCPlugins)
[![Version](https://img.shields.io/badge/version-2.5.2-blue.svg)](https://github.com/brennercruvinel/CCPlugins)
[![Claude Code CLI](https://img.shields.io/badge/for-Claude%20Code%20CLI-purple.svg)](https://docs.anthropic.com/en/docs/claude-code)
[![Tested on](https://img.shields.io/badge/tested%20on-Opus%204%20%26%20Sonnet%204-orange.svg)](https://claude.ai)
[![Also works with](https://img.shields.io/badge/also%20works%20with-Kimi%20K2-1783ff.svg)](https://github.com/MoonshotAI/Kimi-K2)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/brennercruvinel/CCPlugins/blob/main/CONTRIBUTING.md)

**Key Features:**

*   **Automated Workflows:** Streamline development tasks like code formatting, testing, and committing.
*   **Code Quality & Security:** Improve code with vulnerability analysis, code reviews, and more.
*   **Advanced Analysis:** Gain deeper insights with project architecture analysis, senior-level code explanations, and proactive issue detection.
*   **Session & Project Management:** Manage development sessions and integrate with documentation and issue tracking.
*   **Validation & Refinement:** Built-in validation steps ensure completeness and accuracy in refactoring and implementation tasks.
*   **Multi-Agent Architecture:** Leverage specialized sub-agents for security, performance, and architecture analysis.
*   **Git-Safe by Design:** All commands that touch git are designed to prevent AI attribution.

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

A curated set of 24 professional commands optimized for Claude Code CLI:

### 🚀 Development Workflow

```bash
/cleanproject                    # Remove debug artifacts with git safety
/commit                          # Smart conventional commits with analysis
/format                          # Auto-detect and apply project formatter
/scaffold feature-name           # Generate complete features from patterns
/test                            # Run tests with intelligent failure analysis
/implement url/path/feature      # Import and adapt code from any source with validation phase
/refactor                        # Intelligent code restructuring with validation & de-para mapping
```

### 🛡️ Code Quality & Security

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

### 🔍 Advanced Analysis

```bash
/understand            # Analyze entire project architecture and patterns
/explain-like-senior   # Senior-level code explanations with context
/contributing          # Complete contribution readiness analysis
/make-it-pretty        # Improve readability without functional changes
```

### 📋 Session & Project Management

```bash
/session-start         # Begin documented sessions with CLAUDE.md integration
/session-end           # Summarize and preserve session context
/docs                  # Smart documentation management and updates
/todos-to-issues       # Convert code TODOs to GitHub issues
/undo                  # Safe rollback with git checkpoint restore
```

## Enhanced Features

*   **Validation & Refinement:** Ensures completeness and accuracy in key commands.
*   **Extended Thinking:** Utilizes advanced analysis for complex scenarios.
*   **Pragmatic Command Integration:** Provides natural workflow suggestions.

## Real-World Example

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

## How It Works

CCPlugins enhances Claude Code CLI through an architecture designed for developer productivity.

**Execution Flow**
When a command is typed:
1.  **Command Loading**: Claude reads the markdown definition from `~/.claude/commands/`
2.  **Context Analysis**: Analyzes your project structure, technology stack, and current state
3.  **Intelligent Planning**: Creates execution strategy based on your specific situation
4.  **Safe Execution**: Performs actions with automatic checkpoints and validation
5.  **Clear Feedback**: Provides results, next steps, and any warnings

## Technical Notes

*   **Design Philosophy:** Emphasizes simplicity, context awareness, and safety.
*   **Native Tool Integration:** Leverages Claude Code CLI's built-in tools.
*   **Security-First Design:** Includes git checkpoints.
*   **Conversational Interface:** Uses first-person language for improved model performance.

## Performance Metrics

| Task                | Manual Time | With CCPlugins | Time Saved |
| ------------------- | ----------- | -------------- | ---------- |
| Security analysis   | 45-60 min   | 3-5 min        | ~50 min    |
| Architecture review | 30-45 min   | 5-8 min        | ~35 min    |
| Feature scaffolding | 25-40 min   | 2-3 min        | ~30 min    |
| Git commits         | 5-10 min    | 30 sec         | ~9 min     |
| Code cleanup        | 20-30 min   | 1 min          | ~25 min    |
| Import fixing       | 15-25 min   | 1-2 min        | ~20 min    |
| Code review         | 20-30 min   | 2-4 min        | ~20 min    |
| Issue prediction    | 60+ min     | 5-10 min       | ~50 min    |
| TODO resolution     | 30-45 min   | 3-5 min        | ~35 min    |
| Code adaptation     | 40-60 min   | 3-5 min        | ~45 min    |

**Total: 4-5 hours saved per week with professional-grade analysis**

## Requirements

*   Claude Code CLI
*   Python 3.6+ (for installer)
*   Git (for version control commands)

## Advanced Usage

### Creating Custom Commands

Create custom commands by adding markdown files to `~/.claude/commands/`.

### Using Arguments

Commands support arguments via `$ARGUMENTS`.

### CI/CD Integration

Use commands in automated workflows.

### Manual Workflow Integration

Perfect for development routines.

## Security & Git Instructions

All commands that interact with git include security instructions to prevent AI attribution.

## Contributing

We welcome contributions.  See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) file for details.
*Built by a developer tired of typing "please act like a senior engineer" in every conversation.*

## Community

[![Star History Chart](https://api.star-history.com/svg?repos=brennercruvinel/CCPlugins&type=Date)](https://star-history.com/#brennercruvinel/CCPlugins&Date)

---

**Last Updated:** August 2, 2025 (Based on Anthropic Claude Code CLI documentation v2025.08.01)
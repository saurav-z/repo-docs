# Supercharge Your Development with CCPlugins: Automate Boring Tasks (and Save Time!)

Tired of repetitive tasks in your development workflow? **CCPlugins** is your solution, offering a suite of professional commands for the Claude Code CLI, designed to streamline your development process. ([Original Repository](https://github.com/brennercruvinel/CCPlugins))

[![GitHub Repo stars](https://img.shields.io/github/stars/brennercruvinel/CCPlugins?style=social)](https://github.com/brennercruvinel/CCPlugins)
[![Version](https://img.shields.io/badge/version-2.5.2-blue.svg)](https://github.com/brennercruvinel/CCPlugins)
[![Claude Code CLI](https://img.shields.io/badge/for-Claude%20Code%20CLI-purple.svg)](https://docs.anthropic.com/en/docs/claude-code)
[![Tested on](https://img.shields.io/badge/tested%20on-Opus%204%20%26%20Sonnet%204-orange.svg)](https://claude.ai)
[![Also works with](https://img.shields.io/badge/also%20works%20with-Kimi%20K2-1783ff.svg)](https://github.com/MoonshotAI/Kimi-K2)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/brennercruvinel/CCPlugins/blob/main/CONTRIBUTING.md)

## Key Features

*   **Automated Development Workflows:** Simplify common tasks like code formatting, committing, testing, and refactoring with a single command.
*   **Code Quality & Security:** Improve code maintainability and security with automated code reviews, vulnerability scans, and smart import fixing.
*   **Advanced Analysis & Insights:** Gain a deeper understanding of your codebase with project architecture analysis, senior-level code explanations, and proactive issue detection.
*   **Session & Project Management:** Enhance productivity with features like session logging, documentation updates, and streamlined issue creation.
*   **Validation & Refinement Phases:** Ensure the completeness of complex commands with built-in validation, guaranteeing the best possible results.

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

**Development Workflow:**

*   `/cleanproject` - Remove debug artifacts.
*   `/commit` - Smart commits.
*   `/format` - Auto-format your code.
*   `/scaffold feature-name` - Generate features from patterns.
*   `/test` - Run tests with failure analysis.
*   `/implement url/path/feature` - Import and adapt code.
*   `/refactor` - Intelligent code restructuring.

**Code Quality & Security:**

*   `/review` - Multi-agent code analysis.
*   `/security-scan` - Vulnerability analysis.
*   `/predict-issues` - Proactive issue detection.
*   `/remove-comments` - Remove comments.
*   `/fix-imports` - Repair broken imports.
*   `/find-todos` - Locate development tasks.
*   `/create-todos` - Add contextual TODO comments.
*   `/fix-todos` - Implement TODO fixes.

**Advanced Analysis:**

*   `/understand` - Analyze project architecture.
*   `/explain-like-senior` - Senior-level code explanations.
*   `/contributing` - Contribution readiness analysis.
*   `/make-it-pretty` - Improve readability.

**Session & Project Management:**

*   `/session-start` - Begin documented sessions.
*   `/session-end` - Summarize and preserve sessions.
*   `/docs` - Smart documentation management.
*   `/todos-to-issues` - Convert TODOs to GitHub issues.
*   `/undo` - Safe rollback with Git.

## Enhanced Features

*   **Validation & Refinement:** Validate and refine complex commands for completeness, like `/refactor validate` and `/implement validate`.
*   **Extended Thinking:** Advanced analysis for refactoring and security, with deep architectural analysis and sophisticated vulnerability detection.
*   **Pragmatic Command Integration:** Workflow suggestions, such as `/test` after changes and `/commit` at logical checkpoints, with user control.

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

CCPlugins transforms Claude Code CLI into a powerful development assistant.

1.  **Command Loading:** Claude reads the command definition from `~/.claude/commands/`.
2.  **Context Analysis:** Analyzes your project structure.
3.  **Intelligent Planning:** Creates an execution strategy.
4.  **Safe Execution:** Performs actions with checkpoints and validation.
5.  **Clear Feedback:** Provides results and warnings.

### Core Architecture Components

*   **Intelligent Instructions:** First-person language for collaborative reasoning.
*   **Native Tool Integration:** Utilizes Claude Code CLI's capabilities.
*   **Safety-First Design:** Automatic Git checkpoints and rollback capabilities.
*   **Universal Compatibility:** Framework-agnostic and cross-platform support.

### Advanced Features

*   **Session Continuity:** Commands like `/implement` and `/refactor` maintain state across Claude sessions.
*   **Multi-Agent Architecture:** Orchestrates specialized sub-agents for complex commands.
*   **Performance Optimizations:** Smart caching and incremental processing for large codebases.

## Technical Notes

### Design Philosophy

**Why This Approach Works:**

*   **Conversational Commands:** First-person language activates Claude's reasoning.
*   **Build-Agnostic Instructions:** No hardcoded tools.
*   **Think Tool Integration:** Strategic thinking improves decisions.
*   **Native Tools Only:** Uses Claude Code's capabilities.

**Key Principles:**

*   **Simplicity > Complexity:** Start simple.
*   **Context Awareness:** Adapt to your project.
*   **Safety First:** Git checkpoints.
*   **Pattern Recognition:** Learn from your codebase.

### Technical Architecture

**Native Tool Integration:**

Leverages Claude Code CLI's native capabilities:

*   Grep for pattern matching.
*   Glob for file discovery.
*   Read for content analysis.
*   TodoWrite for progress tracking.
*   Sub-agents for specialized analysis.

**Safety-First Design:**

```bash
git add -A
git commit -m "Pre-operation checkpoint" || echo "No changes to commit"
```

**Conversational Interface:** First-person language.
**Framework Agnostic:** Intelligent detection.

### User Commands Indicator

Custom commands appear with a `(user)` tag in Claude Code CLI to distinguish them from built-in commands. This is normal and indicates your commands are properly installed.

```
/commit
    Smart Git Commit (user)    ← Your custom command
/help
    Show help                  ← Built-in command
```

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

Create commands by adding markdown files to `~/.claude/commands/`:

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

Use commands in automated workflows. Examples are provided in the original README.

### Manual Workflow Integration

Perfect for development routines. Examples are provided in the original README.

## Security & Git Instructions

All commands interacting with Git include security instructions to prevent AI attribution.

**Commands with Git protection:**

*   `/commit`, `/scaffold`, `/make-it-pretty`, `/cleanproject`, `/fix-imports`, `/review`, `/security-scan`
*   `/contributing`, `/todos-to-issues`, `/predict-issues`, `/find-todos`, `/create-todos`, `/fix-todos`

These commands will NEVER:

*   Add "Co-authored-by" or AI signatures
*   Include "Generated with Claude Code" messages
*   Modify Git config or credentials
*   Add AI attribution to commits/issues

Modify instructions in command files if needed.

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.
*Built by a developer tired of typing "please act like a senior engineer" in every conversation.*

## Community

[![Star History Chart](https://api.star-history.com/svg?repos=brennercruvinel/CCPlugins&type=Date)](https://star-history.com/#brennercruvinel/CCPlugins&Date)

---

**Last Updated:** August 2, 2025 (Based on Anthropic Claude Code CLI documentation v2025.08.01)
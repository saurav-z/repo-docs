# CCPlugins: Supercharge Your Claude Code CLI for Faster Development 🚀

**Stop wasting time on repetitive coding tasks!** CCPlugins extends your Claude Code CLI with powerful commands to automate development workflows, save you hours, and elevate code quality. [Explore the CCPlugins Repository](https://github.com/brennercruvinel/CCPlugins)

[![GitHub Repo stars](https://img.shields.io/github/stars/brennercruvinel/CCPlugins?style=social)](https://github.com/brennercruvinel/CCPlugins)
[![Version](https://img.shields.io/badge/version-2.5.2-blue.svg)](https://github.com/brennercruvinel/CCPlugins)
[![Claude Code CLI](https://img.shields.io/badge/for-Claude%20Code%20CLI-purple.svg)](https://docs.anthropic.com/en/docs/claude-code)
[![Tested on](https://img.shields.io/badge/tested%20on-Opus%204%20%26%20Sonnet%204-orange.svg)](https://claude.ai)
[![Also works with](https://img.shields.io/badge/also%20works%20with-Kimi%20K2-1783ff.svg)](https://github.com/MoonshotAI/Kimi-K2)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/brennercruvinel/CCPlugins/blob/main/CONTRIBUTING.md)

## Key Features

*   **Automated Workflows:** Streamline development with commands for code generation, testing, refactoring, and more.
*   **Enhanced Code Quality:**  Improve code security and maintainability through advanced analysis and automated fixes.
*   **Time Savings:** Reduce development time by up to 5 hours per week with professional-grade automation.
*   **Context-Aware:** CCPlugins adapts to your project's unique needs and provides smart suggestions.
*   **Safe & Reliable:** Built with Git checkpoints and a focus on safety, minimizing risks during operations.
*   **Cross-Platform Support:** Works seamlessly on macOS, Windows, and Linux.

## Core Functionality

CCPlugins is a curated set of professional commands designed to enhance the Claude Code CLI, offering enterprise-grade development workflows. These commands leverage Claude's contextual understanding while providing structured, predictable outcomes optimized for Opus 4 and Sonnet 4 models.

###  Development Workflow Commands

*   `/cleanproject` - Remove debug artifacts with git safety
*   `/commit` - Smart conventional commits with analysis
*   `/format` - Auto-detect and apply project formatter
*   `/scaffold feature-name` - Generate complete features from patterns
*   `/test` - Run tests with intelligent failure analysis
*   `/implement url/path/feature` - Import and adapt code with validation
*   `/refactor` - Intelligent code restructuring with validation

### Code Quality and Security Commands

*   `/review` - Multi-agent analysis (security, performance, quality)
*   `/security-scan` - Vulnerability analysis and remediation tracking
*   `/predict-issues` - Proactive problem detection with estimates
*   `/remove-comments` - Clean obvious comments, preserve docs
*   `/fix-imports` - Repair broken imports after refactoring
*   `/find-todos` - Locate and organize development tasks
*   `/create-todos` - Add contextual TODO comments
*   `/fix-todos` - Intelligently implement TODO fixes

### Advanced Analysis Commands

*   `/understand` - Analyze project architecture and patterns
*   `/explain-like-senior` - Senior-level code explanations
*   `/contributing` - Contribution readiness analysis
*   `/make-it-pretty` - Improve readability without functional changes

### Session and Project Management Commands

*   `/session-start` - Begin documented sessions with CLAUDE.md
*   `/session-end` - Summarize and preserve session context
*   `/docs` - Smart documentation management and updates
*   `/todos-to-issues` - Convert code TODOs to GitHub issues
*   `/undo` - Safe rollback with git checkpoint restore

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

## Enhanced Capabilities

### Validation and Refinement

Complex commands include validation phases to ensure completeness:

*   `/refactor validate` - Find remaining old patterns and verify migration.
*   `/implement validate` - Check integration completeness, find loose ends.

### Extended Thinking

Advanced analysis for complex scenarios:

*   **Refactoring:** Deep architectural analysis for large-scale changes.
*   **Security:** Sophisticated vulnerability detection with chain analysis.

### Pragmatic Command Integration

Natural workflow suggestions without over-engineering:

*   Suggests `/test` after major changes.
*   Recommends `/commit` at logical checkpoints.
*   Maintains user control, no automatic execution.

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

CCPlugins transforms Claude Code CLI into an intelligent development assistant through a sophisticated yet elegant architecture:

```
Developer → /command → Claude Code CLI → Command Definition → Intelligent Execution
    ↑                                                                       ↓
    ←←←←←←←←←←←←←←←←← Clear Feedback & Results ←←←←←←←←←←←←←←←←←←←
```

### Execution Flow

1.  **Command Loading:** Claude reads the markdown definition from `~/.claude/commands/`.
2.  **Context Analysis:** Analyzes your project structure, technology stack, and current state.
3.  **Intelligent Planning:** Creates an execution strategy based on your situation.
4.  **Safe Execution:** Performs actions with automatic checkpoints and validation.
5.  **Clear Feedback:** Provides results, next steps, and any warnings.

## Technical Architecture and Notes

CCPlugins leverages Claude Code CLI's native capabilities to create intelligent commands:

*   **Intelligent Instructions:** First-person conversational design activates Claude's collaborative reasoning.
*   **Native Tool Integration:** Leverages `Grep`, `Glob`, `Read`, and `Write` tools for efficient operations.
*   **Safety-First Design:** Includes automatic git checkpoints, session persistence, and rollback capabilities.
*   **Universal Compatibility:** Framework-agnostic and adapts to your project conventions.

## Requirements

*   Claude Code CLI
*   Python 3.6+ (for installer)
*   Git (for version control commands)

## Advanced Usage

*   **Creating Custom Commands:** Create custom commands by adding markdown files to `~/.claude/commands/`.
*   **Using Arguments:** Commands support arguments via `$ARGUMENTS`.
*   **CI/CD Integration:** Seamlessly integrate commands into automated workflows.
*   **Manual Workflow Integration:** Perfect for development routines.

### CI/CD Example:

```bash
# Quality pipeline
claude "/security-scan" && claude "/review" && claude "/test"
```

## Contributing

We welcome contributions!  See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) file for details.
*Built by a developer tired of typing "please act like a senior engineer" in every conversation.*

## Community

[![Star History Chart](https://api.star-history.com/svg?repos=brennercruvinel/CCPlugins&type=Date)](https://star-history.com/#brennercruvinel/CCPlugins&Date)

---

**Last Updated:** August 2, 2025 (Based on Anthropic Claude Code CLI documentation v2025.08.01)
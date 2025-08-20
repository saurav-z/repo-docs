<!--  SEO-optimized README for CCPlugins -->

# Supercharge Your Claude Code CLI: Automate Development Tasks with CCPlugins

**Tired of repetitive coding tasks?** CCPlugins extends Claude Code CLI with powerful, professional commands to automate your development workflow, saving you hours each week.  [Explore the original repository](https://github.com/brennercruvinel/CCPlugins).

[![GitHub Repo stars](https://img.shields.io/github/stars/brennercruvinel/CCPlugins?style=social)](https://github.com/brennercruvinel/CCPlugins)
[![Version](https://img.shields.io/badge/version-2.5.2-blue.svg)](https://github.com/brennercruvinel/CCPlugins)
[![Claude Code CLI](https://img.shields.io/badge/for-Claude%20Code%20CLI-purple.svg)](https://docs.anthropic.com/en/docs/claude-code)
[![Tested on](https://img.shields.io/badge/tested%20on-Opus%204%20%26%20Sonnet%204-orange.svg)](https://claude.ai)
[![Also works with](https://img.shields.io/badge/also%20works%20with-Kimi%20K2-1783ff.svg)](https://github.com/MoonshotAI/Kimi-K2)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/brennercruvinel/CCPlugins/blob/main/CONTRIBUTING.md)

## Key Features of CCPlugins:

*   **Automated Workflows:** Streamline your development process with pre-built commands for common tasks.
*   **Enhanced Code Quality:** Improve code security, readability, and maintainability with ease.
*   **Intelligent Analysis:** Leverage advanced analysis tools for refactoring, security scanning, and architecture review.
*   **Validation & Refinement:** Ensure completeness and accuracy with built-in validation phases.
*   **Save Time:**  Reduce manual effort and boost your productivity, reclaiming hours each week.

## Quick Start:

### Installation:

*   **Mac/Linux:**
    ```bash
    curl -sSL https://raw.githubusercontent.com/brennercruvinel/CCPlugins/main/install.sh | bash
    ```

*   **Windows/Cross-platform:**
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

## Core Commands:

**Development Workflow:**

*   `/cleanproject`: Remove debug artifacts.
*   `/commit`: Smart conventional commits.
*   `/format`: Auto-detect and apply project formatter.
*   `/scaffold feature-name`: Generate features from patterns.
*   `/test`: Run tests with failure analysis.
*   `/implement url/path/feature`: Import and adapt code with validation.
*   `/refactor`: Intelligent code restructuring.

**Code Quality & Security:**

*   `/review`: Multi-agent analysis.
*   `/security-scan`: Vulnerability analysis.
*   `/predict-issues`: Proactive problem detection.
*   `/remove-comments`: Clean obvious comments.
*   `/fix-imports`: Repair broken imports.
*   `/find-todos`: Locate and organize tasks.
*   `/create-todos`: Add contextual TODOs.
*   `/fix-todos`: Implement TODO fixes with context.

**Advanced Analysis:**

*   `/understand`: Analyze project architecture.
*   `/explain-like-senior`: Senior-level code explanations.
*   `/contributing`: Contribution readiness analysis.
*   `/make-it-pretty`: Improve readability.

**Session & Project Management:**

*   `/session-start`: Begin documented sessions.
*   `/session-end`: Summarize and preserve context.
*   `/docs`: Smart documentation management.
*   `/todos-to-issues`: Convert TODOs to issues.
*   `/undo`: Safe rollback.

## How CCPlugins Works:

CCPlugins integrates seamlessly with Claude Code CLI, providing intelligent, context-aware commands that understand your project structure and automatically perform complex tasks.

### Architecture:

*   **Developer** → `/command` → **Claude Code CLI** → **Command Definition** → **Intelligent Execution** → **Clear Feedback & Results**.

### Key Components:

*   **Intelligent Instructions:** Uses first-person language.
*   **Native Tool Integration:** Leverages Claude Code CLI's core tools.
*   **Safety-First Design:** Automatic git checkpoints and rollbacks.
*   **Universal Compatibility:** Framework-agnostic, supports various languages.

## Example: Before and After `/cleanproject`:

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

## Technical Notes:

*   **Design Philosophy:** Simple, context-aware, safety-first, and pattern-driven.
*   **Architecture:** Leverages Claude Code CLI's native tools for efficient execution.
*   **User Commands Indicator:** Custom commands are tagged with (user).

## Performance Metrics:

(Example of Time Savings)

*   **Security analysis**: Saved ~50 minutes.
*   **Architecture review**: Saved ~35 minutes.
*   **Total**: Save 4-5 hours a week.

## Requirements:

*   Claude Code CLI
*   Python 3.6+ (for installer)
*   Git (for version control)

## Advanced Usage:

*   **Creating Custom Commands:**  Add markdown files to `~/.claude/commands/`.
*   **Using Arguments:** Access arguments via `$ARGUMENTS`.
*   **CI/CD Integration:** Automate tasks within your CI/CD pipelines.
*   **Manual Workflow Integration:** Integrate commands into your daily workflow.
*   **Security & Git Instructions:** Commits and issues are not AI attributed.

## Contributing:

Contributions are welcome!  See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License:

MIT License - See [LICENSE](LICENSE) for details.

---

**Last Updated:** August 2, 2025 (Based on Anthropic Claude Code CLI documentation v2025.08.01)
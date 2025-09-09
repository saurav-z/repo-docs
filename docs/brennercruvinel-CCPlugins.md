# Supercharge Your Development with CCPlugins: AI-Powered Commands for Claude Code CLI

**Tired of repetitive coding tasks and AI-induced over-engineering?** CCPlugins is a set of professional commands for Claude Code CLI that automate and optimize your development workflow, saving you valuable time and effort.  [Explore the original repository on GitHub](https://github.com/brennercruvinel/CCPlugins).

[![GitHub Repo stars](https://img.shields.io/github/stars/brennercruvinel/CCPlugins?style=social)](https://github.com/brennercruvinel/CCPlugins)
[![Version](https://img.shields.io/badge/version-2.5.2-blue.svg)](https://github.com/brennercruvinel/CCPlugins)
[![Claude Code CLI](https://img.shields.io/badge/for-Claude%20Code%20CLI-purple.svg)](https://docs.anthropic.com/en/docs/claude-code)
[![Tested on](https://img.shields.io/badge/tested%20on-Opus%204%20%26%20Sonnet%204-orange.svg)](https://claude.ai)
[![Also works with](https://img.shields.io/badge/also%20works%20with-Kimi%20K2-1783ff.svg)](https://github.com/MoonshotAI/Kimi-K2)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/brennercruvinel/CCPlugins/blob/main/CONTRIBUTING.md)

## Key Features

*   **Automated Workflows:** Streamline common development tasks like code formatting, testing, and refactoring.
*   **Enhanced Code Quality:** Improve code security and maintainability with built-in analysis and remediation commands.
*   **Time Savings:**  Reduce repetitive tasks and boost productivity, saving you potentially hours per week.
*   **Optimized for Claude Code CLI:** Leverage the power of AI for code generation, review, and improvement.
*   **Safe and Reliable:** Features include git integration with checkpoints, rollback capabilities, and no AI attribution.

## Getting Started

### Installation

**Quick Install (Mac/Linux):**

```bash
curl -sSL https://raw.githubusercontent.com/brennercruvinel/CCPlugins/main/install.sh | bash
```

**Quick Install (Windows/Cross-platform):**

```bash
python install.py
```

**Manual Install**

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

## Command Overview

*   **/cleanproject:** Remove debug artifacts with git safety
*   **/commit:** Smart conventional commits with analysis
*   **/format:** Auto-detect and apply project formatter
*   **/scaffold feature-name:** Generate complete features from patterns
*   **/test:** Run tests with intelligent failure analysis
*   **/implement url/path/feature:** Import and adapt code with validation phase
*   **/refactor:** Code restructuring with validation & de-para mapping
*   **/review:** Multi-agent analysis (security, performance, quality, architecture)
*   **/security-scan:** Vulnerability analysis with extended thinking & remediation
*   **/predict-issues:** Proactive problem detection with timeline estimates
*   **/remove-comments:** Clean obvious comments, preserve valuable docs
*   **/fix-imports:** Repair broken imports after refactoring
*   **/find-todos:** Locate and organize development tasks
*   **/create-todos:** Add contextual TODO comments based on analysis results
*   **/fix-todos:** Intelligently implement TODO fixes with context
*   **/understand:** Analyze entire project architecture and patterns
*   **/explain-like-senior:** Senior-level code explanations with context
*   **/contributing:** Complete contribution readiness analysis
*   **/make-it-pretty:** Improve readability without functional changes
*   **/session-start:** Begin documented sessions with CLAUDE.md integration
*   **/session-end:** Summarize and preserve session context
*   **/docs:** Smart documentation management and updates
*   **/todos-to-issues:** Convert code TODOs to GitHub issues
*   **/undo:** Safe rollback with git checkpoint restore

## Enhanced Features

### Validation and Refinement

Complex commands now include validation phases to ensure completeness:

*   `/refactor validate` - Verify 100% migration.
*   `/implement validate` - Check integration completeness, find loose ends.

### Extended Thinking

Advanced analysis for complex scenarios:

*   **Refactoring:** Deep architectural analysis for large-scale changes
*   **Security:** Sophisticated vulnerability detection with chain analysis

### Pragmatic Command Integration

Natural workflow suggestions without over-engineering:

*   Suggests `/test` after major changes
*   Recommends `/commit` at logical checkpoints
*   Maintains user control, no automatic execution

## How It Works

CCPlugins extends Claude Code CLI by providing a curated set of commands that leverage AI for automated tasks. The system is designed for safety, incorporating git checkpoints before any potentially destructive operation. Commands have validation phases to ensure complete, high quality results.

## Technical Architecture

### Core Components

*   **Intelligent Instructions**: First-person conversational design activates Claude's collaborative reasoning
*   **Native Tool Integration**: Integration with Claude Code CLI’s native tools like grep, glob, read, and write.
*   **Safety-First Design**: Automatic Git checkpoints before destructive operations
*   **Universal Compatibility**: Framework-agnostic with intelligent auto-detection

### Advanced Features

*   **Session Continuity**: Persisting state across multiple sessions for continuity
*   **Multi-Agent Architecture**: Orchestrating specialized sub-agents for various analysis types.
*   **Performance Optimizations**: Smart caching and incremental processing to improve efficiency

## Performance Metrics

| Task                    | Manual Time      | With CCPlugins     | Time Saved      |
| :---------------------- | :--------------- | :----------------- | :-------------- |
| Security analysis       | 45-60 min        | 3-5 min            | ~50 min         |
| Architecture review     | 30-45 min        | 5-8 min            | ~35 min         |
| Feature scaffolding     | 25-40 min        | 2-3 min            | ~30 min         |
| Git commits             | 5-10 min         | 30 sec             | ~9 min          |
| Code cleanup            | 20-30 min        | 1 min              | ~25 min         |
| Import fixing           | 15-25 min        | 1-2 min            | ~20 min         |
| Code review             | 20-30 min        | 2-4 min            | ~20 min         |
| Issue prediction        | 60+ min          | 5-10 min           | ~50 min         |
| TODO resolution         | 30-45 min        | 3-5 min            | ~35 min         |
| Code adaptation         | 40-60 min        | 3-5 min            | ~45 min         |
| **Total Savings** | **4-5 hours per week** |  |  |

## Requirements

*   Claude Code CLI
*   Python 3.6+ (for installer)
*   Git (for version control commands)

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

We welcome contributions!  See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

---

**Last Updated:** August 2, 2025 (Based on Anthropic Claude Code CLI documentation v2025.08.01)
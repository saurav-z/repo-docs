# CCPlugins: Supercharge Your Claude Code CLI Workflow

**Tired of repetitive coding tasks? CCPlugins offers professional commands for the Claude Code CLI, saving you hours each week.** Explore how to automate your development workflow with [CCPlugins](https://github.com/brennercruvinel/CCPlugins), the ultimate tool to streamline your coding experience.

[![GitHub Repo stars](https://img.shields.io/github/stars/brennercruvinel/CCPlugins?style=social)](https://github.com/brennercruvinel/CCPlugins)
[![Version](https://img.shields.io/badge/version-2.5.2-blue.svg)](https://github.com/brennercruvinel/CCPlugins)
[![Claude Code CLI](https://img.shields.io/badge/for-Claude%20Code%20CLI-purple.svg)](https://docs.anthropic.com/en/docs/claude-code)
[![Tested on](https://img.shields.io/badge/tested%20on-Opus%204%20%26%20Sonnet%204-orange.svg)](https://claude.ai)
[![Also works with](https://img.shields.io/badge/also%20works%20with-Kimi%20K2-1783ff.svg)](https://github.com/MoonshotAI/Kimi-K2)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/brennercruvinel/CCPlugins/blob/main/CONTRIBUTING.md)

## Key Features:

*   **Automated Development:** Simplify tasks like code formatting, testing, and project cleanup.
*   **Code Quality & Security:** Enhance your codebase with vulnerability scans, review tools, and proactive issue detection.
*   **Smart Code Analysis:** Understand project architecture and patterns with advanced analysis features.
*   **Enhanced Workflow:** Session management, documentation updates, and efficient TODO management.
*   **Time-Saving:** Save an estimated **4-5 hours per week** with professional-grade analysis and automation.

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

CCPlugins provides a comprehensive suite of commands to streamline your development process:

### 🚀 Development Workflow

*   `/cleanproject`: Remove debug artifacts safely with Git.
*   `/commit`: Create smart, conventional commits with analysis.
*   `/format`: Automatically detect and apply project formatting.
*   `/scaffold feature-name`: Generate complete features from predefined patterns.
*   `/test`: Run tests with intelligent failure analysis.
*   `/implement url/path/feature`: Import and adapt code from various sources with a validation phase.
*   `/refactor`: Restructure code intelligently with validation and de-para mapping.

### 🛡️ Code Quality & Security

*   `/review`: Utilize multi-agent analysis for security, performance, quality, and architecture.
*   `/security-scan`: Perform vulnerability analysis with detailed remediation tracking.
*   `/predict-issues`: Proactively detect potential problems and provide timeline estimates.
*   `/remove-comments`: Clean obvious comments, preserve valuable documentation.
*   `/fix-imports`: Repair broken imports after refactoring.
*   `/find-todos`: Locate and organize development tasks.
*   `/create-todos`: Add context-aware TODO comments based on analysis results.
*   `/fix-todos`: Intelligently implement TODO fixes with context.

### 🔍 Advanced Analysis

*   `/understand`: Analyze the entire project architecture and identify patterns.
*   `/explain-like-senior`: Get senior-level code explanations with context.
*   `/contributing`: Complete contribution readiness analysis.
*   `/make-it-pretty`: Improve readability without modifying functionality.

### 📋 Session & Project Management

*   `/session-start`: Begin documented sessions with CLAUDE.md integration.
*   `/session-end`: Summarize and preserve session context.
*   `/docs`: Smart documentation management and updates.
*   `/todos-to-issues`: Convert code TODOs to GitHub issues.
*   `/undo`: Safely roll back changes with Git checkpoint restore.

## Advanced Features

### Validation & Refinement

Commands like `/refactor` and `/implement` now incorporate validation phases to ensure complete and accurate code transformations:

```bash
/refactor validate  # Verify and validate transformations
/implement validate # Check integration completeness
```

### Extended Thinking

Advanced analysis for complex scenarios with deeper insight.

*   Refactoring: Deep architectural analysis for large-scale changes.
*   Security: Sophisticated vulnerability detection with chain analysis.

### Pragmatic Command Integration

Seamless workflow suggestions without unnecessary automation:

*   Suggestions to run `/test` after significant changes.
*   Recommendation to use `/commit` at strategic checkpoints.
*   Preserves user control, no automatic execution.

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

## Requirements

*   Claude Code CLI
*   Python 3.6+ (for installer)
*   Git (for version control commands)

## Contribute

We welcome contributions! Check out [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

**Last Updated:** August 2, 2025 (Based on Anthropic Claude Code CLI documentation v2025.08.01)
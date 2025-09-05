# CCPlugins: Supercharge Your Claude Code CLI Workflow (Get Started Today!)

**Tired of repetitive tasks and boilerplate code?** CCPlugins is your solution! This powerful set of professional commands instantly boosts your productivity by automating common development tasks within the Claude Code CLI. [Check out the CCPlugins GitHub repository for more details.](https://github.com/brennercruvinel/CCPlugins)

[![GitHub Repo stars](https://img.shields.io/github/stars/brennercruvinel/CCPlugins?style=social)](https://github.com/brennercruvinel/CCPlugins)
[![Version](https://img.shields.io/badge/version-2.5.2-blue.svg)](https://github.com/brennercruvinel/CCPlugins)
[![Claude Code CLI](https://img.shields.io/badge/for-Claude%20Code%20CLI-purple.svg)](https://docs.anthropic.com/en/docs/claude-code)
[![Tested on](https://img.shields.io/badge/tested%20on-Opus%204%20%26%20Sonnet%204-orange.svg)](https://claude.ai)
[![Also works with](https://img.shields.io/badge/also%20works%20with-Kimi%20K2-1783ff.svg)](https://github.com/MoonshotAI/Kimi-K2)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/brennercruvinel/CCPlugins/blob/main/CONTRIBUTING.md)

## Key Features:

*   **Automated Workflows:** Streamline your development with pre-built commands for common tasks.
*   **Code Quality & Security:** Improve code quality and identify vulnerabilities with ease.
*   **Intelligent Analysis:** Leverage advanced analysis for refactoring, security, and more.
*   **Smart Validation & Refinement:** Ensure completeness and accuracy with built-in validation.
*   **Enhanced Performance:** Save time with optimized commands and intelligent caching.
*   **Session Management:** Improve your efficiency with documentation & tracking commands.

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

## Commands (Examples)

*   `/cleanproject`: Remove debug artifacts safely.
*   `/commit`: Create smart, conventional commits.
*   `/format`: Auto-detect and apply project formatting.
*   `/refactor`: Restructure code intelligently.
*   `/security-scan`: Perform vulnerability analysis.
*   `/test`: Run tests with intelligent failure analysis.
*   `/implement`: Adapt code from any source with validation.

(See full list in original README)

## How it Works

CCPlugins extends Claude Code CLI with custom commands that perform complex development tasks. These commands use Claude's capabilities to analyze your code, plan execution, and ensure safety.

## Security & Git Instructions

CCPlugins commands are designed to NEVER add "Co-authored-by", include "Generated with Claude Code" or AI signatures, modify git configs or credentials, or add AI attribution to commits/issues.

##  Real World Example

(Same as in original README)

## Requirements

*   Claude Code CLI
*   Python 3.6+ (for installer)
*   Git (for version control commands)

## Contributing

We welcome contributions!  See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

**Last Updated:** August 2, 2025 (Based on Anthropic Claude Code CLI documentation v2025.08.01)
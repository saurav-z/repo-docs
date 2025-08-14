# CCPlugins: Supercharge Your Claude Code CLI for Faster Development

**Stop wasting time on repetitive tasks and over-engineered solutions.** CCPlugins provides professional commands for Claude Code CLI, saving developers up to 5 hours per week. [Explore the CCPlugins Repository](https://github.com/brennercruvinel/CCPlugins).

[![GitHub Repo stars](https://img.shields.io/github/stars/brennercruvinel/CCPlugins?style=social)]
[![Version](https://img.shields.io/badge/version-2.5.2-blue.svg)](https://github.com/brennercruvinel/CCPlugins)
[![Claude Code CLI](https://img.shields.io/badge/for-Claude%20Code%20CLI-purple.svg)](https://docs.anthropic.com/en/docs/claude-code)
[![Tested on](https://img.shields.io/badge/tested%20on-Opus%204%20%26%20Sonnet%204-orange.svg)](https://claude.ai)
[![Also works with](https://img.shields.io/badge/also%20works%20with-Kimi%20K2-1783ff.svg)](https://github.com/MoonshotAI/Kimi-K2)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/brennercruvinel/CCPlugins/blob/main/CONTRIBUTING.md)

## Key Features

*   **Automated Development Workflows:** Streamline common tasks like code formatting, testing, and project cleaning.
*   **Code Quality & Security Enhancement:** Identify and address vulnerabilities, review code for maintainability, and manage TODOs.
*   **Advanced Code Analysis:** Deeply understand your project's architecture, get senior-level code explanations, and proactively predict issues.
*   **Smart Session & Project Management:** Document your work with session tracking, convert TODOs to issues, and safely rollback changes.
*   **Validation & Refinement:** Complex commands include validation phases to ensure code completeness.
*   **Extended Thinking:** Advanced analysis for complex scenarios like refactoring and security, orchestrating sub-agents for comprehensive results.

## Quick Start

*   **Installation:**  Choose from the simple install method.

    **Mac/Linux:**
    ```bash
    curl -sSL https://raw.githubusercontent.com/brennercruvinel/CCPlugins/main/install.sh | bash
    ```

    **Windows/Cross-platform:**
    ```bash
    python install.py
    ```
*   **Uninstall:**  To remove CCPlugins.

    **Mac/Linux:**
    ```bash
    ./uninstall.sh
    ```

    **Windows/Cross-platform:**
    ```bash
    python uninstall.py
    ```

*   **Command Overview:**

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

## How It Works

CCPlugins enhances Claude Code CLI through its intelligent design that leverages Claude's capabilities, providing a sophisticated development assistant.

1.  **Command Loading:**  Claude reads command definitions.
2.  **Context Analysis:** Your project structure, and technology stack are analyzed.
3.  **Intelligent Planning:** Execution strategy is developed.
4.  **Safe Execution:** Actions are performed with automatic checkpoints and validation.
5.  **Clear Feedback:**  Results and next steps are provided.

## Requirements

*   Claude Code CLI
*   Python 3.6+ (for installer)
*   Git (for version control commands)

## Contributing

Help improve CCPlugins! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) file for details.
*Built by a developer tired of typing "please act like a senior engineer" in every conversation.*

---

**Last Updated:** August 2, 2025 (Based on Anthropic Claude Code CLI documentation v2025.08.01)
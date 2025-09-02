# CCPlugins: Supercharge Your Claude Code CLI Workflow (🚀 Automate Boring Stuff)

Tired of repetitive coding tasks? **CCPlugins** is a suite of professional commands designed to extend the power of Claude Code CLI, saving developers 2-3 hours per week on everyday development chores.  [Explore the original repository here](https://github.com/brennercruvinel/CCPlugins).

## Key Features:

*   🛠️ **Automated Development Workflows:** Streamline your coding process with commands for cleaning projects, smart commits, code formatting, scaffolding features, running tests, and code refactoring.
*   🛡️ **Enhanced Code Quality & Security:**  Ensure robust code with commands for code reviews, security scanning, proactive issue prediction, and improved code readability.
*   🔍 **Advanced Analysis & Session Management:**  Gain deeper insights with commands to understand project architecture, senior-level code explanations, and robust session management.
*   💡 **Validation & Refinement:** Complex commands now include validation phases, ensuring completeness and correctness in your code changes.
*   🤖 **Multi-Agent Architecture:** Leverages sub-agents for specialized tasks like security analysis, performance optimization, and code quality assessments.

## Installation

**Quick Installation:**

*   **Mac/Linux:**
    ```bash
    curl -sSL https://raw.githubusercontent.com/brennercruvinel/CCPlugins/main/install.sh | bash
    ```

*   **Windows/Cross-platform:**
    ```bash
    python install.py
    ```

**Manual Installation:**
```bash
git clone https://github.com/brennercruvinel/CCPlugins.git
cd CCPlugins
python install.py
```

**Uninstallation:**
*   **Mac/Linux:**
    ```bash
    ./uninstall.sh
    ```
*   **Windows/Cross-platform:**
    ```bash
    python uninstall.py
    ```

## Commands

CCPlugins provides a rich set of commands to enhance your Claude Code CLI experience:

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

## Example
(Simplified for brevity)

**Before `/cleanproject`:**
```
src/
├── UserService.js
├── UserService.test.js
├── UserService_backup.js    # Old version
└── debug.log               # Debug output
```

**After `/cleanproject`:**
```
src/
├── UserService.js          # Clean production code
└── UserService.test.js     # Actual tests preserved
```

## Learn More

*   [Contributing](CONTRIBUTING.md): Help improve CCPlugins.
*   [License](LICENSE): MIT License.
---

**Note:** This documentation has been updated to accurately represent CCPlugins' features as of August 2, 2025.
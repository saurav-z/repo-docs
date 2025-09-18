# CCPlugins: Supercharge Your Claude Code CLI Workflow (and Save Hours!)

**Tired of repetitive coding tasks?** CCPlugins provides enterprise-grade commands for Claude Code CLI, automating development and boosting your productivity.  [Explore CCPlugins on GitHub](https://github.com/brennercruvinel/CCPlugins)

[![GitHub Repo stars](https://img.shields.io/github/stars/brennercruvinel/CCPlugins?style=social)](https://github.com/brennercruvinel/CCPlugins)
[![Version](https://img.shields.io/badge/version-2.5.2-blue.svg)](https://github.com/brennercruvinel/CCPlugins)
[![Claude Code CLI](https://img.shields.io/badge/for-Claude%20Code%20CLI-purple.svg)](https://docs.anthropic.com/en/docs/claude-code)
[![Tested on](https://img.shields.io/badge/tested%20on-Opus%204%20%26%20Sonnet%204-orange.svg)](https://claude.ai)
[![Also works with](https://img.shields.io/badge/also%20works%20with-Kimi%20K2-1783ff.svg)](https://github.com/MoonshotAI/Kimi-K2)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/brennercruvinel/CCPlugins/blob/main/CONTRIBUTING.md)

## Key Features:

*   **Automated Workflows:** Streamline development with commands like `/cleanproject`, `/commit`, and `/test`.
*   **Code Quality & Security:** Enhance code with `/review`, `/security-scan`, and `/fix-imports`.
*   **Advanced Analysis:** Gain deeper insights with `/understand` and `/explain-like-senior`.
*   **Validation & Refinement:** Ensure complete and accurate results with built-in validation steps.
*   **Session Management:** Track progress with `/session-start` and `/session-end`.

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

CCPlugins provides 24 professional commands optimized for Claude Code CLI.

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

## 🚀 Boost Your Productivity: Save Up to 5 Hours a Week!

CCPlugins offers significant time savings, freeing up developers to focus on more strategic tasks.

## How It Works

*   **Intelligent Commands:** Leverages Claude Code CLI's power with first-person language.
*   **Native Tool Integration:** Uses Claude Code CLI's features for efficiency.
*   **Safety-First Design:** Includes Git checkpoints and rollback capabilities.
*   **Universal Compatibility:** Works with any programming language or framework.

## Technical Details

### Architecture

```
Developer → /command → Claude Code CLI → Command Definition → Intelligent Execution
    ↑                                                                       ↓
    ←←←←←←←←←←←←←←←←← Clear Feedback & Results ←←←←←←←←←←←←←←←←←←←
```

### Advanced Features

*   **Session Continuity:** Maintains state across Claude sessions.
*   **Multi-Agent Architecture:** Orchestrates specialized sub-agents for analysis.
*   **Performance Optimizations:** Caching, incremental processing, and parallel execution.

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

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) file for details.

---
**Last Updated:** August 2, 2025 (Based on Anthropic Claude Code CLI documentation v2025.08.01)
```

Key improvements and summarization:

*   **SEO Optimization:**  Includes target keywords like "Claude Code CLI," "automation," "development," "productivity," and "AI." The headings are also keyword-rich.
*   **Clear Hook:** Starts with a compelling sentence to grab attention.
*   **Concise Bullet Points:** Highlights the main features in an easy-to-scan format.
*   **Organized Structure:** Uses clear headings and subheadings for readability.
*   **Summary and Emphasis:** Condenses the original README, emphasizing the benefits.
*   **Call to Action:**  Encourages exploration of the GitHub repo.
*   **Removed Redundancy:**  Streamlined some of the more technical details.
*   **Direct Links:**  Keeps links to the GitHub repository readily accessible.
*   **Added Time Savings Section** A key selling point now features prominently.
*   **Consistent Formatting**: Ensures clear distinction between sections and improves readability.
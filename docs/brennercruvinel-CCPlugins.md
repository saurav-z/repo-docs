# CCPlugins: Supercharge Your Claude Code CLI with Enterprise-Grade Automation

**Tired of repetitive coding tasks?** CCPlugins enhances your Claude Code CLI experience, offering a suite of powerful commands to automate your development workflow.  [Explore CCPlugins on GitHub](https://github.com/brennercruvinel/CCPlugins)

[![GitHub Repo stars](https://img.shields.io/github/stars/brennercruvinel/CCPlugins?style=social)](https://github.com/brennercruvinel/CCPlugins)
[![Version](https://img.shields.io/badge/version-2.5.2-blue.svg)](https://github.com/brennercruvinel/CCPlugins)
[![Claude Code CLI](https://img.shields.io/badge/for-Claude%20Code%20CLI-purple.svg)](https://docs.anthropic.com/en/docs/claude-code)
[![Tested on](https://img.shields.io/badge/tested%20on-Opus%204%20%26%20Sonnet%204-orange.svg)](https://claude.ai)
[![Also works with](https://img.shields.io/badge/also%20works%20with-Kimi%20K2-1783ff.svg)](https://github.com/MoonshotAI/Kimi-K2)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/brennercruvinel/CCPlugins/blob/main/CONTRIBUTING.md)

## Key Features

*   **Automated Workflow:** Streamline your development cycle with commands for cleaning projects, committing code, and more.
*   **Code Quality & Security:** Improve code with built-in commands for code reviews, security scans, and fixing imports.
*   **Advanced Analysis:** Gain deeper insights into your projects with architecture analysis and senior-level code explanations.
*   **Validation & Refinement:** Ensure completeness with validation phases in key commands like `/refactor` and `/implement`.
*   **Real-World Time Savings:** Save an estimated 4-5 hours per week with automated analysis and workflow enhancements.
*   **Session Management:** Commands like `/session-start` and `/session-end` help you document and organize your work.
*   **Safe & Reliable:**  Built-in safety features like Git checkpoints and rollback capabilities protect your code.

## Quick Links

*   [🚀 Installation](#installation) - Get started in 30 seconds
*   [💻 Commands](#commands) - See all available commands
*   [🔧 How It Works](#how-it-works) - Understanding the magic
*   [🧠 Technical Notes](#technical-notes) - Why conversational design matters
*   [🤝 Contributing](#contributing) - Help make it better

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
24 professional commands optimized for Claude Code CLI's native capabilities with enhanced validation and refinement phases.

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

### 🔍 Validation & Refinement
Complex commands now include validation phases to ensure completeness:
```bash
/refactor validate   # Find remaining old patterns, verify 100% migration
/implement validate  # Check integration completeness, find loose ends
```

### 🧠 Extended Thinking
Advanced analysis for complex scenarios:
- **Refactoring**: Deep architectural analysis for large-scale changes
- **Security**: Sophisticated vulnerability detection with chain analysis

### 🔗 Pragmatic Command Integration
Natural workflow suggestions without over-engineering:
- Suggests `/test` after major changes
- Recommends `/commit` at logical checkpoints
- Maintains user control, no automatic execution

## Real World Example

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

## 🔧 How It Works

### High-Level Architecture

CCPlugins transforms Claude Code CLI into an intelligent development assistant through a sophisticated yet elegant architecture:

```
Developer → /command → Claude Code CLI → Command Definition → Intelligent Execution
    ↑                                                                       ↓
    ←←←←←←←←←←←←←←←←← Clear Feedback & Results ←←←←←←←←←←←←←←←←←←←
```

### Execution Flow

When you type a command:

1.  **Command Loading**: Claude reads the markdown definition from `~/.claude/commands/`
2.  **Context Analysis**: Analyzes your project structure, technology stack, and current state
3.  **Intelligent Planning**: Creates execution strategy based on your specific situation
4.  **Safe Execution**: Performs actions with automatic checkpoints and validation
5.  **Clear Feedback**: Provides results, next steps, and any warnings

### Core Architecture Components

**🧠 Intelligent Instructions**

*   First-person conversational design activates collaborative reasoning
*   Strategic thinking sections (`<think>`) for complex decision-making
*   Context-aware adaptations without hardcoded assumptions

**🔧 Native Tool Integration**

*   **Grep**: Lightning-fast pattern matching across codebases
*   **Glob**: Intelligent file discovery and project mapping
*   **Read**: Content analysis with full context understanding
*   **Write**: Safe file modifications with automatic backups
*   **TodoWrite**: Progress tracking and task management
*   **Task**: Sub-agent orchestration for specialized analysis

**🛡️ Safety-First Design**

*   Automatic git checkpoints before destructive operations
*   Session persistence for cross-context continuity
*   Rollback capabilities with clear recovery paths
*   No AI attribution in commits or generated content

**🌐 Universal Compatibility**

*   Framework-agnostic with intelligent auto-detection
*   Cross-platform support (Windows, Linux, macOS)
*   Works with any programming language or stack
*   Adapts to your project's conventions and patterns

### Advanced Features

**🔄 Session Continuity**
Commands like `/implement` and `/refactor` maintain state across Claude sessions:
```
# Each command creates its own folder in project root:
refactor/                  # Created by /refactor command
├── plan.md               # Refactoring roadmap
└── state.json            # Completed transformations

implement/                 # Created by /implement command
├── plan.md               # Implementation progress
└── state.json            # Session state and decisions

fix-imports/              # Created by /fix-imports command
├── plan.md               # Import fixes plan
└── state.json            # Resolution progress

security-scan/            # Created by /security-scan command
├── plan.md               # Vulnerabilities and fixes
└── state.json            # Remediation progress

scaffold/                 # Created by /scaffold command
├── plan.md               # Scaffolding plan
└── state.json            # Created files tracking
```

**🤖 Multi-Agent Architecture**
Complex commands orchestrate specialized sub-agents:
- Security analysis agent for vulnerability detection
- Performance optimization agent for bottleneck identification
- Architecture review agent for design pattern analysis
- Code quality agent for maintainability assessment

**📊 Performance Optimizations**
- Reduced verbosity for senior developer efficiency
- Smart caching of project analysis results
- Incremental processing for large codebases
- Parallel execution of independent tasks

## 🧠 Technical Notes

### Design Philosophy

**Why This Approach Works** (Based on Anthropic's Research):
- **Conversational Commands**: First-person language ("I'll help...") activates Claude's collaborative reasoning
- **Build-Agnostic Instructions**: No hardcoded tools = works everywhere
- **Think Tool Integration**: Strategic thinking improves decisions by 50%+ (Anthropic, 2025)
- **Native Tools Only**: Uses Claude Code's actual capabilities, not imaginary APIs

**Key Principles:**
- **Simplicity > Complexity**: Start simple, add only when proven necessary
- **Context Awareness**: Commands adapt to YOUR project, not vice versa
- **Safety First**: Git checkpoints before any destructive operation
- **Pattern Recognition**: Learn from your codebase, not assumptions

### Technical Architecture

**Native Tool Integration:**
All commands leverage Claude Code CLI's native capabilities:
- Grep tool for efficient pattern matching
- Glob tool for file discovery
- Read tool for content analysis
- TodoWrite for progress tracking
- Sub-agents for specialized analysis

**Safety-First Design:**
```bash
git add -A
git commit -m "Pre-operation checkpoint" || echo "No changes to commit"
```

**Conversational Interface:**
Commands use first-person collaborative language ("I'll analyze your code...") rather than imperative commands, creating a natural partnership interaction that improves model performance.

**Framework Agnostic:**
Intelligent detection without hardcoded assumptions enables universal compatibility across technology stacks.

### User Commands Indicator
Custom commands appear with a `(user)` tag in Claude Code CLI to distinguish them from built-in commands. This is normal and indicates your commands are properly installed.

```
/commit
    Smart Git Commit (user)    ← Your custom command
/help
    Show help                  ← Built-in command
```

## Performance Metrics

| Task | Manual Time | With CCPlugins | Time Saved |
|------|-------------|----------------|------------|
| Security analysis | 45-60 min | 3-5 min | ~50 min |
| Architecture review | 30-45 min | 5-8 min | ~35 min |
| Feature scaffolding | 25-40 min | 2-3 min | ~30 min |
| Git commits | 5-10 min | 30 sec | ~9 min |
| Code cleanup | 20-30 min | 1 min | ~25 min |
| Import fixing | 15-25 min | 1-2 min | ~20 min |
| Code review | 20-30 min | 2-4 min | ~20 min |
| Issue prediction | 60+ min | 5-10 min | ~50 min |
| TODO resolution | 30-45 min | 3-5 min | ~35 min |
| Code adaptation | 40-60 min | 3-5 min | ~45 min |

**Total: 4-5 hours saved per week with professional-grade analysis**

## Requirements

- Claude Code CLI
- Python 3.6+ (for installer)
- Git (for version control commands)

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

We welcome contributions that help developers save time. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) file for details.
*Built by a developer tired of typing "please act like a senior engineer" in every conversation.*

## Community

[![Star History Chart](https://api.star-history.com/svg?repos=brennercruvinel/CCPlugins&type=Date)](https://star-history.com/#brennercruvinel/CCPlugins&Date)

---

**Last Updated:** August 2, 2025 (Based on Anthropic Claude Code CLI documentation v2025.08.01)
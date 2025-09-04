# Context Engineering: Build AI-Powered Applications with Precision

**Revolutionize your AI coding workflow with Context Engineering, a system that provides AI assistants with comprehensive information to deliver superior results.** 🔗 [View the original repository](https://github.com/coleam00/context-engineering-intro)

## Key Features:

*   **Comprehensive Context:** Provide AI with documentation, examples, rules, and validation for accurate and consistent code generation.
*   **Reduced Failures:** Minimize AI agent failures by addressing context gaps, leading to more reliable implementations.
*   **Consistent Code:** Enforce project patterns and conventions to ensure uniformity across your codebase.
*   **Complex Implementations:** Enable AI to handle multi-step projects with validation loops and automated error correction.
*   **Automated Workflows:** Leverage automated Product Requirements Prompt (PRP) generation and execution for streamlined development.

## Quick Start

```bash
# 1. Clone this template
git clone https://github.com/coleam00/Context-Engineering-Intro.git
cd Context-Engineering-Intro

# 2. Set up your project rules (optional - template provided)
# Edit CLAUDE.md to add your project-specific guidelines

# 3. Add examples (highly recommended)
# Place relevant code examples in the examples/ folder

# 4. Create your initial feature request
# Edit INITIAL.md with your feature requirements

# 5. Generate a comprehensive PRP (Product Requirements Prompt)
# In Claude Code, run:
/generate-prp INITIAL.md

# 6. Execute the PRP to implement your feature
# In Claude Code, run:
/execute-prp PRPs/your-feature-name.md
```

## Table of Contents

*   [What is Context Engineering?](#what-is-context-engineering)
*   [Template Structure](#template-structure)
*   [Step-by-Step Guide](#step-by-step-guide)
*   [Writing Effective INITIAL.md Files](#writing-effective-initialmd-files)
*   [The PRP Workflow](#the-prp-workflow)
*   [Using Examples Effectively](#using-examples-effectively)
*   [Best Practices](#best-practices)

## What is Context Engineering?

Context Engineering elevates AI coding beyond basic prompt engineering by providing comprehensive context. This system is designed to overcome the limitations of prompt engineering.

### Prompt Engineering vs. Context Engineering

**Prompt Engineering:**

*   Focuses on clever wording and specific phrasing.
*   Limited to how you phrase a task.

**Context Engineering:**

*   Provides a complete system for comprehensive context.
*   Includes documentation, examples, rules, patterns, and validation.

### Why Context Engineering Matters

1.  **Reduces AI Failures:** Addresses context failures, the primary cause of agent failures.
2.  **Ensures Consistency:** Enforces project patterns and conventions.
3.  **Enables Complex Features:** Empowers AI to handle multi-step implementations.
4.  **Self-Correcting:** Utilizes validation loops to enable AI to fix its mistakes.

## Template Structure

```
context-engineering-intro/
├── .claude/
│   ├── commands/
│   │   ├── generate-prp.md    # Generates comprehensive PRPs
│   │   └── execute-prp.md     # Executes PRPs to implement features
│   └── settings.local.json    # Claude Code permissions
├── PRPs/
│   ├── templates/
│   │   └── prp_base.md       # Base template for PRPs
│   └── EXAMPLE_multi_agent_prp.md  # Example of a complete PRP
├── examples/                  # Your code examples (critical!)
├── CLAUDE.md                 # Global rules for AI assistant
├── INITIAL.md               # Template for feature requests
├── INITIAL_EXAMPLE.md       # Example feature request
└── README.md                # This file
```

## Step-by-Step Guide

Follow these steps to get started:

### 1.  Set Up Global Rules (CLAUDE.md)

The `CLAUDE.md` file contains project-wide rules. Customize these rules:

*   **Project Awareness:** Planning docs, task checking.
*   **Code Structure:** File size limits, module organization.
*   **Testing Requirements:** Unit test patterns, coverage expectations.
*   **Style Conventions:** Language preferences, formatting rules.
*   **Documentation Standards:** Docstring formats, commenting practices.

### 2.  Create Your Initial Feature Request

Edit `INITIAL.md` to describe what you want to build:

```markdown
## FEATURE:
[Describe what you want to build - be specific about functionality and requirements]

## EXAMPLES:
[List any example files in the examples/ folder and explain how they should be used]

## DOCUMENTATION:
[Include links to relevant documentation, APIs, or MCP server resources]

## OTHER CONSIDERATIONS:
[Mention any gotchas, specific requirements, or things AI assistants commonly miss]
```

See `INITIAL_EXAMPLE.md` for a complete example.

### 3.  Generate the PRP

PRPs (Product Requirements Prompts) are comprehensive implementation blueprints:

Run in Claude Code:

```bash
/generate-prp INITIAL.md
```

This command:

1.  Reads your feature request
2.  Researches the codebase for patterns
3.  Searches for relevant documentation
4.  Creates a comprehensive PRP in `PRPs/your-feature-name.md`

### 4.  Execute the PRP

```bash
/execute-prp PRPs/your-feature-name.md
```

This command:

1.  Reads all context from the PRP
2.  Creates a detailed implementation plan
3.  Executes each step with validation
4.  Runs tests and fixes any issues
5.  Ensures all success criteria are met

## Writing Effective INITIAL.md Files

Key Sections Explained:

*   **FEATURE:** Be specific and comprehensive
*   **EXAMPLES:** Leverage the `examples/` folder
*   **DOCUMENTATION:** Include relevant resources (APIs, guides, schemas)
*   **OTHER CONSIDERATIONS:** Include authentication, rate limits, pitfalls, etc.

## The PRP Workflow

### How /generate-prp Works

1.  **Research Phase:** Analyzes your codebase, searches for similar implementations.
2.  **Documentation Gathering:** Fetches relevant API docs, includes library documentation.
3.  **Blueprint Creation:** Creates step-by-step implementation plan, includes validation gates and test requirements.
4.  **Quality Check:** Scores confidence level, ensures all context is included.

### How /execute-prp Works

1.  Load Context
2.  Plan
3.  Execute
4.  Validate
5.  Iterate
6.  Complete

See `PRPs/EXAMPLE_multi_agent_prp.md` for a complete example.

## Using Examples Effectively

The `examples/` folder is **critical** for success.

### What to Include in Examples

1.  Code Structure Patterns
2.  Testing Patterns
3.  Integration Patterns
4.  CLI Patterns

## Best Practices

### 1. Be Explicit in INITIAL.md

### 2. Provide Comprehensive Examples

### 3. Use Validation Gates

### 4. Leverage Documentation

### 5. Customize CLAUDE.md

## Resources

*   [Claude Code Documentation](https://docs.anthropic.com/en/docs/claude-code)
*   [Context Engineering Best Practices](https://www.philschmid.de/context-engineering)
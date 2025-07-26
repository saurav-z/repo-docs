# Context Engineering Template: Build AI-powered code with comprehensive context.

Tired of prompt engineering? **This template provides a robust framework for Context Engineering, allowing you to provide AI coding assistants with the necessary information to generate complete, consistent, and high-quality code.**

[![GitHub Repo stars](https://img.shields.io/github/stars/coleam00/context-engineering-intro?style=social)](https://github.com/coleam00/context-engineering-intro)

**[View the original repository](https://github.com/coleam00/context-engineering-intro)**

## Key Features:

*   **Comprehensive Context:** Provide AI with documentation, examples, rules, and validation for superior code generation.
*   **Reduced AI Failures:** Address the root cause of failures by giving the AI the context it needs to succeed.
*   **Project-Specific Consistency:** Enforce your project's patterns and conventions for standardized code.
*   **Enables Complex Features:** Facilitates multi-step implementations through well-defined context and instructions.
*   **Self-Correcting Capabilities:** Validation loops allow AI to identify and resolve its own mistakes.

## Quick Start

```bash
# 1. Clone this template
git clone https://github.com/coleam00/Context-Engineering-Intro.git
cd Context-Engineering-Intro

# 2. Set up project rules (optional - template provided)
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
*   [Resources](#resources)

## What is Context Engineering?

Context Engineering goes beyond basic prompt engineering by providing a complete system for giving AI the context it needs. This involves documentation, code examples, project-specific rules, and validation steps.

### Prompt Engineering vs. Context Engineering

**Prompt Engineering:**

*   Focuses on clever wording and phrasing.
*   Limited in the amount of information it can convey.

**Context Engineering:**

*   Provides a comprehensive system for project-specific context.
*   Includes documentation, examples, project rules, and validation.

### Why Context Engineering Matters

1.  **Reduce AI Failures:** Most failures are due to insufficient context.
2.  **Ensure Consistency:** Maintain project-wide patterns and conventions.
3.  **Enable Complex Features:** Context engineering allows for complex, multi-step implementations.
4.  **Self-Correcting:** Validation loops allow AI to resolve its own errors.

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

### 1. Set Up Global Rules (CLAUDE.md)

`CLAUDE.md` contains project-wide rules that the AI assistant will follow, which can include:

*   Project awareness and task management.
*   Code structure guidelines.
*   Testing requirements.
*   Style and documentation conventions.

**Customize the template to meet your specific project needs.**

### 2. Create Your Initial Feature Request

Edit `INITIAL.md` to describe the desired feature:

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

**See `INITIAL_EXAMPLE.md` for a complete example.**

### 3. Generate the PRP

PRPs (Product Requirements Prompts) are comprehensive implementation plans that include:

*   Detailed context and documentation.
*   Implementation steps with validation.
*   Error handling patterns.
*   Test requirements.

Run in Claude Code:

```bash
/generate-prp INITIAL.md
```

This command does the following:

1.  Reads the feature request.
2.  Researches the codebase for existing patterns.
3.  Searches for relevant documentation.
4.  Creates a comprehensive PRP in `PRPs/your-feature-name.md`.

### 4. Execute the PRP

Execute the generated PRP to implement your feature:

```bash
/execute-prp PRPs/your-feature-name.md
```

The AI assistant will:

1.  Read all context from the PRP.
2.  Create a detailed implementation plan.
3.  Execute each step, including validation.
4.  Run tests and fix any issues.
5.  Ensure all success criteria are met.

## Writing Effective INITIAL.md Files

### Key Sections Explained

**FEATURE**: Be specific and comprehensive in describing the desired feature.

*   **Example:** Instead of "Build a web scraper", use "Build an async web scraper using BeautifulSoup that extracts product data from e-commerce sites, handles rate limiting, and stores results in PostgreSQL".

**EXAMPLES**: Leverage the `/examples` folder.

*   Reference code patterns within the `/examples` folder.
*   Explain how examples should be used.

**DOCUMENTATION**: Provide all relevant resources.

*   Include API documentation URLs, library guides, and database schemas.

**OTHER CONSIDERATIONS**: Capture important details.

*   Include authentication requirements, rate limits, common pitfalls, and performance requirements.

## The PRP Workflow

### How /generate-prp Works

The command performs the following steps:

1.  **Research Phase:** Analyzes your codebase, searches for similar implementations, and identifies conventions.
2.  **Documentation Gathering:** Fetches relevant API docs and adds related information.
3.  **Blueprint Creation:** Creates a step-by-step implementation plan, including validation gates and test requirements.
4.  **Quality Check:** Scores the confidence level and ensures all context is included.

### How /execute-prp Works

1.  **Load Context:** Reads the PRP.
2.  **Plan:** Creates a detailed task list.
3.  **Execute:** Implements each component.
4.  **Validate:** Runs tests and linting.
5.  **Iterate:** Fixes any issues.
6.  **Complete:** Ensures all requirements are met.

## Using Examples Effectively

The `/examples/` folder is **essential** for the success of your AI-powered code generation.

### What to Include in Examples

1.  **Code Structure Patterns**
    *   Module organization, import conventions, and class/function patterns
2.  **Testing Patterns**
    *   Test file structure, mocking approaches, and assertion styles
3.  **Integration Patterns**
    *   API client implementations, database connections, and authentication flows
4.  **CLI Patterns**
    *   Argument parsing, output formatting, and error handling

### Example Structure

```
examples/
├── README.md           # Explains what each example demonstrates
├── cli.py             # CLI implementation pattern
├── agent/             # Agent architecture patterns
│   ├── agent.py      # Agent creation pattern
│   ├── tools.py      # Tool implementation pattern
│   └── providers.py  # Multi-provider pattern
└── tests/            # Testing patterns
    ├── test_agent.py # Unit test patterns
    └── conftest.py   # Pytest configuration
```

## Best Practices

### 1. Be Explicit in INITIAL.md

*   Include specific requirements, don't make assumptions about AI knowledge, and reference examples liberally.

### 2. Provide Comprehensive Examples

*   More examples lead to better implementations, by demonstrating best practices, what to do, and what not to do.  Include error-handling patterns.

### 3. Use Validation Gates

*   PRPs include test commands that must pass to ensure working code from the start.

### 4. Leverage Documentation

*   Include official API docs, and MCP server resources.

### 5. Customize CLAUDE.md

*   Add your specific conventions, include project-specific rules, and define coding standards.

## Resources

*   [Claude Code Documentation](https://docs.anthropic.com/en/docs/claude-code)
*   [Context Engineering Best Practices](https://www.philschmid.de/context-engineering)
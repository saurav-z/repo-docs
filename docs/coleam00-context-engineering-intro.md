# Context Engineering: Supercharge AI Coding with Comprehensive Context

**Unlock 10x better code generation by providing AI assistants with the complete context they need to excel.**

[View the Original Repository](https://github.com/coleam00/context-engineering-intro)

## Key Features

*   **Comprehensive Context:** Move beyond simple prompts and equip AI with documentation, examples, and rules.
*   **Reduced AI Failures:** Address the root cause of AI errors by providing all necessary context.
*   **Consistent Code Generation:** Ensure your AI assistant adheres to your project's patterns and conventions.
*   **Enable Complex Features:** Facilitate multi-step implementations with validated context.
*   **Self-Correcting Mechanism:** Leverage validation loops to allow AI to refine its work.

## Getting Started

Follow these simple steps to begin using Context Engineering:

```bash
# 1. Clone the template
git clone https://github.com/coleam00/Context-Engineering-Intro.git
cd Context-Engineering-Intro

# 2. Set up your project rules (Optional)
# Edit CLAUDE.md to add your project-specific guidelines

# 3. Add examples (Highly Recommended)
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

Context Engineering revolutionizes AI coding by shifting focus from prompt engineering to providing a complete system of context. It's a robust solution for streamlining the coding process.

### Prompt Engineering vs. Context Engineering

**Prompt Engineering:**

*   Relies on clever phrasing and specific word choices.
*   Limited by the way you phrase the task.
*   Analogous to a sticky note.

**Context Engineering:**

*   Provides comprehensive context.
*   Includes documentation, examples, rules, patterns, and validation.
*   Comparable to writing a detailed screenplay with all the necessary details.

### Why Context Engineering Matters

1.  **Reduce AI Failures**: Resolves issues related to agent failures by providing relevant context.
2.  **Ensure Consistency**: Guarantees that AI follows your project standards and conventions.
3.  **Enable Complex Features**: Supports the management of intricate, multi-step projects with correct context.
4.  **Self-Correcting**: Uses validation loops to help AI to fix and refine its own work.

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

The `CLAUDE.md` file defines project-wide rules for the AI assistant. The template includes:

*   **Project Awareness**: Understands planning docs, checks tasks
*   **Code Structure**: Follows file size limits, and module organization.
*   **Testing Requirements**: Uses unit test patterns and coverage expectations.
*   **Style Conventions**: Adheres to language preferences and formatting guidelines.
*   **Documentation Standards**: Follows docstring formats and commenting practices.

**Customize `CLAUDE.md` to suit your project's specific needs.**

### 2. Create Your Initial Feature Request

Describe your desired feature in `INITIAL.md`:

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

**See `INITIAL_EXAMPLE.md` for a comprehensive example.**

### 3. Generate the PRP (Product Requirements Prompt)

PRPs are detailed blueprints that guide implementation:

*   Provide complete context and documentation
*   Define implementation steps and validation
*   Include error handling patterns
*   Specify test requirements

Run the following in Claude Code:

```bash
/generate-prp INITIAL.md
```

**Note:** The slash commands are defined in `.claude/commands/`. Inspect their implementation for details.

This command will:

1.  Read your feature request.
2.  Analyze your codebase for patterns.
3.  Locate relevant documentation.
4.  Create a PRP in `PRPs/your-feature-name.md`.

### 4. Execute the PRP

Implement your feature by executing the generated PRP:

```bash
/execute-prp PRPs/your-feature-name.md
```

The AI assistant will:

1.  Read all context from the PRP.
2.  Create an implementation plan.
3.  Execute each step with validation.
4.  Run tests and address any issues.
5.  Ensure all success criteria are met.

## Writing Effective INITIAL.md Files

### Key Sections Explained

**FEATURE**: Be specific and detailed.

*   ❌ "Build a web scraper"
*   ✅ "Build an async web scraper using BeautifulSoup that extracts product data from e-commerce sites, handles rate limiting, and stores results in PostgreSQL"

**EXAMPLES**: Utilize the examples/ folder.

*   Place code patterns in `examples/`.
*   Reference specific files and patterns.
*   Explain how to mimic these patterns.

**DOCUMENTATION**: Include all relevant resources.

*   Provide API documentation URLs.
*   Include library guides.
*   Add MCP server documentation.
*   Detail database schemas.

**OTHER CONSIDERATIONS**: Capture important details.

*   Outline authentication requirements.
*   Specify rate limits or quotas.
*   Note common pitfalls.
*   Define performance requirements.

## The PRP Workflow

### How `/generate-prp` Works

The command follows these steps:

1.  **Research Phase**: Analyzes the codebase, searches for similar implementations, and identifies conventions.
2.  **Documentation Gathering**: Retrieves relevant API docs and includes gotchas.
3.  **Blueprint Creation**: Creates a step-by-step implementation plan with validation gates and test requirements.
4.  **Quality Check**: Assesses confidence and ensures that all context is included.

### How `/execute-prp` Works

1.  **Load Context**: Reads the entire PRP.
2.  **Plan**: Creates a task list.
3.  **Execute**: Implements components.
4.  **Validate**: Runs tests and linting.
5.  **Iterate**: Fixes any identified issues.
6.  **Complete**: Ensures all requirements are met.

See `PRPs/EXAMPLE_multi_agent_prp.md` for an example of a generated PRP.

## Using Examples Effectively

The `examples/` folder is vital for successful implementations.

### What to Include in Examples

1.  **Code Structure Patterns**: Module organization, import conventions, and class/function patterns.
2.  **Testing Patterns**: Test file structure, mocking approaches, and assertion styles.
3.  **Integration Patterns**: API client implementations, database connections, and authentication flows.
4.  **CLI Patterns**: Argument parsing, output formatting, and error handling.

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

*   Specify your preferences.
*   Include specific requirements and constraints.
*   Reference examples frequently.

### 2. Provide Comprehensive Examples

*   More examples result in better implementations.
*   Demonstrate both what to do and what not to do.
*   Include error handling patterns.

### 3. Use Validation Gates

*   PRPs incorporate test commands that must pass.
*   The AI iterates until validation is successful.
*   This ensures working code from the start.

### 4. Leverage Documentation

*   Incorporate official API docs.
*   Add MCP server resources.
*   Reference specific documentation sections.

### 5. Customize CLAUDE.md

*   Add your project conventions.
*   Include project-specific rules.
*   Define coding standards.

## Resources

*   [Claude Code Documentation](https://docs.anthropic.com/en/docs/claude-code)
*   [Context Engineering Best Practices](https://www.philschmid.de/context-engineering)